"""Fused int4-g64 dequantisation for the packed RV1 artifact (build_rv1_4b_q4b).

WHY THIS FILE EXISTS
--------------------
`loader.py` dequantises a whole tensor into a bf16 weight and hands it back. Applied to 248 host
tensors that builds a 17.1 GiB bf16 state dict, which is DIRTY anonymous memory on MPS and cannot
be evicted -- a 7.71 GiB card (and, on this box, an 18.1 GiB-free window) refuses it.

Here the weight is never materialised. `PackedLinear` holds the artifact's OWN bytes:
  * `packed`  uint8 (out, in/2)  -- two int4 per byte, low nibble = 2i, high = 2i+1
  * `scales`  fp16  (out, in/64) -- absmax/7 per 64-wide group along the INPUT dim
and `forward` dequantises a bounded slice of OUTPUT rows, multiplies, and drops it. Peak transient
is `chunk_rows * in * ~6 bytes` (a few tens of MiB), independent of tensor size.

Numerics are the artifact's own code path, unchanged: the same nibble order, the same
`w = clip(round(w/s), -8, 7) * s` reconstruction in fp32, the same fp32->bf16 rounding as
`loader.dequant`. Chunking is over OUTPUT rows, so every output element is one complete dot
product -- chunk boundaries never split an accumulation.

Layers 0-10 (the compiled chip) are `kind="raw"` in the manifest and are loaded verbatim, bf16,
untouched. The embedding's structural columns (2^40 ballast + +-128 code dims) are stored verbatim
by the packer and are re-interleaved here exactly as `loader.load_state_dict` does; only the live
columns are int4, and they are dequantised *per looked-up row*, so the 3.7 GiB bf16 embedding
never exists either.
"""
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch as st

ART = os.environ.get("QWENPUTER_ART","build_rv1_4b_q4b")
GROUP = 64
CHUNK_BYTES = int(float(os.environ.get("FUSED_CHUNK_MIB", "48")) * 1024 * 1024)  # transient per matmul

# instrumentation: the largest single transient this module ever allocated
_TRANSIENT = {"peak_bytes": 0, "chunks": 0, "calls": 0}


def _note_transient(nbytes):
    if nbytes > _TRANSIENT["peak_bytes"]:
        _TRANSIENT["peak_bytes"] = nbytes
    _TRANSIENT["chunks"] += 1


def transient_peak_mib():
    return _TRANSIENT["peak_bytes"] / 1024 ** 2


def reset_transient():
    _TRANSIENT.update(peak_bytes=0, chunks=0, calls=0)


# --------------------------------------------------------------------------------------- kernels
def dequant_rows(packed, scales, group=GROUP, out_dtype=torch.bfloat16):
    """(r, in/2) uint8 + (r, in/64) fp16 -> (r, in) bf16.  Same arithmetic as loader.dequant."""
    r, hp = packed.shape
    i = hp * 2
    lo = (packed & 0x0F).to(torch.int8)
    hi = ((packed >> 4) & 0x0F).to(torch.int8)
    q = torch.stack((lo, hi), dim=-1).reshape(r, i)
    q = torch.where(q > 7, q - 16, q)
    w = q.reshape(r, i // group, group).to(torch.float32) * scales.to(torch.float32).unsqueeze(-1)
    return w.reshape(r, i).to(out_dtype)


def _chunk_rows(out_features, in_features, chunk_bytes=CHUNK_BYTES):
    # int8 q + fp32 product + bf16 weight  ~= 7 bytes per element of the slice
    per = max(1, in_features * 7)
    return max(16, min(out_features, int(chunk_bytes) // per))


def packed_linear(x, packed, scales, bias=None, chunk_bytes=CHUNK_BYTES):
    """y = x @ W.T with W never materialised; W is dequantised one output-row block at a time."""
    o, hp = packed.shape
    i = hp * 2
    if x.shape[-1] != i:
        raise ValueError(f"packed in_features {i} != x last dim {x.shape[-1]}")
    lead = x.shape[:-1]
    x2 = x.reshape(-1, i)
    y = torch.empty((x2.shape[0], o), dtype=x.dtype, device=x.device)
    r = _chunk_rows(o, i, chunk_bytes)
    _TRANSIENT["calls"] += 1
    for c0 in range(0, o, r):
        c1 = min(o, c0 + r)
        w = dequant_rows(packed[c0:c1], scales[c0:c1])
        _note_transient((c1 - c0) * i * 7)
        y[:, c0:c1] = F.linear(x2, w)
        del w
    if bias is not None:
        y = y + bias
    return y.reshape(*lead, o)


# --------------------------------------------------------------------------------------- modules
class PackedLinear(nn.Module):
    """Drop-in for a bias-free nn.Linear whose weight stays int4-packed in memory."""

    def __init__(self, out_features, in_features, packed, scales, chunk_bytes=CHUNK_BYTES):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chunk_bytes = chunk_bytes
        self.register_buffer("packed", packed, persistent=False)   # uint8 (out, in/2)
        self.register_buffer("scales", scales, persistent=False)   # fp16  (out, in/64)

    def forward(self, x):
        return packed_linear(x, self.packed, self.scales, None, self.chunk_bytes)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"int4-g{GROUP} packed, bias=False")


class PackedEmbedding(nn.Module):
    """embed_tokens with the live columns int4-packed and the structural columns verbatim.

    A lookup only touches the rows asked for, so the dequantised 478656 x 3904 bf16 table
    (3.7 GiB) never exists.
    """

    def __init__(self, packed, scales, struct, struct_cols, n_live, hidden, chunk_bytes=CHUNK_BYTES):
        super().__init__()
        self.n_live = n_live                      # real live columns (pad cols are dropped)
        self.hidden = hidden
        self.struct_cols = list(struct_cols)
        self.live_cols = [c for c in range(hidden) if c not in set(self.struct_cols)]
        self.register_buffer("packed", packed, persistent=False)   # uint8 (vocab, live_pad/2)
        self.register_buffer("scales", scales, persistent=False)   # fp16  (vocab, live_pad/64)
        self.register_buffer("struct", struct, persistent=False)   # bf16  (vocab, n_struct)
        self.register_buffer("_live_idx", torch.tensor(self.live_cols, dtype=torch.long),
                             persistent=False)
        self.register_buffer("_struct_idx", torch.tensor(self.struct_cols, dtype=torch.long),
                             persistent=False)

    def forward(self, ids):
        p = self.packed.index_select(0, ids.reshape(-1))
        s = self.scales.index_select(0, ids.reshape(-1))
        live = dequant_rows(p, s)[:, : self.n_live]
        del p, s
        out = torch.empty(ids.numel(), self.hidden, dtype=self.struct.dtype, device=ids.device)
        out[:, self._live_idx] = live
        out[:, self._struct_idx] = self.struct.index_select(0, ids.reshape(-1))
        del live
        return out.reshape(*ids.shape, self.hidden)

    def extra_repr(self):
        return (f"vocab={self.packed.shape[0]}, hidden={self.hidden}, "
                f"live={self.n_live} int4-g{GROUP}, structural={len(self.struct_cols)} verbatim")


class PackedTiedHead(nn.Module):
    """lm_head tied to a PackedEmbedding: logits = x @ W.T, W never materialised."""

    def __init__(self, emb: PackedEmbedding, chunk_bytes=CHUNK_BYTES):
        super().__init__()
        self.emb = emb
        self.chunk_bytes = chunk_bytes

    def forward(self, x):
        e = self.emb
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.empty((x2.shape[0], e.packed.shape[0]), dtype=x.dtype, device=x.device)
        r = _chunk_rows(out.shape[1], e.packed.shape[1] * 2, self.chunk_bytes)
        xl = x2[:, e._live_idx]
        for c0 in range(0, out.shape[1], r):
            c1 = min(out.shape[1], c0 + r)
            w = dequant_rows(e.packed[c0:c1], e.scales[c0:c1])[:, : e.n_live]
            _note_transient((c1 - c0) * e.n_live * 7)
            out[:, c0:c1] = F.linear(xl, w)
            del w
        # structural columns contribute exactly; 82 dims, evaluated in one small matmul
        out += F.linear(x2[:, e._struct_idx], e.struct)
        return out.reshape(*lead, out.shape[1])

    def extra_repr(self):
        return "tied to a PackedEmbedding; int4-g64, weights never materialised"


# --------------------------------------------------------------------------------------- loader
def _locations(model):
    loc = {}
    for mname, mod in model.named_modules():
        for kind in ("_parameters", "_buffers"):
            for k, v in getattr(mod, kind, {}).items():
                if v is not None:
                    loc[f"{mname}.{k}" if mname else k] = (mod, kind, k)
    return loc


def load_fused_model(path=ART, dev="mps", dtype=torch.bfloat16, chunk_bytes=CHUNK_BYTES, log=print):
    """Build on meta, then materialise: raw tensors verbatim, host linears as packed buffers."""
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    man = json.load(open(os.path.join(path, "pack_manifest.json")))["tensors"]
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    cfg = AutoConfig.from_pretrained(path)

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM._from_config(cfg, dtype=dtype)

    loc = _locations(model)
    missing = sorted(set(man) - set(loc))
    if missing:
        raise RuntimeError(f"manifest names with no module slot: {missing[:5]} ({len(missing)})")

    handles = {}
    fused = []
    try:
        for name, rec in man.items():
            shard = idx[rec.get("q4") or name]
            if shard not in handles:
                handles[shard] = st.safe_open(os.path.join(path, shard), framework="pt")
            fh = handles[shard]
            kind = rec["kind"]
            if kind == "raw":
                t = fh.get_tensor(name).to(dtype=dtype, device=dev)
                mod, _, key = loc[name]
                setattr(mod, key, nn.Parameter(t, requires_grad=False))
                del t
            elif kind == "q4g64":
                o, i = rec["orig_shape"]
                packed = fh.get_tensor(rec["q4"]).to(dev)
                scales = fh.get_tensor(rec["scales"]).to(dev)
                parent, _, leaf = name[: -len(".weight")].rpartition(".")
                old = model.get_submodule(name[: -len(".weight")])
                if not isinstance(old, nn.Linear):
                    raise TypeError(f"{name}: expected nn.Linear, got {type(old).__name__}")
                new = PackedLinear(o, i, packed, scales, chunk_bytes)
                setattr(model.get_submodule(parent), leaf, new)
                del packed, scales
                fused.append(name)
            elif kind == "q4g64_live":
                rows, hidden = rec["orig_shape"]
                packed = fh.get_tensor(rec["q4"]).to(dev)
                scales = fh.get_tensor(rec["scales"]).to(dev)
                struct = fh.get_tensor(rec["struct"]).to(dtype=dtype, device=dev)
                emb = PackedEmbedding(packed, scales, struct, rec["struct_cols"],
                                      rec["live_cols"], hidden, chunk_bytes)
                model.model.embed_tokens = emb
                model.lm_head = PackedTiedHead(emb, chunk_bytes)
                fused.append(name + " (embedding + tied lm_head)")
                del packed, scales, struct
            else:
                raise ValueError(kind)
            if False:  # debug hook
                log(f"[fused] {name}")
    finally:
        handles.clear()

    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=cfg).to(dev)

    left = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if left:
        raise RuntimeError(f"still-meta tensors after load: {left[:5]}")
    model.eval()
    log(f"[fused] {len(fused)} modules packed; meta tensors left: 0")
    return model


# ----------------------------------------------------------------------------------- accounting
def resident_table(path=ART):
    """Bytes the fused model holds, by category -- the VRAM-equivalent footprint."""
    man = json.load(open(os.path.join(path, "pack_manifest.json")))["tensors"]
    ES = {"F16": 2, "BF16": 2, "F32": 4, "I32": 4, "I64": 8, "U8": 1, "I8": 1}

    def layer(k):
        p = k.split(".")
        return int(p[2]) if k.startswith("model.layers.") else None

    cat = {}

    def add(c, b):
        cat[c] = cat.get(c, 0) + b

    for k, r in man.items():
        n = 1
        for d in r["orig_shape"]:
            n *= d
        l = layer(k)
        if r["kind"] == "raw":
            b = n * ES[r["orig_dtype"]]
            add("chip layers 0-10 (bf16, verbatim)" if (l is not None and l <= 10)
                else ("embedding structural cols (verbatim)" if k == "model.embed_tokens.weight"
                      else "norms / conv1d / small (verbatim)"), b)
        elif r["kind"] == "q4g64":
            add("host layers 11-42 (int4-g64 packed)", n // 2 + (n // GROUP) * 2)
        elif r["kind"] == "q4g64_live":
            rows = r["orig_shape"][0]
            pad = r["live_cols"] + r["live_pad"]
            add("embedding live cols (int4-g64 packed)", rows * (pad // 2) + rows * (pad // GROUP) * 2)
            add("embedding structural cols (verbatim)", rows * len(r["struct_cols"]) * ES[r["orig_dtype"]])
    return cat, sum(cat.values())
