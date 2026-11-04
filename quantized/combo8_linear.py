"""combo_linear.load_combo, byte for byte, except the embedding branch.

`qwenputer_combo_20261103T1245/combo_linear.py` is proven faithful (bit-identical to the fused loader
at all 43 layers). This file imports it and repeats its load loop unchanged so that the ONLY
difference between the two runs is the codebook width of `model.embed_tokens` (and its tied head):

    embed_mode="int4"  -- the original branch, verbatim (control: must give argmax 4450)
    embed_mode="int8"  -- embed8.PackedEmbedding8 / PackedTiedHead8, same split, same groups,
                          same fp16 scales, same structural-column handling

Everything else -- chip CSR from the 5.58 MiB container, int4-g64 host layers 11-42, F32
`linear_attn.A_log`/`norm` honoured from `orig_dtype` -- is CL's code path.
"""
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
COMBO_RUN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, COMBO_RUN)

import embed8 as E8  # noqa: E402
import combo_linear as CL  # noqa: E402
import fused_linear as FL  # noqa: E402
import sparse_linear as SL  # noqa: E402

ART = CL.ART
_DT = CL._DT


class PackedTiedHeadFixed(FL.PackedTiedHead):
    """The int4 tied head with the strided-weight defect fixed (see `embed8.PackedTiedHead8`).

    `FL.PackedTiedHead` does `dequant_rows(...)[:, :n_live]` and feeds that non-contiguous view to
    `F.linear`, which on MPS returns wrong logits (argmax 23492, same in bf16 and fp32). Here the
    18 zero pad columns stay in the weight and go into x instead, so the operand is contiguous.
    """

    def forward(self, x):
        e = self.emb
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        v, kpad = e.packed.shape
        kpad *= 2                       # int4: two values per byte
        pad = kpad - e.n_live
        xp = x2[:, e._live_idx]
        if pad:
            xp = torch.cat([xp, torch.zeros((xp.shape[0], pad), dtype=x.dtype, device=x.device)], 1)
        out = torch.empty((x2.shape[0], v), dtype=x.dtype, device=x.device)
        r = max(16, min(v, int(self.chunk_bytes) // max(1, kpad * 7)))
        for c0 in range(0, v, r):
            c1 = min(v, c0 + r)
            w = FL.dequant_rows(e.packed[c0:c1], e.scales[c0:c1])     # contiguous, K = live_pad
            out[:, c0:c1] = F.linear(xp, w)
            del w
        out += F.linear(x2[:, e._struct_idx], e.struct)
        return out.reshape(*lead, v)


def load_combo8(art=ART, sparse=CL.SPARSE, dev="mps", dtype=torch.bfloat16,
                chunk_bytes=FL.CHUNK_BYTES, embed_mode="int8", log=print):
    import safetensors.torch as st
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    chip = SL.SparseChip(sparse)
    man = json.load(open(os.path.join(art, "pack_manifest.json")))["tensors"]
    idx = json.load(open(os.path.join(art, "model.safetensors.index.json")))["weight_map"]
    cfg = AutoConfig.from_pretrained(art)

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM._from_config(cfg, dtype=dtype)

    loc = SL._locations(model)

    csr_names = [n for n, d in chip.desc.items() if d["kind"] == "csr"]
    max_numel = max(chip.desc[n]["rows"] * chip.desc[n]["k"] for n in csr_names)
    scratch = SL.Scratch(max_numel, dev=dev, dtype=dtype)

    handles = {}
    n = {"chip_csr": 0, "chip_raw": 0, "host_q4": 0, "host_raw": 0, "embed": 0}
    embed_stats = None
    try:
        for name, rec in man.items():
            if name in chip.desc:
                d = chip.desc[name]
                modname = name[: -len(".weight")] if name.endswith(".weight") else name
                if d["kind"] == "csr":
                    vals, cols, cnt, shape = chip.csr(name, dev=dev)
                    old = model.get_submodule(modname)
                    if not isinstance(old, nn.Linear):
                        raise TypeError(f"{name}: expected nn.Linear, got {type(old).__name__}")
                    parent, _, leaf = modname.rpartition(".")
                    setattr(model.get_submodule(parent), leaf,
                            SL.CsrLinear(name, vals, cols, cnt, shape, scratch))
                    n["chip_csr"] += 1
                    del vals, cols, cnt
                else:
                    t = chip.raw_tensor(name).to(device=dev)
                    mod, _, key = loc[name]
                    setattr(mod, key, nn.Parameter(t, requires_grad=False))
                    n["chip_raw"] += 1
                    del t
                continue

            shard = idx[rec.get("q4") or name]
            if shard not in handles:
                handles[shard] = st.safe_open(os.path.join(art, shard), framework="pt")
            fh = handles[shard]
            kind = rec["kind"]

            if kind == "raw":
                t = fh.get_tensor(name).to(dtype=_DT[rec["orig_dtype"]], device=dev)
                mod, _, key = loc[name]
                setattr(mod, key, nn.Parameter(t, requires_grad=False))
                n["host_raw"] += 1
                del t
            elif kind == "q4g64":
                o, i = rec["orig_shape"]
                packed = fh.get_tensor(rec["q4"]).to(dev)
                scales = fh.get_tensor(rec["scales"]).to(dev)
                modname = name[: -len(".weight")]
                old = model.get_submodule(modname)
                if not isinstance(old, nn.Linear):
                    raise TypeError(f"{name}: expected nn.Linear, got {type(old).__name__}")
                parent, _, leaf = modname.rpartition(".")
                setattr(model.get_submodule(parent), leaf,
                        FL.PackedLinear(o, i, packed, scales, chunk_bytes))
                n["host_q4"] += 1
                del packed, scales
            elif kind == "q4g64_live":
                if embed_mode == "int8":
                    emb, embed_stats = E8.build_embed8(dev=dev, dtype=dtype, log=log)
                    model.model.embed_tokens = emb
                    model.lm_head = E8.PackedTiedHead8(emb, chunk_bytes)
                    n["embed"] += 1
                elif embed_mode == "int4":
                    rows, hidden = rec["orig_shape"]
                    packed = fh.get_tensor(rec["q4"]).to(dev)
                    scales = fh.get_tensor(rec["scales"]).to(dev)
                    struct = fh.get_tensor(rec["struct"]).to(dtype=dtype, device=dev)
                    emb = FL.PackedEmbedding(packed, scales, struct, rec["struct_cols"],
                                             rec["live_cols"], hidden, chunk_bytes)
                    model.model.embed_tokens = emb
                    model.lm_head = PackedTiedHeadFixed(emb, chunk_bytes)
                    n["embed"] += 1
                    del packed, scales, struct
                else:
                    raise ValueError(embed_mode)
            else:
                raise ValueError(kind)
    finally:
        handles.clear()

    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=cfg).to(dev)

    if embed_mode == "int8":
        if not isinstance(model.lm_head, E8.PackedTiedHead8):
            raise RuntimeError("lm_head is not the int8 tied head")
        if model.lm_head.emb is not model.model.embed_tokens:
            raise RuntimeError("lm_head is not tied to embed_tokens")
    else:
        if not isinstance(model.lm_head, PackedTiedHeadFixed):
            raise RuntimeError("lm_head is not the int4 tied head")
        if model.lm_head.emb is not model.model.embed_tokens:
            raise RuntimeError("lm_head is not tied to embed_tokens")

    left = [n_ for n_, p in list(model.named_parameters()) + list(model.named_buffers())
            if p.is_meta]
    if left:
        raise RuntimeError(f"still-meta tensors after load: {left[:6]} ({len(left)})")

    model.eval()
    log(f"[combo8] chip {n['chip_csr']} CSR + {n['chip_raw']} raw | host {n['host_q4']} int4-g64 + "
        f"{n['host_raw']} raw | embed {n['embed']} {embed_mode} | scratch "
        f"{max_numel * 2 / 1024 ** 2:.2f} MiB | meta left: 0")
    return model, chip, scratch, n, embed_stats


def resident_bytes8(model, scratch):
    """CL.resident_bytes, with the int8 embedding named correctly."""
    out = {}
    seen = set()

    def add(cat, t):
        if id(t) in seen or not isinstance(t, torch.Tensor):
            return
        seen.add(id(t))
        out[cat] = out.get(cat, 0) + t.numel() * t.element_size()

    for mname, mod in model.named_modules():
        if isinstance(mod, SL.CsrLinear):
            cat = "chip layers 0-10: CSR (vals bf16 + cols/rows/flat i64)"
        elif isinstance(mod, FL.PackedLinear):
            cat = "host layers 11-42: int4-g64 packed + scales"
        elif isinstance(mod, (E8.PackedEmbedding8, FL.PackedEmbedding)):
            cat = ("embedding: int8-g64 live + verbatim structural"
                   if isinstance(mod, E8.PackedEmbedding8) else
                   "embedding: int4-g64 live + verbatim structural")
            for k in ("packed", "scales", "struct", "_live_idx", "_struct_idx"):
                add(cat, getattr(mod, k))
            continue
        elif isinstance(mod, (E8.PackedTiedHead8, FL.PackedTiedHead)):
            continue
        else:
            cat = "verbatim (chip 1-D/3-D + norms/conv1d, incl. 48 F32)"
        for k in ("vals", "cols", "rowcnt", "rows_of", "flat", "packed", "scales", "struct"):
            b = getattr(mod, k, None)
            if isinstance(b, torch.Tensor):
                add(cat, b)
        for p in list(mod.parameters(recurse=False)) + list(mod.buffers(recurse=False)):
            add(cat, p)
    add("shared dense scratch (largest chip matrix)", scratch.buf)
    return {k: v / 1024 ** 3 for k, v in sorted(out.items())}, sum(out.values()) / 1024 ** 3
