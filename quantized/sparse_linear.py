"""Serve the compiled chip (layers 0-10) from `sparse_chip.sparse` inside a running QwenPU model.

WHY THIS FILE EXISTS
--------------------
`sparse_probe_20261103` built a 5.58 MiB CSR container for the 154 chip tensors (3.5280 GiB dense,
99.9448 % exact zeros) and proved every one of them re-reads bit-identically. Nothing had ever
put those bytes in front of a forward pass. This module does that and nothing else.

HOW, and why it is bit-exact rather than "close"
------------------------------------------------
The container is lossless, so the *only* defensible requirement is that the model compute exactly
what the dense artifact computes. `CsrLinear.forward` therefore does NOT write a sparse matmul:
it scatters the stored entries back into a dense bf16 block and calls the *same* `F.linear` the
dense module called. What stays sparse is the RESIDENT footprint -- the weights live as
vals(u16-as-bf16) + cols(u16) + row counts(u16), and the dense block exists only for the duration
of one matmul, in ONE shared buffer reused by every chip matrix (peak transient = the largest
single chip matrix, 9216 x 3904 = 71.96 MiB, not 3.53 GiB).

That choice is deliberate. The probe measured that dropping exact zeros is bit-identical *only*
when the surviving terms are accumulated in the original ascending column order, and that a
reordered kernel moves 204 / 2,542,848 bf16 outputs by 1 ulp. The circuit's weights are exact
powers of two spanning 2^0..2^112, so a 1-ulp change in the accumulation is a numeric change to a
bit-sliced boolean net, not a rounding curiosity. Reconstruct-then-multiply sidesteps the question
entirely: same bf16 bytes, same kernel, same order, therefore identical output by construction --
which is then *checked* bitwise against the original tensor (verify.py).

Resident cost: 5.58 MiB of container bytes + one 71.96 MiB scratch buffer, against 3.5280 GiB.
"""
import json
import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SPARSE = os.path.expanduser("~/workspace/localdyno/sparse_probe_20261103/sparse_chip.sparse")
DENSE_BUILD = os.path.expanduser("~/workspace/localdyno/build_rv1_4b")
N_CHIP = 11


# --------------------------------------------------------------------------------------- container
class SparseChip:
    """Reader for the SPRV1 container (154 tensors; 2-D as CSR, 1-D/3-D verbatim)."""

    def __init__(self, path=SPARSE):
        self.path = path
        with open(path, "rb") as fh:
            self.raw = fh.read()
        if self.raw[:5] != b"SPRV1":
            raise ValueError(f"{path}: bad magic {self.raw[:5]!r}")
        n_t, ml = struct.unpack("<II", self.raw[5:13])
        self.desc = {d["name"]: d for d in json.loads(self.raw[13:13 + ml])}
        self.p0 = 13 + ml
        self.n_tensors = len(self.desc)
        assert n_t == self.n_tensors, (n_t, self.n_tensors)

    @property
    def bytes(self):
        return len(self.raw)

    def names(self):
        return sorted(self.desc, key=lambda n: (int(n.split(".")[2]), n))

    def csr(self, name, dev="mps"):
        """-> (vals bf16 [nnz], cols i64 [nnz], rowcnt u16 [rows], (rows, k))"""
        d = self.desc[name]
        if d["kind"] != "csr":
            raise TypeError(f"{name} is kind={d['kind']}, not csr")
        rows, k = d["rows"], d["k"]
        base = self.p0 + d["off_vals"]
        v = torch.from_numpy(np.frombuffer(self.raw, np.uint16, count=d["nzb"],
                                           offset=base).copy()).view(torch.bfloat16).to(dev)
        c = torch.from_numpy(np.frombuffer(self.raw, np.uint16, count=d["nzb"],
                                           offset=self.p0 + d["off_cols"]).copy()).long().to(dev)
        cnt = torch.from_numpy(np.frombuffer(self.raw, np.uint16, count=rows,
                                             offset=self.p0 + d["off_cnt"]).copy())
        return v, c, cnt, (rows, k)

    def raw_tensor(self, name, dtype=torch.bfloat16):
        d = self.desc[name]
        if d["kind"] != "raw":
            raise TypeError(f"{name} is kind={d['kind']}, not raw")
        base = self.p0 + d["off_vals"]
        t = torch.from_numpy(np.frombuffer(self.raw, np.uint16, count=d["nelem"],
                                           offset=base).copy()).view(torch.bfloat16)
        return t.reshape(d["shape"])


# --------------------------------------------------------------------------------------- module
class Scratch:
    """One dense scratch block, shared by every CsrLinear instance.

    A single buffer (not one per module) is what keeps the transient at 71.96 MiB instead of
    ~320 MiB per layer: the previous measurement on this box found a larger freed block raises
    the allocator's retained peak, so fewer, smaller blocks is the safer direction.

    The block is kept all-zero between builds by un-writing only the cells the previous matrix
    occupied (its own (row,col) pairs), never by memsetting 72 MiB. That invariant -- "zero
    everywhere except the current matrix" -- is what makes the next scatter correct, and it
    starts from `torch.zeros` so it holds before the first build.
    """

    def __init__(self, numel, dev="mps", dtype=torch.bfloat16):
        self.buf = torch.zeros(numel, dtype=dtype, device=dev)
        self.numel = numel
        self.builds = 0
        self._last = None      # flat indices of the cells the current occupant wrote

    def build(self, rows, k, flat, vals):
        if self._last is not None:
            self.buf[self._last] = 0
            self._last = None
        n = rows * k
        if n > self.numel:
            raise RuntimeError(f"scratch too small: need {n}, have {self.numel}")
        if flat.numel():
            self.buf[flat] = vals
            self._last = flat
        self.builds += 1
        return self.buf[:n].view(rows, k)


class CsrLinear(nn.Module):
    """Drop-in for a bias-free nn.Linear whose weight is resident as CSR.

    `forward` scatters into the shared scratch block and calls F.linear -- the identical kernel
    on the identical bf16 values, so the output is bit-identical to the dense module's.
    """

    def __init__(self, name, vals, cols, rowcnt, shape, scratch):
        super().__init__()
        self.name = name
        self.out_features, self.in_features = shape
        self.scratch = scratch
        self.register_buffer("vals", vals, persistent=False)          # bf16, nnz
        self.register_buffer("cols", cols, persistent=False)          # int64, nnz
        self.register_buffer("rowcnt", rowcnt, persistent=False)      # u16, rows
        rows = torch.repeat_interleave(torch.arange(shape[0]),
                                       rowcnt.long().cpu()).to(vals.device)
        self.register_buffer("rows_of", rows, persistent=False)       # int64, nnz
        self.register_buffer("flat", rows * self.in_features + cols, persistent=False)
        self.nnz = int(vals.numel())

    def dense(self):
        return self.scratch.build(self.out_features, self.in_features, self.flat, self.vals)

    def forward(self, x):
        return F.linear(x, self.dense())

    def extra_repr(self):
        return (f"{self.out_features}x{self.in_features} CSR nnz={self.nnz} "
                f"({100.0 * self.nnz / (self.out_features * self.in_features):.4f}% dense)")


# --------------------------------------------------------------------------------------- loader
def _locations(model):
    loc = {}
    for mname, mod in model.named_modules():
        for kind in ("_parameters", "_buffers"):
            for k, v in getattr(mod, kind, {}).items():
                if v is not None:
                    loc[f"{mname}.{k}" if mname else k] = (mod, kind, k)
    return loc


def load_sparse_model(path=DENSE_BUILD, sparse=SPARSE, dev="mps", dtype=torch.bfloat16,
                      log=print):
    """Build the artifact on meta, then materialise: host/embed dense, chip layers 0-10 as CSR.

    Chip tensors are never materialised dense at any point in this function -- they arrive from
    the 5.58 MiB container. Peak during load is therefore the 13.62 GiB non-chip footprint plus
    one 71.96 MiB scratch block.
    """
    import safetensors.torch as st
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    chip = SparseChip(sparse)
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    cfg = AutoConfig.from_pretrained(path)

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM._from_config(cfg, dtype=dtype)

    loc = _locations(model)

    # every 2-D chip matrix -> its nnz count, so the scratch block is sized once
    csr_names = [n for n, d in chip.desc.items() if d["kind"] == "csr"]
    max_numel = max(chip.desc[n]["rows"] * chip.desc[n]["k"] for n in csr_names)
    scratch = Scratch(max_numel, dev=dev, dtype=dtype)

    handles = {}
    n_csr = n_raw = 0
    try:
        for name, shard in sorted(idx.items()):
            if name not in loc:
                raise RuntimeError(f"{name} in checkpoint but no module slot")
            if shard not in handles:
                handles[shard] = st.safe_open(os.path.join(path, shard), framework="pt")
            fh = handles[shard]
            if name in chip.desc and chip.desc[name]["kind"] == "csr":
                vals, cols, cnt, shape = chip.csr(name, dev=dev)
                modname = name[: -len(".weight")]
                old = model.get_submodule(modname)
                if not isinstance(old, torch.nn.Linear):
                    raise TypeError(f"{name}: expected nn.Linear, got {type(old).__name__}")
                parent, _, leaf = modname.rpartition(".")
                setattr(model.get_submodule(parent), leaf,
                        CsrLinear(name, vals, cols, cnt, shape, scratch))
                n_csr += 1
                del vals, cols, cnt
            else:
                t = fh.get_tensor(name).to(dtype=dtype, device=dev)
                mod, _, key = loc[name]
                setattr(mod, key, nn.Parameter(t, requires_grad=False))
                n_raw += 1
                del t
    finally:
        handles.clear()

    # rotary_emb was built on meta and carries no checkpoint entry
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=cfg).to(dev)

    # tied embeddings: the checkpoint carries no lm_head, so it is still on meta -- tie it to the
    # embedding BEFORE the meta check (a fresh lm_head would allocate 3.7 GiB of duplicate weights)
    emb = model.model.embed_tokens.weight
    if getattr(model, "lm_head", None) is None:
        raise RuntimeError("model has no lm_head to tie")
    model.lm_head.weight = emb
    if model.lm_head.weight is not emb:
        raise RuntimeError("lm_head did not tie to embed_tokens")

    left = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if left:
        raise RuntimeError(f"still-meta tensors after load: {left[:6]} ({len(left)})")

    model.eval()
    log(f"[sparse] {n_csr} chip matrices as CSR from {os.path.basename(sparse)} "
        f"({chip.bytes / 1024 ** 2:.2f} MiB), {n_raw} host/raw tensors dense bf16; "
        f"scratch {max_numel * 2 / 1024 ** 2:.2f} MiB; meta left: 0")
    return model, chip, scratch
