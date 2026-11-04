"""Embedding + tied head at int8-g64, structural columns verbatim. Everything else unchanged.

WHY THIS FILE EXISTS
--------------------
`qwenputer_combo_20261103T1245` localised the combined artifact's failure to exactly one component:
the embedding. Same loader, same chip, same host, only the embedding swapped:
    int4 embedding -> first-token argmax 4450 (wrong, no tool call)
    bf16 embedding -> first-token argmax 271  (the reference)
The int4 embedding costs 0.9094 GiB live and fits 7.71 GiB; the bf16 embedding costs 3.48 GiB and
does not (8.1 GiB at load). This module is the one-variable test in between: the SAME embedding
tensor, the SAME live/structural column split, the SAME group size, SAME fp16 scales, SAME
dequantised-per-lookup-row design -- only the codebook width changes from 4 to 8 bits.

    int4:  s = absmax/7,   q in [-8,7]     -> |err| <= s/2 = absmax/14
    int8:  s = absmax/127, q in [-127,127] -> |err| <= s/2 = absmax/254      (18.14x finer)

The structural columns (61 x 2^40 ballast + 21 x +-128 code markers) are NOT quantised at all:
they are copied verbatim bf16 from the packed artifact's own `.struct` tensor, which `pack_q4b.py`
wrote as `t[:, STRUCT].contiguous()` -- bit-identical to `build_rv1_4b`. That is checked here
(`verify_struct`) rather than assumed, because structural corruption and precision loss look
identical from the logits.

Source of the 3822 live columns is the dense bf16 artifact `build_rv1_4b`, streamed row-block by
row-block with `safe_open(...).get_slice(...)` so the 3.48 GiB table is never resident.
"""
import json
import os

import safetensors.torch as st
import torch
import torch.nn as nn
import torch.nn.functional as F

DENSE = os.path.expanduser("~/workspace/localdyno/build_rv1_4b")
Q4B = os.environ.get("QWENPUTER_ART","build_rv1_4b_q4b")
GROUP = 64
FP16_MIN_NORMAL = 6.103515625e-05


# ------------------------------------------------------------------------------ quantise / dequant
def quant8_rows(t):
    """bf16 (r, i) with i % 64 == 0 -> int8 (r, i) + fp16 scales (r, i/64). Symmetric absmax/127.

    Quantisation uses the fp16-ROUNDED scale so that `q * scale` reconstructs exactly what
    `dequant8_rows` computes -- the same discipline as `pack_q4b.q4`.
    """
    r, i = t.shape
    assert i % GROUP == 0, i
    wg = t.float().reshape(r, i // GROUP, GROUP)
    # Floor at FP16_MIN_NORMAL (6.1035e-5), NOT 1e-8: 31.6% of the embedding's 64-wide groups are
    # exactly zero, and 1e-8 rounds to 0 in fp16, which makes the division 0/0 -> nan -> clamped
    # garbage. With this floor an all-zero group gets q=0 and reconstructs 0 exactly; any nonzero
    # group has absmax >= 0.0141 (measured), so the floor never binds on real data.
    scale = (wg.abs().amax(dim=2) / 127.0).clamp_min(FP16_MIN_NORMAL).to(torch.float16)
    sf = scale.to(torch.float32).unsqueeze(2)          # the scale dequant will actually use
    q = torch.round(wg / sf).clamp_(-127, 127).to(torch.int8).reshape(r, i)
    return q.contiguous(), scale.contiguous()


def dequant8_rows(packed, scales, out_dtype=torch.bfloat16):
    """(r, i) int8 + (r, i/64) fp16 -> (r, i) bf16. Same reconstruction as loader.dequant."""
    r, i = packed.shape
    w = packed.to(torch.float32).reshape(r, i // GROUP, GROUP) * scales.to(torch.float32).unsqueeze(-1)
    return w.reshape(r, i).to(out_dtype)


# -------------------------------------------------------------------------------------- modules
class PackedEmbedding8(nn.Module):
    """embed_tokens with live columns int8 (one signed byte per weight), structural verbatim.

    A lookup touches only the rows asked for, so the dequantised 478656 x 3904 bf16 table
    (3.7 GiB) never exists.
    """

    def __init__(self, packed, scales, struct, struct_cols, n_live, hidden, chunk_bytes=None):
        super().__init__()
        self.n_live = n_live
        self.hidden = hidden
        self.struct_cols = list(struct_cols)
        self.live_cols = [c for c in range(hidden) if c not in set(self.struct_cols)]
        self.chunk_bytes = chunk_bytes
        self.register_buffer("packed", packed, persistent=False)   # int8 (vocab, live_pad)
        self.register_buffer("scales", scales, persistent=False)   # fp16 (vocab, live_pad/64)
        self.register_buffer("struct", struct, persistent=False)   # bf16 (vocab, n_struct)
        self.register_buffer("_live_idx", torch.tensor(self.live_cols, dtype=torch.long),
                             persistent=False)
        self.register_buffer("_struct_idx", torch.tensor(self.struct_cols, dtype=torch.long),
                             persistent=False)

    def forward(self, ids):
        flat = ids.reshape(-1)
        p = self.packed.index_select(0, flat)
        s = self.scales.index_select(0, flat)
        live = dequant8_rows(p, s)[:, : self.n_live]
        del p, s
        out = torch.empty(flat.numel(), self.hidden, dtype=self.struct.dtype, device=ids.device)
        out[:, self._live_idx] = live
        out[:, self._struct_idx] = self.struct.index_select(0, flat)
        del live
        return out.reshape(*ids.shape, self.hidden)

    def extra_repr(self):
        return (f"vocab={self.packed.shape[0]}, hidden={self.hidden}, "
                f"live={self.n_live} int8-g{GROUP}, structural={len(self.struct_cols)} verbatim")


class PackedTiedHead8(nn.Module):
    """lm_head tied to a PackedEmbedding8: logits = x @ W.T, W never materialised.

    THE DEFECT THIS FIXES (measured, probes 1-3): the shipped `fused_linear.PackedTiedHead` builds
    `w = dequant_rows(...)[:, : n_live]` -- a NON-CONTIGUOUS view with row stride = live_pad (3840)
    and width = n_live (3822) -- and passes it directly to `F.linear`. On MPS that returns wrong
    values (argmax 23492 instead of 271, identical in bf16 AND fp32, at int4 AND int8), which is
    what the previous run read as "the int4 embedding breaks the chip". The same numbers, made
    contiguous, give 271.

    Fix costs one 15 KiB pad vector per call instead of a 14 MiB weight copy per chunk: the pad
    columns (18 zeros the packer already carries) are put back into x, so the weight operand is the
    full-width contiguous dequant output. Verified to give bit-identical logits to the full-width
    single matmul in the artifact's column order (`split_pad` == `fused` == 271 in probe3).
    """

    def __init__(self, emb: PackedEmbedding8, chunk_bytes):
        super().__init__()
        self.emb = emb
        self.chunk_bytes = max(1, int(chunk_bytes))

    def forward(self, x):
        e = self.emb
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        v, kpad = e.packed.shape
        pad = kpad - e.n_live
        xp = x2[:, e._live_idx]
        if pad:
            xp = torch.cat([xp, torch.zeros((xp.shape[0], pad), dtype=x.dtype, device=x.device)], 1)
        out = torch.empty((x2.shape[0], v), dtype=x.dtype, device=x.device)
        r = max(16, min(v, self.chunk_bytes // max(1, kpad * 7)))
        for c0 in range(0, v, r):
            c1 = min(v, c0 + r)
            w = dequant8_rows(e.packed[c0:c1], e.scales[c0:c1])   # contiguous, K = live_pad
            out[:, c0:c1] = F.linear(xp, w)
            del w
        # structural columns: measured exactly 0 for this model (x has zeros in all 82 dims), kept
        # anyway so the arithmetic is complete if that ever changes
        out += F.linear(x2[:, e._struct_idx], e.struct)
        return out.reshape(*lead, v)

    def extra_repr(self):
        return "tied to a PackedEmbedding8; int8-g64, weights never materialised, K padded to 64"


# --------------------------------------------------------------------------------------- build
def column_split():
    cf = json.load(open(os.path.join(DENSE, "chipformer.json")))
    hid = cf["hidden"]
    b0, bn = cf["chip_dims_offset"], cf["ballast_dims"]
    struct = sorted(set(range(b0, b0 + bn)) | set(cf["code_dims"]))
    live = [c for c in range(hid) if c not in set(struct)]
    pad = (-len(live)) % GROUP
    return hid, struct, live, pad


def build_embed8(dev="mps", dtype=torch.bfloat16, block=8192, log=print):
    """Stream build_rv1_4b's bf16 embedding into an int8-g64 packed table on `dev`.

    Returns (PackedEmbedding8, stats). The dense table is never resident: one `block`-row slice
    (8192 x 3904 bf16 = 61 MiB) is read, quantised, copied, dropped.
    """
    hid, struct_cols, live, pad = column_split()
    idx = json.load(open(os.path.join(DENSE, "model.safetensors.index.json")))["weight_map"]
    shard = idx["model.embed_tokens.weight"]
    path = os.path.join(DENSE, shard)

    with st.safe_open(path, framework="pt") as h:
        sl = h.get_slice("model.embed_tokens.weight")
        vocab, hid_s = sl.get_shape()
        assert hid_s == hid, (hid_s, hid)

        i_pad = len(live) + pad
        packed = torch.empty((vocab, i_pad), dtype=torch.int8, device=dev)
        scales = torch.empty((vocab, i_pad // GROUP), dtype=torch.float16, device=dev)

        live_idx_cpu = torch.tensor(live, dtype=torch.long)
        s_min = float("inf")
        s_max = 0.0
        n_zero_scale = 0
        err_absmax = 0.0
        src_absmax = 0.0
        for r0 in range(0, vocab, block):
            r1 = min(vocab, r0 + block)
            t = sl[r0:r1]                                   # (b, hid) bf16, CPU
            lv = t.index_select(1, live_idx_cpu)
            if pad:
                lv = torch.cat([lv, torch.zeros(lv.shape[0], pad, dtype=lv.dtype)], 1)
            q, s = quant8_rows(lv.contiguous())
            dq = dequant8_rows(q, s, out_dtype=torch.float32)
            src = lv.float()
            err_absmax = max(err_absmax, float((dq - src).abs().max()))
            src_absmax = max(src_absmax, float(src.abs().max()))
            s_min = min(s_min, float(s.float().min()))
            s_max = max(s_max, float(s.float().max()))
            n_zero_scale += int((s == 0).sum())
            packed[r0:r1] = q.to(dev)
            scales[r0:r1] = s.to(dev)
            del t, lv, q, s, dq, src
            if (r0 // block) % 15 == 0:
                log(f"  [embed8] {r1}/{vocab} rows")

    # structural columns verbatim from the packed artifact (pack_q4b wrote them unquantised)
    q4idx = json.load(open(os.path.join(Q4B, "model.safetensors.index.json")))["weight_map"]
    with st.safe_open(os.path.join(Q4B, q4idx["model.embed_tokens.weight.struct"]),
                      framework="pt") as h:
        struct = h.get_tensor("model.embed_tokens.weight.struct").to(dtype=dtype, device=dev)
    assert struct.shape == (vocab, len(struct_cols)), struct.shape

    emb = PackedEmbedding8(packed, scales, struct, struct_cols, len(live), hid)
    stats = {"vocab": vocab, "hidden": hid, "live_cols": len(live), "live_pad": i_pad,
             "struct_cols": len(struct_cols), "pad": pad,
             "q_bytes_gib": round(packed.numel() / 1024 ** 3, 4),
             "scale_bytes_gib": round(scales.numel() * 2 / 1024 ** 3, 4),
             "struct_bytes_gib": round(struct.numel() * 2 / 1024 ** 3, 4),
             "max_abs_err_vs_bf16_source": round(err_absmax, 6),
             "src_absmax_live": round(src_absmax, 6),
             "rel_err": round(err_absmax / src_absmax, 6) if src_absmax else None,
             "scale_min": s_min, "scale_max": s_max, "zero_scales": n_zero_scale}
    log(f"[embed8] {json.dumps(stats)}")
    return emb, stats


def verify_struct(emb, rows=4096):
    """Prove the structural columns in memory are the source's, bit for bit (not just close)."""
    hid, struct_cols, live, pad = column_split()
    idx = json.load(open(os.path.join(DENSE, "model.safetensors.index.json")))["weight_map"]
    with st.safe_open(os.path.join(DENSE, idx["model.embed_tokens.weight"]), framework="pt") as h:
        sl = h.get_slice("model.embed_tokens.weight")
        t = sl[0:rows].index_select(1, torch.tensor(struct_cols, dtype=torch.long))
    got = emb.struct[:rows].to("cpu")
    same = bool((t == got).all())
    return {"rows": rows, "bit_identical": same,
            "max_abs": float((t.float() - got.float()).abs().max()), "cols": len(struct_cols)}
