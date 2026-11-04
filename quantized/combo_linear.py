"""One process, three weight formats: sparse CSR chip + int4-g64 host + int4-g64 embedding.

WHY THIS FILE EXISTS
--------------------
Two runs already exist and each is proven on its own:

  * `qwenputer_sparse_20261103T1120`  -- chip layers 0-10 served from a 5.58 MiB CSR container
    (never materialised dense), host + embedding left bf16. 5/5 exact, peak device 15.85 GiB.
  * `qwenputer_fused_q4b_20261103T0930` -- host layers 11-42 and the embedding int4-g64 with a
    fused dequantising matmul. Peak device 7.96 GiB, but the chip was loaded DENSE (3.528 GiB).

Neither run answers the question that decides whether the artifact fits a 7.71 GiB card, because
each leaves one of the two big blocks at full size. This module puts both savings in ONE model:

    model.layers[0..10]   CsrLinear      (88 CSR matrices from sparse_chip.sparse + 66 raw bf16)
    model.layers[11..42]  PackedLinear   (248 int4-g64 tensors, dequantised 48 MiB at a time)
    embed_tokens          PackedEmbedding(listup dequantises only the rows asked for)
    lm_head               PackedTiedHead (same buffer, vocab-blocked)

Nothing is re-derived. `sparse_linear` supplies SparseChip / Scratch / CsrLinear and the meta
build; `fused_linear` supplies PackedLinear / PackedEmbedding / PackedTiedHead and the artifact's
own nibble order. This file is only the placement policy: which name comes from which source.

Both source modules are imported from their own run directories, unmodified. Column order in the
CSR matmul is whatever the container's ascending-column scatter produces, which is the property
that keeps the chip bit-exact (a reordered kernel moves 204/2,542,848 outputs by 1 ulp).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPARSE_RUN = os.path.expanduser("~/workspace/localdyno/qwenputer_sparse_20261103T1120")
FUSED_RUN = os.path.expanduser("~/workspace/localdyno/qwenputer_fused_q4b_20261103T0930")
for p in (SPARSE_RUN, FUSED_RUN):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import fused_linear as FL  # noqa: E402
import sparse_linear as SL  # noqa: E402

ART = os.environ.get("QWENPUTER_ART","build_rv1_4b_q4b")   # int4 host + embed
DENSE_BUILD = os.path.expanduser("~/workspace/localdyno/build_rv1_4b")  # reference only
SPARSE = SL.SPARSE                                                    # 5.58 MiB chip container
N_CHIP = SL.N_CHIP
_DT = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
       "I32": torch.int32, "I64": torch.int64, "U8": torch.uint8, "I8": torch.int8}


def load_combo(art=ART, sparse=SPARSE, dev="mps", dtype=torch.bfloat16,
               chunk_bytes=FL.CHUNK_BYTES, log=print, embed_mode="packed"):
    """Build on meta, then fill from two sources. No tensor is ever materialised twice.

    Chip names are taken from the CSR container (never from the safetensors shards), host names
    from the int4 manifests. The only dense bf16 chip state is the 66 1-D/3-D tensors (0.86 MiB)
    and one shared 68.6 MiB scratch block.
    """
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
    chip_raw_names, host_raw_names = [], []
    try:
        for name, rec in man.items():
            # ---------------- chip layers 0-10: from the CSR container, never from disk ------
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
                    chip_raw_names.append(name)
                    n["chip_raw"] += 1
                    del t
                continue

            # ---------------- everything else: the int4 artifact -----------------------------
            shard = idx[rec.get("q4") or name]
            if shard not in handles:
                handles[shard] = st.safe_open(os.path.join(art, shard), framework="pt")
            fh = handles[shard]
            kind = rec["kind"]

            if kind == "raw":
                # NOT cast to bf16: 48 host tensors are F32 in the artifact (linear_attn.A_log and
                # linear_attn.norm, whose exp() drives the DeltaNet decay). `loader.py`, the eager
                # path that scored 10/10, keeps `fh.get_tensor(name)` as-is; the fused loaders cast
                # everything to bf16, which is a 0.4% perturbation of exp(-A_log) per layer.
                t = fh.get_tensor(name).to(dtype=_DT[rec["orig_dtype"]], device=dev)
                mod, _, key = loc[name]
                setattr(mod, key, nn.Parameter(t, requires_grad=False))
                host_raw_names.append(name)
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
                rows, hidden = rec["orig_shape"]
                if embed_mode == "bf16":
                    # diagnostic arm: the embedding stays the artifact's own bf16 table (3.48 GiB)
                    # so the chip's input residual is exact. Everything else is unchanged.
                    didx = json.load(open(os.path.join(DENSE_BUILD,
                                                       "model.safetensors.index.json")))["weight_map"]
                    dsh = didx["model.embed_tokens.weight"]
                    h = st.safe_open(os.path.join(DENSE_BUILD, dsh), framework="pt")
                    w = h.get_tensor("model.embed_tokens.weight").to(dtype=dtype, device=dev)
                    del h
                    emb = nn.Embedding(rows, hidden, device=dev, dtype=dtype)
                    emb.weight = nn.Parameter(w, requires_grad=False)
                    model.model.embed_tokens = emb
                    model.lm_head.weight = emb.weight
                    n["embed"] += 1
                    del w
                    continue
                packed = fh.get_tensor(rec["q4"]).to(dev)
                scales = fh.get_tensor(rec["scales"]).to(dev)
                struct = fh.get_tensor(rec["struct"]).to(dtype=dtype, device=dev)
                emb = FL.PackedEmbedding(packed, scales, struct, rec["struct_cols"],
                                         rec["live_cols"], hidden, chunk_bytes)
                model.model.embed_tokens = emb
                model.lm_head = FL.PackedTiedHead(emb, chunk_bytes)
                n["embed"] += 1
                del packed, scales, struct
            else:
                raise ValueError(kind)
    finally:
        handles.clear()

    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=cfg).to(dev)

    # the tied head must be the SAME embedding object (a missed tie leaves a meta lm_head and
    # silently produces argmax 0 -- the failure that cost the fused run a bf16 control pass)
    if embed_mode == "packed":
        if not isinstance(model.lm_head, FL.PackedTiedHead):
            raise RuntimeError("lm_head is not the packed tied head")
        if model.lm_head.emb is not model.model.embed_tokens:
            raise RuntimeError("lm_head is not tied to embed_tokens")
    else:
        if model.lm_head.weight is not model.model.embed_tokens.weight:
            raise RuntimeError("bf16 arm: lm_head not tied to embed_tokens")

    left = [n_ for n_, p in list(model.named_parameters()) + list(model.named_buffers())
            if p.is_meta]
    if left:
        raise RuntimeError(f"still-meta tensors after load: {left[:6]} ({len(left)})")

    model.eval()
    log(f"[combo] chip {n['chip_csr']} CSR + {n['chip_raw']} raw from "
        f"{os.path.basename(sparse)} ({chip.bytes / 1024 ** 2:.2f} MiB) | host "
        f"{n['host_q4']} int4-g64 + {n['host_raw']} raw | embed {n['embed']} packed+tied | "
        f"scratch {max_numel * 2 / 1024 ** 2:.2f} MiB | meta left: 0")
    return model, chip, scratch, n


# ---------------------------------------------------------------------------------- accounting
def resident_bytes(model, scratch):
    """What the model actually holds on device, by component -- measured off the live modules."""
    out = {}
    seen = set()

    def add(cat, t):
        if id(t) in seen or not isinstance(t, torch.Tensor):
            return
        seen.add(id(t))
        out[cat] = out.get(cat, 0) + t.numel() * t.element_size()

    for mname, mod in model.named_modules():
        if isinstance(mod, SL.CsrLinear):
            cat = "chip CSR matrices (vals bf16 + cols/rows/flat i64)"
        elif isinstance(mod, FL.PackedLinear):
            cat = "host layers 11-42 (int4-g64 packed + scales)"
        elif isinstance(mod, FL.PackedEmbedding):
            cat = "embedding (int4 live packed + scales)"
        elif isinstance(mod, FL.PackedTiedHead):
            continue
        else:
            cat = "verbatim bf16 (chip 1-D/3-D + norms/conv1d)"
        for k in ("vals", "cols", "rowcnt", "rows_of", "flat", "packed", "scales", "struct"):
            b = getattr(mod, k, None)
            if isinstance(b, torch.Tensor):
                add(cat, b)
        for p in list(mod.parameters(recurse=False)) + list(mod.buffers(recurse=False)):
            add(cat, p)
    if isinstance(model.model.embed_tokens, FL.PackedEmbedding):
        e = model.model.embed_tokens
        for k in ("packed", "scales", "struct", "_live_idx", "_struct_idx"):
            add("embedding (int4 live packed + scales)", getattr(e, k))
    add("shared dense scratch (largest chip matrix)", scratch.buf)
    return {k: v / 1024 ** 3 for k, v in sorted(out.items())}, sum(out.values()) / 1024 ** 3


def chunk_ops(model):
    """How many dequantised chunks one host forward costs at this chunk size."""
    tot = 0
    for mod in model.modules():
        if isinstance(mod, FL.PackedLinear):
            r = FL._chunk_rows(mod.out_features, mod.in_features, mod.chunk_bytes)
            tot += -(-mod.out_features // r)
        elif isinstance(mod, FL.PackedTiedHead):
            r = FL._chunk_rows(mod.emb.packed.shape[0], mod.emb.packed.shape[1] * 2,
                               mod.chunk_bytes)
            tot += -(-mod.emb.packed.shape[0] // r)
    return tot
