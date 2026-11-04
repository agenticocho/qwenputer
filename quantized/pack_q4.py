#!/usr/bin/env python3
"""Pack build_rv1_4b: layers 0-10 verbatim bf16, host layers 11-42 + embed to int4 g64.

Output layout (new dir):
  model-XXXXX-of-YYYYY.safetensors   raw/packed tensors
  pack_manifest.json                 per-tensor recipe (kind, orig shape, storage)
  model.safetensors.index.json       name -> shard
  loader.py                          reconstructs a bf16 state_dict
"""
import json, os, struct, sys, time, hashlib
import torch, safetensors.torch as st

SRC  = os.path.expanduser("~/workspace/localdyno/build_rv1_4b")
DST  = os.path.expanduser("~/workspace/localdyno/build_rv1_4b_q4")
G    = 64
TGT  = 2_000_000_000   # bytes per output shard

os.makedirs(DST, exist_ok=True)

# ---- source inventory ----
src_files = sorted(f for f in os.listdir(SRC) if f.endswith(".safetensors"))
meta = {}
for f in src_files:
    with open(os.path.join(SRC, f), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    for k, v in hdr.items():
        meta[k] = (v["dtype"], tuple(v["shape"]), f)

def layer_of(k):
    if not k.startswith("model.layers."):
        return None
    return int(k.split(".")[2])

def packable(k, sh):
    """2D weight, input dim divisible by G, and in a region we are allowed to quantise."""
    if len(sh) != 2 or sh[1] % G != 0:
        return False
    if k == "model.embed_tokens.weight":
        return True
    l = layer_of(k)
    return l is not None and l >= 11

def q4(w):
    o, i = w.shape
    wg = w.float().reshape(o, i // G, G)
    absmax = wg.abs().amax(dim=2)
    scale = (absmax / 7.0).clamp_min(1e-8)
    q = torch.round(wg / scale.unsqueeze(2)).clamp_(-8, 7).to(torch.int8).reshape(o, i)
    q = (q & 0xF).to(torch.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return packed, scale.to(torch.float16).contiguous()

# ---- pass 1: pack ----
t0 = time.time()
manifest, shard_map = {}, {}
buf, buf_bytes, shard_no = {}, 0, 0
out_files = []

def flush():
    global buf, buf_bytes, shard_no
    if not buf:
        return
    name = f"model-{shard_no+1:05d}-of-XXXXX.safetensors"
    p = os.path.join(DST, name)
    st.save_file(buf, p)
    for k in buf:
        shard_map[k] = name
    with open(p, "rb") as fh:
        _n = struct.unpack("<Q", fh.read(8))[0]
        _hdr = json.loads(fh.read(_n))
    _hdr.pop("__metadata__", None)
    # total_size means TENSOR bytes (nelems x itemsize), not a census of files: getsize
    # counts the header and the 8-byte prefix. Keep both, labelled.
    _payload = max((v["data_offsets"][1] for v in _hdr.values()), default=0)
    out_files.append((name, os.path.getsize(p), _payload))
    print(f"  wrote {name}  {os.path.getsize(p)/1e9:.3f} GB  ({len(buf)} tensors)", flush=True)
    shard_no += 1
    buf, buf_bytes = {}, 0

written_src = set()
for f in src_files:
    with st.safe_open(os.path.join(SRC, f), framework="pt") as fh:
        for k in fh.keys():
            if k in written_src:
                continue
            written_src.add(k)
            dt, sh, _ = meta[k]
            t = fh.get_tensor(k)
            assert t.shape == torch.Size(sh), (k, t.shape, sh)
            if packable(k, sh):
                p, s = q4(t)
                buf[k + ".q4"] = p
                buf[k + ".scales"] = s
                manifest[k] = {"kind": "q4g64", "orig_shape": list(sh),
                               "orig_dtype": dt, "q4": k + ".q4", "scales": k + ".scales"}
                buf_bytes += p.numel() + s.numel() * 2
            else:
                buf[k] = t
                manifest[k] = {"kind": "raw", "orig_shape": list(sh), "orig_dtype": dt}
                buf_bytes += t.numel() * t.element_size()
            del t
            if buf_bytes >= TGT:
                flush()
flush()

# ---- rewrite shard names with real count ----
n = len(out_files)
for i, (old, _, _) in enumerate(out_files):
    new = f"model-{i+1:05d}-of-{n:05d}.safetensors"
    if old != new:
        os.rename(os.path.join(DST, old), os.path.join(DST, new))
        for k, v in shard_map.items():
            if v == old:
                shard_map[k] = new

idx = {"metadata": {"total_size": sum(pl for _, _, pl in out_files),
                    "total_file_bytes": sum(z for _, z, _ in out_files)},
       "weight_map": {k: shard_map[k] for k in sorted(shard_map)}}
json.dump(idx, open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=2)
json.dump({"format": "int4_groupwise", "group_size": G, "axis": "input_dim",
           "packing": "two int4 per uint8, element 2i in low nibble",
           "codebook": "symmetric, w = clip(round(w/(absmax/7)), -8, 7) * (absmax/7)",
           "scales_dtype": "float16",
           "tensors": manifest}, open(os.path.join(DST, "pack_manifest.json"), "w"), indent=1)

# ---- carry over non-weight files ----
import shutil
for f in os.listdir(SRC):
    if f.endswith(".safetensors") or f.startswith("model.safetensors.index"):
        continue
    if f in ("config.json", "generation_config.json", "chat_template.jinja", "tokenizer.json",
             "tokenizer_config.json", "chipformer.json", "LICENSE"):
        shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))

print(f"\npacked in {time.time()-t0:.1f}s  shards={n}  tensors_in={len(manifest)}")
