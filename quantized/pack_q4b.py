#!/usr/bin/env python3
"""Pack build_rv1_4b -> int4-g64 artifact, preserving structural ('load-bearing') bits exactly.

Policy
------
* layers 0-10 (compiled chip)            -> verbatim bf16, bit-identical
* layers 11-42 2D weights                -> int4 g64 (absmax/7 symmetric, fp16 scales)
* model.embed_tokens
    - structural cols = ballast [chip_dims_offset, +ballast_dims)  (constant 2^40)
                      + code_dims (binary +-128 markers)
      -> kept verbatim bf16, bit-identical
    - all other (live) cols             -> int4 g64
* everything else (norms, biases, conv1d, A_log, dt_bias) -> verbatim
"""
import json, os, struct, shutil, time
import torch, safetensors.torch as st

SRC = os.path.expanduser("~/workspace/localdyno/build_rv1_4b")
DST = os.environ.get("QWENPUTER_ART","build_rv1_4b_q4b")
G   = 64
TGT = 2_000_000_000

cf = json.load(open(os.path.join(SRC, "chipformer.json")))
HID = cf["hidden"]
CODE = cf["code_dims"]
B0, BN = cf["chip_dims_offset"], cf["ballast_dims"]
STRUCT = sorted(set(range(B0, B0 + BN)) | set(CODE))
LIVE = [c for c in range(HID) if c not in set(STRUCT)]
LIVE_PAD = (-len(LIVE)) % G
print(f"hidden={HID} structural_cols={len(STRUCT)} live_cols={len(LIVE)} pad_to={len(LIVE)+LIVE_PAD}")

os.makedirs(DST, exist_ok=True)

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
    return int(k.split(".")[2]) if k.startswith("model.layers.") else None

def packable(k, sh):
    if len(sh) != 2 or sh[1] % G != 0:
        return False
    l = layer_of(k)
    return l is not None and l >= 11

def q4(w):
    o, i = w.shape
    assert i % G == 0
    wg = w.float().reshape(o, i // G, G)
    scale = (wg.abs().amax(dim=2) / 7.0).clamp_min(1e-8)
    q = torch.round(wg / scale.unsqueeze(2)).clamp_(-8, 7).to(torch.int8).reshape(o, i)
    packed = ((q & 0xF).to(torch.uint8)[:, 0::2] | ((q & 0xF).to(torch.uint8)[:, 1::2] << 4)).contiguous()
    sc = scale.to(torch.float16).contiguous()
    assert torch.isfinite(sc).all(), "fp16 scale overflow"
    return packed, sc

t0 = time.time()
manifest, shard_map, buf, buf_bytes, out_files = {}, {}, {}, 0, []
shard_no = 0

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
    # total_size must mean TENSOR bytes (nelems x itemsize), the convention the source
    # pack's own index follows. os.path.getsize() counts the header + 8-byte prefix, so
    # summing it makes the field a census of files and leaves any "index vs payload"
    # reader explaining a 96,320 B residue as overhead. Keep both, labelled.
    _payload = max((v["data_offsets"][1] for v in _hdr.values()), default=0)
    out_files.append((name, os.path.getsize(p), _payload))
    print(f"  {name}  {os.path.getsize(p)/1e9:.3f} GB  ({len(buf)} tensors)", flush=True)
    shard_no += 1
    buf, buf_bytes = {}, 0

seen = set()
for f in src_files:
    with st.safe_open(os.path.join(SRC, f), framework="pt") as fh:
        for k in fh.keys():
            if k in seen:
                continue
            seen.add(k)
            dt, sh, _ = meta[k]
            t = fh.get_tensor(k)
            assert t.shape == torch.Size(sh)

            if k == "model.embed_tokens.weight":
                struct_t = t[:, STRUCT].contiguous()
                live_t = t[:, LIVE]
                if LIVE_PAD:
                    live_t = torch.cat([live_t, torch.zeros(live_t.shape[0], LIVE_PAD, dtype=live_t.dtype)], 1)
                p, s = q4(live_t.contiguous())
                buf[k + ".live.q4"] = p
                buf[k + ".live.scales"] = s
                buf[k + ".struct"] = struct_t
                manifest[k] = {"kind": "q4g64_live", "orig_shape": list(sh), "orig_dtype": dt,
                               "struct_cols": STRUCT, "live_cols": len(LIVE), "live_pad": LIVE_PAD,
                               "q4": k + ".live.q4", "scales": k + ".live.scales", "struct": k + ".struct"}
                buf_bytes += p.numel() + s.numel() * 2 + struct_t.numel() * 2
                del struct_t, live_t
            elif packable(k, sh):
                p, s = q4(t)
                buf[k + ".q4"] = p
                buf[k + ".scales"] = s
                manifest[k] = {"kind": "q4g64", "orig_shape": list(sh), "orig_dtype": dt,
                               "q4": k + ".q4", "scales": k + ".scales"}
                buf_bytes += p.numel() + s.numel() * 2
            else:
                buf[k] = t
                manifest[k] = {"kind": "raw", "orig_shape": list(sh), "orig_dtype": dt}
                buf_bytes += t.numel() * t.element_size()
            del t
            if buf_bytes >= TGT:
                flush()
flush()

n = len(out_files)
for i, (old, _, _) in enumerate(out_files):
    new = f"model-{i+1:05d}-of-{n:05d}.safetensors"
    if old != new:
        os.rename(os.path.join(DST, old), os.path.join(DST, new))
        for k, v in shard_map.items():
            if v == old:
                shard_map[k] = new

json.dump({"metadata": {"total_size": sum(pl for _, _, pl in out_files),
                        "total_file_bytes": sum(z for _, z, _ in out_files)},
           "weight_map": {k: shard_map[k] for k in sorted(shard_map)}},
          open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=2)
json.dump({"format": "int4_groupwise", "group_size": G, "axis": "input_dim",
           "packing": "two int4 per uint8; element 2i low nibble, 2i+1 high nibble",
           "codebook": "w ~= clip(round(w/(absmax/7)),-8,7) * (absmax/7); absmax per 64-dim group; fp16 scales",
           "preserved_exact": "layers 0-10, all norms/biases/conv1d, embed structural cols",
           "embed_structural_cols": STRUCT,
           "tensors": manifest},
          open(os.path.join(DST, "pack_manifest.json"), "w"), indent=1)

for f in os.listdir(SRC):
    if f.endswith(".safetensors") or f.startswith("model.safetensors.index"):
        continue
    if f in ("config.json", "generation_config.json", "chat_template.jinja", "tokenizer.json",
             "tokenizer_config.json", "chipformer.json", "LICENSE"):
        shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))

print(f"\npacked in {time.time()-t0:.1f}s  shards={n}  tensors={len(manifest)}")
