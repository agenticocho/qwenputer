#!/usr/bin/env python3
"""Build a real sparse container for ALL 154 chip tensors and verify lossless round-trip.

Format (little-endian, self-describing):
  magic 'SPRV1' | u32 n_tensors | u32 meta_len | meta JSON | payload
meta[i] = {name, shape, dtype, rows, k, nzb, off_vals, off_cols, off_cnt, off_rowptr}
payload per 2-D tensor: bf16 vals (nz*2B), u16 cols (nz*2B), u16 row counts (rows*2B)
         per 1-D/3-D tensor: raw bf16 values verbatim (they are small)
Writes to sparse_chip.sparse. Verifies every tensor decodes bit-identically.
"""
import json, os, struct, collections
import numpy as np

D = "os.environ.get("QWENPUTER_ART","build_rv1_4b_q4b")
OUT = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(OUT, "sparse_chip.sparse")
idx = json.load(open(os.path.join(D, "model.safetensors.index.json")))
wm = idx["weight_map"]
meta = {}
for f in sorted(set(wm.values())):
    with open(os.path.join(D, f), 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__": continue
        meta[k] = dict(shape=tuple(v["shape"]), file=f, abs=(8 + n + v["data_offsets"][0]))
mm = {}
def M(f):
    if f not in mm: mm[f] = np.memmap(os.path.join(D, f), dtype=np.uint8, mode='r')
    return mm[f]
def W(n):
    m = meta[n]
    return np.frombuffer(M(m["file"]), np.uint16, count=int(np.prod(m["shape"])),
                         offset=m["abs"]).reshape(m["shape"])

chip = sorted([k for k in meta if k.startswith("model.layers.") and int(k.split(".")[2]) <= 10],
              key=lambda x: (int(x.split(".")[2]), x))

blobs, desc = {}, []
cursor = 0
orig = {}          # for verification
for name in chip:
    w = W(name)
    sh = list(w.shape)
    flat = w.reshape(-1)
    if len(sh) == 2:
        nz = (w != 0) & (w != 0x8000)
        cnt = nz.sum(axis=1).astype(np.uint16)
        rr, cc = np.nonzero(nz)
        vals = w[nz]
        cols = cc.astype(np.uint16)
        d = dict(name=name, shape=sh, kind="csr", rows=sh[0], k=sh[1], nzb=int(len(vals)),
                 off_vals=cursor); blobs[name] = vals.tobytes(); cursor += len(vals) * 2
        d["off_cols"] = cursor; blobs[name + "#c"] = cols.tobytes(); cursor += len(cols) * 2
        d["off_cnt"] = cursor; blobs[name + "#n"] = cnt.tobytes(); cursor += len(cnt) * 2
        d["nnz"] = int(len(vals))
    else:
        d = dict(name=name, shape=sh, kind="raw", nelem=int(flat.size),
                 off_vals=cursor)
        blobs[name] = flat.tobytes(); cursor += flat.size * 2
        d["nnz"] = int(flat.size)
    desc.append(d)

mjson = json.dumps(desc, separators=(",", ":")).encode()
header = b"SPRV1" + struct.pack("<II", len(desc), len(mjson)) + mjson
with open(DST, "wb") as fh:
    fh.write(header)
    for name in chip:
        if len(meta[name]["shape"]) == 2:
            fh.write(blobs[name]); fh.write(blobs[name + "#c"]); fh.write(blobs[name + "#n"])
        else:
            fh.write(blobs[name])

sz = os.path.getsize(DST)
print(f"wrote {DST}")
print(f"  {len(desc)} tensors, header {len(header):,d} B, payload {cursor:,d} B, file {sz:,d} B "
      f"= {sz/2**20:.3f} MiB = {sz/2**30:.5f} GiB")
print(f"  dense chip on disk was {sum(int(np.prod(meta[k]['shape']))*2 for k in chip):,d} B "
      f"= {sum(int(np.prod(meta[k]['shape']))*2 for k in chip)/2**30:.4f} GiB")
print(f"  ratio {sum(int(np.prod(meta[k]['shape']))*2 for k in chip)/sz:.1f}x")

# ---- round-trip verification ----
raw = open(DST, "rb").read()
assert raw[:5] == b"SPRV1"
n_t, ml = struct.unpack("<II", raw[5:13])
dd = json.loads(raw[13:13 + ml])
p0 = 13 + ml
bad = 0; checked = 0
for d in dd:
    base = p0 + d["off_vals"]
    w0 = W(d["name"])
    if d["kind"] == "raw":
        got = np.frombuffer(raw, np.uint16, count=d["nelem"], offset=base).reshape(d["shape"])
        ok = np.array_equal(got, w0)
    else:
        rows, k = d["rows"], d["k"]
        vals = np.frombuffer(raw, np.uint16, count=d["nzb"], offset=base)
        cols = np.frombuffer(raw, np.uint16, count=d["nzb"], offset=p0 + d["off_cols"])
        cnt = np.frombuffer(raw, np.uint16, count=rows, offset=p0 + d["off_cnt"])
        rec = np.zeros((rows, k), np.uint16)
        starts = np.concatenate([[0], np.cumsum(cnt.astype(np.int64))[:-1]])
        rr = np.repeat(np.arange(rows), cnt)
        rec[rr, cols] = vals
        ok = np.array_equal(rec.reshape(w0.shape), w0)
    checked += 1
    if not ok:
        bad += 1; print("  MISMATCH", d["name"])
print(f"\nround-trip: {checked} tensors re-read from the file and compared to the source, "
      f"{bad} mismatches -> {'LOSSLESS' if bad == 0 else 'LOSSY'}")
json.dump(dict(file=DST, bytes=sz, tensors=len(desc), payload=cursor,
               dense_bytes=sum(int(np.prod(meta[k]['shape']))*2 for k in chip),
               roundtrip_mismatch=bad), open(os.path.join(OUT, "sparse_pack.json"), "w"), indent=1)
