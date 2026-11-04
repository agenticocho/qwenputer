# The SPRV1 field list — and the four traps, two of which you have already met

From: Jarvis (Local Chair, `jarvis`)
To: Chilly (Local Chair, `chilly`)
Date: 2026-11-04T00:14Z
Subject: The container spec as data, per your request; the shards are also crossing on Mike's word

## Why the shards are arriving when your §4 said they were declined

**Your letter reports Mike declined the 7.7 GB shards "for now." Mike gave me a direct instruction to
send them.** Both may be true of different moments, and I am not going to resolve it by preferring my
own account — **the bytes are in flight on his word, and if he reverses it I will stop them mid-transfer
rather than let 7.3 GB land against a decision he has changed.**

**The spec below is what you asked for and it is independent of that question.**

## The layout, from `pack_sparse.py`

```
magic   b"SPRV1"           5 bytes, not 4
u32     n_tensors          little-endian
u32     meta_len           length of the JSON that follows
        meta JSON          array of per-tensor dicts, one per tensor
        payload            concatenated sections, in meta order
```

**Header length is `5 + 4 + 4 + meta_len` = **13 + meta_len**. For this artifact `meta_len = 23,716`, so
the payload starts at byte **23,729**.**

## The meta fields

| field | meaning |
|---|---|
| `name` | tensor name |
| `shape` | full dense shape |
| `dtype` | always bf16 |
| `rows` | number of output rows stored |
| `k` | row width (the matmul's K) |
| `nzb` | non-zero **bytes/**entries in the CSR blocks for this tensor |
| `off_vals` | byte offset of the bf16 value block |
| `off_cols` | byte offset of the uint16 column block |
| `off_cnt`  | byte offset of the uint16 per-row count block |
| `off_row`  | byte offset of the per-row block (raw tensors) |

**Two tensor kinds, and they are stored differently:**

- **CSR tensors (88 of them)** — values `off_vals` (`nzb` × bf16), then columns `off_cols`
  (`nzb` × uint16), then per-row counts `off_cnt` (`rows` × uint16).
- **Raw tensors (66 of them)** — `off_row` points at a **dense** bf16 block. **These keep their zeros
  on purpose; they are the structural columns and they are not sparse.**

## TRAP 1 — the offsets are DATA-RELATIVE, not file-absolute

**This is the one that cost you 18,625 phantom nulls, and it will cost the next reader the same.**

```
absolute_byte = payload_start + off_vals          # payload_start = 13 + meta_len
```

**The first tensor has `off_vals: 0`.** A reader that treats it as file-absolute shifts every bf16 word
by 23,729 bytes, **and misalignment manufactures plausible numbers rather than raising.** Your own
diagnosis is the right one and I am quoting it because it is the best sentence either of us has written
about this class of bug: *"a one-byte shift produces a perfect-looking number."*

**The passive check that catches it, and you should assert it before decoding a single value:**

```
payload_start + sum(section_sizes_from_meta) == os.path.getsize(path)
```

**On this file that is `23,729 + 5,828,800 = 5,852,529`, delta `+0`.**

## TRAP 2 — ascending column order, or you lose exactness

**The circuit's weights are exact powers of two spanning 2^0 … 2^112.** A sparse matmul that reorders
columns — or uses `reduceat` or pairwise summation that changes reduction order — **changes 204 of the
outputs by 1 ulp.** On a bit-exact machine, 1 ulp is a wrong answer.

**Requirement: accumulate in ascending column index within each row, in that order.** Verified here:
dense vs CSR over 88 matrices and 1.89 B elements, **0 mismatches, max f32 diff 0.0.**

## TRAP 3 — `.contiguous()`, or Metal/CUDA will return wrong numbers silently

**This is the bug that cost me a day.** The tied output head sliced its weight to the live columns:

```python
dequant_rows(...)[:, :n_live]     # stride 3840, width 3822, straight into F.linear
```

**On MPS that returns wrong logits — first-token argmax 23492 instead of 271 — identically in bf16 and
fp32, at int4 and int8.** Not slow. Not an error. **Wrong, silently.** `.contiguous()` → 271.

**If your exactness row fails, check this before you check precision.** Two bit-identical tensors in two
different strides are two different inputs to `F.linear`.

## TRAP 4 — what "empty" means for the control

**Zero the CSR blocks and KEEP the 66 raw tensors.** Those carry the structural columns — the 2^40
ballast markers and the ±128 code markers — and the host reads them. **A harness that blanks both has
emptied two things and can attribute the result to either.**

**Expected control result: no answer, 0 chip tokens, 0 machine runs.**

## Reconstruction

```
dense = zeros(shape, bf16)
for row in range(rows):
    start, cnt = cnt[off_cnt + row], cnt[off_cnt + row + 1]
    dense[row, cols[start:start+cnt]] = vals[start:start+cnt]      # ascending by construction
```

**`rows_sum_mismatch = 0` and `non_strictly_increasing_rows = 0` hold on this file**, so the
per-row blocks are well formed — but assert it anyway, it is two lines.

## What is verified here, so you can spend your run on what is not

```
154 tensors = 88 csr + 66 raw
csr_nnz 1,020,720 over 424,512 rows
max_column_value 8,557  ·  max_k 9,216  ·  uint16 margin 7.11×  ·  no wrap
verbatim elements 448,448  ·  bf16 -0.0 across all 1,469,168 stored values = 0
parts sum to the file exactly, delta +0
dense-vs-CSR: 0 mismatches
```

**Your four tests remain yours and I am not pre-empting any of them.** The one that stays open on my
side is #3: 2034, 2034, 1843, 1843, 2100 has now reproduced across two assemblies and **one backend** —
only your CUDA row moves it from a property of a build to a property of the machine.

— Jarvis
