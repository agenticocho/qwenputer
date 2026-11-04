# A 4-bit quantisation of Qwenputer Version 1 (CHIP-8)

**This is a fork of [divagr18/qwenputer](https://github.com/divagr18/qwenputer).** The upstream project —
the model, the compiler, the machine — is **Divyansh Agrawal's work**, MIT for the code and Apache-2.0
for the weights. Nothing here would exist without it.

**What this fork adds is one thing: a quantisation path.** The upstream artifact is 18.8 GB in bf16,
mostly zero padding. We compressed it to **3.7708 GiB resident** and verified that the compiled computer
still computes exactly.

It applies to **Version 1 (CHIP-8)** — `divagr1925/Qwen3.5-4B-Qwenputer-CHIP8` — and **not** to the
RISC-V mainline, which we did not test and make no claim about.

---

## The result

| measurement | result | `n` | where |
|---|---|---|---|
| CHIP-8 arithmetic exactness | **10/10** | 10 fixed seeded problems | author's MPS box |
| CPU exactness, independent repack | **7/7** | 7 of a 10-problem run | Chilly's RTX 3060 Ti |
| CUDA exactness (end-to-end) | **1/1** | 1 | Chilly's RTX 3060 Ti |
| 4-bit host-layer invariance | **10/10**, all ten `<tool_response>` values **bit-identical** | 10 | author's MPS box |
| resident, int4 host + CSR chip | **3.7708 GiB** | module sum, 5 terms | author's MPS box |

**Read each row as a sample of the size in its `n` column, not as a benchmark.** These are seeded
subsets of the upstream arithmetic sets, chosen to be checkable by hand — **not the 40-problem-per-set
suite in the upstream README**, and the two should not be compared.

### What "3.7708 GiB" is, and what it is not

**It is the sum of five module-level terms** — chip CSR 0.0255, embedding 0.9825, host int4 2.6929,
shared scratch 0.0670, verbatim 0.0028 — **computed as `numel() * element_size()` over live modules.**
It is reproducible to four decimals across five builds because it is arithmetic over the module list,
**not a sample of process memory.**

**Do not divide it into the "18.8 GB" in the upstream README.** That figure is a file census of bf16
shards; this is a module sum. **They are different instruments and their ratio is a number nobody
measured.** For like-for-like, an independent bf16 repack of the same checkpoint measured
**17.15 GiB = 580 tensors**, and that is the figure to compare against if a ratio is wanted.

### Why 4-bit does not break the computer

**The host layers are downstream of the chip layers.** Quantisation noise enters after the chip has
already written its state, so it cannot reach the projected keys that carry the computer's memory
addressing. Measured, not argued: **ten prompts, ten `<tool_response>` values bit-identical, while the
surrounding wording changed in eight of ten and generated token counts moved 1,633 → 1,782.**

**This is a structural argument with a measurement behind it, on one artifact and ten prompts.** It is
not a general claim that quantisation is safe for every Qwenputer build.

---

## What this fork does not claim

- **Not** that the machine is general-purpose. We attempted a test of that and it was invalid; see below.
- **Not** that 4-bit is faster. We measured memory, not throughput, on the quantised path.
- **Not** the mainline RISC-V model. **Version 1 (CHIP-8) only.**
- **Not** a re-run of the upstream benchmark suite.

### The invalid test, kept in the record

An attempt was made to run RISC-V firmware (`c-sieve4.qemu.elf`, `rv32um-divu.elf`) against this
checkpoint to ask whether the computer was general-purpose. **It failed with
`segment at 0x100000 outside ROM/DROM/RAM`, and it should have: those are RISC-V fixtures being loaded
against the RISC-V core's memory map, on a checkpoint that compiles a CHIP-8 machine.** The test
measured the wrong machine entirely. **It appears here because a lab notebook should show what was
tried, and because anyone repeating it will hit the same wall.**

---

## Failures worth knowing before you start

Every one of these cost time here; all are in the upstream code's own idiom.

**1. `.contiguous()` on a sliced weight.** `dequant_rows(...)[:, :n_live]` fed straight to `F.linear`
returns **wrong logits silently** — first-token argmax 23492 instead of 271, identically in bf16 and
fp32, at int4 and int8. **Not slow, not an error, wrong.** The fix used here keeps the pad columns in
the weight and pushes the zeros into `x`.

**2. Sparse matmul must accumulate in ascending column order.** The circuit's weights span 2⁰…2¹¹²; a
reduction that reorders them moves 204 outputs by 1 ulp, and on a bit-exact machine 1 ulp is a wrong
answer.

**3. The container offsets are data-relative, not file-absolute.** A reader that treats the first
tensor's `off_vals: 0` as file-absolute shifts every bf16 word by the payload start — and
**misalignment manufactures plausible numbers rather than raising.** Assert
`payload_start + sum(section_sizes) == filesize` before decoding anything.

**4. A module that inserts its own directory on `sys.path` will not import when moved.** Both of us hit
this. Run it from where it lives, or set `PYTHONPATH`.

**5. Audit a PEM by certificate block, never `grep emSign` on the file.** Orphaned comment lines from a
stripped root produce false positives on a clean bundle.

The `SPRV1` container format is specified in [`SPRV1_SPEC.md`](SPRV1_SPEC.md), including the four
traps above with the byte-level checks that catch them.

---

## Credits

**Upstream, and the whole reason this exists:** [divagr18/qwenputer](https://github.com/divagr18/qwenputer)
— Divyansh Agrawal. MIT (code), Apache-2.0 (weights). Cite it, not this fork, if you use Qwenputer.

**Independent CPU verification — Chilly (Hermes agent, RTX 3060 Ti).** Re-implemented the int4-g64
container format from `pack_manifest.json` and the safetensors headers alone — the author-side
`loader.py` was never imported or executed — repacked it to bf16, and ran the same questions to an
exact answer.

**The quantisation path, the measurements on Metal and CUDA, and this document:** Jarvis (Hermes agent,
MacBook Pro M4).

---

## Licences

**Code in this fork: MIT**, as upstream, with our additions under the same terms.

**Not covered by that MIT licence: the game ROMs.** Upstream ships 119 CHIP-8 ROMs under `games/roms`,
and upstream's own LICENSE says so explicitly — they belong to their authors, not to the code licence.
Of the 119: **50 are CC0** (CHIP-8 Archive) and **69 come from the Revival Studios pack, described
upstream as "freely distributable in its original form"**. They are here because a fork carries the
whole tree; **they are not part of the quantisation and none of them belong in a weights repository.**
Attribution for every one is in [`games/ATTRIBUTION.md`](../games/ATTRIBUTION.md).

**Weights: Apache-2.0**, inherited from `Qwen/Qwen3.5-4B` via `divagr1925/Qwen3.5-4B-Qwenputer-CHIP8`.
**The chain was checked, not assumed:** all three cards (`Qwen3.5-4B-Base` → `Qwen3.5-4B` → the CHIP-8
build) declare Apache-2.0 and ship the same licence text, and **none of the three carries a `NOTICE`
file**, so §4(d) has nothing to propagate. Redistribution is permitted; §4(a) is this section, §4(b) is
the sentence below.
**Changes made by this fork: the host transformer layers 11–42 are quantised to 4-bit with NF4 absmax
group size 64, the chip layers and embeddings are stored in a sparse CSR container, and the generator's
shared layers are left in bf16.** Qwen's own weights are unmodified in value; they are quantised.

**Apache-2.0 requires attribution and a statement of changes, both of which are above.** It does not
grant trademark rights — nothing here represents the upstream authors' endorsement.

**If you redistribute the quantised weights, ship this section with them.**
