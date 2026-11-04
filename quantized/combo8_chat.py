#!/usr/bin/env python3
"""The combination run: sparse chip + int4 host + int4 embedding in ONE process.

Same decoder, same prompts, same grading as `qwenputer_sparse_20261103T1120/sparse_subset.py`
(the artifact's own `chipformer.rv.chat.RvChat`, greedy, `check_head=True`, `lock=Session(RV32
reference)`); the only change is that `SparseRvChat`'s loader is replaced by `combo_linear.load_combo`,
which takes the chip from the 5.58 MiB CSR container AND the host/embedding from the int4 pack.

Modes:
  smoke    load, then a 16-token decode with per-token timing (cost model + load memory)
  run      the 5 reference questions, graded, with peak device memory
  control  same container with every chip matrix emptied (expect no answer, 0 chip tokens)
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SRC = os.path.expanduser("~/qwenputer/experimental/riscv")
sys.path.insert(0, SRC)

import torch  # noqa: E402

import combo8_linear as C8  # noqa: E402
import fused_linear as FL  # noqa: E402
import sparse_linear as SL  # noqa: E402
from chipformer.rv.chat import RvChat  # noqa: E402

MODEL = os.environ.get("QWENPUTER_ART","build_rv1_4b_q4b")
EMBED_MODE = os.environ.get("EMBED_MODE", "int8")
REF = os.path.expanduser("~/workspace/localdyno/qwenputer_subset_20261102T060108/subset.json")
FW = os.path.join(SRC, "tests/data/rv/calc-bridge.elf")
MAX_NEW = 4000
NUDGE = ("You have a calc tool for exact arithmetic. Use it for every arithmetic step instead "
         "of calculating by hand.")

SHIM_CHECK = {}


# ------------------------------------------------------------------ toolchain shim (verbatim)
def install_inject_shim():
    """rv1's hf.inject is written for transformers >=5.19 (`recurrent_states[0]`); on 5.5.4 the
    state is one [1, H, K, V] tensor, so that index drops the batch axis. The same 4-line shim as
    qwenputer_rv1_20261103T062259 / qwenputer_sparse_20261103T1120, verified by read-back."""
    from chipformer.rv import hf as HF

    def inject(c, cache):
        r = c.target.value_heads // c.target.key_heads
        for ni, (L, j) in HF.head_slots(c).items():
            rs = cache.layers[L].recurrent_states
            if isinstance(rs, (list, tuple)):
                st = rs[0]
            elif getattr(rs, "ndim", 0) >= 5:
                st = rs[0]
            else:
                st = rs
            init = c.nodes[ni].init or []
            for x in range(len(c.nodes[ni].vheads())):
                S = st[0, j * r + x]
                S.zero_()
                for row, vec in (init[x] if x < len(init) else {}).items():
                    S[row] = torch.tensor(vec, dtype=S.dtype, device=S.device)
        SHIM_CHECK.clear()
        SHIM_CHECK.update(verify_inject(c, cache))

    HF.inject = inject


def verify_inject(c, cache):
    from chipformer.rv import hf as HF
    r = c.target.value_heads // c.target.key_heads
    n = bad = 0
    for ni, (L, j) in HF.head_slots(c).items():
        rs = cache.layers[L].recurrent_states
        st = rs[0] if isinstance(rs, (list, tuple)) or getattr(rs, "ndim", 0) >= 5 else rs
        init = c.nodes[ni].init or []
        for x in range(len(c.nodes[ni].vheads())):
            S = st[0, j * r + x]
            for row, vec in (init[x] if x < len(init) else {}).items():
                n += 1
                want = torch.tensor(vec, dtype=S.dtype, device=S.device)
                if not bool((S[row] == want).all()):
                    bad += 1
    return {"entries": n, "mismatches": bad}


# ------------------------------------------------------------------ the decoder
class ComboRvChat(RvChat):
    """RvChat over one model whose 43 layers are served in two different formats."""

    def __init__(self, path=MODEL, sparse=SL.SPARSE, device="mps", chunk_bytes=None):
        import transformers
        from chipformer.q35.fast import exact_matmuls
        from chipformer.q35.ir import Target
        from chipformer.rv import bridge as BR, core
        from transformers import AutoTokenizer

        exact_matmuls()
        self.path = Path(path)
        self.meta = json.loads((self.path / "chipformer.json").read_text())
        self.dev = device
        cb = FL.CHUNK_BYTES if chunk_bytes is None else chunk_bytes
        self.chunk_bytes = cb
        t0 = time.time()
        self.model, self.chip, self.scratch, self.loaded, self.embed_stats = C8.load_combo8(
            path, sparse, dev=device, chunk_bytes=cb, embed_mode=EMBED_MODE)
        self.load_s = time.time() - t0
        self.lm = self.model.model
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.T = BR.Text(self.path / "tokenizer.json")
        cfg = self.model.config
        self.c = core.build(Target(cfg.linear_num_key_heads, cfg.linear_num_value_heads,
                                   cfg.intermediate_size), self.T)
        self.n_chip, self.v0 = self.meta["chip_layers"], self.meta["chip_token_offset"]
        self.code_dims = torch.tensor(self.meta["code_dims"], device=self.dev)
        self.eos = {248046, 248044}
        # MPS: no CUDA graph and no triton executor -- the HF-module path, as in both prior runs
        self.fast = False
        self.executor = False
        self.verify = False


# ------------------------------------------------------------------ helpers
def answer_number(text):
    nums = re.findall(r"\d[\d,]*", text.replace("**", ""))
    return int(nums[-1].replace(",", "")) if nums else None


class MemSampler:
    """Peak `torch.mps.driver_allocated_memory()` -- the device-side figure a card budges on."""

    def __init__(self, hz=2.0):
        self.dt = 1.0 / hz
        self.peak = 0
        self.peak_current = 0
        self.series = []
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        while not self._stop.is_set():
            try:
                d = torch.mps.driver_allocated_memory()
                c = torch.mps.current_allocated_memory()
            except Exception:
                break
            self.series.append((round(time.time(), 2), d, c))
            self.peak = max(self.peak, d)
            self.peak_current = max(self.peak_current, c)
            self._stop.wait(self.dt)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2.0)


def snapshot(tag):
    torch.mps.synchronize()
    return {"tag": tag, "driver_gib": round(torch.mps.driver_allocated_memory() / 1024 ** 3, 4),
            "current_gib": round(torch.mps.current_allocated_memory() / 1024 ** 3, 4),
            "recommended_gib": round(torch.mps.recommended_max_memory() / 1024 ** 3, 3)}


def empty_chip(chat):
    """Negative control: every chip matrix emptied. The scratch block then rebuilds all-zero."""
    n = 0
    for mod in chat.model.modules():
        if isinstance(mod, SL.CsrLinear):
            mod.flat = torch.empty(0, dtype=torch.long, device=mod.flat.device)
            mod.vals = torch.empty(0, dtype=torch.bfloat16, device=mod.vals.device)
            mod.nnz = 0
            n += 1
    left = sum(int((m.vals != 0).sum()) for m in chat.model.modules()
               if isinstance(m, SL.CsrLinear))
    return {"matrices_emptied": n, "nonzero_vals_remaining": left}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="run", choices=["run", "smoke", "control"])
    ap.add_argument("--dev", default="mps")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--chunk-mib", type=float, default=48.0)
    ap.add_argument("--max-wall", type=float, default=4200.0, help="stop cleanly at this many s")
    ap.add_argument("--only", type=int, default=None, help="single prompt index for smoke/control")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)

    ref = json.load(open(REF))["rows"]
    if args.only is not None:
        items = [(ref[args.only]["q"], ref[args.only]["op"], ref[args.only]["expected"])]
    else:
        items = [(r["q"], r["op"], r["expected"]) for r in ref[: args.n]]

    rec = {"artifact": MODEL, "chip_container": SL.SPARSE,
           "chip_container_bytes": os.path.getsize(SL.SPARSE),
           "chunk_mib": args.chunk_mib, "mode": args.mode,
           "harness": f"chipformer.rv.chat.RvChat; chip 0-10 CSR + host 11-42 int4-g64 + embed {EMBED_MODE} + contiguous-head fix",
           "firmware": FW, "max_new": args.max_new, "device": args.dev, "n": len(items),
           "prompts_from": REF, "rows": [], "ok": False}
    t0 = time.time()
    sampler = MemSampler(hz=2.0)

    def flush():
        rec.setdefault("correct", sum(1 for r in rec["rows"] if r["correct"]))
        rec.setdefault("total", len(rec["rows"]))
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["peak_driver_gib"] = round(sampler.peak / 1024 ** 3, 4)
        rec["peak_current_gib"] = round(sampler.peak_current / 1024 ** 3, 4)
        tmp = os.path.join(out, "subset.json.tmp")
        json.dump(rec, open(tmp, "w"), indent=2)
        os.replace(tmp, os.path.join(out, "subset.json"))
        with open(os.path.join(out, "rows.jsonl"), "w") as f:
            for r in rec["rows"]:
                f.write(json.dumps(r) + "\n")

    sampler.__enter__()
    try:
        import transformers
        from chipformer.rv import bridge as BR, elf, ref as RVREF

        install_inject_shim()
        rec["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                           "transformers": transformers.__version__}
        rec["mps_before"] = snapshot("before")
        print(f"[env] torch {torch.__version__} transformers {transformers.__version__} "
              f"dev {args.dev} chunk {args.chunk_mib} MiB", flush=True)

        chat = ComboRvChat(device=args.dev, chunk_bytes=int(args.chunk_mib * 1024 * 1024))
        rec["load_s"] = round(chat.load_s, 1)
        rec["loaded"] = chat.loaded
        rec["scratch_mib"] = round(chat.scratch.numel * 2 / 1024 ** 2, 2)
        rec["chunk_ops_per_forward"] = C8.CL.chunk_ops(chat.model)
        table, total = C8.resident_bytes8(chat.model, chat.scratch)
        rec["resident_gib"] = {k: round(v, 4) for k, v in table.items()}
        rec["resident_total_gib"] = round(total, 4)
        rec["embed_stats"] = chat.embed_stats
        rec["after_load"] = snapshot("after_load")
        print(f"[load] {chat.load_s:.1f}s  {rec['after_load']}", flush=True)
        print(f"[resident] {json.dumps(rec['resident_gib'], indent=1)} total {total:.4f} GiB",
              flush=True)
        print(f"[chunk ops / host forward] {rec['chunk_ops_per_forward']}", flush=True)
        if args.mode == "control":
            rec["control"] = empty_chip(chat)
            print(f"[control] {rec['control']}", flush=True)
        flush()

        if args.mode == "smoke":
            q = items[0][0]
            ids = chat.prompt_ids([{"role": "system", "content": NUDGE},
                                   {"role": "user", "content": q}])
            rec["prompt_tokens"] = len(ids)
            stamps, kinds = [], []

            def on_token(i):
                stamps.append(time.time())
                kinds.append("chip" if i >= chat.v0 else "text")

            m = RVREF.Machine.from_program(elf.program(FW))
            BR.park(m)
            t = time.time()
            r = chat.generate(ids, m, args.max_new, max_chip=60000,
                              check_head=False, lock=None, on_token=on_token)
            dt = time.time() - t
            deltas = [b - a for a, b in zip(stamps, stamps[1:])]
            text_d = [d for d, k in zip(deltas, kinds) if k == "text"]
            chip_d = [d for d, k in zip(deltas, kinds) if k == "chip"]
            rec["smoke"] = {"prompt_tokens": len(ids), "wall_s": round(dt, 1),
                            "text_tokens": r["text_tokens"], "chip_tokens": r["chip_tokens"],
                            "n_timed": len(deltas),
                            "text_s_per_tok": round(sum(text_d) / len(text_d), 3) if text_d else None,
                            "chip_s_per_tok": round(sum(chip_d) / len(chip_d), 3) if chip_d else None,
                            "first_delta_s": round(deltas[0], 2) if deltas else None,
                            "gen_text_tail": chat.text_of(r["tokens"])[-300:],
                            "transient_peak_mib": round(FL.transient_peak_mib(), 1),
                            "transient_chunks": FL._TRANSIENT["chunks"]}
            rec["after_smoke"] = snapshot("smoke")
            rec["ok"] = True
            print(f"[smoke] {json.dumps(rec['smoke'])}", flush=True)
            print(f"[smoke] {rec['after_smoke']}", flush=True)
            flush()
            return 0

        for i, (q, op, expected) in enumerate(items):
            if time.time() - t0 > args.max_wall:
                rec["stopped"] = f"wall budget {args.max_wall}s hit before prompt {i}"
                print(f"[stop] {rec['stopped']}", flush=True)
                break
            exp_int = int(expected)
            ids = chat.prompt_ids([{"role": "system", "content": NUDGE},
                                   {"role": "user", "content": q}])
            m = RVREF.Machine.from_program(elf.program(FW))
            BR.park(m)
            m2 = RVREF.Machine.from_program(elf.program(FW))
            BR.park(m2)
            lock = BR.Session(chat.T, m2)
            lock.history(ids)
            b0 = chat.scratch.builds
            t = time.time()
            seen = {"chip": 0, "text": 0}
            last = [t]

            def on_token(tok, _seen=seen, _last=last):
                _seen["chip" if tok >= chat.v0 else "text"] += 1
                if (sum(_seen.values())) % 50 == 0:
                    print(f"      [prog] {sum(_seen.values())} toks "
                          f"({_seen['chip']} chip, {_seen['text']} text) "
                          f"{time.time() - t:.0f}s", flush=True)

            r = chat.generate(ids, m, args.max_new, max_chip=60000, check_head=True,
                              lock=lock, on_token=on_token)
            dt = time.time() - t
            text = chat.text_of(r["tokens"])
            got = answer_number(text)
            runs = [chat.T.decode_bytes(e).decode("utf-8", "replace") for e in lock.runs]
            values = [(s.split("\n")[2] if len(s.split("\n")) > 2 else s.strip()) for s in runs]
            row = {"i": i, "q": q, "op": op, "expected": str(expected), "got": got,
                   "correct": got == exp_int, "wall_s": round(dt, 1),
                   "prompt_tokens": len(ids),
                   "gen_text_tokens": r["text_tokens"], "gen_chip_tokens": r["chip_tokens"],
                   "chip_s": r["chip_seconds"], "text_s": r["text_seconds"],
                   "lock_error": r["lock_error"], "head_mismatch": r["head_mismatch"],
                   "machine_runs": values,
                   "machine_all_exact": (all(v.strip() == str(exp_int) for v in values)
                                         if values else None),
                   "inject": dict(SHIM_CHECK),
                   "csr_rebuilds": chat.scratch.builds - b0,
                   "mem_after": snapshot(f"p{i}"),
                   "text_tail": text[-400:]}
            rec["rows"].append(row)
            flush()
            print(f"  [{i+1}/{len(items)}] {q[:36]:38s} want {expected} got {got} "
                  f"{'OK' if row['correct'] else 'MISS'}  {dt:6.0f}s "
                  f"text={r['text_tokens']} chip={r['chip_tokens']} "
                  f"drv={row['mem_after']['driver_gib']:.2f}GiB "
                  f"runs={values} lock={r['lock_error']} head={r['head_mismatch']}", flush=True)

        rec["ok"] = True
        rec["correct"] = sum(1 for r in rec["rows"] if r["correct"])
        rec["total"] = len(rec["rows"])
        rec["exact_rate"] = round(rec["correct"] / rec["total"], 4) if rec["total"] else None
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-4000:]
        print("FAILED:", rec["error"], file=sys.stderr, flush=True)
        print(rec["traceback"], file=sys.stderr, flush=True)
    finally:
        rec["total_s"] = round(time.time() - t0, 1)
        sampler.__exit__()
        rec["peak_driver_gib"] = round(sampler.peak / 1024 ** 3, 4)
        rec["peak_current_gib"] = round(sampler.peak_current / 1024 ** 3, 4)
        rec["mem_series_n"] = len(sampler.series)
        rec["transient_peak_mib"] = round(FL.transient_peak_mib(), 1)
        rec.setdefault("correct", sum(1 for r in rec["rows"] if r["correct"]))
        rec.setdefault("total", len(rec["rows"]))
        rec["after_run"] = snapshot("end")
        flush()

    if rec.get("ok"):
        print(f"\n  RESULT {rec['correct']}/{rec['total']} exact  "
              f"peak driver {rec['peak_driver_gib']:.2f} GiB  "
              f"{rec['total_s']/60:.1f} min", flush=True)
    return 0 if rec.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
