"""T004 — what each optimisation actually reduces, one factor at a time.

The campaign has measured these optimisations in separate runs and it became possible to
attribute a gain to the wrong cause. Two concrete cases:

* E4's `step_only` compiles `denoise_step` and leaves `config.compile_model` unset, so
  TF32 is never enabled; E11f loaded the policy with `compile_mode=`, which DOES set it
  (`modeling_pi0.py:591-595` calls `torch.set_float32_matmul_precision("high")` one line
  before `torch.compile`). The prefix then looked 1.41x faster "under compile" when the
  cause was the precision flip.
* Compiling `sample_actions` (LeRobot's own path) and compiling `denoise_step` (our
  `step_only` variant) are different interventions with different scopes, and an isolated
  action-expert loop that calls `denoise_step` directly is untouched by the former.

This runs ONE ladder in which each rung differs from the one above it by exactly one
factor, and reports every region, so the attribution is read off the table instead of
argued.

**Why the compiled rungs can still be split per region.** A region boundary *inside* a
compiled callable forces a graph break, which is why a compiled `sample_actions` cannot be
instrumented at all. But `bench/regions.py` wraps three specific callables from the
outside — `paligemma_with_expert.embed_image` (`vision`), `paligemma_with_expert.forward`
(`vlm_prefix`, prefill branch) and `PI0Pytorch.denoise_step` (`ae_step[i]`). Compiling
those same callables and instrumenting afterwards puts every region boundary *outside* a
compiled graph, so the split survives. That is the trick T003's `step_only` used for the
expert; rungs 4-7 extend it to vision and the VLM prefix, which no run has done before —
T003 could only report their fixed cost falling 485.1 -> 433.6 ms (1.12x) as one number.

    0  eager, matmul_precision=highest              baseline
    1  eager, matmul_precision=high                 + TF32
    2  compile denoise_step, no CUDA graphs         + compilation   (action expert)
    3  compile denoise_step, reduce-overhead        + CUDA graphs   (action expert)
    4  ... also compile embed_image, no graphs      + compilation   (vision)
    5  ... also embed_image, reduce-overhead        + CUDA graphs   (vision)
    6  ... also compile pg.forward, no graphs       + compilation   (VLM prefix)
    7  ... also pg.forward, reduce-overhead         + CUDA graphs   (VLM prefix)
    8  compile sample_actions, reduce-overhead      LeRobot's own path, e2e only
    9  compile sample_actions, max-autotune         + autotuning,    e2e only

Rungs 0-7 are CUMULATIVE on one policy: rung k+1 adds one more compiled callable to the
same model, so each marginal speedup is attributable to that callable alone, and all of
them keep their regions. Rung 7 is therefore a fully compiled model WITH a per-region
split, which `tasks/T004` says does not exist yet. Rungs 8-9 need a fresh policy — they
compile the outer `sample_actions`, which subsumes the other three — and report
end-to-end only; compare them with rung 7's e2e, never with its per-region numbers.

    python bench/opt_ladder.py --outdir results/<date>_E4c_pi0_opt_ladder --steps 4,10
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
import regions as R  # noqa: E402
from common import (REPO, collect_raw, compress_and_record, sh_out,  # noqa: E402
                    software_versions, summarize, write_json)

# (key, label, precision, [(attr_owner, attr_name, mode, clone_out)], fresh, regions_ok)
#
# clone_out exists for ONE measured reason. `embed_image` is called once per camera —
# three times per inference — and a reduce-overhead CUDA graph reuses a single static
# output buffer, so camera 1's embedding is overwritten by camera 2 before the caller has
# consumed it:
#     RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a
#     subsequent run
# Cloning the graph's output makes it measurable. The clone is ~1 MB per camera and is
# counted in the rung's own time, so the number is honest about what it cost; the rung is
# labelled so nobody reads it as a free win.
# attr_owner: "model" = policy.model, "pg" = policy.model.paligemma_with_expert
NOGRAPH = "max-autotune-no-cudagraphs"
GRAPHS = "reduce-overhead"
RUNGS = [
    ("eager_fp32",   "eager (matmul_precision=highest)",          "highest", [], False, True),
    ("eager_tf32",   "+ TF32 (matmul_precision=high)",            "high",    [], False, True),
    ("ae_compile",   "+ compile denoise_step (no CUDA graphs)",   "high",
     [("model", "denoise_step", NOGRAPH, False)], True, True),
    ("ae_graphs",    "+ CUDA graphs on denoise_step",             "high",
     [("model", "denoise_step", GRAPHS, False)], True, True),
    ("vis_compile",  "+ compile embed_image (no CUDA graphs)",    "high",
     [("model", "denoise_step", GRAPHS, False), ("pg", "embed_image", NOGRAPH, False)], True, True),
    ("vis_graphs",   "+ CUDA graphs on embed_image (output cloned)", "high",
     [("model", "denoise_step", GRAPHS, False), ("pg", "embed_image", GRAPHS, True)], True, True),
    ("vlm_compile",  "+ compile pg.forward (no CUDA graphs)",     "high",
     [("model", "denoise_step", GRAPHS, False), ("pg", "embed_image", GRAPHS, True),
      ("pg", "forward", NOGRAPH, False)], True, True),
    ("vlm_graphs",   "+ CUDA graphs on pg.forward",               "high",
     [("model", "denoise_step", GRAPHS, False), ("pg", "embed_image", GRAPHS, True),
      ("pg", "forward", GRAPHS, False)], True, True),
    ("whole_graphs", "compile sample_actions (LeRobot's path)",   "high",
     [("model", "sample_actions", GRAPHS, False)], True, False),
    # CORRECTION 2026-09-22: this rung was first blamed for an OOM kill. It was not the
    # cause. A second run, with whole_autotune excluded, was OOM-killed at the SAME rung
    # index (9) on a different rung, with a near-identical signature — 31.2 vs 31.7 GB
    # resident, 115.9 vs 115.6 GB virtual. The cause is memory accumulating ACROSS rungs:
    # each loads a fresh 8.28 GiB policy, and `del policy; empty_cache()` does not free it
    # because torch.compile retains references and on Orin the GPU memory IS system
    # memory. Nine models is ~74 GiB against 61 GB of RAM. Fixed by running each rung in
    # its own process (--single-rung), so the OS reclaims on exit. Kept out of the default
    # set only because T003 showed autotuning buys nothing here (1-4 % between stock
    # modes) at 5x the compile time.
    ("whole_autotune", "+ Triton autotuning (max-autotune)",      "high",
     [("model", "sample_actions", "max-autotune", False)], True, False),
]


def write_tables(out: Path, rows: list[dict], steps: list[int]) -> None:
    """Write the CSVs after EVERY rung, not once at the end.

    The first full run was OOM-killed on its last rung (max-autotune on sample_actions
    reached 31.7 GB resident on a 61 GB machine). Because the tables were written only
    after the loop, all nine completed rungs were lost with it — three hours of
    measurement, recoverable only as text in a log. Partial results are now on disk from
    the first rung onward.
    """
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:                      # rows from different mechanisms differ in columns
        for k in r:
            if k not in keys:
                keys.append(k)
    with (out / "opt_ladder.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in rows])

    eff = []
    for N in steps:
        rs = [r for r in rows if r["N"] == N]
        for prev, cur in zip(rs, rs[1:]):
            e = {"N": N, "step": f"{prev['rung']} -> {cur['rung']}", "adds": cur["label"]}
            for k in ("e2e_median", "vision", "vlm_prefix", "ae_total"):
                col = k + "_ms"
                a, b = prev.get(col), cur.get(col)
                try:
                    e[k + "_speedup"] = round(float(a) / float(b), 4) if a and b else ""
                except (TypeError, ValueError, ZeroDivisionError):
                    e[k + "_speedup"] = ""
            eff.append(e)
    if eff:
        with (out / "opt_ladder_marginal.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(eff[0].keys()))
            w.writeheader()
            w.writerows(eff)



def mem_state() -> dict:
    """Free memory and swap, from /proc/meminfo. Recorded per rung.

    On Orin the GPU memory IS system memory, so a rung that starts under pressure can be
    slowed by reclaim or pushed into swap without any error appearing. Two runs of this
    ladder were OOM-killed at rung 9 before rungs were given their own processes; the
    numbers happened to survive intact (two independent runs agreed to 1.81 % worst case),
    but that was luck, not design. Recording the state makes it checkable instead.
    """
    mi = {}
    for ln in Path("/proc/meminfo").read_text().splitlines():
        k, _, v = ln.partition(":")
        mi[k] = int(v.split()[0]) if v.split() else 0
    return {"mem_available_GiB": round(mi.get("MemAvailable", 0) / 1048576, 2),
            "mem_free_GiB": round(mi.get("MemFree", 0) / 1048576, 2),
            "swap_used_GiB": round((mi.get("SwapTotal", 0) - mi.get("SwapFree", 0))
                                   / 1048576, 3)}


MIN_AVAIL_GIB = 20.0     # the policy is 8.28 GiB; compile and activations need headroom



def owner_of(policy, which: str):
    return policy.model if which == "model" else policy.model.paligemma_with_expert


def finish(out: Path, args, gate, rows, notes, steps, samples=None, verified=None,
           real=None, shipped=None):
    """Write raw samples and meta.json. Shared by the per-rung parent and --in-process."""
    samples = samples or []
    sp = out / "raw" / "latency_samples.csv"
    with sp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "rung", "iter", "ms"])
        w.writeheader()
        w.writerows(samples or [])
    orig = compress_and_record(sp)

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E4c (optimisation ladder)", "task": "T004",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": P.CHECKPOINT, "model_repo": P.CHECKPOINT, "model_commit": P.REVISION,
        "model_revision": P.REVISION, "checkpoint": P.CHECKPOINT, "precision": "bfloat16",
        "compile_mode": "per rung, see opt_ladder.csv", "batch": 1,
        "num_denoise_steps": steps, "prefix_tokens": 816, "suffix_tokens": 51,
        "n_warmup": args.warmup, "n_timed": args.timed,
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"), "python": sw.get("python"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "weights_verified": verified or real, "shipped_config": shipped,
        "command": f"python bench/opt_ladder.py --outdir {args.outdir} "
                   f"--steps {args.steps} --warmup {args.warmup} --timed {args.timed}",
        "throttled": False,
        "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
        "originals": [orig], "not_measured": notes, "raw_files": [],
        "notes": [
            "A fresh policy is loaded for every rung. torch.compile is not reversible, and "
            "a compiled callable left on the module would contaminate the rungs below it.",
            "Rungs 0-3 keep regions on; 4-5 cannot, because a region boundary inside a "
            "compiled sample_actions forces a graph break. Compare rung 4/5's e2e with "
            "rung 3's e2e, never with its per-region numbers.",
            "`step_compile`/`whole_compile` use max-autotune-no-cudagraphs to get "
            "compilation WITHOUT CUDA graphs; the next rung adds reduce-overhead, whose "
            "difference from it is the CUDA-graph contribution. T003 found the stock modes "
            "differ by 1-4 % otherwise, so the mode change is a graph switch, not a "
            "codegen change.",
            "TF32 is set explicitly on every rung rather than inherited from "
            "config.compile_model, so that the precision factor is separated from the "
            "compilation factor instead of moving with it.",
        ],
    })
    meta = json.loads((out / "meta.json").read_text())
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out} ({len(rows)} rungs)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", default="4,10")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--in-process", action="store_true",
                    help="run every rung in this process (the old behaviour that OOMed)")
    ap.add_argument("--single-rung", default=None,
                    help="internal: run exactly this one rung and write its row to "
                         "raw/rows/<rung>_N<N>.json, then exit. The parent spawns one "
                         "process per rung so the OS reclaims the 8.28 GiB policy on exit; "
                         "in-process `del policy` does not, and nine rungs OOM-killed this "
                         "box twice at rung 9.")
    ap.add_argument("--single-N", type=int, default=None, help="internal, with --single-rung")
    ap.add_argument("--rungs",
                    default=",".join(r[0] for r in RUNGS if r[0] != "whole_autotune"),
                    help="default excludes whole_autotune, which OOM-kills this device")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    want = [r for r in RUNGS if r[0] in args.rungs.split(",")]
    steps = [int(x) for x in args.steps.split(",") if x]

    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)
    rows, notes, samples = [], [], []
    verified = None

    if not args.single_rung and not args.in_process:
        # One process per rung. `del policy; torch.cuda.empty_cache()` does NOT free the
        # 8.28 GiB policy — torch.compile keeps references, and on Orin the GPU memory is
        # system memory — so nine rungs in one process reached 31 GB resident and the
        # kernel killed the run, twice, both times at rung 9. A process exit always frees.
        d = out / "raw" / "rows"
        d.mkdir(parents=True, exist_ok=True)
        for N in steps:
            for key, label, *_ in want:
                f = d / f"{key}_N{N}.json"
                if f.exists():
                    print(f"=== N={N} {key}: already done, skipping", flush=True)
                    continue
                cmd = [sys.executable, __file__, "--outdir", str(out), "--single-rung", key,
                       "--single-N", str(N), "--warmup", str(args.warmup),
                       "--timed", str(args.timed)]
                print(f"\n=== N={N} {key}: spawning", flush=True)
                r = subprocess.run(cmd)
                # Let the kernel finish reclaiming the 8.28 GiB the child held before the
                # next rung loads its own. The exit frees it, but not instantaneously, and
                # a rung that starts while reclaim is still running measures reclaim.
                for _ in range(30):
                    if mem_state()["mem_available_GiB"] >= MIN_AVAIL_GIB:
                        break
                    time.sleep(2)
                m = mem_state()
                print(f"  settled: {m['mem_available_GiB']:.1f} GiB available, "
                      f"{m['swap_used_GiB']:.2f} GiB swap", flush=True)
                if r.returncode != 0:
                    notes.append(f"N={N} {key}: subprocess exited {r.returncode} "
                                 f"(137/-9 = OOM-killed by the kernel)")
                    print(f"  rung failed, exit {r.returncode}; continuing", flush=True)
        for f in sorted(d.glob("*.json")):
            r_ = json.loads(f.read_text())
            r_.setdefault("provenance", "own process, this run")
            rows.append(r_)
        rows.sort(key=lambda r: (int(r["N"]), [w[0] for w in want].index(r["rung"])))
        write_tables(out, rows, steps)
        print(f"\nmerged {len(rows)} rung(s) into {out}/opt_ladder.csv")
        return finish(out, args, gate, rows, notes, steps)

    if args.single_rung:
        want = [r for r in want if r[0] == args.single_rung]
        steps = [args.single_N]
        if not want:
            raise SystemExit(f"unknown rung {args.single_rung!r}")

    for N in steps:
        for key, label, prec, targets, fresh, regions_ok in want:
            print(f"\n=== N={N}  {key}: {label}", flush=True)
            m0 = mem_state()
            print(f"  memory before: {m0['mem_available_GiB']:.1f} GiB available, "
                  f"{m0['swap_used_GiB']:.2f} GiB swap in use", flush=True)
            if m0["mem_available_GiB"] < MIN_AVAIL_GIB:
                raise SystemExit(
                    f"refusing to measure: only {m0['mem_available_GiB']:.1f} GiB available, "
                    f"below the {MIN_AVAIL_GIB} GiB floor. A rung started under memory "
                    f"pressure produces a number that looks fine and is not comparable.")
            # Every rung that compiles anything loads a fresh policy: torch.compile is not
            # reversible, so a rung cannot be un-done to build the next one on top.
            policy, cfg, shipped = P.load_policy(num_inference_steps=N)
            if verified is None:
                verified = P.assert_real_weights(policy)
                print("weights:", json.dumps(verified), flush=True)
            real = verified
            fx = P.load_fixture()
            batch, _, _ = P.build_batch(policy, fx)
            torch.set_float32_matmul_precision(prec)

            compiled_desc, compile_s, fails = [], 0.0, []
            for which, attr, mode, clone_out in targets:
                ownr = owner_of(policy, which)
                try:
                    t0 = time.time()
                    c = torch.compile(getattr(ownr, attr), mode=mode)
                    if clone_out:
                        def _cloned(*a, _c=c, **k):
                            r = _c(*a, **k)
                            return r.clone() if torch.is_tensor(r) else r
                        setattr(ownr, attr, _cloned)
                    else:
                        setattr(ownr, attr, c)
                    compile_s += time.time() - t0
                    compiled_desc.append(f"{which}.{attr}@{mode}"
                                         + ("+clone" if clone_out else ""))
                except Exception as e:                            # noqa: BLE001
                    fails.append(f"{which}.{attr}@{mode}: {type(e).__name__}: {str(e)[:160]}")
            compile_s = round(compile_s, 3) if targets else ""

            # instrument AFTER compiling, so every region boundary is OUTSIDE a compiled
            # callable and the per-region split survives (the step_only trick, generalised)
            rec = undo = None
            if regions_ok:
                rec = R.Recorder(enable_layers=False, layers_in_step=0)
                undo = R.instrument(policy, rec)

            err = None
            try:
                for _ in range(args.warmup):
                    policy.predict_action_chunk(batch)
                torch.cuda.synchronize()
                # Reset the recorder EVERY iteration and take the median across them, as
                # bench/run_e3.py:90-102 does. results_ms() sums repeats of a name, so
                # starting it once and reading the total gives the sum over the whole timed
                # loop: a first version reported vision 499.9 / vlm_prefix 1171.8 /
                # ae_total 1707.0 ms against an end-to-end of 409.5 ms over 5 iterations.
                per_iter, s_ = [], []
                for _ in range(args.timed):
                    if rec is not None:
                        rec.reset()
                        rec.start()
                    t0 = time.perf_counter()
                    policy.predict_action_chunk(batch)
                    torch.cuda.synchronize()
                    s_.append((time.perf_counter() - t0) * 1e3)
                    if rec is not None:
                        rec.stop()
                        per_iter.append(rec.results_ms())
                st = summarize(s_)
                reg = {}
                if per_iter:
                    for k in {k for d in per_iter for k in d}:
                        vals = [d[k] for d in per_iter if k in d]
                        reg[k] = summarize(vals)["median_ms"]
            except Exception as e:                                # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:200]}"
                st, s_, reg = {"median_ms": float("nan"), "p5_ms": float("nan"),
                               "p95_ms": float("nan"), "n": 0}, [], {}
                print(f"  NOT MEASURED: {err}", flush=True)
            if undo:
                undo()

            row = {"N": N, "rung": key, "label": label, "matmul_precision": prec,
                   "compiled": ";".join(compiled_desc), "regions": "on" if regions_ok
                   else "off (compiled sample_actions cannot be instrumented)",
                   "compile_call_s": compile_s,
                   "e2e_median_ms": "" if err else round(st["median_ms"], 4),
                   "e2e_p5_ms": "" if err else round(st["p5_ms"], 4),
                   "e2e_p95_ms": "" if err else round(st["p95_ms"], 4),
                   "n": st["n"], "error": err or "",
                   "mem_available_GiB_before": m0["mem_available_GiB"],
                   "mem_available_GiB_after": mem_state()["mem_available_GiB"],
                   "swap_used_GiB_after": mem_state()["swap_used_GiB"]}
            for k in ("vision", "vlm_prefix", "ae_total"):
                row[k + "_ms"] = round(reg[k], 4) if k in reg else ""
            rows.append(row)
            samples += [{"N": N, "rung": key, "iter": i, "ms": v} for i, v in enumerate(s_)]
            if not err:
                print(f"  e2e {st['median_ms']:8.2f} ms   vision {row['vision_ms'] or '—':>10} "
                      f"vlm_prefix {row['vlm_prefix_ms'] or '—':>10} "
                      f"ae_total {row['ae_total_ms'] or '—':>10}"
                      + (f"   (compile {compile_s} s)" if targets else ""), flush=True)
            for f_ in fails:
                notes.append(f"N={N} {key}: {f_}")
                print(f"  compile failed: {f_}", flush=True)
            if err:
                notes.append(f"N={N} {key}: {err}")
            if not regions_ok:
                notes.append(f"N={N} {key}: regions off — a region boundary inside a "
                             f"compiled sample_actions forces a graph break, so only "
                             f"end-to-end is measurable for this rung.")
            if args.single_rung:
                d = out / "raw" / "rows"
                d.mkdir(parents=True, exist_ok=True)
                write_json(d / f"{key}_N{N}.json", row)
                print(f"wrote {d / f'{key}_N{N}.json'}")
                return 0
            write_tables(out, rows, steps)     # survive a kill on any later rung
            del policy
            torch.cuda.empty_cache()

    return finish(out, args, gate, rows, notes, steps, samples, verified)


if __name__ == "__main__":
    raise SystemExit(main())
