"""T010 / E11f — throughput, GPU idle inside the expert, and what happens when two things share
the GPU (protocol §13.7, plan part 6).

The hybrid design overlaps the next frame's GPU prefix with the current frame's memory-side
denoising, and assumes the two do not delay each other (`host_layer_slowdown = 1.0`,
**[ASSUMPTION]**). Three things are measured here:

  (a) the three throughputs — prefix-only, action-expert-only, full inference;
  (b) `gpu_idle_ae_total` — the action-expert region time minus the union of the kernel
      intervals inside it, from an nsys capture. This is the GPU time already free for the next
      frame's prefix even in the ordinary all-GPU run. T004's E5 may reuse these captures;
  (c) the prefix loop running concurrently with the per-layer host-segment sequence, **as two
      processes and as two CUDA streams**, reporting median, p95 and max of the short activity
      alone and under concurrency.

    python bench/pipelining.py --outdir results/<date>_E11f_pi0_pipelining
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as MR  # noqa: E402
import pi0_common as P  # noqa: E402
import phase_loops as PL  # noqa: E402
from e11_summaries import e11f  # noqa: E402
from common import (REPO, collect_raw, sh_out, software_versions, summarize,  # noqa: E402
                    write_json)
from host_segments import LayerCapture, ModelInputs, build_segments, intermediates  # noqa: E402


def _open_csv(path_glob: str):
    """Open an nsys export, transparently handling the zstd-compressed form.

    Large exports are committed compressed (protocol §6), so a plain `*.csv` glob misses them —
    which silently turned every gpu_idle row into "no data" once the traces were compressed.
    """
    import glob as _glob
    import io
    import subprocess as _sp
    hits = sorted(_glob.glob(path_glob)) or sorted(_glob.glob(path_glob + ".zst"))
    if not hits:
        return None, None
    f = hits[0]
    if f.endswith(".zst"):
        out = _sp.run(["zstd", "-dc", f], capture_output=True, timeout=600)
        return io.StringIO(out.stdout.decode("utf-8", "replace")), f
    return open(f, newline=""), f


def gpu_idle_from_nsys(nsys_dir: Path, tag: str, n_inferences: int = 2) -> dict | None:
    """region time - union of kernel intervals, from the nsys exports.

    `cuda_gpu_trace` gives every kernel's start and duration; `nvtx_gpu_proj_trace` gives the
    NVTX ranges projected onto the GPU timeline. The union is taken so overlapping kernels are
    not double-counted.
    """
    trace, _tn = _open_csv(str(nsys_dir / f"{tag}*cuda_gpu_trace.csv"))
    proj, _pn = _open_csv(str(nsys_dir / f"{tag}*nvtx_gpu_proj_trace.csv"))
    if trace is None or proj is None:
        return None

    def num(x):
        try:
            return float(str(x).replace(",", ""))
        except (TypeError, ValueError):
            return None

    kern = []
    for r in csv.DictReader(trace):
        st = num(r.get("Start (ns)") or r.get("Start"))
        du = num(r.get("Duration (ns)") or r.get("Duration"))
        name = r.get("Name") or ""
        if st is None or du is None or not name:
            continue
        kern.append((st, st + du))
    spans = []
    for r in csv.DictReader(proj):
        # nsys prefixes projected NVTX ranges with their domain, e.g. ":ae_step[0]"
        nm = (r.get("Name") or r.get("Range") or "").lstrip(":").strip()
        if not nm.startswith("ae_step") or nm.startswith("ae_step_"):
            continue                       # the bracket-free alias duplicates the same span
        # ...and the projected trace names its columns differently from cuda_gpu_trace
        st = num(r.get("Projected Start (ns)") or r.get("Start (ns)") or r.get("Start"))
        du = num(r.get("Projected Duration (ns)") or r.get("Duration (ns)")
                 or r.get("Duration"))
        if st is None or du is None:
            continue
        spans.append((st, st + du))
    if not kern or not spans:
        return None
    # The ae_step spans cover EVERY captured inference. Taking min(start)..max(end) across all
    # of them spans the gap between inferences too — i.e. the next inference's vision and VLM
    # prefix, which are GPU-saturated — and dilutes the idle fraction badly (50-66 % instead of
    # the real 84 %). Group the spans into one chunk per inference and measure each on its own.
    spans.sort()
    per_inf = max(1, len(spans) // max(1, n_inferences))
    chunks = [spans[i:i + per_inf] for i in range(0, len(spans), per_inf)]
    walls, busies, nks = [], [], []
    for ch in chunks:
        lo, hi = ch[0][0], ch[-1][1]
        inside = sorted((max(a, lo), min(b, hi)) for a, b in kern if b > lo and a < hi)
        busy, cur_s, cur_e = 0.0, None, None
        for a, b in inside:
            if cur_s is None:
                cur_s, cur_e = a, b
            elif a <= cur_e:
                cur_e = max(cur_e, b)
            else:
                busy += cur_e - cur_s
                cur_s, cur_e = a, b
        if cur_s is not None:
            busy += cur_e - cur_s
        walls.append(hi - lo)
        busies.append(busy)
        nks.append(len(inside))
    wall = sum(walls) / len(walls)
    busy = sum(busies) / len(busies)
    idle = wall - busy
    return {"ae_total_ms": wall / 1e6, "kernel_busy_ms": busy / 1e6,
            "gpu_idle_ms": idle / 1e6,
            "gpu_idle_frac": idle / wall if wall else 0.0,
            "n_kernels_in_region": round(sum(nks) / len(nks), 1),
            "n_ae_step_spans": len(spans), "n_inferences": len(chunks),
            "per_inference": True}


def host_sequence_fn(policy, batch, layers_per_inf: int, pause_ms: float, spec):
    """The per-layer host-segment sequence, `layers_per_inf` times with a pause between
    segments — the plan's 18 x N layers per inference."""
    cap = LayerCapture(policy, 0, spec)
    with cap, torch.no_grad():
        policy.predict_action_chunk(batch)
    t = intermediates(policy, cap)
    ex = spec.resolve(policy, spec.expert_layers_path.rsplit('.', 1)[0])
    S = t["x"].shape[1]
    t["_partials"] = torch.randn(4, S, ex.config.hidden_size, device="cuda", dtype=t["x"].dtype)
    segs = [s for s in build_segments(t, S, ex.config.hidden_size, ex.config.intermediate_size,
                                      ex.config.num_attention_heads, ex.config.head_dim,
                                      t["scores"].shape[-1])
            if s["segment"] in ("rmsnorm1", "rope", "softmax", "residual_rmsnorm2",
                                "gelu_gate_mul_up", "partials_residual")]

    def run():
        for _ in range(layers_per_inf):
            for sd in segs:
                sd["fn"]()
                if pause_ms:
                    time.sleep(pause_ms / 1e3)
        torch.cuda.synchronize()

    return run, len(segs)



def emit_model_inputs(out: Path, trows, crows, irows, steps: int,
                      bench_commit: str, spec) -> int:
    """Every model_inputs.csv row E11f is responsible for, from the three CSVs.

    Shared by the measuring path and by --from-csv, so a re-emission cannot drift
    from what a fresh run would write.
    """
    mi = ModelInputs(spec.name, spec.revision or "", bench_commit,
                     "MAXN+jetson_clocks", 816, 51, steps)
    qmap = {"prefix": "prefix_throughput", "ae": "ae_throughput", "full": "e2e_throughput"}
    for r in trows:
        st = "eager" if r["mode"] == "eager" else "compiled"
        mi.add(qmap[r["phase"]], r["throughput_inf_per_s"], "inf_per_s", "value", r["n"],
               "throughput.csv", variant=f"mode={r['mode']}", exec_style=st,
               timing="wall_loop")
    for r in crows:
        if r["slowdown"] != "":
            st = "eager" if r["mode"] == "eager" else "compiled"
            mi.add("host_layer_slowdown", r["slowdown"], "ratio", "value", r["n"],
                   "concurrency.csv", key="prefix_loop",
                   variant=f"sharing={r['sharing']};mode={r['mode']}", exec_style=st,
                   timing="wall_isolated")
    # §13.8's `timing` is a closed enum and does not contain "nsys". Both idle figures rest
    # on a wall-clock loop measurement — the corrected one on the UNPROFILED loop, the raw
    # one on the profiled wall — so both are `wall_loop`, and `basis=` in `variant` says
    # which. (An earlier version wrote timing="nsys" and check_model_inputs.py rejected it,
    # and timing="analytic" for the raw figure, which was worse: nothing about it is analytic.)
    for r in irows:
        st = "eager" if r["mode"] == "eager" else "compiled"
        if r.get("gpu_idle_frac_corrected") not in ("", None):
            base = (f"mode={r['mode']};N={r['N']};"
                    f"basis=unprofiled_wall_minus_profiled_kernel_busy")
            mi.add("gpu_idle_ae_total", r["gpu_idle_ms_corrected"], "ms", "value", 1,
                   "gpu_idle.csv", variant=base, exec_style=st, timing="wall_loop")
            mi.add("gpu_idle_frac_ae_total", r["gpu_idle_frac_corrected"], "ratio", "value",
                   1, "gpu_idle.csv", variant=base, exec_style=st, timing="wall_loop")
        if r.get("gpu_idle_ms") not in ("", None):
            base = f"mode={r['mode']};N={r['N']};basis=profiled_wall"
            mi.add("gpu_idle_ae_total", r["gpu_idle_ms"], "ms", "value", 1, "gpu_idle.csv",
                   variant=base, exec_style=st, timing="wall_loop")
            mi.add("gpu_idle_frac_ae_total", r["gpu_idle_frac"], "ratio", "value", 1,
                   "gpu_idle.csv", variant=base, exec_style=st, timing="wall_loop")
    return mi.write(out / "model_inputs.csv")


def rebuild_from_csv(out: Path, args) -> int:
    """Re-emit model_inputs.csv and summary.md from this run's committed CSVs.

    Nothing is measured. The CSVs are the record of the run; this only rebuilds the
    derived hand-off file when the way a row is *labelled* turns out to be wrong.
    """
    def rd(name):
        p_ = out / name
        return list(csv.DictReader(p_.open())) if p_.exists() else []

    trows, crows, irows = rd("throughput.csv"), rd("concurrency.csv"), rd("gpu_idle.csv")
    if not (trows or crows or irows):
        print(f"{out}: no CSVs to rebuild from")
        return 1
    meta_p = out / "meta.json"
    if not meta_p.exists():
        print(f"{out}: meta.json is missing; refusing to invent one")
        return 1
    meta = json.loads(meta_p.read_text())
    # The re-emit path has no CLI, so the model comes from the run's own meta.json --
    # never from a default, which would relabel another model's rows as pi0.
    n_mi = emit_model_inputs(out, trows, crows, irows, int(meta.get("num_denoise_steps", 4)),
                             meta.get("bench_commit") or sh_out(f"git -C {REPO} rev-parse HEAD"),
                             MR.get(meta.get("model", "pi0")))
    e11f(out)
    meta["model_inputs_rows"] = n_mi
    meta.setdefault("notes", []).append(
        f"model_inputs.csv and summary.md were rebuilt from this run's own CSVs on "
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} with "
        f"`python bench/pipelining.py --outdir {out} --from-csv` at bench commit "
        f"{sh_out(f'git -C {REPO} rev-parse HEAD')}. No measurement was repeated: the fix "
        f"was to the `timing` label of the gpu_idle rows, which §13.8's closed enum does "
        f"not allow to be 'nsys'. The measured values are unchanged.")
    meta["raw_files"] = collect_raw(out, meta.get("command", ""))
    write_json(meta_p, meta)
    print(f"rebuilt {out}: {n_mi} model_inputs rows (nothing measured)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi0",
                    help="pi0 | pi05 | smolvla. T011 extends E11 beyond pi0; the expert layout and the checkpoint-specific processors come from bench/model_registry.py.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--modes", default="eager,reduce-overhead")
    ap.add_argument("--conc-reps", type=int, default=30)
    ap.add_argument("--pause-ms", type=float, default=2.0)
    ap.add_argument("--from-csv", action="store_true",
                    help="measure nothing: re-read this run's own throughput.csv, "
                         "concurrency.csv and gpu_idle.csv and rebuild model_inputs.csv and "
                         "summary.md from them. For correcting how a measurement is "
                         "*described* (a wrong `timing` label, a missing row) without "
                         "pretending to have measured it again. meta.json keeps the original "
                         "run's device state and gains a note saying this happened.")
    args = ap.parse_args()
    spec = MR.get(args.model)
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "nsys").mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        return rebuild_from_csv(out, args)

    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)

    trows, crows, irows = [], [], []
    for mode in [m for m in args.modes.split(",") if m]:
        cm = None if mode == "eager" else mode
        policy, cfg, shipped = MR.load_policy(spec, n=args.steps, compile_mode=cm)
        real = P.assert_real_weights(policy)
        fx = P.load_fixture()
        batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                revision=spec.revision)
        print(f"\n=== {mode}: throughput ===", flush=True)
        for phase in ("prefix", "ae", "full"):
            fn = PL.make_phase(policy, batch, phase, args.steps, spec)
            r = PL.run_loop(fn, args.seconds, phase)
            r.pop("samples")
            trows.append({"mode": mode, "phase": phase,
                          "effective_exec_style": PL.effective_exec_style(phase, cm), **r})
            print(f"  {phase:8s} {r['throughput_inf_per_s']:7.3f} inf/s  "
                  f"median {r['median_ms']:8.2f} ms", flush=True)

        # --- (c) concurrency
        print(f"=== {mode}: concurrency ===", flush=True)
        seq_fn, n_segs = host_sequence_fn(policy, batch, 18, args.pause_ms, spec)
        prefix_fn = PL.make_phase(policy, batch, "prefix", args.steps, spec)

        # The mapping's 2 ms pause x 6 segments x 18 layers puts 216 ms of time.sleep inside
        # a ~250 ms measurement. A raw before/after ratio is diluted by 250/34 = 7.3x and
        # cannot represent a slowdown above ~1.14 however hard the GPU is contended, so every
        # concurrency row also carries the ratio on the NON-SLEEP portion.
        sleep_ms = 18 * len([s_ for s_ in ("rmsnorm1", "rope", "softmax", "residual_rmsnorm2",
                                           "gelu_gate_mul_up", "partials_residual")]) \
            * args.pause_ms

        def bench_seq(label, sharing):
            for _ in range(3):
                seq_fn()
            torch.cuda.synchronize()
            ms = []
            for _ in range(args.conc_reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                seq_fn()
                ms.append((time.perf_counter() - t0) * 1e3)
            s = summarize(ms)
            crows.append({"mode": mode, "sharing": sharing, "activity": "host_sequence",
                          "condition": label, "median_ms": round(s["median_ms"], 4),
                          "p95_ms": round(s["p95_ms"], 4), "max_ms": round(s["max_ms"], 4),
                          "n": s["n"], "slowdown": "",
                          "sleep_ms": round(sleep_ms, 3),
                          "non_sleep_ms": round(s["median_ms"] - sleep_ms, 4)})
            return s["median_ms"]

        alone = bench_seq("alone", "none")

        # --- two CUDA streams in ONE process, launched from ONE thread (§13.7).
        # The first version created a high-priority stream, never used it, and ran the
        # contender on a second Python THREAD -- so both activities went to the default
        # stream and the 1.9x it measured was largely GIL contention. Here the contender is
        # issued onto s_hi from this thread, between chunks of the measured sequence, so the
        # two really do share the GPU through two streams.
        lo, hi = torch.cuda.current_stream().priority_range()
        s_hi = torch.cuda.Stream(priority=lo)          # lo is the HIGHEST priority value
        prio_ok = (s_hi.priority == lo)
        print(f"  stream priority range {(lo, hi)}; requested {lo}, got {s_hi.priority} "
              f"-> accepted={prio_ok}", flush=True)

        def seq_with_contender():
            with torch.cuda.stream(s_hi):
                prefix_fn()
            seq_fn()
            torch.cuda.synchronize()

        for _ in range(2):
            seq_with_contender()
        torch.cuda.synchronize()
        ms = []
        for _ in range(args.conc_reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            seq_with_contender()
            ms.append((time.perf_counter() - t0) * 1e3)
        st = summarize(ms)
        crows.append({"mode": mode, "sharing": "two_streams", "activity": "host_sequence",
                      "condition": "with_prefix_loop", "median_ms": round(st["median_ms"], 4),
                      "p95_ms": round(st["p95_ms"], 4), "max_ms": round(st["max_ms"], 4),
                      "n": st["n"], "slowdown": ""})
        med = st["median_ms"]
        crows[-1]["sleep_ms"] = round(sleep_ms, 3)
        crows[-1]["non_sleep_ms"] = round(med - sleep_ms, 4)
        crows[-1]["slowdown"] = round(med / alone, 4)
        crows[-1]["slowdown_non_sleep"] = round((med - sleep_ms) / (alone - sleep_ms), 4)
        print(f"  two_streams   host_sequence {med:8.2f} ms  slowdown {med/alone:.3f} "
              f"(non-sleep {crows[-1]['slowdown_non_sleep']:.3f})", flush=True)

        # two processes
        proc = subprocess.Popen(
            [sys.executable, str(REPO / "bench" / "phase_loops.py"), "--phase", "prefix",
             "--seconds", str(args.conc_reps * 2 + 30), "--steps", str(args.steps)]
            + ([] if cm is None else ["--compile-mode", cm]),
            stdout=open(out / "raw" / f"contender_{mode}.log", "w"),
            stderr=subprocess.STDOUT)
        time.sleep(45 if cm else 25)               # let it load (and compile) before timing
        med2 = bench_seq("with_prefix_loop", "two_processes")
        proc.terminate()
        proc.wait(timeout=30)
        crows[-1]["sleep_ms"] = round(sleep_ms, 3)
        crows[-1]["non_sleep_ms"] = round(med2 - sleep_ms, 4)
        crows[-1]["slowdown"] = round(med2 / alone, 4)
        crows[-1]["slowdown_non_sleep"] = round((med2 - sleep_ms) / (alone - sleep_ms), 4)
        print(f"  two_processes host_sequence {med2:8.2f} ms  slowdown {med2/alone:.3f} "
              f"(non-sleep {crows[-1]['slowdown_non_sleep']:.3f})", flush=True)
        # --- (b0) the UNPROFILED baseline for each N, in this same process.
        # nsys inflates wall time: it instruments every CUDA API call, and this workload is
        # launch-bound, so the region it reports is much longer than the same region unprofiled.
        # Kernel durations are not inflated (nsys reads them from the GPU), so an idle fraction
        # of (profiled wall - busy)/profiled wall counts nsys's own overhead as model idle.
        # Measure the same loop here, with no profiler attached, and correct with it.
        unprof = {}
        for n in (4, 10):
            fn = PL.make_phase(policy, batch, "ae", n, spec)
            r = PL.run_loop(fn, 20.0, f"ae_unprofiled_N{n}")
            r.pop("samples")
            unprof[n] = r["median_ms"]
            print(f"  unprofiled ae loop N={n}: {r['median_ms']:.2f} ms "
                  f"(p5 {r['p5_ms']:.2f}, p95 {r['p95_ms']:.2f}, n={r['n']})", flush=True)
        del policy
        torch.cuda.empty_cache()

        # --- (b) nsys, N = 4 and N = 10
        for n in (4, 10):
            tag = f"{mode}_N{n}"
            cmd = ["bash", str(REPO / "bench" / "run_nsys.sh"), str(out), tag, "--",
                   sys.executable, str(REPO / "bench" / "nsys_target.py"),
                   "--steps", str(n), "--inferences", "2"]
            if cm:
                cmd += ["--compile-mode", cm, "--no-regions"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            (out / "raw" / f"nsys_{tag}_driver.log").write_text(r.stdout + r.stderr)
            g = gpu_idle_from_nsys(out / "nsys", tag)
            if g:
                u = unprof.get(n)
                g["ae_total_ms_unprofiled"] = round(u, 4) if u else ""
                g["profiling_inflation"] = round(g["ae_total_ms"] / u, 4) if u else ""
                # the number to quote: nsys's own overhead is not GPU idle in the model
                g["gpu_idle_ms_corrected"] = round(u - g["kernel_busy_ms"], 4) if u else ""
                g["gpu_idle_frac_corrected"] = (
                    round((u - g["kernel_busy_ms"]) / u, 4) if u else "")
                irows.append({"mode": mode, "N": n, **g})
                print(f"  nsys {tag}: ae_total {g['ae_total_ms']:.2f} ms  busy "
                      f"{g['kernel_busy_ms']:.2f}  idle {g['gpu_idle_ms']:.2f} "
                      f"({100*g['gpu_idle_frac']:.1f} %)  ->  unprofiled "
                      f"{g['ae_total_ms_unprofiled']} ms (x{g['profiling_inflation']}), "
                      f"corrected idle {g['gpu_idle_frac_corrected']}", flush=True)
            else:
                irows.append({"mode": mode, "N": n, "ae_total_ms": "", "kernel_busy_ms": "",
                              "gpu_idle_ms": "", "gpu_idle_frac": "",
                              "ae_total_ms_unprofiled": round(unprof.get(n, 0), 4) or "",
                              "profiling_inflation": "", "gpu_idle_ms_corrected": "",
                              "gpu_idle_frac_corrected": "",
                              "n_kernels_in_region": "", "n_ae_step_spans": "",
                              "note": "nsys exports missing or no ae_step NVTX ranges "
                                      "(compiled captures run with --no-regions because the "
                                      "context managers graph-break the model)"})
                print(f"  nsys {tag}: no usable exports", flush=True)

    for name, rows_ in (("throughput.csv", trows), ("concurrency.csv", crows),
                        ("gpu_idle.csv", irows)):
        if rows_:
            keys = sorted({k for r in rows_ for k in r})
            with (out / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows([{k: r.get(k, "") for k in keys} for r in rows_])

    n_mi = emit_model_inputs(out, trows, crows, irows, args.steps,
                             sh_out(f"git -C {REPO} rev-parse HEAD"), spec)

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E11f", "task": "T010",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": spec.name, "model_repo": spec.checkpoint,
        "model_commit": spec.revision, "model_revision": spec.revision,
        "checkpoint": spec.checkpoint, "precision": "bfloat16",
        "compile_mode": args.modes, "batch": 1, "num_denoise_steps": args.steps,
        "prefix_tokens": 816, "suffix_tokens": 51,
        "n_warmup": 5, "n_timed": f"{args.seconds} s loops; {args.conc_reps} concurrency reps",
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
        "nsys_version": sh_out("nsys --version"),
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
        "torchvision": sw.get("torchvision"), "triton": sw.get("triton"),
        "python": sw.get("python"), "venv": sw.get("executable"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "command": f"python bench/pipelining.py --outdir {args.outdir}",
        "high_priority_stream_accepted": prio_ok,
        "throttled": False,
        "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
        "model_inputs_rows": n_mi, "raw_files": [],
        "notes": [
            "gpu_idle_ae_total is region time minus the UNION of kernel intervals inside the "
            "ae_step NVTX spans, so overlapping kernels are not double-counted. It is the GPU "
            "time already free for the next frame's prefix in the ordinary all-GPU run.",
            "T004's E5 may reuse these captures rather than taking new ones; the .nsys-rep "
            "files are under raw/ and the exports under nsys/.",
            "Compiled captures run with --no-regions: bench/regions.py's context managers "
            "force a graph break under torch.compile (bench/run_e4.py), so a compiled run "
            "with regions on would not be the graph the model actually executes. That means "
            "no ae_step NVTX spans exist for compiled, and gpu_idle is reported for eager.",
            "Concurrency is measured both ways §13.7 asks for: two CUDA streams in one "
            "process and two processes. Whether the high-priority stream argument was "
            "accepted is recorded.",
        ],
    })
    meta = json.loads((out / "meta.json").read_text())
    e11f(out)
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out} ({n_mi} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
