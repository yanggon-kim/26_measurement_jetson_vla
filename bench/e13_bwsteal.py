"""T016 step 1a / 1b (EV-02): pi0's compiled bf16_both PREFIX while an SM-free load steals bandwidth.

1a (--mode sweep): for each target load rate x (GB/s of DRAM traffic, read + write), bracketed
    reference (load off) -> loaded -> reference, n frames each, same process, same SM layout (the load
    takes no SMs: bench/bwload.py, copy engine, own process). sigma(x) = A_loaded / mean(A_ref_before,
    A_ref_after).
1b (--mode duty): the same, with the load ON only in a window of each prefix frame (frame-aligned via
    a shared frame-start timestamp): [0, win) from the start, or the last `win` ms (--window end).

Prefix = the UNMODIFIED rung prefix: corun.prefix_stage(ctx) -> phase_builders.prefix_once on the
whole GPU with pi0 at E7's bf16_both rung (e12b_common.load_rung checks it against E7's meta), timed
with CUDA events around each frame, one frame at a time (sync after each), 5 real fixture frames in
rotation. Every frame's KV output is finite-checked on the device (no host sync inside the loop).

Rates: tegrastats EMC_FREQ (20 ms) calibrated IN THIS SESSION (bench/e12b_m4.py --mode calib, run
first into <out>/calib/), and the load's own byte count. 'left' = the calibration's highest measured
streaming rate (the bus a streaming load sustains alone this session) minus the load's rate DURING.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import bwload                     # noqa: E402
import corun as CR               # noqa: E402
import e12b_common as C          # noqa: E402
import e12b_corun as H           # noqa: E402
import power_core as PC          # noqa: E402

EMC_RE = re.compile(r"EMC_FREQ (\d+)%@(\d+)")
N = 10


class Teg:
    def __init__(self, log: Path):
        PC.stop_tegrastats()
        if log.exists():
            log.unlink()
        self.log = log
        self.p = subprocess.Popen(["sudo", "-n", "tegrastats", "--interval", "20", "--logfile",
                                   str(log)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.3)

    def stop(self):
        PC.stop_tegrastats(self.p)
        return [int(m.group(1)) for m in (EMC_RE.search(l) for l in
                self.log.read_text(errors="replace").splitlines()) if m] if self.log.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mode", choices=("sweep", "duty"), default="sweep")
    ap.add_argument("--load", choices=("ce", "cpu", "both"), default="ce",
                    help="both = copy engine (first half of --load-cores) + CPU copy workers (second half), "
                         "target split evenly; A1 allows both together when one alone cannot reach the rates")
    ap.add_argument("--rates", default="0,20,40,50,60,70,80,88,94,-1",
                    help="target DRAM-traffic GB/s of the load; -1 = as fast as it goes")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--win-ms", type=float, default=93.0)
    ap.add_argument("--window", choices=("start", "end"), default="start")
    ap.add_argument("--calib-seconds", type=float, default=15.0)
    ap.add_argument("--calib", default="", help="reuse an emc_calibration.json instead of refitting")
    ap.add_argument("--harness-cores", default="0-7")
    ap.add_argument("--load-cores", default="8,9,10,11")
    args = ap.parse_args()
    out = REPO / args.outdir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty (protocol §5)")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    g0 = C.gate_full()
    if "MAXN" not in g0["power_mode"] or g0["sm_count"] != 16:
        raise SystemExit("MAXN / 16 SMs only (T016 NN4)")

    # in-session EMC calibration, before the model is resident (same method as E12b M4)
    if args.calib:
        cal_path = Path(args.calib)
    else:
        r = subprocess.run([sys.executable, str(REPO / "bench/e12b_m4.py"), "--mode", "calib",
                            "--outdir", str((out / "calib").relative_to(REPO)),
                            "--seconds", str(args.calib_seconds)], cwd=REPO)
        if r.returncode:
            raise SystemExit("calibration failed")
        cal_path = out / "calib" / "emc_calibration.json"
    fit = json.loads(cal_path.read_text())
    stream_max = max(p["GBps"] for p in fit["points"])
    cal = lambda pct: fit["slope_GBps_per_pct"] * pct + fit["intercept_GBps"]   # noqa: E731
    print(f"  calibration: GB/s = {fit['slope_GBps_per_pct']:.3f} x EMC% + "
          f"{fit['intercept_GBps']:.2f}; streaming max this session {stream_max:.1f} GB/s", flush=True)

    def cores(s):
        return [int(x) for x in s.split(",")] if "," in s else list(range(
            int(s.split("-")[0]), int(s.split("-")[1]) + 1))
    os.sched_setaffinity(0, cores(args.harness_cores))
    lc = cores(args.load_cores)
    if args.load == "both":
        # the copy engine needs one host core to issue; the rest go to CPU copy workers
        load = bwload.Multi([bwload.Load("ce", cores=lc[:1]),
                             bwload.Load("cpu", cores=lc[1:], cpu_workers=len(lc) - 1)])
    else:
        load = bwload.Load(args.load, cores=lc)

    policy, spec, cfg, note = C.load_rung(N)
    frames, manifest = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    fin = C.Finite()

    def frame(k):
        ctx = frames[k % len(frames)]["ctx"]
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        load.frame_start()
        e0.record()
        pkv, pad = CR.prefix_stage(ctx)
        e1.record()
        fin.add(H.leaves(pkv)[-1])
        torch.cuda.synchronize()
        return e0.elapsed_time(e1)

    with torch.no_grad():
        for k in range(20):                                   # warm-up / compile
            frame(k)
        a_ref0 = statistics.median(frame(k) for k in range(10))

        def segment(tag, rate, window, w_start, win):
            load.set(rate, window, w_start, win)
            time.sleep(0.2)
            teg = Teg(out / "raw" / f"teg_{tag}.log")
            b0, t0 = load.copied_bytes(), time.perf_counter()
            ms = [frame(k) for k in range(args.frames)]
            dt = time.perf_counter() - t0
            b1 = load.copied_bytes()
            emc = teg.stop()
            load.set(0)
            return ms, emc, (b1 - b0) * 2 / dt / 1e9       # load's own DRAM traffic, GB/s

        def load_alone(tag, rate, window, w_start, win, secs=5.0):
            load.set(rate, window, w_start, win)
            teg = Teg(out / "raw" / f"teg_{tag}_alone.log")
            b0, t0 = load.copied_bytes(), time.perf_counter()
            if window:                                      # emulate the frame cadence without a prefix
                while time.perf_counter() - t0 < secs:
                    load.frame_start(); time.sleep(a_ref0 / 1e3)
            else:
                time.sleep(secs)
            dt = time.perf_counter() - t0
            gb = (load.copied_bytes() - b0) * 2 / dt / 1e9
            emc = teg.stop()
            load.set(0)
            return emc, gb

        rows, samples = [], []
        rates = [float(x) for x in args.rates.split(",")]
        if args.mode == "duty":
            points = [(r, 1, 0.0 if args.window == "start" else max(0.0, a_ref0 - args.win_ms),
                       args.win_ms) for r in rates]
        else:
            points = [(r, 0, 0.0, 0.0) for r in rates]
        for i, (rate, window, w_start, win) in enumerate(points):
            tag = f"p{i}_{'max' if rate < 0 else int(rate)}"
            emc_alone, gb_alone = load_alone(tag, rate, window, w_start, win) if rate else ([], 0.0)
            ref_a, emc_ra, _ = segment(f"{tag}_refA", 0, 0, 0, 0)
            ld, emc_ld, gb_during = segment(f"{tag}_load", rate, window, w_start, win)
            ref_b, emc_rb, _ = segment(f"{tag}_refB", 0, 0, 0, 0)
            sa, sl_, sb = C.summ(ref_a), C.summ(ld), C.summ(ref_b)
            ref = (sa["median_ms"] + sb["median_ms"]) / 2
            med = lambda v: statistics.median(v) if v else float("nan")   # noqa: E731
            row = {"point": i, "mode": args.mode, "load": args.load,
                   "target_GBps": "max" if rate < 0 else rate,
                   "window": ("always" if not window else f"{args.window} {win:.0f} ms"),
                   "load_alone_GBps_own": round(gb_alone, 2),
                   "load_alone_GBps_emc": round(cal(med(emc_alone)), 2) if emc_alone else "",
                   "load_during_GBps_own": round(gb_during, 2),
                   "emc_during_total_GBps": round(cal(med(emc_ld)), 2) if emc_ld else "",
                   "emc_ref_GBps": round(cal(med(emc_ra + emc_rb)), 2) if emc_ra else "",
                   "stream_max_GBps": round(stream_max, 2),
                   "left_GBps": round(stream_max - gb_during, 2),
                   "ref_before_ms": sa["median_ms"], "ref_after_ms": sb["median_ms"],
                   "ref_drift_pct": round(100 * (sb["median_ms"] - sa["median_ms"]) / sa["median_ms"], 3),
                   "loaded_median_ms": sl_["median_ms"], "loaded_p5_ms": sl_["p5_ms"],
                   "loaded_p95_ms": sl_["p95_ms"], "n": sl_["n"],
                   "sigma": round(sl_["median_ms"] / ref, 4)}
            rows.append(row)
            for cond, v in (("refA", ref_a), ("load", ld), ("refB", ref_b)):
                samples += [{"point": i, "cond": cond, "frame": j, "ms": round(x, 4)}
                            for j, x in enumerate(v)]
            print(f"  [{i}] target {row['target_GBps']:>5} {row['window']:12s} during "
                  f"{gb_during:6.1f} GB/s (left {row['left_GBps']:6.1f})  ref {sa['median_ms']:.2f}/"
                  f"{sb['median_ms']:.2f}  loaded {sl_['median_ms']:.2f} ms  sigma {row['sigma']:.4f}",
                  flush=True)
    nonfinite = fin.read()
    load.close()
    with (out / "sigma.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out / "raw" / "latency_samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["point", "cond", "frame", "ms"]); w.writeheader(); w.writerows(samples)
    thr = C.throttle_record(g0)
    C.write_meta(out / "meta.json", {
        "experiment": f"T016 step {'1a' if args.mode == 'sweep' else '1b'} (EV-02): prefix under SM-free bandwidth steal",
        "bench_commit": C.bench_commit(), "model_commit": C.CKPT_COMMIT, "checkpoint": spec.checkpoint,
        "checkpoint_commit": C.CKPT_COMMIT, "rung": "bf16_both", "rung_note": note,
        "variant": f"bf16_both (unmodified rung prefix, whole GPU) + load_{args.load}",
        "power_mode": g0["power_mode"], "sm_count": g0["sm_count"],
        "granted_partitions": "none (whole GPU; the load takes no SMs)",
        "gpu_clock_hz": g0["gpu_cur_freq_hz"], "emc_rate_hz": g0["emc_rate_hz"], "gate_start": g0, **thr,
        "load": {"kind": args.load, "process": "separate (multiprocessing spawn), pinned to cores "
                 + args.load_cores, "pids": load.pids, "chunk_MB": bwload.CHUNK_MB,
                 "rate_definition": "DRAM traffic = 2 x bytes copied (read + write, host and device "
                                    "memory are the same LPDDR5)",
                 "shared_with_prefix": "LPDDR5 bus + memory controller; GPU memory hub / SMMU (ce); "
                                       "CUDA driver (ce, separate context); CPU L3/DSU (cpu); host cores "
                                       "are disjoint (harness " + args.harness_cores + ")"},
        "harness_cores": args.harness_cores, "N": N, "frames_per_segment": args.frames,
        "reference": "same process, load off, bracketed before and after each loaded segment",
        "sigma_definition": "loaded median / mean(ref_before median, ref_after median)",
        "calibration": {"file": str(cal_path.relative_to(REPO)) if cal_path.is_relative_to(REPO) else str(cal_path),
                        **{k: fit[k] for k in ("slope_GBps_per_pct", "intercept_GBps")},
                        "stream_max_GBps": stream_max},
        "n_nonfinite": nonfinite, "valid": nonfinite == 0, "frames_manifest": manifest,
        "command": " ".join([sys.executable, *sys.argv]), "torch": torch.__version__,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(f"  nonfinite {nonfinite}; wrote {out}", flush=True)
    return 0 if nonfinite == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
