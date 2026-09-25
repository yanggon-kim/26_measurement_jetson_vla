"""E12b M4 -- attribute slowdowns to the memory system (spec §6 M4).

--mode calib        full-GPU copy hog at several known rates (sleep cycles between copies), each
                    >= 20 s with tegrastats at 20 ms and the four rails at 20 Hz. Converts
                    tegrastats EMC_FREQ utilisation (%) to GB/s by a least-squares line.
--mode rung         bandwidth sources during the M2 configuration's conditions: prefix alone,
                    expert alone, serial, co-run (full GPU).
--mode graphs_both  the same for M3's configuration on the full GPU and in one split (--split),
                    plus the hog-max control in that split.
Every condition runs >= --seconds with tegrastats and the rails; separate from the M2/M3 timing
runs so the headline timings carry no sampler (E7 measured tegrastats at ~0.19 %).
Output: bandwidth.csv (variant, split, condition, source, metric, median, p5, p95, n,
calibrated_GBps, notes) and, for calib, emc_calibration.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import e12b_common as C          # noqa: E402
import power_core as PC          # noqa: E402
import power_phases as PP        # noqa: E402

N = 10
EMC_RE = re.compile(r"EMC_FREQ (\d+)%@(\d+)")


def with_samplers(fn_loop, seconds, log: Path):
    """Run fn_loop repeatedly for `seconds` with tegrastats (20 ms) and rails (20 Hz)."""
    PC.stop_tegrastats()
    if log.exists():
        log.unlink()
    teg = subprocess.Popen(["sudo", "-n", "tegrastats", "--interval", "20", "--logfile", str(log)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rails = PC.Rails(20.0)
    rails.start()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        fn_loop()
        n += 1
    torch.cuda.synchronize()
    rails.stop.set(); rails.join(timeout=5)
    PC.stop_tegrastats(teg)
    emc = []
    for line in log.read_text(errors="replace").splitlines() if log.exists() else []:
        m = EMC_RE.search(line)
        if m:
            emc.append(int(m.group(1)))
    mem = [s.get(PP.MEM_RAIL) for s in rails.samples if s.get(PP.MEM_RAIL) is not None]
    return emc, mem, n


def stats(v):
    if not v:
        return {"median": "", "p5": "", "p95": "", "n": 0}
    s = sorted(v)
    return {"median": statistics.median(s), "p5": s[int(0.05 * (len(s) - 1))],
            "p95": s[int(0.95 * (len(s) - 1))], "n": len(s)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("calib", "rung", "graphs_both"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--split", default="12/4")
    ap.add_argument("--calib", default="", help="emc_calibration.json from a calib run")
    args = ap.parse_args()
    out = REPO / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    g0 = C.gate_full()
    if "MAXN" not in g0["power_mode"]:
        raise SystemExit("MAXN only")
    rows = []
    fit = json.loads(Path(args.calib).read_text()) if args.calib else None

    def cal(pct):
        return round(fit["slope_GBps_per_pct"] * pct + fit["intercept_GBps"], 2) if fit else ""

    def record(variant, split, cond, emc, mem, notes=""):
        e, m = stats(emc), stats(mem)
        rows.append({"variant": variant, "split": split, "condition": cond,
                     "source": "tegrastats_emc", "metric": "EMC_FREQ utilisation %",
                     **{k: e[k] for k in ("median", "p5", "p95", "n")},
                     "calibrated_GBps": cal(e["median"]) if e["n"] else "", "notes": notes})
        rows.append({"variant": variant, "split": split, "condition": cond,
                     "source": "mem_rail", "metric": f"{PP.MEM_RAIL} (regulator input) mW",
                     **{k: m[k] for k in ("median", "p5", "p95", "n")},
                     "calibrated_GBps": "", "notes": notes})
        print(f"  {split:9s} {cond:22s} EMC {e['median']}% (n={e['n']})  -> "
              f"{cal(e['median']) if e['n'] else '-'} GB/s   mem rail {m['median']} mW", flush=True)

    if args.mode == "calib":
        idle_emc, idle_mem, _ = with_samplers(lambda: time.sleep(0.05), args.seconds,
                                              out / "raw" / "teg_idle.log")
        record("idle", "full_gpu", "idle", idle_emc, idle_mem)
        nel = 128 * 2**20 // 2
        a = torch.empty(nel, dtype=torch.bfloat16, device="cuda").normal_()
        b = torch.empty_like(a)
        pts = []
        for sleep in (0, 200_000, 600_000, 1_500_000, 4_000_000):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                for _ in range(2):
                    b.copy_(a)
                    if sleep:
                        torch.cuda._sleep(sleep)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    for _ in range(4):
                        b.copy_(a)
                        if sleep:
                            torch.cuda._sleep(sleep)
                ts = []
                for _ in range(10):
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record(s); g.replay(); e1.record(s); torch.cuda.synchronize()
                    ts.append(e0.elapsed_time(e1))
                gbs = 4 * 2 * 128 * 2**20 / (statistics.median(ts) / 1e3) / 1e9
                emc, mem, _ = with_samplers(g.replay, args.seconds,
                                            out / "raw" / f"teg_hog_{sleep}.log")
            record(f"hog_{gbs:.1f}GBps", "full_gpu", f"hog_sleep_{sleep}", emc, mem,
                   notes=f"hog achieved {gbs:.2f} GB/s alone (CUDA events)")
            pts.append((statistics.median(emc) if emc else None, gbs, sleep))
        good = [(e, g_) for e, g_, _ in pts if e is not None]
        n = len(good)
        mx = sum(e for e, _ in good) / n
        my = sum(g_ for _, g_ in good) / n
        sl = sum((e - mx) * (g_ - my) for e, g_ in good) / max(1e-9, sum((e - mx) ** 2 for e, _ in good))
        fit = {"slope_GBps_per_pct": sl, "intercept_GBps": my - sl * mx,
               "points": [{"emc_pct": e, "GBps": g_, "sleep": s_} for e, g_, s_ in pts],
               "idle_emc_pct_median": statistics.median(idle_emc) if idle_emc else None,
               "note": "least squares of hog GB/s (CUDA events, alone) on tegrastats EMC_FREQ % "
                       "(20 ms interval). EMC utilisation counts ALL memory clients, so a "
                       "stage's calibrated GB/s includes the CPU's and other engines' traffic."}
        (out / "emc_calibration.json").write_text(json.dumps(fit, indent=2) + "\n")
        with (out / "emc_calibration.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["emc_pct", "GBps", "sleep"])
            w.writeheader(); w.writerows(fit["points"])
        for r in rows:
            if r["source"] == "tegrastats_emc" and r["median"] != "":
                r["calibrated_GBps"] = round(sl * r["median"] + fit["intercept_GBps"], 2)
        print(f"  fit: GB/s = {sl:.3f} x EMC% + {fit['intercept_GBps']:.2f}", flush=True)
    else:
        import corun as CR                                              # noqa: PLC0415
        import e12b_corun as H                                          # noqa: PLC0415
        policy, spec, cfg, note = C.load_rung(N)
        note["expert"] = H.recompile_default(policy, spec, "denoise_step")
        if args.mode == "graphs_both":
            note["embed_image"] = H.recompile_default(policy, spec, "embed_image")
            note["memo"] = H.memo_681(policy)
        frames, _ = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
        fin = C.Finite()
        configs = [None] if args.mode == "rung" else [None, tuple(int(x) for x in args.split.split("/"))]
        for sp in configs:
            h = H.Harness(policy, spec, frames, N,
                          config="rung_prefix" if args.mode == "rung" else "graphs_both",
                          split=sp, order="prefix_first")
            spn = "full_gpu" if sp is None else f"{h.granted[0]}/{h.granted[1]}"
            for cond in ("prefix_alone", "expert_alone", "serial", "pipeline"):
                h.run(cond, 6, fin)
                emc, mem, _ = with_samplers(lambda c=cond: h.run(c, 10, fin), args.seconds,
                                            out / "raw" / f"teg_{args.mode}_{spn.replace('/', '-')}_{cond}.log")
                record(h.label(), spn, {"pipeline": "corun" if sp is None else "corun_part"}.get(
                    cond, cond), emc, mem)
            if sp is not None:
                import e12b_m3 as M3                                     # noqa: PLC0415
                hog = M3.Hog(h.ge, 0)
                hms, gbs = hog.alone()
                h._A_part = 1.0

                def hogloop():
                    with torch.no_grad(), h.ge.switch():
                        for _ in range(20):
                            hog.graph.replay()
                    ev = h._events(1)
                    h.issue_prefix(0, [None], ev["pd"], ev["p0"], ev["p1"])
                emc, mem, _ = with_samplers(hogloop, args.seconds, out / "raw" / f"teg_hog_{spn.replace('/', '-')}.log")
                record(h.label().replace("prefix_first", "hog_max"), spn, "hog_part", emc, mem,
                       notes=f"hog alone in {h.granted[1]} SMs: {gbs:.1f} GB/s")
            del h
            torch.cuda.synchronize()
    fn = out / "bandwidth.csv"
    exists = fn.exists()
    with fn.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    thr = C.throttle_record(g0)
    mp = out / f"meta_{args.mode}.json"
    mp.write_text(json.dumps({"mode": args.mode, "gate_start": g0, **thr,
                              "command": " ".join([sys.executable, *sys.argv]),
                              "bench_commit": C.bench_commit()}, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
