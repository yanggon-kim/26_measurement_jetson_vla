"""T016 step 4 (EV-10): why pi0 bf16_both's E7 stage loops do not add up to the full loop.

E7 (results/20260924_E7_pi0_N10_bf16_both_MAXN): prefix 6395.8 + ae 3112.5 = 9508.3 mJ above idle vs
full 13 666.2 mJ (~30 % missing). Hypothesis (from the committed raw data): power_phases.measure starts
the rail sampler BEFORE phase_loops.run_loop's warm-up, so each phase's mean power includes the warm-up --
and for `prefix` the first call compiles pg.forward (max-autotune-no-cudagraphs), ~45 s at idle power.

--reanalyse   derived, no GPU: the committed rail_samples.csv trimmed to each loop's timed window (the
              last seconds*hz samples of the phase: the sampler stops right after the loop).
--measure     new session: every phase warmed (and compiled) first; ONE idle baseline; then the three
              loops interleaved (prefix, ae, full) x rounds, each with the sampler started AFTER warm-up
              + sync; per rail; GPU clock, EMC rate and temperatures logged at every loop boundary.
              Energy per frame two ways: (mean power - idle) x median latency (E7's definition) and
              integrated energy above idle / iterations.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
E7 = REPO / "results/20260924_E7_pi0_N10_bf16_both_MAXN"


def reanalyse(out: Path, seconds=60.0, hz=20.0):
    rows = list(csv.DictReader(open(E7 / "raw" / "rail_samples.csv")))
    pw = list(csv.DictReader(open(E7 / "power.csv")))
    lat = {r["phase"]: float(r.get("latency_median_ms") or r.get("median_ms") or 0)
           for r in pw if r.get("rail") == "module"}
    rails = [k for k in rows[0] if k not in ("phase", "t")]
    by = {}
    for r in rows:
        by.setdefault(r["phase"], []).append(r)
    keep = int(seconds * hz)

    def mod(samples):
        return statistics.fmean(sum(float(s[k]) for k in rails if s[k] not in ("", None)) for s in samples)
    idle = mod(by["idle"])
    res = []
    for ph in ("prefix", "ae", "full"):
        allm, trm = mod(by[ph]), mod(by[ph][-keep:])
        res.append({"phase": ph, "samples_all": len(by[ph]), "samples_timed": min(keep, len(by[ph])),
                    "latency_median_ms": lat.get(ph, ""),
                    "module_mW_all": round(allm, 1), "module_mW_timed": round(trm, 1),
                    "idle_mW": round(idle, 1),
                    "E_above_idle_mJ_all": round((allm - idle) * lat.get(ph, 0) / 1e3, 1),
                    "E_above_idle_mJ_timed": round((trm - idle) * lat.get(ph, 0) / 1e3, 1)})
    e = {r["phase"]: r for r in res}
    for k in ("all", "timed"):
        s = e["prefix"][f"E_above_idle_mJ_{k}"] + e["ae"][f"E_above_idle_mJ_{k}"]
        f = e["full"][f"E_above_idle_mJ_{k}"]
        res.append({"phase": f"residual_{k}", "E_above_idle_mJ_all" if k == "all" else "E_above_idle_mJ_timed":
                    round(f - s, 1), "latency_median_ms": f"(full - prefix - ae)/full = {100 * (f - s) / f:.2f} %"})
    with (out / "reanalysis_E7_raw.csv").open("w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in res for k in r))
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(res)
    print(json.dumps(res, indent=1))


def measure(out: Path, seconds: float, rounds: int):
    import torch                                                         # noqa: PLC0415
    import e12b_common as C                                              # noqa: PLC0415
    import model_registry as MR                                          # noqa: PLC0415
    import opt_ladder as OL                                              # noqa: PLC0415
    import phase_loops as PL                                             # noqa: PLC0415
    import pi0_common as P                                               # noqa: PLC0415
    import power_phases as PP                                            # noqa: PLC0415
    g0 = C.gate_full()
    if "MAXN" not in g0["power_mode"]:
        raise SystemExit("MAXN only")

    def rails_for(fn, secs, tag):
        """sampler started AFTER the caller warmed fn; returns means, samples, loop result."""
        PP.stop_tegrastats()
        teg = __import__("subprocess").Popen(["sudo", "-n", "tegrastats", "--interval", "100", "--logfile",
                                              str(out / "raw" / f"teg_{tag}.log")])
        s = PP.Rails(); s.start()
        t0 = time.perf_counter()
        if fn is None:
            time.sleep(secs); r = {"median_ms": 0.0, "iterations": 0}
        else:
            r = PL.run_loop(fn, secs, tag, warmup=0)
        t1 = time.perf_counter()
        s.stop.set(); s.join(timeout=2)
        PP.stop_tegrastats(teg)
        rk = [k for k in s.samples[0] if k != "t"]
        means = {k: statistics.fmean(x[k] for x in s.samples if x.get(k) is not None) for k in rk}
        return means, s.samples, r, t1 - t0

    state = []

    def snap(tag):
        g = C.gate_full()
        state.append({"tag": tag, "t_utc": g["t_utc"], "gpu_hz": g["gpu_cur_freq_hz"],
                      "emc_hz": g["emc_rate_hz"], "temps_mC": json.dumps(g["temps_mC"])})

    idle_means, idle_s, _, _ = rails_for(None, seconds, "idle")
    spec = MR.get("pi0")
    policy, cfg, _ = MR.load_policy(spec, n=10, compile_mode=None)
    P.assert_real_weights(policy)
    compiled, prep, prec, comp_s, fails, _ = OL.apply_rung(policy, spec, "bf16_both")
    if fails:
        raise SystemExit(f"rung failed: {fails}")
    batch, _, _ = P.build_batch(policy, P.load_fixture(), checkpoint=spec.checkpoint, revision=spec.revision)
    fns = {ph: PL.make_phase(policy, batch, ph, 10, spec) for ph in ("prefix", "ae", "full")}
    with torch.no_grad():
        for ph, fn in fns.items():                      # warm + compile EVERY phase before any sampling
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
        # the idle baseline once more, now that the model is resident (the E7 idle was before load)
        idle2_means, idle2_s, _, _ = rails_for(None, seconds, "idle_resident")
        rows, samples = [], [{"loop": "idle", **x} for x in idle_s] + [{"loop": "idle_resident", **x} for x in idle2_s]
        mod_idle = sum(idle_means.values())
        for rd in range(rounds):
            for ph in ("prefix", "ae", "full"):
                snap(f"{ph}{rd}_start")
                means, s, r, wall = rails_for(fns[ph], seconds, f"{ph}{rd}")
                snap(f"{ph}{rd}_end")
                mod = sum(means.values())
                e_int = sum((sum(x[k] for k in means) - mod_idle) for x in s) / len(s) * wall / max(1, r["iterations"])
                rows.append({"loop": f"{ph}{rd}", "phase": ph, "round": rd, "iterations": r["iterations"],
                             "median_ms": r["median_ms"], "wall_s": round(wall, 3),
                             "module_mW": round(mod, 1), "idle_module_mW": round(mod_idle, 1),
                             "idle_resident_module_mW": round(sum(idle2_means.values()), 1),
                             **{f"{k}_mW": round(v, 1) for k, v in means.items()},
                             "E_above_idle_mJ_x_median": round((mod - mod_idle) * r["median_ms"] / 1e3, 1),
                             "E_above_idle_mJ_integrated": round(e_int, 1)})
                samples += [{"loop": f"{ph}{rd}", **x} for x in s]
                print(f"  {ph}{rd}: {r['median_ms']:.2f} ms  {mod:.0f} mW  "
                      f"{rows[-1]['E_above_idle_mJ_x_median']:.1f} mJ (x median)  "
                      f"{rows[-1]['E_above_idle_mJ_integrated']:.1f} mJ (integrated)", flush=True)
    res = []
    for rd in range(rounds):
        e = {r["phase"]: r for r in rows if r["round"] == rd}
        for k in ("E_above_idle_mJ_x_median", "E_above_idle_mJ_integrated"):
            s_ = e["prefix"][k] + e["ae"][k]
            res.append({"round": rd, "definition": k, "prefix": e["prefix"][k], "ae": e["ae"][k],
                        "sum": round(s_, 1), "full": e["full"][k],
                        "residual_pct_of_full": round(100 * (e["full"][k] - s_) / e["full"][k], 2)})
    for name, data in (("loops.csv", rows), ("reconcile.csv", res), ("state.csv", state)):
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    with (out / "raw" / "rail_samples.csv").open("w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in samples for k in r))
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(samples)
    thr = C.throttle_record(g0)
    C.write_meta(out / "meta.json", {
        "experiment": "T016 step 4 (EV-10): E7 energy reconciliation, pi0 bf16_both N=10",
        "bench_commit": C.bench_commit(), "model_commit": C.CKPT_COMMIT, "checkpoint": spec.checkpoint,
        "checkpoint_commit": C.CKPT_COMMIT, "rung": "bf16_both", "variant": "bf16_both (E7 phase loops)",
        "power_mode": g0["power_mode"], "sm_count": g0["sm_count"], "granted_partitions": "none (whole GPU)",
        "gpu_clock_hz": g0["gpu_cur_freq_hz"], "emc_rate_hz": g0["emc_rate_hz"], "gate_start": g0, **thr,
        "method": "all phases warmed and compiled first; one idle baseline before load (as E7) and one "
                  "with the model resident; loops interleaved prefix/ae/full x rounds; sampler started "
                  "after warm-up + sync (E7 started it before run_loop's warm-up)",
        "seconds_per_loop": seconds, "rounds": rounds, "rails": "power_phases.Rails (20 Hz INA3221)",
        "reconcile": res, "command": " ".join([sys.executable, *sys.argv]),
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps(res, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--reanalyse", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    out = REPO / a.outdir
    (out / "raw").mkdir(parents=True, exist_ok=True)
    if a.reanalyse:
        reanalyse(out)
    if a.measure:
        measure(out, a.seconds, a.rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
