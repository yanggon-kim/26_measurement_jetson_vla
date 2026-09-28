"""E12b M5 -- energy per frame, WHOLE-FRAME loops only (spec §6 M5).

E7's phase loops do not add up (prefix + ae 418.64 ms vs full 424.68; 9 509 vs 13 666 mJ above
idle), so only whole-frame loops are compared. One session: 60 s idle, then >= 60 s each of
  (a) serial `full`, stock -- policy.predict_action_chunk, the E7 anchor
  (b) serial `full`, fair configuration, through the co-run harness (serial condition)
  (c) the best VALID co-run (pipeline condition of the chosen configuration)
Four sysfs rails at 20 Hz (power_core, as E7) with tegrastats alongside. Energy per frame =
mean power x period, total and above idle, per rail, four-rail module sum and the three-rail sum
without the memory rail. Idle subtraction: each rail's in-session 60 s idle mean. Every action
tensor in every arm is finite-checked on the device.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import corun as CR               # noqa: E402
import e12b_common as C          # noqa: E402
import e12b_corun as H           # noqa: E402
import power_core as PC          # noqa: E402
import power_phases as PP        # noqa: E402

N = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--corun-config", default="rung_prefix", choices=("rung_prefix", "graphs_both"))
    ap.add_argument("--corun-split", default="", help="e.g. 12/4 for a green split; empty = full GPU")
    ap.add_argument("--frames-per-call", type=int, default=30,
                    help="frames per timed call; a pipeline has a fill and a drain per call, so "
                         "10 (the first version) inflated the co-run period to 323 ms vs M3's "
                         "steady 307")
    args = ap.parse_args()
    out = REPO / args.outdir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty (protocol §5)")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    g0 = C.gate_full()
    if "MAXN" not in g0["power_mode"]:
        raise SystemExit("MAXN only")
    print(f"  {args.seconds:.0f} s idle baseline first (before the model is resident)", flush=True)
    idle, _, idle_s = PP.measure(None, args.seconds, "idle", out / "raw" / "tegrastats_idle.log")
    policy, spec, cfg, rung_note = C.load_rung(N)
    fx = json.loads((REPO / "bench/fixtures/e12/obs_0.json").read_text())
    batch0, _, _ = C.P.build_batch(policy, fx, fixtures=REPO / "bench/fixtures/e12",
                                   checkpoint=spec.checkpoint, revision=spec.revision)
    fin = C.Finite()

    def stock_full():
        fin.add(policy.predict_action_chunk(batch0))
    for _ in range(10):
        stock_full()
    torch.cuda.synchronize()
    arms = [("a_stock_full", stock_full, 1, "bf16_both+stock+shipping_full")]

    # (b) and (c) need the harness's compilation; the stock arm is measured FIRST, before
    # denoise_step is recompiled for the own-graph expert.
    rows, samples, results = [], [], {}
    frames = None

    def measure_arm(name, fn, frames_per_call, label):
        fin0 = fin.read()
        means, r, s = PP.measure(fn, args.seconds, name, out / "raw" / f"tegrastats_{name}.log")
        nf = fin.read() - fin0
        per = r["median_ms"] / frames_per_call
        mod, mod0 = sum(means.values()), sum(idle.values())
        nomem = sum(v for k, v in means.items() if k != PP.MEM_RAIL)
        nomem0 = sum(v for k, v in idle.items() if k != PP.MEM_RAIL)
        results[name] = {"label": label, "period_ms": round(per, 4), "module_mW": round(mod, 1),
                         "energy_per_frame_mJ": round(mod * per / 1e3, 3),
                         "energy_per_frame_above_idle_mJ": round((mod - mod0) * per / 1e3, 3),
                         "three_rail_no_mem_mJ": round(nomem * per / 1e3, 3),
                         "three_rail_no_mem_above_idle_mJ": round((nomem - nomem0) * per / 1e3, 3),
                         "n_nonfinite": nf, "valid": nf == 0}
        for rail, mw in list(means.items()) + [("module", mod), ("module_without_memory_rail", nomem)]:
            base = {"module": mod0, "module_without_memory_rail": nomem0}.get(rail, idle.get(rail))
            rows.append({"arm": name, "variant": label, "rail": rail + (
                " (regulator input)" if rail == PP.MEM_RAIL else ""),
                "mean_mW": round(mw, 2), "idle_mW": round(base, 2),
                "above_idle_mW": round(mw - base, 2), "period_ms": round(per, 4),
                "energy_per_frame_mJ": round(mw * per / 1e3, 4),
                "energy_per_frame_above_idle_mJ": round((mw - base) * per / 1e3, 4),
                "n_samples": len(s), "valid": nf == 0})
        samples.extend({"arm": name, **x} for x in s)
        print(f"  {name:16s} {per:8.2f} ms/frame  {mod:7.0f} mW  "
              f"{results[name]['energy_per_frame_mJ']:9.1f} mJ/frame  nonfinite {nf}", flush=True)

    with torch.no_grad():
        measure_arm(*arms[0][:2], arms[0][2], arms[0][3])
        rung_note["expert"] = H.recompile_default(policy, spec, "denoise_step")
        if args.corun_config == "graphs_both":
            rung_note["embed_image"] = H.recompile_default(policy, spec, "embed_image")
            rung_note["memo"] = H.memo_681(policy)
        frames, manifest = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
        split = tuple(int(x) for x in args.corun_split.split("/")) if args.corun_split else None
        # (b) is the fair SERIAL frame on the WHOLE GPU, through the same configuration's harness.
        # The first version measured it through the partitioned harness -- prefix on 12 SMs then
        # expert on 4, one after the other: 526 ms instead of ~353 -- which made the co-run look
        # 13 % cheaper in energy by comparison. That run is kept in scratch/superseded/.
        hf = H.Harness(policy, spec, frames, N, config=args.corun_config, split=None,
                       order="prefix_first")
        h = hf if split is None else H.Harness(policy, spec, frames, N, config=args.corun_config,
                                               split=split, order="prefix_first")
        F = args.frames_per_call
        hf.run("serial", 6, fin); h.run("pipeline", 6, fin)
        measure_arm("b_fair_serial", lambda: hf.run("serial", F, fin), F,
                    hf.label() + "+serial_full_gpu")
        measure_arm("c_best_corun", lambda: h.run("pipeline", F, fin), F, h.label())

    PC.write_power_csv(out, rows, [{"phase": "idle", **x} for x in idle_s] + samples)
    b, c = results["b_fair_serial"], results["c_best_corun"]
    res = {"arms": results,
           "corun_over_fair_serial_energy": round(c["energy_per_frame_mJ"] / b["energy_per_frame_mJ"], 4),
           "corun_over_fair_serial_above_idle": round(c["energy_per_frame_above_idle_mJ"] / b["energy_per_frame_above_idle_mJ"], 4),
           "outcome_10_corun_le_0.95_fair_serial": c["energy_per_frame_mJ"] <= 0.95 * b["energy_per_frame_mJ"],
           "idle_subtraction": "each rail's in-session 60 s idle mean, measured before the model "
                               "was loaded"}
    (out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    thr = C.throttle_record(g0)
    C.write_meta(out / "meta.json", {
        "experiment": "E12b M5 -- energy per frame, whole-frame loops (T015)",
        "bench_commit": C.bench_commit(), "model_commit": C.CKPT_COMMIT,
        "checkpoint": spec.checkpoint, "checkpoint_commit": C.CKPT_COMMIT,
        "rung": "bf16_both", "rung_note": rung_note,
        "variant": " | ".join(v["label"] for v in results.values()),
        "power_mode": g0["power_mode"], "sm_count": g0["sm_count"],
        "granted_partitions": f"{h.granted[0]}/{h.granted[1]}" if split else "none (full GPU)",
        "gpu_clock_hz": g0["gpu_cur_freq_hz"], "emc_rate_hz": g0["emc_rate_hz"],
        "gate_start": g0, **thr, "N": N, "seconds_per_arm": args.seconds,
        "rails": "four sysfs INA3221 rails at 20 Hz (power_core) + tegrastats at 100 ms",
        "results": res, "command": " ".join([sys.executable, *sys.argv]),
        "torch": torch.__version__, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
