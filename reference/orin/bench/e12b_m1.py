"""E12b M1 -- the fair GPU expert: compiled `stock` vs compiled `hoist` (spec §6 M1).

  stock  bf16_both exactly as E7, the per-step `torch.tensor(<float>, device)` timestep (shipping)
  hoist  bf16_both, the N timestep tensors built once (corun.timestep_tensors); nothing else

1. EQUIVALENCE FIRST, compiled path: hoist vs stock on 5 distinct LIBERO observations x their own
   noise seeds, same prefix cache; plus stock vs stock (same inputs) as the run-to-run floor.
   Bitwise identity is the expectation because the timestep VALUES are identical.
2. TIMINGS, bracketed stock -> hoist -> stock, each loop >= 100 frames and >= 60 s after 20 warm-up:
     prefix       corun.prefix_stage (= phase_builders.prefix_once, E7's prefix loop)
     ae           the expert on a fixed cache, stock or hoist timestep
     full_ship    policy.predict_action_chunk (E7's `full`; stock only, the anchor)
     full_stages  prefix_stage + expert_stage -- the same two callables back to back, so
                  full_stages(stock) vs full_stages(hoist) differ ONLY by the hoist
   Every action tensor of every timed loop is finite-checked on the device, read after the loop.
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

N = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-frames", type=int, default=100)
    ap.add_argument("--min-seconds", type=float, default=60.0)
    args = ap.parse_args()
    out = REPO / args.outdir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty (protocol §5)")
    (out / "raw").mkdir(parents=True, exist_ok=True)

    g0 = C.gate_full()
    if "MAXN" not in g0["power_mode"] or g0["sm_count"] != 16:
        raise SystemExit(f"gate: {g0['power_mode']} / {g0['sm_count']} SMs -- MAXN only (NN5)")
    print(f"  gate: {g0['power_mode']} GPU {g0['gpu_cur_freq_hz']} EMC {g0['emc_rate_hz']}",
          flush=True)
    policy, spec, cfg, rung_note = C.load_rung(N)
    print(f"  rung matches E7: {rung_note['compiled']}", flush=True)
    frames, manifest = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)

    # ---- 1. equivalence, compiled path --------------------------------------------------
    eq_rows = []
    fin = C.Finite()
    with torch.no_grad():
        caches = []
        for fr in frames:
            pkv, pad = CR.prefix_stage(fr["ctx"])
            caches.append((pkv, pad))
        for fr, (pkv, pad) in zip(frames, caches):
            a_st = CR.expert_stage(fr["ctx"], pkv, pad, fr["noise"], N, e7_timesteps=True).clone()
            a_st2 = CR.expert_stage(fr["ctx"], pkv, pad, fr["noise"], N, e7_timesteps=True).clone()
            a_ho = CR.expert_stage(fr["ctx"], pkv, pad, fr["noise"], N, e7_timesteps=False).clone()
            torch.cuda.synchronize()
            for chk, x, y in (("hoist", a_ho, a_st), ("stock_run_to_run", a_st2, a_st)):
                d = (x.float() - y.float()).abs()
                rel = (d / y.float().abs().clamp_min(1e-6)).max().item()
                nf = int((~torch.isfinite(x)).sum().item() + (~torch.isfinite(y)).sum().item())
                eq_rows.append({"variant": "bf16_both+hoist" if chk == "hoist" else "bf16_both+stock",
                                "path": "compiled", "check": chk, "split": "full_gpu",
                                "frame": fr["k"], "obs_id": fr["obs_id"], "seed": fr["seed"],
                                "max_abs_diff": d.max().item(), "max_rel_diff": rel,
                                "bitwise_identical": bool(torch.equal(x, y)),
                                "n_nonfinite": nf,
                                "verdict": "PASS" if (torch.equal(x, y) and nf == 0) else "FAIL"})
        # scale: how different are consecutive frames' actions (the check's power)
        scale = max((CR.expert_stage(frames[i]["ctx"], *caches[i], frames[i]["noise"], N).float()
                     - CR.expert_stage(frames[i + 1]["ctx"], *caches[i + 1],
                                       frames[i + 1]["noise"], N).float()).abs().max().item()
                    for i in range(len(frames) - 1))
    hoist_pass = all(r["verdict"] == "PASS" for r in eq_rows if r["check"] == "hoist")
    floor_ok = all(r["verdict"] == "PASS" for r in eq_rows if r["check"] == "stock_run_to_run")
    with (out / "equivalence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eq_rows[0]))
        w.writeheader(); w.writerows(eq_rows)
    print(f"  hoist equivalence: {'PASS' if hoist_pass else 'FAIL'} "
          f"(stock run-to-run {'bitwise' if floor_ok else 'NOT bitwise'}); frame scale {scale:.3f}",
          flush=True)

    # ---- 2. timings, bracketed ----------------------------------------------------------
    ctx0 = frames[0]["ctx"]
    pkv0, pad0 = caches[0]
    noise0 = frames[0]["noise"]
    # full_ship needs the preprocessed batch itself (build_ctx keeps only derived tensors);
    # built from obs_0 exactly as frame 0's ctx was.
    fx = json.loads((REPO / "bench/fixtures/e12/obs_0.json").read_text())
    batch0, _, _ = C.P.build_batch(policy, fx, fixtures=REPO / "bench/fixtures/e12",
                                   checkpoint=spec.checkpoint, revision=spec.revision)

    def mk(which, stock):
        if which == "prefix":
            return lambda: CR.prefix_stage(ctx0)
        if which == "ae":
            def f():
                a = CR.expert_stage(ctx0, pkv0, pad0, noise0, N, e7_timesteps=stock)
                fin.add(a)
            return f
        if which == "full_stages":
            def f():
                pkv, pad = CR.prefix_stage(ctx0)
                a = CR.expert_stage(ctx0, pkv, pad, noise0, N, e7_timesteps=stock)
                fin.add(a)
            return f
        if which == "full_ship":
            def f():
                a = policy.predict_action_chunk(batch0)
                fin.add(a)
            return f
        raise ValueError(which)

    plan = [("stock", "prefix"), ("stock", "ae"), ("stock", "full_ship"), ("stock", "full_stages"),
            ("hoist", "prefix"), ("hoist", "ae"), ("hoist", "full_stages"),
            ("stock", "prefix"), ("stock", "ae"), ("stock", "full_ship"), ("stock", "full_stages")]
    if batch0 is None:
        plan = [p for p in plan if p[1] != "full_ship"]
        print("  *** ctx carries no batch: full_ship skipped", flush=True)
    samples, summary = [], []
    with torch.no_grad():
        for bi, (var, which) in enumerate(plan):
            label = f"bf16_both+{var}+{'shipping_full' if which == 'full_ship' else 'stages'}"
            fin0 = fin.read()
            ms = C.loop(mk(which, var == "stock"), args.min_frames, args.min_seconds)
            nf = fin.read() - fin0
            block = "pre" if bi < 4 else ("hoist" if var == "hoist" else "post")
            s = C.summ(ms)
            summary.append({"variant": label, "loop": which, "block": block, **s,
                            "n_nonfinite": nf, "valid": nf == 0})
            samples += [{"variant": label, "split": "full_gpu", "condition": "isolated",
                         "stage": which, "frame": i, "ms": round(x, 4), "valid": nf == 0}
                        for i, x in enumerate(ms)]
            print(f"  {block:5s} {var:5s} {which:12s} median {s['median_ms']:8.2f} ms "
                  f"n={s['n']} nonfinite={nf}", flush=True)

    with (out / "latency_samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0]))
        w.writeheader(); w.writerows(samples)
    with (out / "loops_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)

    def med(var, which):
        v = [r["median_ms"] for r in summary if r["variant"].startswith(f"bf16_both+{var}")
             and r["loop"] == which]
        return sum(v) / len(v) if v else None

    B_st, B_ho = med("stock", "ae"), med("hoist", "ae")
    res = {"B_stock_ms": B_st, "B_hoist_ms": B_ho, "A_ms": med("stock", "prefix"),
           "full_ship_stock_ms": med("stock", "full_ship"),
           "full_stages_stock_ms": med("stock", "full_stages"),
           "full_stages_hoist_ms": med("hoist", "full_stages"),
           "hoist_saves_ms_per_frame": (med("stock", "full_stages") or 0) - (med("hoist", "full_stages") or 0),
           "reference_E7": {"A": 251.28, "B": 167.36, "full": 424.68},
           "reference_reconciliation_B_hoist": 111.02,
           "outcome_7_fair_B_le_93.1ms": bool(B_ho is not None and B_ho <= 93.1),
           "outcome_8_93.1_lt_B_le_102.4": bool(B_ho is not None and 93.1 < B_ho <= 102.4),
           "outcome_9_hoist_fails_equivalence": not hoist_pass}
    (out / "summary.json").write_text(json.dumps(res, indent=2) + "\n")

    md = ["# E12b M1 — hoist equivalence (compiled `bf16_both` path)", "",
          f"Compared: `expert_stage(..., e7_timesteps=False)` (hoist) against `e7_timesteps=True` "
          f"(stock, the shipping per-step timestep), both through the compiled rung, same prefix "
          f"cache and noise, on {len(frames)} distinct LIBERO observations with their own seeds. "
          f"Floor: stock against itself with identical inputs.", "",
          f"* hoist vs stock: **{'PASS — bitwise identical on every frame' if hoist_pass else 'FAIL'}**",
          f"* stock vs stock (run-to-run floor): {'bitwise' if floor_ok else 'NOT bitwise'}",
          f"* power of the check: consecutive frames' actions differ by up to {scale:.4f} "
          f"(max abs), so a hoist that changed the computation would be visible", "",
          "Per-frame rows: `equivalence.csv`."]
    (out / "equivalence.md").write_text("\n".join(md) + "\n")

    thr = C.throttle_record(g0)
    C.write_meta(out / "meta.json", {
        "experiment": "E12b M1 -- the fair GPU expert (T015)",
        "bench_commit": C.bench_commit(), "model_commit": C.CKPT_COMMIT,
        "checkpoint": spec.checkpoint, "checkpoint_commit": C.CKPT_COMMIT,
        "rung": "bf16_both", "rung_note": rung_note,
        "variant": "bf16_both+stock | bf16_both+hoist (per row)",
        "power_mode": g0["power_mode"], "sm_count": g0["sm_count"],
        "granted_partitions": "none (full GPU, 16 SMs)",
        "gpu_clock_hz": g0["gpu_cur_freq_hz"], "emc_rate_hz": g0["emc_rate_hz"],
        "gate_start": g0, **thr, "N": N, "batch": 1,
        "loops": "bracketed stock -> hoist -> stock; each >= %d frames and >= %.0f s after 20 "
                 "warm-up; per-iteration sync (E7 run_loop semantics)" % (args.min_frames,
                                                                        args.min_seconds),
        "frames_manifest": manifest, "results": res,
        "command": " ".join([sys.executable, *sys.argv]),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps(res, indent=2), flush=True)
    return 0 if hoist_pass else 9


if __name__ == "__main__":
    raise SystemExit(main())
