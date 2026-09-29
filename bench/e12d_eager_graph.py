"""E12d arm (b) -- the EAGER expert captured as ONE CUDA graph, prefix eager on its own stream.

A causal test of T014's host-issue explanation. The eager expert issues ~968 kernels per denoising
step from Python; if that issue is what keeps the two streams from overlapping, capturing the SAME
eager kernels into one torch.cuda.CUDAGraph (one launch per frame) must free the host to issue the
prefix while the expert runs, and T_pipe must fall below A_iso + B_iso. If it does not fall, the
host-issue explanation is wrong.

Everything except the expert's launch mechanism is T014's eager configuration:
  * pi0 eager (no torch.compile anywhere), N = 10, corun.patch_pi0_nosync (label pi0_nosync, as the
    committed T014 eager run) -- it also removes modeling_pi0.py:746's per-step host sync, which a
    graph capture could not contain;
  * the prefix is corun.prefix_stage, issued eagerly on its own stream;
  * the expert is e12b_corun.Harness's owngraph_expert + dbuf_kv construction, UNCHANGED: the
    hoisted N-step loop of the eager model.denoise_step captured once per KV-buffer parity in the
    harness's own graph pool, each frame's KV + pad copied on the prefix stream into buffer k % 2,
    prefix(k) waiting for expert(k-2). Only the label differs (no compile rung).

Order (each gating the next), as E12b M2: equivalence through the harness (pipelined vs serial
per frame, bitwise; stale control must be detected; non-finite count); a cross-check of the graph
expert against the plain eager serial path (corun.run_serial) on the same frames; then bracketed
timings, n >= 100 per condition, a >= 60 s pipeline loop for T_pipe, and the plain (non-graph)
eager expert B in the same session as the reference the graph is compared with.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import corun as CR               # noqa: E402
import e12b_common as C          # noqa: E402
import e12b_corun as H           # noqa: E402
import e12b_m2 as M2             # noqa: E402

N = 10


class EagerGraphHarness(H.Harness):
    def label(self) -> str:
        return "+".join(["eager", "pi0_nosync", "hoist", "owngraph_expert_eager", "dbuf_kv",
                         "prefix_eager", self.order])


def med(v):
    return statistics.median(v) if v else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--smoke", action="store_true", help="equivalence + 12-frame conditions only")
    args = ap.parse_args()
    out = REPO / args.outdir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty (protocol §5)")
    (out / "raw").mkdir(parents=True, exist_ok=True)

    import model_registry as MR                                        # noqa: PLC0415
    g0 = C.gate_full()
    if "MAXN" not in g0["power_mode"] or g0["sm_count"] != 16:
        raise SystemExit("MAXN / 16 SMs only")
    spec = MR.get("pi0")
    policy, cfg, _ = MR.load_policy(spec, n=N, compile_mode=None)
    P_real = CR.P.assert_real_weights(policy, spec=spec)
    nosync = CR.patch_pi0_nosync(policy)
    print("  " + nosync, flush=True)
    frames, manifest = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    t_build = time.perf_counter()
    capture_error = "none"
    try:
        h = EagerGraphHarness(policy, spec, frames, N, config="rung_prefix", order="prefix_first")
    except Exception as e:                                              # noqa: BLE001
        capture_error = f"{type(e).__name__}: {str(e).splitlines()[0][:400]}"
        (out / "raw" / "capture_error.txt").write_text(capture_error + "\n")
        print(f"  *** capture FAILED: {capture_error}", flush=True)
        raise
    print(f"  harness built in {time.perf_counter() - t_build:.1f} s: {h.label()}", flush=True)
    peak_mem = torch.cuda.max_memory_allocated() / 2**30
    fin = C.Finite()

    # ---- 1. equivalence through the harness (E12b M2's function, unchanged) ---------------
    eq_rows, ok, scale = M2.equivalence(h, fin)
    for r in eq_rows:
        r["path"] = "eager"
    # ---- 1b. the graph expert vs the plain eager serial path (no graph), same frames ------
    keep_g = {}
    h.run("serial", 6, fin, keep=keep_g)
    ref, _ = CR.run_serial([frames[k % len(frames)] for k in range(6)], N, keep_actions=True)
    torch.cuda.synchronize()
    cross = []
    for k in range(6):
        a, b = ref[k].float(), keep_g[k].float()
        d = (a - b).abs()
        cross.append({"variant": h.label(), "path": "eager", "check": "graph_vs_plain_eager_serial",
                      "split": "full_gpu", "frame": k, "obs_id": frames[k % 5]["obs_id"],
                      "seed": frames[k % 5]["seed"], "max_abs_diff": d.max().item(),
                      "max_rel_diff": (d / a.abs().clamp_min(1e-6)).max().item(),
                      "bitwise_identical": bool(torch.equal(a, b)),
                      "n_nonfinite": int((~torch.isfinite(b)).sum().item()),
                      "verdict": "BITWISE" if torch.equal(a, b) else "DIFFERS"})
    eq_rows += cross
    with (out / "equivalence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eq_rows[0]))
        w.writeheader(); w.writerows(eq_rows)
    for r in eq_rows:
        print(f"    {r['check']:28s} frame {r['frame']} max|d| {r['max_abs_diff']:.3e} "
              f"nonfinite {r['n_nonfinite']} -> {r['verdict']}", flush=True)
    xb = all(r["bitwise_identical"] for r in cross)
    (out / "equivalence.md").write_text("\n".join([
        "# E12d arm (b) — equivalence, eager expert as one CUDA graph", "",
        f"Variant `{h.label()}`.", "",
        f"* pipelined vs serial through the harness: **{'bitwise identical on every frame' if all(r['verdict']=='PASS' for r in eq_rows if r['check']=='pipeline') else 'NOT identical — INVALID'}**",
        f"* stale control (expert reads the other KV buffer): **{sum(r['verdict']=='DETECTED' for r in eq_rows)} of "
        f"{sum(r['check']=='stale_control' and r['frame']>0 for r in eq_rows)} detected**",
        f"* graph expert vs the plain eager serial path (corun.run_serial, no graph), 6 frames: "
        f"**{'bitwise identical' if xb else 'differs, max|d| ' + format(max(r['max_abs_diff'] for r in cross), '.3e')}**",
        f"* power: consecutive serial frames differ by up to {scale:.4f}",
        f"* non-finite elements in compared actions: {sum(r['n_nonfinite'] for r in eq_rows)}", "",
        "Rows: `equivalence.csv`."]) + "\n")
    print(f"  equivalence {'PASS' if ok else 'FAIL'}; graph-vs-plain bitwise {xb}", flush=True)

    # ---- 2. timings --------------------------------------------------------------------
    F = 12 if args.smoke else args.frames
    samples, host_rows, cs = [], [], {}

    def do(cond, order="prefix_first", Fr=None, tag=""):
        h.order = order
        Fr = Fr or F
        n0 = fin.read()
        h.run(cond, 10, fin)
        r = h.run(cond, Fr, fin)
        nf = fin.read() - n0
        key = f"{cond}{tag}"
        pre = [x["prefix_ms"] for x in r["records"] if "prefix_ms" in x]
        exp = [x["expert_ms"] for x in r["records"] if "expert_ms" in x]
        frm = [x["frame_ms"] for x in r["records"] if "frame_ms" in x]
        cs[key] = {"prefix": med(pre[5:]), "expert": med(exp[5:]), "frame": med(frm[5:]),
                   "prefix_all": pre[5:], "expert_all": exp[5:], "frame_all": frm[5:],
                   "period": r["period_ms"], "host_issue_med": med(r["host_issue_ms"]),
                   "valid": ok and nf == 0, "n": Fr, "nonfinite": nf, "label": h.label()}
        cond_n = {"prefix_alone": "isolated", "expert_alone": "isolated", "serial": "serial",
                  "pipeline": "corun"}[cond]
        for x in r["records"]:
            for st in ("prefix", "expert", "frame"):
                if f"{st}_ms" in x:
                    samples.append({"variant": h.label(), "condition": cond_n, "block": key,
                                    "stage": st, "frame": x["frame"], "ms": round(x[f"{st}_ms"], 4),
                                    "valid": cs[key]["valid"]})
        for i, hv in enumerate(r["host_issue_ms"]):
            host_rows.append({"block": key, "frame": i, "issue_ms": round(hv, 4),
                              "period_ms": round(r["period_ms"], 4)})
        print(f"  {key:24s} prefix {cs[key]['prefix']:8.2f}  expert {cs[key]['expert']:8.2f}  "
              f"frame {cs[key]['frame']:8.2f}  period {r['period_ms']:8.2f}  host-issue "
              f"{cs[key]['host_issue_med']:7.2f} ms  nonfinite {nf}", flush=True)

    def plain_expert(tag):
        """The NON-graph eager expert, same frames, same stream discipline as corun.run_isolated."""
        n = 12 if args.smoke else args.frames
        s = CR.run_isolated([frames[k % 5] for k in range(n)], N, "expert", h.se)
        v = [x["ms"] for x in s][5:]
        cs[f"plain_expert{tag}"] = {"expert": med(v), "expert_all": v, "n": n}
        for x in s:
            samples.append({"variant": "eager+pi0_nosync+hoist (no graph)", "condition": "isolated",
                            "block": f"plain_expert{tag}", "stage": "expert", "frame": x["frame"],
                            "ms": round(x["ms"], 4), "valid": True})
        print(f"  plain_expert{tag:14s} expert {med(v):8.2f}  (non-graph eager, reference)", flush=True)

    with torch.no_grad():
        plain_expert("_pre")
        for c in ("prefix_alone", "expert_alone", "serial"):
            do(c, tag="_pre")
        do("pipeline", "prefix_first", tag="_prefix_first")
        do("pipeline", "expert_first", tag="_expert_first")
        if not args.smoke:
            est = cs["pipeline_prefix_first"]["period"]
            do("pipeline", "prefix_first", Fr=max(F, int(args.seconds * 1000 / max(est, 1)) + 1),
               tag="_60s")
        for c in ("prefix_alone", "expert_alone", "serial"):
            do(c, tag="_post")
        plain_expert("_post")

    def both(key, f):
        return (cs[f"{key}_pre"][f] + cs[f"{key}_post"][f]) / 2

    A, B, Tser = both("prefix_alone", "prefix"), both("expert_alone", "expert"), both("serial", "frame")
    Bplain = (cs["plain_expert_pre"]["expert"] + cs["plain_expert_post"]["expert"]) / 2
    drift = {k: round(abs(cs[f"{k}_post"][f] - cs[f"{k}_pre"][f]) / cs[f"{k}_pre"][f] * 100, 3)
             for k, f in (("prefix_alone", "prefix"), ("expert_alone", "expert"), ("serial", "frame"))}
    drift["plain_expert"] = round(abs(cs["plain_expert_post"]["expert"] - cs["plain_expert_pre"]["expert"])
                                  / cs["plain_expert_pre"]["expert"] * 100, 3)
    ratios = []
    for key in [k for k in ("pipeline_prefix_first", "pipeline_expert_first", "pipeline_60s") if k in cs]:
        c = cs[key]
        Tp, mx = c["period"], max(A, B)
        ratios.append({
            "experiment": "E12d", "arm": "eager_graph_expert", "variant": c["label"],
            "power_mode": g0["power_mode"], "sm_count": 16, "N": N,
            "A_iso_ms": round(A, 4), "B_iso_ms": round(B, 4), "T_serial_ms": round(Tser, 4),
            "A_co_ms": round(c["prefix"], 4), "B_co_ms": round(c["expert"], 4),
            "T_pipe_ms": round(Tp, 4), "s": round(c["expert"] / B, 4), "p": round(c["prefix"] / A, 4),
            "r": round(A / B, 4), "O": round(Tp / mx, 4), "phi": round((Tser - Tp) / (Tser - mx), 4),
            "speedup_vs_serial": round(Tser / Tp, 4),
            "B_iso_plain_eager_ms": round(Bplain, 4),
            "A_co_p5_ms": round(sorted(c["prefix_all"])[int(0.05 * len(c["prefix_all"]))], 4),
            "A_co_p95_ms": round(sorted(c["prefix_all"])[int(0.95 * (len(c["prefix_all"]) - 1))], 4),
            "B_co_p5_ms": round(sorted(c["expert_all"])[int(0.05 * len(c["expert_all"]))], 4),
            "B_co_p95_ms": round(sorted(c["expert_all"])[int(0.95 * (len(c["expert_all"]) - 1))], 4),
            "host_issue_median_ms": round(c["host_issue_med"], 4), "n": c["n"],
            "reference": "same_session_same_harness", "valid": c["valid"], "block": key})
    iso_rows = []
    for key, v in cs.items():
        for st in ("prefix", "expert", "frame"):
            vals = v.get(f"{st}_all") or []
            if vals:
                ss = C.summ(vals)
                iso_rows.append({"block": key, "stage": st, **ss})
    for name, rows_ in (("latency_samples.csv", samples), ("host_issue.csv", host_rows),
                        ("corun_ratios.csv", ratios), ("condition_summary.csv", iso_rows)):
        if rows_:
            with ((out / "raw" / name) if name in ("latency_samples.csv", "host_issue.csv") else (out / name)
                  ).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_[0]))
                w.writeheader(); w.writerows(rows_)
    for r in ratios:
        print(f"  {r['block']:22s} O={r['O']} phi={r['phi']} p={r['p']} s={r['s']} r={r['r']} "
              f"T_pipe={r['T_pipe_ms']} (A {r['A_iso_ms']}, B {r['B_iso_ms']}, Tser {r['T_serial_ms']}, "
              f"B plain eager {r['B_iso_plain_eager_ms']})", flush=True)
    thr = C.throttle_record(g0)
    C.write_meta(out / "meta.json", {
        "experiment": "E12d arm (b) -- eager expert captured as one CUDA graph, eager prefix, in-process "
                      "two-stream co-run (owner-requested T014 follow-up)",
        "task": "owner request 2026-09-28 (T014 follow-up)",
        "bench_commit": C.bench_commit(), "model_commit": C.CKPT_COMMIT, "checkpoint": spec.checkpoint,
        "checkpoint_commit": C.CKPT_COMMIT, "rung": "eager (no torch.compile)", "compile_mode": "none (eager)",
        "variant": h.label().replace("prefix_first", "<order>"), "harness_note": nosync,
        "power_mode": g0["power_mode"], "sm_count": g0["sm_count"],
        "granted_partitions": "none (full GPU, 16 SMs)", "gpu_clock_hz": g0["gpu_cur_freq_hz"],
        "emc_rate_hz": g0["emc_rate_hz"], "gate_start": g0, **thr, "N": N, "batch": 1,
        "frames_per_condition": F, "smoke": bool(args.smoke), "equivalence_pass": ok,
        "graph_vs_plain_eager_bitwise": xb, "capture_error": capture_error,
        "bracket_drift_pct": drift, "peak_memory_allocated_GiB": round(peak_mem, 3),
        "weights_verified": P_real, "frames_manifest": manifest,
        "decisions": {"pipeline": "free-running, gated only by CUDA events",
                      "kv_handoff": "dbuf_kv (E12b): KV + pad copied on the prefix stream into expert "
                                    "buffer k%2; prefix(k) waits for expert(k-2); copy counted in the prefix",
                      "first_5_frames_dropped": "per-stage medians skip frames 0-4 of each run; T_pipe is "
                                                "wall / frames over the whole run",
                      "plain_expert": "the non-graph eager expert through corun.run_isolated, bracketed, as "
                                      "the reference the graph is compared with"},
        "command": " ".join([sys.executable, *sys.argv]), "torch": torch.__version__,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
