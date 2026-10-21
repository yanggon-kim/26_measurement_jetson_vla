"""E12b M2 -- a correct compiled co-run on the whole GPU (spec §6 M2).

Headline configuration (closest to baseline): the prefix EXACTLY as the rung compiles it (its
embed_image cudagraph tree and max-autotune pg.forward, on the prefix stream) and the fair
expert (hoist) as this harness's own CUDA graph with its own pool, KV handed over through a
double buffer. Label: bf16_both+hoist+owngraph_expert+dbuf_kv+<order>.

Order of work, each gating the next:
  1. equivalence through the timing harness: pipelined vs serial per frame (6 frames cycling the
     5 LIBERO observations with per-frame seeds), and the stale control (expert reads the other
     buffer), which MUST be detected; non-finite count on every action.
  2. timings, bracketed: prefix_alone, expert_alone, serial -> pipeline (prefix_first) ->
     pipeline (expert_first) -> prefix_alone, expert_alone, serial again; n >= 100 each, plus a
     >= 60 s pipeline loop for T_pipe as throughput.
"""
from __future__ import annotations

import argparse
import csv
import json
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

N = 10


def equivalence(h, fin, F=6):
    keep_s, keep_p, keep_x = {}, {}, {}
    h.run("serial", F, fin, keep=keep_s)
    h.run("pipeline", F, fin, keep=keep_p)
    h.run("pipeline", F, fin, keep=keep_x, stale=True)
    torch.cuda.synchronize()
    rows, scale = [], 0.0
    for k in range(F):
        a, b, x = keep_s[k].float(), keep_p[k].float(), keep_x[k].float()
        fr = h.frames[k % len(h.frames)]
        if k:
            scale = max(scale, (a - keep_s[k - 1].float()).abs().max().item())
        for chk, y in (("pipeline", b), ("stale_control", x)):
            d = (y - a).abs()
            nf = int((~torch.isfinite(y)).sum().item() + (~torch.isfinite(a)).sum().item())
            bit = bool(torch.equal(y, a))
            if chk == "pipeline":
                verdict = "PASS" if (bit and nf == 0) else "FAIL"
            else:   # the control must DIFFER; frame 0 has no other frame in the other buffer
                verdict = ("n/a (frame 0)" if k == 0 else ("DETECTED" if not bit else "MISSED"))
            rows.append({"variant": h.label(), "path": "compiled", "check": chk,
                         "split": "full_gpu" if not h.split else f"{h.granted[0]}/{h.granted[1]}",
                         "frame": k, "obs_id": fr["obs_id"], "seed": fr["seed"],
                         "max_abs_diff": d.max().item(),
                         "max_rel_diff": (d / a.abs().clamp_min(1e-6)).max().item(),
                         "bitwise_identical": bit, "n_nonfinite": nf, "verdict": verdict})
    ok = (all(r["verdict"] == "PASS" for r in rows if r["check"] == "pipeline")
          and all(r["verdict"] in ("DETECTED", "n/a (frame 0)") for r in rows
                  if r["check"] == "stale_control"))
    return rows, ok, scale


def med(v):
    return statistics.median(v) if v else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--m1", default="", help="M1 summary.json for the stock-harness comparison")
    args = ap.parse_args()
    out = REPO / args.outdir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty (protocol §5)")
    (out / "raw").mkdir(parents=True, exist_ok=True)

    g0 = C.gate_full()
    if "MAXN" not in g0["power_mode"] or g0["sm_count"] != 16:
        raise SystemExit("MAXN / 16 SMs only (NN5)")
    policy, spec, cfg, rung_note = C.load_rung(N)
    rung_note["expert"] = H.recompile_default(policy, spec, "denoise_step")
    frames, manifest = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    t_build = time.perf_counter()
    h = H.Harness(policy, spec, frames, N, config="rung_prefix", order="prefix_first")
    print(f"  harness built in {time.perf_counter() - t_build:.1f} s: {h.label()}", flush=True)
    peak_mem = torch.cuda.max_memory_allocated() / 2**30
    fin = C.Finite()

    # ---- 1. equivalence -----------------------------------------------------------------
    eq_rows, ok, scale = equivalence(h, fin)
    with (out / "equivalence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eq_rows[0]))
        w.writeheader(); w.writerows(eq_rows)
    for r in eq_rows:
        print(f"    {r['check']:14s} frame {r['frame']} max|d| {r['max_abs_diff']:.3e} "
              f"nonfinite {r['n_nonfinite']} -> {r['verdict']}", flush=True)
    print(f"  equivalence {'PASS' if ok else 'FAIL'}; serial frame-to-frame scale {scale:.4f}",
          flush=True)
    (out / "equivalence.md").write_text("\n".join([
        "# E12b M2 — pipeline equivalence, compiled path, whole GPU", "",
        f"Variant `{h.label()}`. Through the timing harness itself: the same captured graphs run "
        "serially (frame k complete before k+1) and pipelined (prefix(k) with expert(k-1)); "
        "6 frames cycling the 5 LIBERO observations, each with its own noise seed.", "",
        f"* pipelined vs serial: **{'bitwise identical on every frame' if all(r['verdict']=='PASS' for r in eq_rows if r['check']=='pipeline') else 'NOT identical — configuration INVALID'}**",
        f"* stale control (expert reads the other KV buffer = another frame's cache): "
        f"**{sum(r['verdict']=='DETECTED' for r in eq_rows)} of "
        f"{sum(r['check']=='stale_control' and r['frame']>0 for r in eq_rows)} detected**",
        f"* power: consecutive serial frames' actions differ by up to {scale:.4f} (max abs)",
        f"* non-finite elements in any compared action: "
        f"{sum(r['n_nonfinite'] for r in eq_rows)}", "",
        "Rows: `equivalence.csv`."]) + "\n")
    if not ok:
        print("  *** equivalence FAILED: timings are still taken but every row is valid=false",
              flush=True)

    # ---- 2. timings ---------------------------------------------------------------------
    samples, host_rows, cond_summary = [], [], {}

    def do(cond, order=None, F=None, tag=""):
        if order:
            h.order = order
        F = F or args.frames
        fin0 = fin.read()
        h.run(cond, 10, fin)                           # warm-up of this condition
        r = h.run(cond, F, fin)
        nf = fin.read() - fin0
        valid = ok and nf == 0
        lab = h.label()
        key = f"{cond}{tag}"
        pre = [x.get("prefix_ms") for x in r["records"] if x.get("prefix_ms") is not None]
        exp = [x.get("expert_ms") for x in r["records"] if x.get("expert_ms") is not None]
        frm = [x.get("frame_ms") for x in r["records"] if x.get("frame_ms") is not None]
        cond_summary.setdefault(key, []).append({
            "prefix": med(pre[5:]), "expert": med(exp[5:]), "frame": med(frm[5:]),
            "period": r["period_ms"], "host_issue_med": med(r["host_issue_ms"]),
            "valid": valid, "n": F, "nonfinite": nf, "label": lab})
        condname = {"prefix_alone": "isolated", "expert_alone": "isolated",
                    "serial": "serial", "pipeline": "corun"}[cond]
        for x in r["records"]:
            for st in ("prefix", "expert", "frame"):
                v = x.get(f"{st}_ms")
                if v is not None:
                    samples.append({"variant": lab, "split": "full_gpu", "condition": condname,
                                    "stage": st, "frame": x["frame"], "ms": round(v, 4),
                                    "valid": valid, "block": key})
        for i, hv in enumerate(r["host_issue_ms"]):
            host_rows.append({"variant": lab, "condition": key, "frame": i,
                              "issue_ms": round(hv, 4), "period_ms": round(r["period_ms"], 4),
                              "order": h.order})
        print(f"  {key:22s} prefix {med(pre[5:]):8.2f}  expert {med(exp[5:]):8.2f}  "
              f"frame {med(frm[5:]):8.2f}  period {r['period_ms']:8.2f}  host-issue "
              f"{med(r['host_issue_ms']):7.2f} ms  nonfinite {nf}", flush=True)
        return r

    with torch.no_grad():
        for c in ("prefix_alone", "expert_alone", "serial"):
            do(c, "prefix_first", tag="_pre")
        do("pipeline", "prefix_first", tag="_prefix_first")
        do("pipeline", "expert_first", tag="_expert_first")
        # >= 60 s throughput loop, prefix-first
        est = cond_summary["pipeline_prefix_first"][0]["period"]
        F60 = max(args.frames, int(args.seconds * 1000 / max(est, 1)) + 1)
        do("pipeline", "prefix_first", F=F60, tag="_60s")
        for c in ("prefix_alone", "expert_alone", "serial"):
            do(c, "prefix_first", tag="_post")

    def both(key, field):
        v = [x[field] for x in cond_summary.get(f"{key}_pre", []) + cond_summary.get(f"{key}_post", [])]
        return sum(v) / len(v) if v else float("nan")

    A = both("prefix_alone", "prefix")
    B = both("expert_alone", "expert")
    Tser = both("serial", "frame")
    drift = {k: abs(cond_summary[f"{k}_post"][0][f] - cond_summary[f"{k}_pre"][0][f])
             / cond_summary[f"{k}_pre"][0][f] * 100
             for k, f in (("prefix_alone", "prefix"), ("expert_alone", "expert"),
                          ("serial", "frame"))}
    ratios = []
    m1 = json.loads(Path(args.m1).read_text()) if args.m1 else {}
    for key in ("pipeline_prefix_first", "pipeline_expert_first", "pipeline_60s"):
        c = cond_summary[key][0]
        Tp = c["period"]
        mx = max(A, B)
        row = {"mechanism": "compiled_owngraph_expert", "variant": c["label"],
               "split": "full_gpu", "prefix_sms": 16, "expert_sms": 16,
               "power_mode": g0["power_mode"], "sm_count": 16, "N": N, "rung": "bf16_both",
               "A_iso_ms": round(A, 4), "B_iso_ms": round(B, 4), "T_serial_ms": round(Tser, 4),
               "A_co_ms": round(c["prefix"], 4), "B_co_ms": round(c["expert"], 4),
               "T_pipe_ms": round(Tp, 4), "s": round(c["expert"] / B, 4),
               "p": round(c["prefix"] / A, 4), "r": round(A / B, 4),
               "O": round(Tp / mx, 4), "phi": round((Tser - Tp) / (Tser - mx), 4),
               "T_serial_over_T_pipe": round(Tser / Tp, 4),
               "A_part_ms": "", "B_part_ms": "", "p_part": "", "s_part": "", "p_hog": "",
               "reference": "fair_full_gpu_same_harness", "valid": c["valid"], "n": c["n"],
               "host_issue_median_ms": round(c["host_issue_med"], 4), "block": key}
        ratios.append(row)
        if m1.get("A_ms") and m1.get("B_hoist_ms"):
            mx1 = max(m1["A_ms"], m1["B_hoist_ms"])
            ratios.append({**row, "reference": "stock_harness", "A_iso_ms": m1["A_ms"],
                           "B_iso_ms": m1["B_hoist_ms"], "T_serial_ms": m1["full_stages_hoist_ms"],
                           "O": round(Tp / mx1, 4),
                           "phi": round((m1["full_stages_hoist_ms"] - Tp)
                                        / (m1["full_stages_hoist_ms"] - mx1), 4),
                           "s": round(c["expert"] / m1["B_hoist_ms"], 4),
                           "p": round(c["prefix"] / m1["A_ms"], 4),
                           "r": round(m1["A_ms"] / m1["B_hoist_ms"], 4)})
    for name, rows_ in (("latency_samples.csv", samples), ("host_issue.csv", host_rows),
                        ("corun_ratios.csv", ratios)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0]))
            w.writeheader(); w.writerows(rows_)
    for r in ratios:
        print(f"  {r['block']:22s} ref={r['reference']:28s} O={r['O']}  phi={r['phi']}  "
              f"p={r['p']}  s={r['s']}  r={r['r']}  T_pipe={r['T_pipe_ms']}", flush=True)

    thr = C.throttle_record(g0)
    C.write_meta(out / "meta.json", {
        "experiment": "E12b M2 -- compiled co-run, whole GPU (T015)",
        "bench_commit": C.bench_commit(), "model_commit": C.CKPT_COMMIT,
        "checkpoint": spec.checkpoint, "checkpoint_commit": C.CKPT_COMMIT,
        "rung": "bf16_both", "rung_note": rung_note, "variant": h.label().replace(
            "prefix_first", "<order>"), "power_mode": g0["power_mode"],
        "sm_count": g0["sm_count"], "granted_partitions": "none (full GPU, 16 SMs)",
        "gpu_clock_hz": g0["gpu_cur_freq_hz"], "emc_rate_hz": g0["emc_rate_hz"],
        "gate_start": g0, **thr, "N": N, "batch": 1, "frames_per_condition": args.frames,
        "equivalence_pass": ok, "bracket_drift_pct": drift,
        "peak_memory_allocated_GiB": round(peak_mem, 3),
        "frames_manifest": manifest, "decisions": {
            "pipeline": "free-running, gated only by CUDA events (a per-frame barrier would "
                        "serialise the stages)",
            "kv_handoff": "dbuf_kv: KV + pad copied on the prefix stream into expert buffer k%2; "
                          "prefix(k) waits for expert(k-2)",
            "first_5_frames_dropped": "per-stage medians skip frames 0-4 of each run (pipeline "
                                      "fill); T_pipe is wall / frames over the whole run",
            "orders": "prefix_first is the headline; expert_first measured alongside"},
        "command": " ".join([sys.executable, *sys.argv]), "torch": torch.__version__,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
