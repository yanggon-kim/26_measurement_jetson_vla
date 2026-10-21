"""E12b M3 -- both stages TOGETHER inside green-context partitions, and the bandwidth-hog control.

Configuration graphs_both (both stages this harness's own graphs), because M0 measured that a
graph captured on the primary context escapes a green partition -- the rung's embed_image tree
would. Every graph here is captured INSIDE its own green context.

  0. full GPU, same harness and graphs: A_iso, B_iso, T_serial, T_pipe (the reference every
     split's O is divided by), with its own equivalence check;
  per split (prefix SMs / expert SMs, what the driver GRANTED is what is reported):
  1. equivalence: pipelined vs serial per frame + stale control, in this split's graphs;
  2. A_part (prefix alone in its partition), B_part (expert alone in its), serial, pipeline
     -> A_co,part, B_co,part, T_pipe,part, p_part, s_part, O, phi;
  3. p_hog: prefix alone in its partition while an in-process copy hog replays on the EXPERT
     partition's stream (same process -- two contexts without MPS time-slice), at >= 2 rates:
     `max` (copies back to back) and `matched` (copies interleaved with torch.cuda._sleep so the
     hog alone moves roughly the fair expert's average demand). The rate each achieves ALONE in
     its partition is recorded.
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
HOG_MB = 128                      # per copy; 2 x 128 MB buffers, well above the 4 MB L2


def med(v):
    return statistics.median(v) if v else float("nan")


class Hog:
    """A device-copy graph captured INSIDE a green context (so it is confined; M0)."""

    def __init__(self, gctx, sleep_cycles: int, copies: int = 4):
        self.g = gctx
        nel = HOG_MB * 2**20 // 2
        with gctx.activate():
            self.a = torch.empty(nel, dtype=torch.bfloat16, device="cuda").normal_()
            self.b = torch.empty_like(self.a)
            for _ in range(2):
                self._body(sleep_cycles, copies)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=torch.cuda.current_stream()):
                self._body(sleep_cycles, copies)
        self.bytes = 2 * HOG_MB * 2**20 * copies       # read + write per copy
        self.sleep, self.copies = sleep_cycles, copies

    def _body(self, sleep, copies):
        for _ in range(copies):
            self.b.copy_(self.a)
            if sleep:
                torch.cuda._sleep(sleep)

    def alone(self, reps=20):
        ts = []
        with self.g.activate() as s:
            for _ in range(3):
                self.graph.replay()
            torch.cuda.synchronize()
            for _ in range(reps):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(s); self.graph.replay(); e1.record(s)
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1))
        ms = med(ts)
        return ms, self.bytes / (ms / 1e3) / 1e9


def hog_prefix(h, hog, hog_ms, F, fin):
    """prefix alone in its partition while the hog covers every prefix frame."""
    ev = h._events(F)
    reps = max(2, int(h._A_part / max(hog_ms, 1e-3)) + 2)
    torch.cuda.synchronize()
    for k in range(F):
        with torch.no_grad(), h.ge.switch():
            for _ in range(reps):
                hog.graph.replay()
        h.issue_prefix(k, [None] * F, ev["pd"], ev["p0"], ev["p1"])
    torch.cuda.synchronize()
    return [ev["p0"][k].elapsed_time(ev["p1"][k]) for k in range(F)], reps


def run_config(h, fin, frames_n, tag, out_rows, samples, eq_all, ref=None, hog_rates=None):
    eq_rows, ok, scale = M2.equivalence(h, fin)
    eq_all += eq_rows
    split = "full_gpu" if not h.split else f"{h.granted[0]}/{h.granted[1]}"
    print(f"  [{split}] equivalence {'PASS' if ok else 'FAIL'} "
          f"(stale detected {sum(r['verdict']=='DETECTED' for r in eq_rows)}/"
          f"{sum(r['check']=='stale_control' and r['frame']>0 for r in eq_rows)}, "
          f"scale {scale:.3f})", flush=True)
    res = {}
    with torch.no_grad():
        for cond in ("prefix_alone", "expert_alone", "serial", "pipeline", "prefix_alone",
                     "expert_alone"):
            fin0 = fin.read()
            h.run(cond, 8, fin)
            r = h.run(cond, frames_n, fin)
            nf = fin.read() - fin0
            pre = [x.get("prefix_ms") for x in r["records"][5:] if x.get("prefix_ms") is not None]
            exp = [x.get("expert_ms") for x in r["records"][5:] if x.get("expert_ms") is not None]
            frm = [x.get("frame_ms") for x in r["records"][5:] if x.get("frame_ms") is not None]
            res.setdefault(cond, []).append({"prefix": med(pre), "expert": med(exp),
                                             "frame": med(frm), "period": r["period_ms"],
                                             "host": med(r["host_issue_ms"]), "nf": nf})
            cn = {"prefix_alone": "alone_part" if h.split else "isolated",
                  "expert_alone": "alone_part" if h.split else "isolated",
                  "serial": "serial", "pipeline": "corun_part" if h.split else "corun"}[cond]
            for x in r["records"]:
                for st in ("prefix", "expert", "frame"):
                    v = x.get(f"{st}_ms")
                    if v is not None:
                        samples.append({"variant": h.label(), "split": split, "condition": cn,
                                        "stage": st, "frame": x["frame"], "ms": round(v, 4),
                                        "valid": ok and nf == 0})
            print(f"  [{split}] {cond:13s} prefix {med(pre):8.2f} expert {med(exp):8.2f} "
                  f"frame {med(frm):8.2f} period {r['period_ms']:8.2f} host {med(r['host_issue_ms']):6.2f}"
                  f" nonfinite {nf}", flush=True)
    A = statistics.fmean(x["prefix"] for x in res["prefix_alone"])
    B = statistics.fmean(x["expert"] for x in res["expert_alone"])
    Ts = res["serial"][0]["frame"]
    pl = res["pipeline"][0]
    valid = ok and all(x["nf"] == 0 for v in res.values() for x in v)
    row = {"mechanism": "green_contexts_corun" if h.split else "compiled_graphs_both",
           "variant": h.label(), "split": split,
           "prefix_sms": h.granted[0], "expert_sms": h.granted[1],
           "N": N, "rung": "bf16_both", "valid": valid,
           "A_co_ms": round(pl["prefix"], 4), "B_co_ms": round(pl["expert"], 4),
           "T_pipe_ms": round(pl["period"], 4), "T_serial_ms_this_config": round(Ts, 4),
           "host_issue_median_ms": round(pl["host"], 4),
           "bracket_drift_prefix_pct": round(abs(res["prefix_alone"][1]["prefix"] - res["prefix_alone"][0]["prefix"]) / res["prefix_alone"][0]["prefix"] * 100, 3),
           "bracket_drift_expert_pct": round(abs(res["expert_alone"][1]["expert"] - res["expert_alone"][0]["expert"]) / res["expert_alone"][0]["expert"] * 100, 3)}
    if ref is None:        # this IS the full-GPU reference
        mx = max(A, B)
        row.update({"A_iso_ms": round(A, 4), "B_iso_ms": round(B, 4), "T_serial_ms": round(Ts, 4),
                    "s": round(pl["expert"] / B, 4), "p": round(pl["prefix"] / A, 4),
                    "r": round(A / B, 4), "O": round(pl["period"] / mx, 4),
                    "phi": round((Ts - pl["period"]) / (Ts - mx), 4),
                    "A_part_ms": "", "B_part_ms": "", "p_part": "", "s_part": "", "p_hog": "",
                    "reference": "fair_full_gpu_same_harness"})
    else:
        mx = max(ref["A"], ref["B"])
        row.update({"A_iso_ms": round(ref["A"], 4), "B_iso_ms": round(ref["B"], 4),
                    "T_serial_ms": round(ref["T"], 4),
                    "A_part_ms": round(A, 4), "B_part_ms": round(B, 4),
                    "p_part": round(pl["prefix"] / A, 4), "s_part": round(pl["expert"] / B, 4),
                    "s": round(pl["expert"] / ref["B"], 4), "p": round(pl["prefix"] / ref["A"], 4),
                    "r": round(ref["A"] / ref["B"], 4), "O": round(pl["period"] / mx, 4),
                    "phi": round((ref["T"] - pl["period"]) / (ref["T"] - mx), 4), "p_hog": "",
                    "reference": "fair_full_gpu_same_harness"})
    h._A_part = A
    out_rows.append(row)
    print(f"  [{split}] => O={row['O']} phi={row['phi']} p={row['p']} s={row['s']}"
          + (f" p_part={row['p_part']} s_part={row['s_part']}" if ref else ""), flush=True)
    # hog control, split runs only
    if h.split and hog_rates:
        for name, sleep in hog_rates:
            hog = Hog(h.ge, sleep)
            hms, gbs = hog.alone()
            pre, reps = hog_prefix(h, hog, hms, frames_n, fin)
            ph = med(pre[5:]) / A
            out_rows.append({**row, "mechanism": "bandwidth_hog_control",
                             "variant": h.label().replace(h.order, f"hog_{name}"),
                             "p_hog": round(ph, 4), "hog_rate_alone_GBps": round(gbs, 2),
                             "hog_graph_ms": round(hms, 3), "hog_replays_per_frame": reps,
                             "A_co_ms": round(med(pre[5:]), 4), "B_co_ms": "", "T_pipe_ms": "",
                             "p_part": "", "s_part": "", "O": "", "phi": ""})
            for i, v in enumerate(pre):
                samples.append({"variant": h.label().replace(h.order, f"hog_{name}"),
                                "split": split, "condition": "hog_part", "stage": "prefix",
                                "frame": i, "ms": round(v, 4), "valid": True})
            print(f"  [{split}] hog {name:8s} alone {gbs:7.1f} GB/s in {h.granted[1]} SMs -> "
                  f"prefix {med(pre[5:]):8.2f} ms, p_hog = {ph:.4f}", flush=True)
            del hog
            torch.cuda.synchronize()
    return {"A": A, "B": B, "T": Ts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--splits", default="12/4,8/8,10/6,14/2")
    ap.add_argument("--matched-sleep", type=int, default=0,
                    help="torch.cuda._sleep cycles between hog copies for the `matched` rate; "
                         "0 = calibrate to ~57 GB/s (the fair expert's average demand)")
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
    rung_note["embed_image"] = H.recompile_default(policy, spec, "embed_image")
    rung_note["memo"] = H.memo_681(policy)
    frames, manifest = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    fin = C.Finite()
    rows, samples, eq_all, granted = [], [], [], {}

    h0 = H.Harness(policy, spec, frames, N, config="graphs_both", order="prefix_first")
    ref = run_config(h0, fin, args.frames, "full", rows, samples, eq_all)
    ref_B_bytes = 638046208                        # E6 matched-rung bf16_both ae_step bytes
    target = ref_B_bytes / (ref["B"] / N / 1e3) / 1e9
    del h0
    torch.cuda.synchronize()

    for sp in args.splits.split(","):
        a, b = (int(x) for x in sp.split("/"))
        try:
            h = H.Harness(policy, spec, frames, N, config="graphs_both", split=(a, b),
                          order="prefix_first")
        except Exception as e:        # noqa: BLE001
            print(f"  [{sp}] harness FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append({"mechanism": "green_contexts_corun", "split": sp, "valid": False,
                         "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        granted[sp] = f"{h.granted[0]}/{h.granted[1]}"
        # matched hog rate: calibrate the sleep so the hog ALONE in this expert partition moves
        # about the fair expert's average DRAM demand
        sleep = args.matched_sleep
        if not sleep:
            lo, hi = 0, 2_000_000
            for _ in range(10):
                mid = (lo + hi) // 2
                hg = Hog(h.ge, mid)
                _, gbs = hg.alone(reps=5)
                del hg
                if gbs > target:
                    lo = mid
                else:
                    hi = mid
            sleep = hi
        run_config(h, fin, args.frames, sp, rows, samples, eq_all, ref=ref,
                   hog_rates=[("max", 0), ("matched", sleep)])
        del h
        torch.cuda.synchronize()

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    for name, rr, kk in (("corun_ratios.csv", rows, keys), ("latency_samples.csv", samples, None),
                         ("equivalence.csv", eq_all, None)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=kk or list(rr[0]))
            w.writeheader(); w.writerows(rr)
    eq_ok = {}
    for r in eq_all:
        eq_ok.setdefault(r["split"], True)
        if r["check"] == "pipeline" and r["verdict"] != "PASS":
            eq_ok[r["split"]] = False
        if r["check"] == "stale_control" and r["verdict"] == "MISSED":
            eq_ok[r["split"]] = False
    (out / "equivalence.md").write_text("\n".join(
        ["# E12b M3 — pipeline equivalence per split (graphs_both, captured in-partition)", "",
         "Pipelined vs serial per frame through the timing harness; stale control = the expert "
         "reads the other KV buffer (another frame's cache).", ""]
        + [f"* **{s}**: {'PASS (bitwise, stale control detected)' if v else 'FAIL — rows invalid'}"
           for s, v in eq_ok.items()] + ["", "Rows: `equivalence.csv`."]) + "\n")
    thr = C.throttle_record(g0)
    C.write_meta(out / "meta.json", {
        "experiment": "E12b M3 -- co-run inside green-context partitions + hog control (T015)",
        "bench_commit": C.bench_commit(), "model_commit": C.CKPT_COMMIT,
        "checkpoint": spec.checkpoint, "checkpoint_commit": C.CKPT_COMMIT,
        "rung": "bf16_both", "rung_note": rung_note,
        "variant": "bf16_both+hoist+owngraph_expert+dbuf_kv+owngraph_prefix+memo_681+green_<p>_<e>+prefix_first (per row)",
        "power_mode": g0["power_mode"], "sm_count": g0["sm_count"],
        "granted_partitions": granted or "none granted", "gpu_clock_hz": g0["gpu_cur_freq_hz"],
        "emc_rate_hz": g0["emc_rate_hz"], "gate_start": g0, **thr, "N": N,
        "frames_per_condition": args.frames, "equivalence_by_split": eq_ok,
        "hog": {"copy_MB": HOG_MB, "copies_per_graph": 4,
                "matched_target_GBps": round(target, 2),
                "matched_target_basis": "E6 matched-rung bf16_both ae_step bytes 638046208 / this "
                                        "session's full-GPU B_iso per step",
                "placement": "same process, captured and replayed inside the EXPERT partition's "
                             "green context, concurrently with the prefix alone in its partition"},
        "frames_manifest": manifest,
        "command": " ".join([sys.executable, *sys.argv]), "torch": torch.__version__,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
