"""Analytic co-run feasibility: can a VLA's prefix and action stage share one Orin, by resources alone?

Owner question (2026-09-28): "Let's say we have some amount of compute and some amount of bandwidth and capacity
requirements for action and prefix at the same time. Check whether they can be run on one Jetson." This answers it
with no new measurement: every demand and every supply below is read from a committed file, and every derived
quantity says how it was derived.

Model. Per frame, stage A = prefix (vision + VLM prefill) and stage B = the action stage (all N denoising steps).
  demand   F_X   FLOPs per frame            (E6 analytic, results/20260925_E6_rungs/<model>_<rung>/floor.csv)
           M_X   DRAM bytes per frame       analytic minimum (weights once; expert weights + KV once PER STEP) and,
                                            where measured, ncu graph-level bytes (E12b for π0, E14 for GR00T)
           C     capacity                   parameter bytes + KV (analytic) and the measured torch peak (E11d)
  supply   P     33.85 TFLOP/s bf16, BW 187.47 GB/s, 65.88 GB total (results/20260922_E1_peak/hw_constants.json)

Floors on the frame period T (each is a LOWER bound; FLOPs that E6 did not count only make it looser):
  serial roofline      T_ser ≥ t_A + t_B,  t_X = max(F_X / P, M_X / BW)
  co-run (fluid)       T_co  ≥ max((F_A + F_B) / P, (M_A + M_B) / BW)   — perfect sharing of both resources
  overlap headroom     T_ser / T_co: the most ANY co-run schedule can gain over serial, by resources alone
  rate feasibility     at a frame rate f: need (F_A+F_B)·f ≤ P, (M_A+M_B)·f ≤ BW, C ≤ memory — necessary, not sufficient
For π0 the measured co-run experiments add two tighter floors (fluid with the measured-bytes demand; SM space from the
measured per-partition stage times, E12b M3) and the eager host-issue floor (E5 kernels, E12 stage times).

    python3 bench/analytic_corun.py --out results/<date>_ANALYTIC_corun_feasibility
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
R = REPO / "results"
HW = json.loads((R / "20260922_E1_peak/hw_constants.json").read_text())
P_BF16 = HW["measured_peaks"]["peak_bf16_TFLOPps"] * 1e12          # FLOP/s
BW = HW["measured_peaks"]["peak_memory_bandwidth_GBps"] * 1e9      # B/s
MEM = HW["total_memory_bytes"]
BW_HOG = 178.5e9          # the most the bus delivered to a bandwidth hog (E12b M4, EMC-calibrated)

# model -> (E6 rung dir, N, E7v config label, E11d peak-memory dir)
MODELS = {
    "pi0": ("pi0_bf16_both", 10, "bf16_both", "*_E11d_pi0_reduce-overhead_prefixsweep"),
    "pi05": ("pi05_bf16_both_graphs", 10, "bf16_both_graphs", "*_E11d_pi05_reduce-overhead_prefixsweep"),
    "smolvla": ("smolvla_vlm_graphs", 10, "vlm_graphs", "*_E11d_smolvla_reduce-overhead_prefixsweep"),
    "groot_n17": ("groot_n17_vlm_compile", 4, "vlm_compile", "*_E11d_groot_n17_vlm_compile_prefixsweep"),
}


def newest(pattern: str) -> Path:
    c = sorted(p for p in R.glob(pattern) if p.name[:8].isdigit())
    if not c:
        raise SystemExit(f"no committed directory matches {pattern}")
    return c[-1]


def floor_rows(rung: str) -> dict:
    out = {}
    for r in csv.DictReader(open(R / "20260925_E6_rungs" / rung / "floor.csv")):
        if r["peak_source"] == "measured":
            out[r["region"]] = (float(r["bytes_read"] or 0), float(r["flops"] or 0), r["note"])
    return out


def demand(model: str) -> dict:
    rung, n, _, _ = MODELS[model]
    fr = floor_rows(rung)
    fi = json.loads((R / "20260925_E6_rungs" / rung / "floor_inputs.json").read_text())
    notes = []
    if model == "groot_n17":
        # E6 did not count GR00T FLOPs. Derived here with E6's own convention (2 x tokens x params, matrix work only):
        # language prefill over P tokens, DiT over S tokens per step, plus the 16 cross-attention blocks' to_k / to_v on
        # the P backbone tokens every step (results/20260928_E11c_groot_n17_handover/kv_reuse.md). Vision NOT counted.
        pb = fi["region_parameter_bytes"]
        P_tok, S_tok = 273, 41              # results/20260928_E11c_groot_n17_handover/handover.csv, boundary.csv
        f_lang = 2 * P_tok * pb["backbone_language"] / 2
        f_step = 2 * S_tok * pb["action_head_dit"] / 2 + 2 * P_tok * 16 * 2 * 2048 * 1536
        m_a = fr["vlm_backbone"][0]
        m_step = fr["ae_step"][0]
        f_a, f_b = f_lang, n * f_step
        notes.append("GR00T FLOPs derived here (E6 did not count them): 2·P·language params + N·(2·S·DiT params + "
                     "2·P·(16 cross blocks · to_k/to_v 2048→1536)); vision FLOPs not counted → compute floors are loose")
    else:
        vis_b, vis_f, _ = fr["vision"]
        vf = R / "20260925_E6_rungs" / rung / "vision_flops.json"
        if vis_f == 0 and vf.exists():
            vis_f = float(json.loads(vf.read_text())["flops_vision_region"])
        elif vis_f == 0:
            notes.append("vision FLOPs not counted (no vision_flops.json) → compute floors are loose")
        f_a = vis_f + fr["vlm_prefix"][1]
        m_a = vis_b + fr["vlm_prefix"][0]
        f_b = n * fr["ae_step"][1]
        m_step = fr["ae_step"][0]
    d = {"model": model, "rung": rung, "N": n, "F_A": f_a, "F_B": f_b, "M_A_analytic": m_a,
         "M_B_analytic": n * m_step, "notes": notes}
    # measured bytes (ncu, graph level), where they exist
    if model == "pi0":
        rows = {r["stage"]: r for r in csv.DictReader(open(R / "20260925_E12b_pi0_dram_bytes_MAXN/dram_bytes.csv"))}
        d["M_A_measured"] = float(rows["prefix_alone"]["measured_total_GB_per_frame"]) * 1e9
        d["M_B_measured"] = float(rows["expert_alone"]["measured_total_GB_per_frame"]) * 1e9
        d["measured_bytes_source"] = "results/20260925_E12b_pi0_dram_bytes_MAXN/dram_bytes.csv"
    elif model == "groot_n17":
        g = {}
        for r in csv.DictReader(open(R / "20260926_E14_groot_n17_savings_MAXN/bytes/dram_bytes.csv")):
            if r["variant"] == "ref":
                g[r["part"]] = (float(r["reads_GB"]) + float(r["writes_upper_GB"])) * 1e9
        d["M_A_measured"] = g["vision"] + g["vlm_prefix"] + g["other"]
        d["M_B_measured"] = g["ae_total"]
        d["measured_bytes_source"] = "results/20260926_E14_groot_n17_savings_MAXN/bytes/dram_bytes.csv (variant ref)"
    # capacity
    pb = fi.get("region_parameter_bytes", {})
    params = sum(v for k, v in pb.items() if "kv_cache" not in k and k != "backbone_language")
    d["capacity_params_B"] = params
    sw = newest(MODELS[model][3])
    d["capacity_peak_alloc_B"] = max(float(r["peak_mem_bytes"]) for r in csv.DictReader(open(sw / "prefix_sweep.csv"))
                                     if r.get("peak_mem_bytes"))
    d["capacity_source"] = f"{sw.name}/prefix_sweep.csv peak_mem_bytes (torch max allocated)"
    # measured stage times (E7v, MAXN, same configs)
    for r in csv.DictReader(open(R / "20260928_E7v_summary/e7v_split.csv")):
        if r["model"] == model and r["config"] == MODELS[model][2]:
            d["A_ms"] = float(r["vision_ms"]) + float(r["language_ms"])
            d["B_ms"] = float(r["action_ms"])
            d["full_ms"] = float(r["full_ms"])
    return d


def bounds(d: dict, which: str) -> dict:
    ma, mb = d[f"M_A_{which}"], d[f"M_B_{which}"]
    ta = max(d["F_A"] / P_BF16, ma / BW) * 1e3
    tb = max(d["F_B"] / P_BF16, mb / BW) * 1e3
    tc = max((d["F_A"] + d["F_B"]) / P_BF16, (ma + mb) / BW) * 1e3
    lim = "compute" if (d["F_A"] + d["F_B"]) / P_BF16 >= (ma + mb) / BW else "bandwidth"
    return {"bytes": which, "t_A_floor_ms": ta, "t_B_floor_ms": tb, "serial_floor_ms": ta + tb,
            "corun_floor_ms": tc, "corun_limited_by": lim, "overlap_headroom": (ta + tb) / tc,
            "compute_only_ms": (d["F_A"] + d["F_B"]) / P_BF16 * 1e3, "bw_only_ms": (ma + mb) / BW * 1e3}


def pi0_detail() -> list[dict]:
    """Tighter floors for π0, from the committed co-run measurements."""
    rows = []
    g = newest("*_E12b_pi0_green_corun_N10_MAXN")
    split, full = [], None
    for r in csv.DictReader(open(g / "corun_ratios.csv")):
        if not r["variant"].endswith("prefix_first") or not r["T_pipe_ms"]:
            continue
        if r["split"] == "full_gpu":
            full = r
        else:
            split.append(r)
    a, b = float(full["A_iso_ms"]), float(full["B_iso_ms"])
    bb = {x["stage"]: x for x in csv.DictReader(open(R / "20260925_E12b_pi0_dram_bytes_MAXN/dram_bytes.csv"))}
    mA = float(bb["prefix_alone"]["measured_total_GB_per_frame"]) * 1e9
    mB = float(bb["expert_alone"]["measured_total_GB_per_frame"]) * 1e9
    fluid = max(a, b, (mA + mB) / BW_HOG * 1e3)
    rows.append({"config": "compiled (graphs_both, fair expert)", "floor": "fluid, measured bytes at the hog's 178.5 GB/s",
                 "T_floor_ms": fluid, "O_floor": fluid / max(a, b),
                 "derivation": f"max(A={a:.1f}, B={b:.1f}, (M_A+M_B)/178.5 GB/s = {(mA + mB) / BW_HOG * 1e3:.1f}) ms"})
    best = min(split, key=lambda r: max(float(r["A_part_ms"]), float(r["B_part_ms"])))
    smf = max(float(best["A_part_ms"]), float(best["B_part_ms"]))
    rows.append({"config": "compiled (graphs_both, fair expert)", "floor": "SM space: best split, zero memory coupling",
                 "T_floor_ms": smf, "O_floor": smf / max(a, b),
                 "derivation": f"min over measured splits of max(A_part, B_part); best {best['split']} = "
                               f"max({float(best['A_part_ms']):.1f}, {float(best['B_part_ms']):.1f})"})
    meas = min(float(r["T_pipe_ms"]) for r in split + [full])
    rows.append({"config": "compiled (graphs_both, fair expert)", "floor": "MEASURED best co-run (T_pipe)",
                 "T_floor_ms": meas, "O_floor": meas / max(a, b), "derivation": f"{g.name}/corun_ratios.csv"})
    # eager: host issue (E5) and GPU busy time
    e5 = {r["region"]: r for r in csv.DictReader(open(newest("*_E5_pi0_eager_N10") / "timeline_summary.csv"))}
    gpu_a = float(e5["vision"]["kernel_union_ms"]) + float(e5["vlm_prefix"]["kernel_union_ms"])
    gpu_b = float(e5["ae_step"]["kernel_union_ms"])          # all N steps of one inference (instances_per_inference=10)
    eg = next(csv.DictReader(open(newest("*_E12_pi0_corun_N10_MAXN_eager_nosync") / "corun_ratios.csv")))
    ea, eb = float(eg["A_iso_ms"]), float(eg["B_iso_ms"])
    rows.append({"config": "eager", "floor": "one issuing thread (program order): T ≥ A + B", "T_floor_ms": ea + eb,
                 "O_floor": (ea + eb) / max(ea, eb),
                 "derivation": f"A={ea:.1f}, B={eb:.1f} (E12); the expert is host-bound (E5: {float(e5['ae_step']['gpu_idle_frac']) * 100:.0f} % GPU idle,"
                               f" {float(e5['ae_step']['kernels_per_unit']):.0f} kernels/step) so its issue occupies the thread for ≈ B"})
    two = max(ea, eb, gpu_a + gpu_b)
    rows.append({"config": "eager", "floor": "two independent issuers, GPU serialised: T ≥ max(A, B, GPU_A + GPU_B)",
                 "T_floor_ms": two, "O_floor": two / max(ea, eb),
                 "derivation": f"GPU busy per frame from E5 (nsys): prefix {gpu_a:.1f} + expert {gpu_b:.1f} ms; "
                               "prediction for a two-process / graph-captured-expert co-run (E12d measures it)"})
    rows.append({"config": "eager", "floor": "MEASURED in-process co-run (T_pipe)", "T_floor_ms": float(eg["T_pipe_ms"]),
                 "O_floor": float(eg["T_pipe_ms"]) / max(ea, eb), "derivation": "E12 corun_ratios.csv"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = REPO / ap.parse_args().out
    (out / "raw").mkdir(parents=True, exist_ok=True)
    dem, bnd, feas = [], [], []
    for m in MODELS:
        d = demand(m)
        dem.append({k: v for k, v in d.items() if k != "notes"} | {"notes": "; ".join(d["notes"])})
        for which in ("analytic", "measured"):
            if f"M_A_{which}" not in d:
                continue
            b = bounds(d, which)
            bnd.append({"model": m, **b, "A_meas_ms": d["A_ms"], "B_meas_ms": d["B_ms"],
                        "serial_meas_ms": d["A_ms"] + d["B_ms"], "ideal_overlap_meas_ms": max(d["A_ms"], d["B_ms"])})
            for hz in (2, 10, 20):
                feas.append({"model": m, "bytes": which, "rate_Hz": hz,
                             "compute_util_needed": (d["F_A"] + d["F_B"]) * hz / P_BF16,
                             "bw_util_needed": (d[f"M_A_{which}"] + d[f"M_B_{which}"]) * hz / BW,
                             "capacity_frac_params": d["capacity_params_B"] / MEM,
                             "capacity_frac_peak_alloc": d["capacity_peak_alloc_B"] / MEM,
                             "resources_allow": (d["F_A"] + d["F_B"]) * hz <= P_BF16
                             and (d[f"M_A_{which}"] + d[f"M_B_{which}"]) * hz <= BW,
                             "measured_serial_meets_rate": (d["A_ms"] + d["B_ms"]) <= 1e3 / hz,
                             "measured_ideal_overlap_meets_rate": max(d["A_ms"], d["B_ms"]) <= 1e3 / hz})
    det = pi0_detail()
    for name, rows in (("demand.csv", dem), ("bounds.csv", bnd), ("rate_feasibility.csv", feas), ("pi0_floors.csv", det)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    meta = {"experiment": "analytic co-run feasibility (owner request, 2026-09-28)", "kind": "derived only — no new measurement",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": "python3 bench/analytic_corun.py --out " + str(out.relative_to(REPO)),
            "supply": {"P_bf16_TFLOPps": P_BF16 / 1e12, "BW_GBps": BW / 1e9, "BW_hog_GBps": BW_HOG / 1e9,
                       "memory_bytes": MEM, "source": "results/20260922_E1_peak/hw_constants.json; E12b M4 (hog)"},
            "power_mode": "MAXN (every source run)", "models": MODELS,
            "raw_files": [], "notes": [__doc__.split("\n\n")[1]]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({"bounds": bnd, "pi0": det}, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
