"""T006 step 5-6: join ncu (form D) with nsys, per kernel and per part; roofline tables and plot for one model x mode.

Inputs:
  --nsys   results/<date>_E5_<m>_<mode>_nsys/  (kernels.csv: every kernel, region, nsys duration)
  --ncu    results/<date>_E6_<m>_<mode>/ncu/   (<part>_s<start>_raw.csv per window, T003-validated metrics)
  --analytic results/<date>_E6_dims/raw/<m>_<mode>/  (accounting.json: analytic FLOPs and bytes per region)
  --t005-row raw/rows/<rung>_N<N>.json of the same configuration in T005 (CUDA-event region medians)
Join (protocol §5): ncu rows of one part, in launch order, against the nsys kernels of the same part in inference_0,
in launch order. The kernel names must agree (checked per row, mismatches counted). Time per kernel comes from nsys,
never from ncu's gpu__time_duration (kept only as a cross-check column).
Per kernel: DRAM bytes = (lts__d_sectors_fill_sysmem + lts__t_sectors_aperture_sysmem_op_write) x 32; tensor FLOPs = Σ
sm__ops_path_tensor_src_*; SIMT FLOPs = add + mul + 2·fma over f / h / d; intensity; achieved rates; the ceiling of
the dtype the kernel computes in (hw_constants measured: tensor bf16 / fp16 / tf32, else SIMT fp32); bound by ridge.
Outputs in --out: kernel_roofline.csv, roofline.csv (per part: source ncu with nsys-sum time, source ncu with
CUDA-event time, source analytic), floor.csv, roofline.png, and coverage per part.
"""
import argparse
import csv
import glob
import json
import os
import re

TENSOR = {"bf16_dst_fp32": "bf16", "fp16_dst_fp16": "fp16", "fp16_dst_fp32": "fp16", "tf32_dst_fp32": "tf32", "int8": "int8",
          "fp64": "fp64", "fp4_dst_fp32": "fp8", "fp4_fp6_fp8_dst_fp16": "fp8", "fp4_fp6_fp8_dst_fp32": "fp8"}


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return 0.0


def read_ncu(path):
    rows = list(csv.reader(open(path)))
    if len(rows) < 3:
        return []
    head, units = rows[0], rows[1]
    out = []
    for r in rows[2:]:
        d = dict(zip(head, r))
        d["_units"] = dict(zip(head, units))
        out.append(d)
    return out


def ns(d, key):
    v, u = num(d.get(key, 0)), d["_units"].get(key, "ns")
    return v * {"ns": 1, "nsecond": 1, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6, "s": 1e9, "second": 1e9}.get(u, 1)


def norm(n):
    """The kernel's base function name. nsys and ncu demangle differently (nsys: `(int)128`, `(bool)0`,
    `at::native::<unnamed>::`, `cudnn::cnn::`; ncu: `128`, `0`, `at::<unnamed>::`, `cudnn::`), so the full strings of
    the same kernel differ; the identifier just before the template / argument list does not."""
    n = re.sub(r"^void\s+", "", (n or "").strip()).replace("<unnamed>", "").replace("(anonymous namespace)", "")
    head = re.split(r"[<(]", n, 1)[0]
    return head.split("::")[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--nsys", required=True)
    ap.add_argument("--ncu", required=True)
    ap.add_argument("--analytic", required=True)
    ap.add_argument("--hw", default="results/20261022_E1_peak/hw_constants.json")
    ap.add_argument("--t005-row", default="", help="raw/rows/<rung>_N<N>.json of the same configuration (CUDA-event times)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hw = json.load(open(a.hw))
    bw = hw["measured_bw_GBps"]["best"]["value"]
    tf = {k: v["value"] for k, v in hw["measured_tflops"].items() if isinstance(v, dict) and "value" in v}
    region_ms = {}
    if a.t005_row:
        row = json.load(open(a.t005_row))
        st = (row.get("by_N") or {}).get(str(row["N"]), {}).get("regions", {})
        for k in ("vision", "vlm_prefix", "ae_total", "backbone_lm_head", "vlm_backbone"):
            if k in st:
                region_ms[k] = st[k]["median_ms"]
        if "ae_step_individual[0]" in st:
            region_ms["ae_step[0]"] = st["ae_step_individual[0]"]["median_ms"]
    nk = [r for r in csv.DictReader(open(os.path.join(a.nsys, "kernels.csv"))) if r["inference"] == "inference_0"]
    parts = sorted({os.path.basename(p).rsplit("_s", 1)[0] for p in glob.glob(os.path.join(a.ncu, "*_raw.csv"))})
    krows, prow, cov = [], [], {}
    for part in parts:
        nreg = {"ae_step0": "ae_step[0]"}.get(part, part)
        ncu_rows = []
        for p in sorted(glob.glob(os.path.join(a.ncu, f"{part}_s*_raw.csv")),
                        key=lambda p: int(re.search(r"_s(\d+)_raw", p).group(1))):
            ncu_rows += read_ncu(p)
        ns_part = [r for r in nk if r["region"] == nreg]
        tot_ns = sum(int(r["duration_ns"]) for r in ns_part)
        joined_ns, mism = 0, 0
        for i, d in enumerate(ncu_rows):
            nsr = ns_part[i] if i < len(ns_part) else None
            name = d.get("Kernel Name", d.get("launch__kernel_name", ""))
            match = nsr is not None and norm(nsr["name"]) == norm(name)
            if nsr is not None and not match:
                mism += 1
            dur = int(nsr["duration_ns"]) if nsr is not None else None
            rd = num(d.get("lts__d_sectors_fill_sysmem.sum")) * 32
            wr = num(d.get("lts__t_sectors_aperture_sysmem_op_write.sum")) * 32
            tens = {TENSOR[k[len("sm__ops_path_tensor_src_"):-4]]: num(v) for k, v in d.items()
                    if k.startswith("sm__ops_path_tensor_src_") and k.endswith(".sum")
                    and k[len("sm__ops_path_tensor_src_"):-4] in TENSOR}
            tsum = sum(tens.values())
            simt = sum(num(d.get(f"smsp__sass_thread_inst_executed_op_{p}{o}_pred_on.sum", 0)) * (2 if o == "fma" else 1)
                       for p in "fhd" for o in ("add", "mul", "fma"))
            fl = tsum + simt
            dt = max(tens, key=tens.get) if tsum > 0 else "fp32_simt"
            peak = tf.get(dt if dt != "fp32_simt" else "fp32", tf.get("fp32"))
            t = (dur or ns(d, "gpu__time_duration.sum")) * 1e-9
            byt = rd + wr
            inten = fl / byt if byt else None
            ridge = peak * 1e12 / (bw * 1e9)
            if dur:
                joined_ns += dur
            krows.append({"model": a.model, "mode": a.mode, "part": nreg, "idx_in_part": i, "kernel": name[:200],
                          "nsys_name_match": match, "grid": nsr["grid"] if nsr else "", "block": nsr["block"] if nsr else "",
                          "nsys_duration_ns": dur if dur is not None else "", "ncu_duration_ns": round(ns(d, "gpu__time_duration.sum")),
                          "dram_read_B": int(rd), "dram_write_B": int(wr), "tensor_flops": int(tsum), "simt_flops": int(simt),
                          "flops": int(fl), "compute_dtype": dt, "intensity_flop_per_B": round(inten, 3) if inten else "",
                          "achieved_TFLOPps": round(fl / t / 1e12, 4) if t else "", "achieved_GBps": round(byt / t / 1e9, 3) if t else "",
                          "ceiling_TFLOPps": peak, "ceiling_GBps": bw, "ridge_flop_per_B": round(ridge, 2),
                          "pct_peak_compute": round(fl / t / 1e12 / peak * 100, 2) if t else "",
                          "pct_peak_bw": round(byt / t / 1e9 / bw * 100, 2) if t else "",
                          "bound_by_intensity": ("compute" if inten and inten >= ridge else "memory") if inten else "n/a (no DRAM bytes)",
                          "sm_pct": num(d.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")),
                          "mem_pct": num(d.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")),
                          "occupancy_pct": num(d.get("sm__warps_active.avg.pct_of_peak_sustained_active"))})
        cov[nreg] = {"nsys_kernels": len(ns_part), "ncu_kernels": len(ncu_rows), "name_mismatches": mism,
                     "nsys_kernel_ms": tot_ns / 1e6, "joined_kernel_ms": joined_ns / 1e6,
                     "coverage_pct": round(joined_ns / tot_ns * 100, 2) if tot_ns else 0.0}
        ks = [k for k in krows if k["part"] == nreg]
        F = sum(k["flops"] for k in ks)
        B = sum(k["dram_read_B"] + k["dram_write_B"] for k in ks)
        dts = {}
        for k in ks:
            dts[k["compute_dtype"]] = dts.get(k["compute_dtype"], 0) + k["flops"]
        dom = max(dts, key=dts.get) if dts else "bf16"
        peak = tf.get(dom if dom != "fp32_simt" else "fp32", tf.get("bf16"))
        for src, ms in (("ncu, time = nsys kernel sum", joined_ns / 1e6), ("ncu, time = CUDA-event region", region_ms.get(nreg))):
            if not ms:
                continue
            prow.append({"model": a.model, "mode": a.mode, "part": nreg, "source": src, "flops": F, "dram_bytes": B,
                         "intensity": round(F / B, 3) if B else "", "time_ms": round(ms, 4),
                         "achieved_TFLOPps": round(F / ms / 1e9, 4), "achieved_GBps": round(B / ms / 1e6, 3),
                         "dominant_dtype": dom, "ceiling_TFLOPps": peak, "ceiling_GBps": bw,
                         "pct_peak_compute": round(F / ms / 1e9 / peak * 100, 2), "pct_peak_bw": round(B / ms / 1e6 / bw * 100, 2),
                         "ridge": round(peak * 1e3 / bw, 2), "bound": ("compute" if B and F / B >= peak * 1e3 / bw else "memory"),
                         "coverage_pct": cov[nreg]["coverage_pct"]})
    acc = json.load(open(os.path.join(a.analytic, "accounting.json")))
    floors = []
    for part in sorted(cov):
        F = acc["flops"].get(part, 0)
        by = acc["param_bytes_by_dtype"].get(part, {})
        B = acc["param_bytes"].get(part, 0) + (sum(acc["kv_bytes"].values()) if part.startswith(("ae_step", "ae_total")) else 0)
        steps = sorted(k for k in acc["param_bytes"] if k.startswith("ae_step["))
        if part == "ae_total" and steps:
            # Fix 2026-10-25: param_bytes["ae_total"] counts each parameter once, but the expert's weights (1.1-2.2 GB) do
            # not fit in L2 and are streamed from DRAM at EVERY denoising step. The floor reads them once per step; the
            # parameters outside the steps (state / action encoders) once; the hand-over (VLM features) once per step.
            once = acc["param_bytes"]["ae_total"] - acc["param_bytes"][steps[0]]
            B = sum(acc["param_bytes"][s] for s in steps) + once + len(steps) * sum(acc["kv_bytes"].values())
        ncall = int(acc.get("calls", {}).get(part, 1) or 1)
        l2 = json.load(open(a.hw)).get("l2_bytes", 0)
        if part != "ae_total" and ncall > 1 and acc["param_bytes"].get(part, 0) > l2:
            # Fix 2026-10-26: a region entered several times per inference (π0 / π0.5 / SmolVLA: embed_image once per
            # camera) streams its weights from DRAM at every call when they do not fit in L2; the floor counts each call.
            B = acc["param_bytes"][part] * ncall + (sum(acc["kv_bytes"].values()) if part.startswith("ae_step") else 0)
        if part == "backbone_lm_head" and not B:
            # Fix 2026-10-26: the accounting hook ran the lm_head with vlm_backbone innermost on its stack, so its weight
            # bytes were never attributed to backbone_lm_head (its FLOPs were). Take them from the C3 dims row of the
            # lm_head module: weights + activation in + logits out.
            dp = os.path.join(a.analytic, "dims.csv")
            lm = [r for r in csv.DictReader(open(dp)) if r["module"].endswith("lm_head")] if os.path.exists(dp) else []
            B = sum(int(r["weight_bytes"]) + int(r["act_bytes"]) + int(r["out_bytes"]) for r in lm)
            by = {r["weight_dtype"]: int(r["weight_bytes"]) for r in lm}
        dom = max(by, key=by.get) if by else "torch.bfloat16"
        key = {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32"}.get(dom, "bf16")
        peak = tf[key]
        fm, fc = B / (bw * 1e9) * 1e3, F / (peak * 1e12) * 1e3
        ms = region_ms.get(part)
        prow.append({"model": a.model, "mode": a.mode, "part": part, "source": "analytic (parameters touched per step + hand-over; "
                     "FlopCounterMode)", "flops": F, "dram_bytes": B, "intensity": round(F / B, 3) if B else "",
                     "time_ms": ms or "", "achieved_TFLOPps": round(F / ms / 1e9, 4) if ms else "",
                     "achieved_GBps": round(B / ms / 1e6, 3) if ms else "", "dominant_dtype": key, "ceiling_TFLOPps": peak,
                     "ceiling_GBps": bw, "pct_peak_compute": round(F / ms / 1e9 / peak * 100, 2) if ms else "",
                     "pct_peak_bw": round(B / ms / 1e6 / bw * 100, 2) if ms else "", "ridge": round(peak * 1e3 / bw, 2),
                     "bound": "memory" if fm >= fc else "compute", "coverage_pct": ""})
        floors.append({"model": a.model, "mode": a.mode, "part": part, "flops": F, "bytes": B, "floor_mem_ms": round(fm, 4),
                       "floor_compute_ms": round(fc, 4), "floor_ms": round(max(fm, fc), 4),
                       "bound": "memory" if fm >= fc else "compute", "measured_ms": ms or "",
                       "gap_ratio": round(ms / max(fm, fc), 3) if ms and max(fm, fc) else "",
                       "ceilings": f"measured: {a.hw} ({key} {peak} TFLOP/s, {bw} GB/s)"})
    os.makedirs(a.out, exist_ok=True)
    for name, rows in (("kernel_roofline.csv", krows), ("roofline.csv", prow), ("floor.csv", floors)):
        if rows:
            keys = []
            for r in rows:
                keys += [k for k in r if k not in keys]
            with open(os.path.join(a.out, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
    json.dump(cov, open(os.path.join(a.out, "coverage.json"), "w"), indent=1)
    plot(a, krows, prow, tf, bw)
    print(json.dumps(cov, indent=1))


def plot(a, krows, prow, tf, bw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [10 ** (i / 20) for i in range(-40, 101)]
    for dt, ls in (("bf16", "-"), ("tf32", "--"), ("fp32", ":")):
        if dt in tf:
            ax.plot(xs, [min(tf[dt], x * bw / 1e3) for x in xs], ls, color="k", lw=1, label=f"{dt} roof {tf[dt]} TFLOP/s")
    col = {"vision": "tab:blue", "vlm_prefix": "tab:orange", "ae_step[0]": "tab:green", "ae_total": "tab:green",
           "backbone_lm_head": "tab:gray"}
    for k in krows:
        if k["intensity_flop_per_B"] and k["achieved_TFLOPps"]:
            ax.scatter(k["intensity_flop_per_B"], k["achieved_TFLOPps"], s=4, alpha=0.3, color=col.get(k["part"], "tab:purple"))
    for p in prow:
        if p["source"].startswith("ncu, time = CUDA") and p["intensity"]:
            ax.scatter(p["intensity"], p["achieved_TFLOPps"], s=80, marker="*", color=col.get(p["part"], "tab:purple"),
                       edgecolor="k", label=f"{p['part']} (part)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("intensity (FLOP / DRAM byte, ncu)")
    ax.set_ylabel("achieved TFLOP/s (nsys time)")
    ax.set_title(f"{a.model} {a.mode}: kernels (dots) and parts (stars), measured roofs")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(a.out, "roofline.png"), dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
