"""T003 step 5: the complete hw_constants.json (protocol §8) from T002's static half + E1 + the counter validation.

Ceiling rules (STATUS.md question 2026-10-22, conservative reading until answered):
  * bandwidth: per kernel and size, each repeat's median GB/s; `best` = the highest repeat median over kernels, sizes
    and processes among SM kernels and the copy engine (both listed); spread_ratio = max / min of the repeat medians of
    that best (kernel, size) across all repeats of all processes.
  * compute: per dtype, the highest BURST TFLOP/s (median of the calls before the row's first soctherm OC event) over
    n and processes is the ceiling `value`; the matching sustained median (calls under OC throttling) and the all-call
    median are listed beside it. Rows that failed the correctness check are excluded.
  * ridge = ceiling TFLOP/s ÷ best GB/s, per dtype.
Every value carries the CSV row it came from.

python bench/hw_constants_build.py --e1 results/<E1> --t002 results/<E0_platform>/hw_constants.json
"""
import argparse
import csv
import glob
import json
import os

SPEC_URL = "https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/ (accessed 2026-10-22)"


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--t002", required=True)
    ap.add_argument("--b2b", default="", help="directory of back-to-back probes (bench/oc_throttle_probe.py --operands distinct)")
    a = ap.parse_args()
    hw = json.load(open(a.t002))
    bw_rows, fl_rows = [], []
    for p in sorted(glob.glob(os.path.join(a.e1, "latency_summary_bw_*.csv"))):
        bw_rows += [dict(r, _file=os.path.relpath(p)) for r in csv.DictReader(open(p))]
    for p in sorted(glob.glob(os.path.join(a.e1, "latency_summary_gemm_*.csv"))):
        fl_rows += [dict(r, _file=os.path.relpath(p)) for r in csv.DictReader(open(p))]

    # ---- bandwidth
    by_kernel = {}
    for r in bw_rows:
        if f(r["GBps"]) is None:
            continue
        by_kernel.setdefault(r["kernel"], []).append(r)
    kern_best = {}
    for k, rs in by_kernel.items():
        b = max(rs, key=lambda r: f(r["GBps"]))
        same = [f(r["GBps"]) for r in rs if r["shape"] == b["shape"]]
        kern_best[k] = {"value": f(b["GBps"]), "shape": b["shape"], "engine": b["engine"], "process": b["proc"],
                        "row": f"{b['_file']}: kernel={k} shape={b['shape']} proc={b['proc']} {b['note'][:20]}",
                        "repeat_medians_at_that_size": {"min": min(same), "max": max(same), "n": len(same)},
                        "spread_ratio": round(max(same) / min(same), 4),
                        "kernel_names": b["kernel_names"]}
    best_k = max(kern_best, key=lambda k: kern_best[k]["value"])
    sm_best_k = max((k for k in kern_best if kern_best[k]["engine"] == "SM kernel"), key=lambda k: kern_best[k]["value"])
    procs = sorted({r["proc"] for r in bw_rows})
    per_proc = {p: max(f(r["GBps"]) for r in bw_rows if r["proc"] == p and r["kernel"] == best_k and
                       r["shape"] == kern_best[best_k]["shape"]) for p in procs}
    hw["measured_bw_GBps"] = {
        "copy": kern_best.get("copy_sm_mul1"), "copy_engine_d2d": kern_best.get("copy_d2d"),
        "axpy": kern_best.get("axpy_inplace"), "read": kern_best.get("read_sum"),
        "add_out_of_place": kern_best.get("add_out_of_place"), "write_fill": kern_best.get("fill_write"),
        "best": {"value": kern_best[best_k]["value"], "kernel": best_k, **{k: v for k, v in kern_best[best_k].items() if k != "value"}},
        "best_sm_kernel": {"value": kern_best[sm_best_k]["value"], "kernel": sm_best_k},
        "spread_ratio": kern_best[best_k]["spread_ratio"],
        "per_process_best_of_best_kernel": per_proc,
        "between_process_ratio": round(max(per_proc.values()) / min(per_proc.values()), 4),
        "median_over_all_repeats_of_best_kernel": round(sorted(f(r["GBps"]) for r in bw_rows if r["kernel"] == best_k)[
            len([r for r in bw_rows if r["kernel"] == best_k]) // 2], 2),
        "selection_note": "best = the highest of the per-repeat medians (3 sizes x 5 repeats x 2 processes) -- an optimistic "
                          "estimator by construction (a peak); the median over all repeats of the same kernel is listed",
        "n_processes": len(procs),
        "definition": "GB/s = bytes read + written ÷ median per-call time of one repeat (100 calls); best = highest repeat median"}

    # ---- compute
    tf = {}
    for r in fl_rows:
        if r.get("correct") != "True" or f(r.get("burst_TFLOPps")) is None and f(r.get("TFLOPps")) is None:
            continue
        val = f(r["burst_TFLOPps"]) if f(r.get("burst_TFLOPps")) is not None else None
        if val is None:
            continue
        cur = tf.get(r["dtype"])
        if cur is None or val > cur["value"]:
            tf[r["dtype"]] = {"value": val, "shape": r["shape"], "process": r["proc"],
                              "burst_n_calls": int(r["burst_n"]), "sustained_TFLOPps": f(r["sustained_TFLOPps"]),
                              "sustained_n_calls": int(r["sustained_n"]) if r["sustained_n"] else 0,
                              "all_calls_median_TFLOPps": f(r["TFLOPps"]), "oc_event_delta": int(r["oc_event_delta"]),
                              "kernel": [k for k in r["kernel_names"].split(" | ") if "Memset" not in k],
                              "rel_err": r["rel_err"], "row": f"{r['_file']}: gemm {r['dtype']} {r['shape']} proc={r['proc']}"}
    # back-to-back probes: the same ceiling shapes issued without per-call synchronisation
    b2b = {}
    if a.b2b:
        import statistics as st
        for p in sorted(glob.glob(os.path.join(a.b2b, "backtoback_*.csv"))):
            name = os.path.basename(p)[len("backtoback_"):-4]
            dt, n = name.rsplit("_", 1)
            rows = list(csv.DictReader(open(p)))
            tfv = [float(r["tflops"]) for r in rows]
            first_oc = next((i for i, r in enumerate(rows) if int(r["oc3_delta"]) > 0), len(rows))
            b2b.setdefault(dt, {})[f"{n}^3"] = {
                "first_slice_TFLOPps": tfv[0], "median_TFLOPps": round(st.median(tfv), 3), "max_slice_TFLOPps": max(tfv),
                "max_slice_index": tfv.index(max(tfv)),
                "sustained_median_TFLOPps": round(st.median(tfv[first_oc:]), 3) if first_oc < len(rows) else None,
                "first_slice_with_oc_event": first_oc if first_oc < len(rows) else "none",
                "oc3_events": sum(int(r["oc3_delta"]) for r in rows), "slices": len(rows), "slice_s": 0.5,
                "gpu_C_last": float(rows[-1]["gpu_C"]), "source": os.path.relpath(p)}
    sustained_best = {}
    for r in fl_rows:
        v = f(r.get("sustained_TFLOPps"))
        if r.get("correct") == "True" and v is not None and (r["dtype"] not in sustained_best or v > sustained_best[r["dtype"]]["value"]):
            sustained_best[r["dtype"]] = {"value": v, "shape": r["shape"], "process": r["proc"],
                                          "row": f"{r['_file']}: gemm {r['dtype']} {r['shape']} proc={r['proc']}"}
    for dt, v in tf.items():
        v["back_to_back"] = b2b.get(dt, {}).get(v["shape"], "n/a: not probed back to back")
        v["isolated_calls_TFLOPps"] = v["value"]
        v["value_mode"] = "isolated calls (E1, synchronised per call, burst = before any OC event)"
        bb = v["back_to_back"]
        if isinstance(bb, dict) and bb["max_slice_TFLOPps"] > v["value"]:
            v["value"] = bb["max_slice_TFLOPps"]
            v["value_mode"] = ("back to back, the fastest 0.5 s slice (exceeds the isolated-call rate; tensor throughput "
                               "on this device is state-dependent, see value_mode_note)")
            v["row"] = bb["source"] + f": slice {bb['max_slice_index']}"
        v["value_mode_note"] = ("Tensor GEMM rates depend on a state the clocks do not show: int8 8192^3 back to back ran "
                                "258 TOP/s for 20 s, then 313-331 TOP/s with VDD_GPU rising 83 -> 99.7 W at the same "
                                "1575 MHz devfreq; bf16 / fp8 ran slower back to back than as isolated calls. ncu documents "
                                "a 'Tensor Core boost state' (--pipeline-boost-state); a boost state is the hypothesis, "
                                "not measured. The ceiling is the highest rate measured in either mode.")
        bb = v["back_to_back"]
        v["sustained_TFLOPps"] = (bb.get("sustained_median_TFLOPps") or bb.get("median_TFLOPps")) if isinstance(bb, dict) else None
        v["sustained_definition"] = ("back-to-back A @ B at the ceiling shape for 30 s (bench/oc_throttle_probe.py "
                                     "--operands distinct): median of the slices from the first soctherm OC event on, or "
                                     "of all slices if none occurred")
    hw["measured_tflops"] = {
        "fp32": tf.get("fp32"), "tf32": tf.get("tf32"), "bf16": tf.get("bf16"), "fp16": tf.get("fp16"),
        "int8": tf.get("int8"), "fp8_record_only": tf.get("fp8_e4m3"),
        "sustained_best_per_dtype": sustained_best,
        "definition": "value = the highest rate measured in either mode at the dtype's best shape: isolated calls (E1, "
                      "median of the calls before the row's first soctherm OC event, after a cool-down) or back to back "
                      "(the fastest 0.5 s slice); sustained_TFLOPps = back-to-back median from the first "
                      "OC event on (≈ 100 W VDD_GPU limit). Both modes are listed per dtype."}
    bw = hw["measured_bw_GBps"]["best"]["value"]
    hw["ridge_flop_per_byte"] = {k: round(v["value"] * 1e12 / (bw * 1e9), 2)
                                 for k, v in (("fp32", tf.get("fp32")), ("tf32", tf.get("tf32")),
                                              ("bf16", tf.get("bf16")), ("fp16", tf.get("fp16")),
                                              ("int8", tf.get("int8")), ("fp8_record_only", tf.get("fp8_e4m3"))) if v}
    hw["ridge_flop_per_byte"]["definition"] = "measured ceiling TFLOP/s ÷ measured best GB/s"
    hw["spec_tflops"] = {"fp4_sparse": 2070, "fp4_sparse_note": "\"2070 TFLOPS (FP4—Sparse)\"; NVIDIA publishes no dense or "
                         "per-dtype tensor figure for Thor", "fp32_simt_derived": round(2560 * 2 * 1.575e9 / 1e12, 3),
                         "fp32_simt_derived_note": "derived, not published: 2,560 CUDA cores (NVIDIA spec page: "
                                                   "'2560-core NVIDIA Blackwell architecture GPU') × 2 FLOP/FMA × 1.575 GHz "
                                                   "(locked clock, results/20261022_E0_platform_MAXN/clock_probe.csv); 20 SMs × "
                                                   "128 FP32 lanes = 2,560 agrees with the device's SM count",
                         "spec_source": SPEC_URL}
    hw["memory"]["spec_GBps"] = 273
    hw["memory"]["spec_source"] = SPEC_URL + ": \"128 GB 256-bit LPDDR5X\", \"273 GB/s\""
    hw["memory"]["type"] = "LPDDR5X (NVIDIA spec)"
    hw["memory"]["bus_width_bits"] = "256 (NVIDIA spec; ncu device__attribute_global_memory_bus_width reads 0)"
    val = os.path.join(a.e1, "validation.csv")
    if os.path.exists(val):
        rows = list(csv.DictReader(open(val)))
        streaming = [r for r in rows if r["case"] in ("copy_sm", "axpy", "read_sum", "gemv") and r["launch"] == "0"]
        rd_fill = [f(r["ratio_read_fill"]) for r in streaming]
        rd_miss = [f(r["ratio_read_miss"]) for r in streaming]
        wr = [f(r["ratio_write"]) for r in streaming if r["case"] in ("copy_sm", "axpy")]
        hw["counter_validation"] = {
            "dram_bytes_ratio_read": {"lts__d_sectors_fill_sysmem": [min(rd_fill), max(rd_fill)],
                                      "lts__t_sectors_aperture_sysmem_op_read_lookup_miss": [min(rd_miss), max(rd_miss)],
                                      "cases": [r["case"] for r in streaming]},
            "dram_bytes_ratio_write": {"lts__t_sectors_aperture_sysmem_op_write": [min(wr), max(wr)],
                                       "cases": ["copy_sm", "axpy"]},
            "flops_ratio_per_dtype": {r["dtype"]: {"tensor": f(r["ratio_tensor_flops"]), "simt": f(r["ratio_simt_flops"]),
                                                   "case": r["case"], "kernel": r["kernel"]}
                                      for r in rows if r["case"].startswith("gemm_")},
            "skinny_bf16_tensor_ratio": next(f(r["ratio_tensor_flops"]) for r in rows if r["case"] == "skinny_bf16"),
            "chosen": {"dram_read_bytes": "lts__d_sectors_fill_sysmem.sum x 32 B",
                       "dram_write_bytes": "lts__t_sectors_aperture_sysmem_op_write.sum x 32 B",
                       "tensor_flops": "sum of sm__ops_path_tensor_src_*.sum (counts executed, tile-padded MMA work: "
                                       "1.0078 at 4096^3 bf16 = 4128/4096 from 96-row tiles; 1.098 at M = 51 padded to 56)",
                       "simt_flops": "fadd + fmul + 2 x ffma (+ h*, d*) smsp__sass_thread_inst_executed_op_*_pred_on.sum"},
            "source_file": val}
    hw["source_runs"] = sorted(set(hw.get("source_runs", []) + [a.e1]))
    out = os.path.join(a.e1, "hw_constants.json")
    json.dump(hw, open(out, "w"), indent=1)
    print(json.dumps({"bw_best": hw["measured_bw_GBps"]["best"]["value"], "bw_kernel": best_k,
                      "tflops": {k: (v or {}).get("value") for k, v in hw["measured_tflops"].items() if isinstance(v, dict) and "value" in (v or {})},
                      "ridge": hw["ridge_flop_per_byte"]}, indent=1))


if __name__ == "__main__":
    main()
