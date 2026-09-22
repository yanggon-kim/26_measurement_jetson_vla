"""T005 / E1 — what this silicon can actually do: peak bandwidth and peak FLOP/s.

Every hardware constant in a profiling guide belongs to another GPU (protocol §8). This
measures the Orin's own, and fills the `measured` rows that T002/T003 deliberately left empty
in every `floor.csv`.

  1. bandwidth as the GPU sees it — device-to-device `clone()` at several sizes, and a
     bandwidth-bound elementwise kernel `y += a*x` on 1 GiB bf16;
  2. compute — square GEMM n x n for n in {1024, 2048, 4096, 8192} in bf16, fp16, fp32, tf32
     and int8, each checked for numerical correctness before its time is believed;
  3. `hw_constants.json` with the measured peaks, the ridge point, and the two **efficiency
     fractions** that docs/00_context.md currently carries as `[ASSUMPTION]` (0.4 of peak
     FLOP/s and 0.7 of peak bandwidth) — E11's analytic host model is built on those.

Every timed row records the GPU clock and die temperature on both sides, so `throttled` is
evidence rather than an assertion (protocol §1).

    python bench/micro_peak.py --outdir results/<date>_E1_peak
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
from common import (REPO, collect_raw, sh_out, software_versions,  # noqa: E402
                    summarize, write_json)

SPEC_GBPS = 204.8
SPEC_BF16_TFLOPS = 42.0

# docs/00_context.md gives `achieved_tflops = peak x eff_flops_frac` and
# `achieved_bw = ext_bw x eff_bw_frac` with both fractions marked **[ASSUMPTION]**.
# Replacing them with measurements is a main reason E1 exists.
ASSUMED_EFF_FLOPS_FRAC = 0.4
ASSUMED_EFF_BW_FRAC = 0.7

GPU_FREQ = "/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq"
THERMAL = Path("/sys/class/thermal")

FIELDS = ["kernel", "dtype", "shape", "bytes", "flops",
          "ms", "median_ms", "mean_ms", "p5_ms", "p95_ms", "min_ms", "max_ms",
          "GBps", "TFLOPps", "n", "correct", "rel_err",
          "gpu_hz_before", "gpu_hz_after", "gpu_C_before", "gpu_C_after", "tj_C_after",
          "note"]


def dev_sample() -> dict:
    """GPU clock and the readable thermal zones."""
    out = {"gpu_hz": int(Path(GPU_FREQ).read_text().strip())}
    for z in sorted(THERMAL.glob("thermal_zone*")):
        try:
            out[(z / "type").read_text().strip()] = int((z / "temp").read_text().strip()) / 1e3
        except Exception:
            # cv0/cv1/cv2 return EAGAIN while the CV engines are idle. The read surfaces as a
            # TypeError out of the codec layer rather than an OSError, so catch broadly.
            continue
    return out


def timed(fn, warmup: int = 20, n: int = 100):
    """Per-call GPU time via CUDA events (protocol §2), with device state on both sides."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    d0 = dev_sample()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(n)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in ev], d0, dev_sample()


ALL_SAMPLES: list[dict] = []          # every timed call, dumped to raw/ (protocol §3.5)


def row(kernel, dtype, shape, samples, d0, d1, *, nbytes=0, flops=0,
        correct="", rel_err="", note="") -> dict:
    for i, v in enumerate(samples):
        ALL_SAMPLES.append({"kernel": kernel, "dtype": dtype, "shape": shape,
                            "iter": i, "ms": v})
    s = summarize(samples)
    sec = s["median_ms"] / 1e3
    return {"kernel": kernel, "dtype": dtype, "shape": shape,
            "bytes": nbytes, "flops": flops,
            "ms": round(s["median_ms"], 6),
            "median_ms": round(s["median_ms"], 6), "mean_ms": round(s["mean_ms"], 6),
            "p5_ms": round(s["p5_ms"], 6), "p95_ms": round(s["p95_ms"], 6),
            "min_ms": round(s["min_ms"], 6), "max_ms": round(s["max_ms"], 6),
            "GBps": round(nbytes / sec / 1e9, 2) if nbytes else 0.0,
            "TFLOPps": round(flops / sec / 1e12, 4) if flops else 0.0,
            "n": s["n"], "correct": correct, "rel_err": rel_err,
            "gpu_hz_before": d0["gpu_hz"], "gpu_hz_after": d1["gpu_hz"],
            "gpu_C_before": d0.get("gpu-thermal"), "gpu_C_after": d1.get("gpu-thermal"),
            "tj_C_after": d1.get("tj-thermal"), "note": note}


def bandwidth_rows(repeats: int = 3) -> list[dict]:
    """Several access patterns, because "peak bandwidth" is whatever the best one achieves.

    A read-modify-write (`add_`) is not the same as a pure copy, and a copy that allocates its
    own output is not the same as one writing into a preallocated buffer. Measuring only one
    pattern would under-report the peak; measuring all of them says which to use as the
    constant and by how much the others fall short.
    """
    n = 1024 * 1024 * 1024 // 2                          # 1 GiB of bf16
    G = n * 2
    x = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
    y = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
    z = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    xf = torch.empty(n // 2, device="cuda", dtype=torch.float32).normal_()
    zf = torch.empty(n // 2, device="cuda", dtype=torch.float32)

    # (tag, callable, bytes moved at the DRAM pins, dtype, note)
    pats = [
        ("add_out_of_place", lambda: torch.add(x, y, out=z), 3 * G, "bfloat16",
         "z = x + y into a preallocated z; reads x and y, writes z. No allocation in the "
         "timed loop and no read-for-ownership on the output."),
        ("fill_", lambda: z.fill_(1.0), G, "bfloat16", "pure write, 1 GiB"),
        ("mul_scalar_out_of_place", lambda: torch.mul(x, 2.0, out=z), 2 * G, "bfloat16",
         "z = 2*x; one stream in, one out"),
        ("axpy_inplace", lambda: y.add_(x, alpha=1.0001), 3 * G, "bfloat16",
         "y += a*x; read-modify-write on y"),
        ("copy_out_of_place", lambda: z.copy_(x), 2 * G, "bfloat16",
         "z.copy_(x) into a preallocated z"),
        ("copy_fp32", lambda: zf.copy_(xf), 2 * G, "float32",
         "same bytes, 32-bit elements — checks whether element width matters"),
        ("mul_scalar_fp32", lambda: torch.mul(xf, 2.0, out=zf), 2 * G, "float32", ""),
        ("sum_reduction", lambda: x.sum(), G, "bfloat16",
         "pure read; a reduction, so partly limited by the reduction tree rather than DRAM"),
    ]
    rows = []
    for tag, fn, nb, dt, note in pats:
        # repeat the whole measurement so the row carries run-to-run spread, not just
        # within-run jitter — this device shows more of the former than the latter.
        best, allms = None, []
        for _ in range(repeats):
            ms, d0, d1 = timed(fn, 20, 100)
            allms += ms
            r = row(tag, dt, str(n), ms, d0, d1, nbytes=nb, note=note)
            if best is None or r["GBps"] > best["GBps"]:
                best = r
        spread = summarize(allms)
        best["note"] = (f"{note} | {repeats} repeats x {best['n']} calls; GB/s is the median "
                        f"of the BEST repeat (an optimistic estimator, because this is a peak). "
                        f"Pooled median {spread['median_ms']:.2f} ms, "
                        f"p5-p95 {spread['p5_ms']:.2f}-{spread['p95_ms']:.2f} ms.").strip(" |")
        rows.append(best)
    del x, y, z, xf, zf
    torch.cuda.empty_cache()

    # the allocating copy, kept because it shows what the caching allocator costs
    for mib in (256, 1024, 2048):
        n2 = mib * 1024 * 1024 // 2
        x = torch.empty(n2, device="cuda", dtype=torch.bfloat16).normal_()
        ms, d0, d1 = timed(lambda: x.clone(), 20, 100)
        rows.append(row("clone_allocating", "bfloat16", str(n2), ms, d0, d1,
                        nbytes=2 * n2 * 2,
                        note=f"{mib} MiB; each call allocates a fresh output, so the caching "
                             f"allocator is inside the timed region. Not a peak."))
        del x
        torch.cuda.empty_cache()
    return rows


def gemm_rows() -> list[dict]:
    """Square GEMMs. Each configuration is checked against an fp32 reference first: an
    unverified FLOP/s is just a number, and a silently-wrong kernel would look fast."""
    cfgs = [("bfloat16", torch.bfloat16, False), ("float16", torch.float16, False),
            ("float32", torch.float32, False), ("tf32", torch.float32, True),
            ("int8", torch.int8, False)]
    rows = []
    for n in (1024, 2048, 4096, 8192):
        for name, dt, tf32 in cfgs:
            prev = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = tf32
                if dt is torch.int8:
                    a = torch.randint(-8, 8, (n, n), device="cuda", dtype=torch.int8)
                    b = torch.randint(-8, 8, (n, n), device="cuda", dtype=torch.int8)
                    fn = lambda: torch._int_mm(a, b)  # noqa: E731
                    got, ref = fn().float(), (a.float() @ b.float())
                else:
                    a = torch.empty(n, n, device="cuda", dtype=dt).normal_()
                    b = torch.empty(n, n, device="cuda", dtype=dt).normal_()
                    fn = lambda: a @ b  # noqa: E731
                    got, ref = fn().float(), (a.float() @ b.float())
                rel = ((got - ref).norm() / ref.norm()).item()
                # bf16 carries ~3 decimal digits, fp16 ~3-4, tf32 ~3, fp32/int8 exact-ish
                tol = {"bfloat16": 1e-2, "float16": 1e-2, "tf32": 1e-2,
                       "float32": 1e-5, "int8": 0.0}[name]
                ok = rel <= tol
                del got, ref
                ms, d0, d1 = timed(fn, 20, 100)
                el = 1 if dt is torch.int8 else (4 if dt is torch.float32 else 2)
                note = ""
                if name == "int8":
                    note = ("torch._int_mm; a SOFTWARE-PATH number, not the hardware's dense "
                            "INT8 peak — see notes in meta.json")
                elif name == "tf32":
                    note = "fp32 tensors with allow_tf32=True (tensor cores, 10-bit mantissa)"
                elif name == "float32":
                    note = "true IEEE fp32, allow_tf32=False (CUDA cores)"
                rows.append(row("gemm_square", name, f"{n}x{n}x{n}", ms, d0, d1,
                                nbytes=3 * n * n * el, flops=2 * n ** 3,
                                correct=ok, rel_err=f"{rel:.3e}", note=note))
                del a, b
            except Exception as e:
                rows.append({**dict.fromkeys(FIELDS, ""), "kernel": "gemm_square",
                             "dtype": name, "shape": f"{n}x{n}x{n}", "n": 0,
                             "note": f"FAILED: {type(e).__name__}: {str(e)[:90]}"})
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev
                torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    gate = P.gate()
    print("gate:", json.dumps(gate))
    print("\n=== bandwidth ===", flush=True)
    rows = bandwidth_rows()
    for r in rows:
        print(f"  {r['kernel']:14s} {r['bytes']/1e9:6.2f} GB  {r['ms']:8.3f} ms  "
              f"{r['GBps']:7.1f} GB/s  ({100*r['GBps']/SPEC_GBPS:5.1f} % of spec)")

    print("\n=== compute ===", flush=True)
    g = gemm_rows()
    for r in g:
        if str(r["note"]).startswith("FAILED"):
            print(f"  {r['dtype']:9s} {r['shape']:18s} {r['note']}")
        else:
            flag = "" if r["correct"] is True else "  <-- NUMERICALLY WRONG, ignore"
            print(f"  {r['dtype']:9s} {r['shape']:18s} {r['ms']:9.3f} ms  "
                  f"{r['TFLOPps']:7.2f} TFLOP/s  rel={r['rel_err']}{flag}")
    rows += g

    with (out / "latency_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # --- throttle evidence: did the clock ever leave its pinned value?
    hz = {int(r[k]) for r in rows for k in ("gpu_hz_before", "gpu_hz_after") if r[k] != ""}
    throttled = hz != {gate["gpu_cur_freq_hz"]}
    tmax = max((r["gpu_C_after"] for r in rows if isinstance(r["gpu_C_after"], float)),
               default=None)

    ok = [r for r in rows if r["kernel"] == "gemm_square" and r["correct"] is True]
    peak_bw = max(r["GBps"] for r in rows if r["kernel"] != "gemm_square")
    bw_row = max((r for r in rows if r["kernel"] != "gemm_square"), key=lambda r: r["GBps"])
    best = {}
    for dt in ("bfloat16", "float16", "float32", "tf32", "int8"):
        c = [r for r in ok if r["dtype"] == dt]
        if c:
            best[dt] = max(c, key=lambda r: r["TFLOPps"])
    peak_bf16 = best.get("bfloat16", {}).get("TFLOPps", 0.0)
    ridge = (peak_bf16 * 1e12) / (peak_bw * 1e9) if peak_bw else 0.0
    eff_flops = peak_bf16 / SPEC_BF16_TFLOPS
    eff_bw = peak_bw / SPEC_GBPS

    print(f"\n  measured peak bandwidth : {peak_bw:7.1f} GB/s   "
          f"({100*eff_bw:.1f} % of the {SPEC_GBPS} spec)  [{bw_row['kernel']}]")
    print(f"  measured peak bf16      : {peak_bf16:7.2f} TFLOP/s "
          f"({100*eff_flops:.1f} % of the {SPEC_BF16_TFLOPS} spec)  "
          f"[{best.get('bfloat16', {}).get('shape')}]")
    print(f"  measured ridge point    : {ridge:7.1f} FLOP/B  (spec-derived ~205)")
    print(f"  GPU clock during the run: {sorted(hz)} Hz -> throttled={throttled}")
    print(f"  peak GPU die temperature: {tmax} C")
    print("\n  the two [ASSUMPTION] fractions in docs/00_context.md, now measured:")
    print(f"    eff_flops_frac  assumed {ASSUMED_EFF_FLOPS_FRAC:.2f}  measured "
          f"{eff_flops:.3f}   ({eff_flops/ASSUMED_EFF_FLOPS_FRAC:.2f}x the assumption)")
    print(f"    eff_bw_frac     assumed {ASSUMED_EFF_BW_FRAC:.2f}  measured "
          f"{eff_bw:.3f}   ({eff_bw/ASSUMED_EFF_BW_FRAC:.2f}x the assumption)")

    hw = json.loads((REPO / "results" / "20260921_E0_platform_maxn"
                     / "hw_constants.json").read_text())
    hw["measured_peaks"] = {
        "_status": f"measured by T005/E1 at MAXN, {time.strftime('%Y-%m-%d', time.gmtime())}",
        "_method": "CUDA-event timing at MAXN, jetson_clocks applied, GPU clock pinned at "
                   f"{gate['gpu_cur_freq_hz']} Hz. GEMM rows: median of 100 timed calls "
                   "after 20 warm-ups. Bandwidth rows: 3 independent repeats of "
                   "(20 warm-ups + 100 timed calls); the reported GB/s is the median of the "
                   "BEST repeat, i.e. a deliberately optimistic best-of-three estimator, "
                   "because this is a PEAK. The pooled median and p5-p95 across all three "
                   "repeats are in each row's `note`, and every individual call is in "
                   "raw/latency_samples.csv. Every GEMM was verified against an fp32 "
                   "reference before its time was used.",
        "_power_mode": gate["power_mode"], "_sm_count": gate["sm_count"],
        "peak_memory_bandwidth_GBps": peak_bw,
        "peak_memory_bandwidth_kernel": bw_row["kernel"],
        "peak_bf16_TFLOPps": peak_bf16,
        "peak_bf16_shape": best.get("bfloat16", {}).get("shape"),
        "peak_fp16_TFLOPps": best.get("float16", {}).get("TFLOPps"),
        "peak_fp32_TFLOPps": best.get("float32", {}).get("TFLOPps"),
        "peak_tf32_TFLOPps": best.get("tf32", {}).get("TFLOPps"),
        "peak_int8_TOPps_torch_int_mm": best.get("int8", {}).get("TFLOPps"),
        "peak_int8_CAVEAT": "torch._int_mm does not reach the tensor cores on sm_87; this is "
                            "a PyTorch-path throughput, NOT the hardware's dense INT8 peak. "
                            "Do not use it as a hardware constant.",
        "ridge_point_FLOP_per_byte": round(ridge, 2),
        "pct_of_spec_bandwidth": round(100 * eff_bw, 1),
        "pct_of_spec_bf16": round(100 * eff_flops, 1),
    }
    hw["efficiency_fractions"] = {
        "_purpose": "docs/00_context.md carries these as **[ASSUMPTION]** in E11's analytic "
                    "host model (`achieved_tflops = peak x eff_flops_frac`, "
                    "`achieved_bw_GBps = ext_bw x eff_bw_frac`). These are the measured "
                    "values. Both assumptions are pessimistic.",
        "eff_flops_frac_assumed": ASSUMED_EFF_FLOPS_FRAC,
        "eff_flops_frac_measured": round(eff_flops, 4),
        "eff_bw_frac_assumed": ASSUMED_EFF_BW_FRAC,
        "eff_bw_frac_measured": round(eff_bw, 4),
        "_caveat": "Measured on dense square GEMM and a streaming kernel — the best case. "
                   "The action expert's own shapes are small and skinny and do far worse; "
                   "E2 (micro_ae_kernels.py) measures what THOSE shapes achieve.",
    }
    write_json(out / "hw_constants.json", hw)

    with (out / "raw" / "latency_samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "dtype", "shape", "iter", "ms"])
        w.writeheader()
        w.writerows(ALL_SAMPLES)
    print(f"  wrote {len(ALL_SAMPLES)} per-iteration samples to raw/latency_samples.csv")

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E1", "task": "T005",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": None, "precision": "one row per dtype", "compile_mode": None,
        "batch": 1, "n_warmup": 20, "n_timed": 100,
        "n_repeats_bandwidth": 3,
        "estimator": "GEMM rows: median. Bandwidth rows: best-of-3 repeat medians (a peak).",
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "jetson_clocks": "applied before the run; GPU locked min==max",
        "emc_hz": int(sh_out(
            "sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
        "python": sw.get("python"), "venv": sw.get("executable"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "command": f"python bench/micro_peak.py --outdir {args.outdir}",
        "throttled": throttled,
        "throttled_evidence": f"GPU clock sampled before and after all {len(rows)} timed rows; "
                              f"observed values {sorted(hz)} Hz against the pinned "
                              f"{gate['gpu_cur_freq_hz']} Hz. Peak GPU die temperature "
                              f"{tmax} C. Per-row samples are columns gpu_hz_* / gpu_C_* / "
                              f"tj_C_after of latency_summary.csv.",
        "measured_peaks": hw["measured_peaks"],
        "efficiency_fractions": hw["efficiency_fractions"],
        "raw_files": collect_raw(out, "bench/micro_peak.py"),
        "notes": [
            "Fills the `measured` rows every floor.csv has left empty since T002.",
            "Bandwidth counts read+write for clone and read+read+write for axpy.",
            "Every GEMM was checked against an fp32 reference; `correct` and `rel_err` are "
            "per row, and only correct rows were considered for a peak.",
            "int8 via torch._int_mm reaches only ~10 TOP/s, roughly an eighth of the "
            "hardware's dense INT8 capability. The result is bit-exact, so this is a "
            "kernel-selection limit in PyTorch on sm_87, not a hardware limit. Recorded as a "
            "software-path number and explicitly NOT offered as a hardware constant.",
            "bf16 peaks at n=2048 and falls off at n=8192. That is not throttling: the GPU "
            "clock is sampled on both sides of every timed row and never left its pinned "
            "value. `bench/sustained_load_probe.py` holds the 8192^2 bf16 GEMM under load "
            "and records clock, temperature and throughput over time; its output is "
            "committed as raw/sustained_load.csv. The fall-off is cuBLAS kernel selection "
            "at large n.",
            "RUN-TO-RUN BANDWIDTH VARIATION, unexplained. Repeated measurements of the "
            "same kernel in fresh processes span 165.1-188.6 GB/s (22 samples, "
            "raw/bandwidth_modes.jsonl) -- a spread of 1.14x -- and three whole-E1 passes "
            "gave peaks of 161.8, 183.4 and 189.1 GB/s (raw/earlier_passes/). The "
            "distribution is continuous rather than two clean modes. Across every sample the "
            "GPU clock (1300.5 MHz), the EMC clock (3199 MHz, sampled DURING the timed loop) "
            "and the die temperature (45-47 C) were constant, as were the buffers' device "
            "pointers. Sustained load does not throttle it either (raw/sustained_load.csv). "
            "Five checks are backed by these committed files; several further checks "
            "(competing processes, accumulator value drift, buffer reuse vs reallocation, "
            "allocator pool size and ordering, re-applying jetson_clocks, a 4-minute idle) "
            "were run interactively and their output was NOT retained -- they are reasoning, "
            "not evidence. Re-check after the power-mode reboots.",
            "peak_memory_bandwidth_GBps is the best value observed in THIS run's own rows "
            "(max over latency_summary.csv), which is what `peak` should mean for a roofline. "
            "Consequence: a bandwidth-derived floor carries roughly +/-7 %, and a "
            "measured-vs-floor ratio should use a peak from the same session. The sensitivity "
            "runs in the safe direction for E2's conclusion: a LOWER peak raises the floor and "
            "lowers the measured multiple of it.",
        ],
    })
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
