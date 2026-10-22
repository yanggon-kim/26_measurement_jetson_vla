"""T003 / E1: this Thor's peak bandwidth and peak FLOP/s (the roofline ceilings), per process.

Ported in spirit from reference/orin/bench/micro_peak.py, with three Thor-specific changes:
  * VDD_GPU is current-limited (≈ 100 W, T002): a sustained heavy kernel triggers soctherm oc3 throttling that devfreq
    does not show. Every timed call is followed by a read of the OC counters, so each row reports
      burst     = calls before the first OC event of the row (after a cool-down),
      sustained = calls after it,
      all       = every call,
    and the OC event delta. hw_constants takes the burst rate as the ceiling and lists the sustained one (STATUS.md).
  * bandwidth at ≥ 3 sizes ≥ 1 GiB, ≥ 5 repeats per process; run this script in ≥ 2 processes (--proc-tag).
  * GEMMs up to n = 16384, fp32 (`highest`), TF32, bf16, fp16, int8 (`torch._int_mm`), FP8 e4m3 (`torch._scaled_mm`,
    record only); every configuration checked for correctness first; the kernel name from torch.profiler.

Per call: CUDA events around one call, synchronised after each call (so the OC read belongs to that call).

python bench/micro_peak.py --part bw   --proc-tag p1 --out results/<run>
python bench/micro_peak.py --part gemm --proc-tag p1 --out results/<run>
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate  # noqa: E402

GiB = 1 << 30
SUMMARY_FIELDS = ["proc", "kernel", "dtype", "shape", "bytes", "flops", "n", "median_ms", "mean_ms", "p5_ms", "p95_ms",
                  "min_ms", "max_ms", "GBps", "TFLOPps", "burst_n", "burst_median_ms", "burst_GBps", "burst_TFLOPps",
                  "sustained_n", "sustained_median_ms", "sustained_GBps", "sustained_TFLOPps", "oc_event_delta",
                  "throttled", "gpu_hz_before", "gpu_hz_after", "gpu_C_before", "gpu_C_after", "correct", "rel_err",
                  "engine", "kernel_names", "note"]


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def gpu_c():
    try:
        return int(open("/sys/class/thermal/thermal_zone1/temp").read()) / 1e3
    except Exception:
        return "n/a"


def timed(fn, warmup=20, n=100, cooldown_s=4.0):
    """Cool down, warm up, then n calls each bracketed by CUDA events and followed by an OC-counter read."""
    time.sleep(cooldown_s)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    hz0, c0 = int(open(f"{gate.DEVFREQ}/cur_freq").read()), gpu_c()
    oc0 = gate.oc_counts()
    ms, oc_cum = [], []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
        oc = gate.oc_counts()
        oc_cum.append(sum(oc[k] - oc0[k] for k in oc0))
    dev = {"gpu_hz_before": hz0, "gpu_hz_after": int(open(f"{gate.DEVFREQ}/cur_freq").read()),
           "gpu_C_before": c0, "gpu_C_after": gpu_c(), "oc_event_delta": oc_cum[-1] if oc_cum else 0}
    return ms, oc_cum, dev


def kernel_names(fn):
    """Device-side events of one call (kernels and memcpys), from torch.profiler."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
    return names


def summarise(proc, kernel, dtype, shape, ms, oc_cum, dev, nbytes=0, flops=0, **extra):
    burst = [m for m, o in zip(ms, oc_cum) if o == 0]
    sust = [m for m, o in zip(ms, oc_cum) if o > 0]

    def rate(med, kind):
        if not med:
            return ""
        s = med / 1e3
        return round(nbytes / s / 1e9, 2) if kind == "bw" else round(flops / s / 1e12, 3)

    med = statistics.median(ms)
    bm = statistics.median(burst) if burst else ""
    sm = statistics.median(sust) if sust else ""
    r = {"proc": proc, "kernel": kernel, "dtype": dtype, "shape": shape, "bytes": nbytes, "flops": flops, "n": len(ms),
         "median_ms": round(med, 6), "mean_ms": round(statistics.mean(ms), 6), "p5_ms": round(pct(ms, .05), 6),
         "p95_ms": round(pct(ms, .95), 6), "min_ms": round(min(ms), 6), "max_ms": round(max(ms), 6),
         "GBps": rate(med, "bw") if nbytes else "", "TFLOPps": rate(med, "fl") if flops else "",
         "burst_n": len(burst), "burst_median_ms": round(bm, 6) if burst else "",
         "burst_GBps": rate(bm, "bw") if nbytes and burst else "", "burst_TFLOPps": rate(bm, "fl") if flops and burst else "",
         "sustained_n": len(sust), "sustained_median_ms": round(sm, 6) if sust else "",
         "sustained_GBps": rate(sm, "bw") if nbytes and sust else "",
         "sustained_TFLOPps": rate(sm, "fl") if flops and sust else "",
         "throttled": dev["oc_event_delta"] > 0}
    r.update(dev)
    r.update(extra)
    return r


def bandwidth(proc, repeats, sizes_gib, samples_out):
    rows = []
    for gib in sizes_gib:
        n = int(gib * GiB) // 2                       # bf16 elements
        G = n * 2
        x = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
        y = torch.empty(n, device="cuda", dtype=torch.bfloat16).normal_()
        z = torch.empty(n, device="cuda", dtype=torch.bfloat16)
        pats = [
            ("copy_d2d", lambda: z.copy_(x), 2 * G, "z.copy_(x): read x, write z"),
            ("copy_sm_mul1", lambda: torch.mul(x, 1.0, out=z), 2 * G, "z = 1.0 * x: the same traffic as an SM kernel"),
            ("axpy_inplace", lambda: y.add_(x, alpha=1.0001), 3 * G, "y += a*x: read x, read y, write y"),
            ("add_out_of_place", lambda: torch.add(x, y, out=z), 3 * G, "z = x + y: read x, y, write z"),
            ("read_sum", lambda: x.sum(), G, "x.sum(): read only (reduction)"),
            ("fill_write", lambda: z.fill_(1.0), G, "z.fill_(1): write only"),
        ]
        for tag, fn, nb, note in pats:
            names = kernel_names(fn)
            engine = "copy engine (memcpy)" if names and all("Memcpy" in k or "memcpy" in k for k in names) else "SM kernel"
            for rep in range(repeats):
                ms, oc, dev = timed(fn, 20, 100, cooldown_s=2.0)
                for i, (m, o) in enumerate(zip(ms, oc)):
                    samples_out.append({"proc": proc, "kernel": tag, "dtype": "bfloat16", "shape": f"{gib}GiB",
                                        "repeat": rep, "iter": i, "ms": m, "oc_cum": o})
                rows.append(summarise(proc, tag, "bfloat16", f"{gib}GiB", ms, oc, dev, nbytes=nb,
                                      engine=engine, kernel_names=" | ".join(names), note=f"repeat {rep}; {note}"))
                print(f"{proc} {tag} {gib}GiB rep{rep}: {rows[-1]['GBps']} GB/s (burst {rows[-1]['burst_GBps']}) "
                      f"oc+{dev['oc_event_delta']}", flush=True)
        del x, y, z
        torch.cuda.empty_cache()
    return rows


def gemm_case(name, n):
    """(callable, flops, correctness check) for one dtype at size n."""
    dev = "cuda"
    if name == "int8":
        a = torch.randint(-8, 8, (n, n), device=dev, dtype=torch.int8)
        b = torch.randint(-8, 8, (n, n), device=dev, dtype=torch.int8)
        fn = lambda: torch._int_mm(a, b)  # noqa: E731
        ref = lambda: a[:256].float() @ b.float()  # noqa: E731  exact for these magnitudes (< 2^24)
        got = lambda: fn()[:256].float()  # noqa: E731
        tol = 0.0
    elif name == "fp8_e4m3":
        a = (torch.randn(n, n, device=dev) * 0.5).to(torch.float8_e4m3fn)
        b = (torch.randn(n, n, device=dev) * 0.5).to(torch.float8_e4m3fn).t()   # column-major second operand
        one = torch.ones((), device=dev)
        fn = lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)  # noqa: E731
        ref = lambda: a[:256].float() @ b.float()  # noqa: E731
        got = lambda: fn()[:256].float()  # noqa: E731
        tol = 1e-2
    else:
        dt = {"fp32": torch.float32, "tf32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]
        a = torch.randn(n, n, device=dev, dtype=dt)
        b = torch.randn(n, n, device=dev, dtype=dt)
        fn = lambda: a @ b  # noqa: E731
        ref = lambda: a[:256].double() @ b.double()  # noqa: E731
        got = lambda: fn()[:256].double()  # noqa: E731
        tol = {"fp32": 1e-5, "tf32": 1e-2, "bf16": 1e-2, "fp16": 1e-2}[name]
    return fn, 2 * n ** 3, ref, got, tol


def gemm(proc, sizes, dtypes, samples_out):
    rows = []
    flags = {}
    for name in dtypes:
        for n in sizes:
            if name == "tf32":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.set_float32_matmul_precision("highest")
            flags[f"{name}_{n}"] = {"allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                    "float32_matmul_precision": torch.get_float32_matmul_precision()}
            try:
                fn, fl, ref, got, tol = gemm_case(name, n)
                g, r = got(), ref()
                rel = ((g - r).norm() / r.norm()).item()
                ok = rel <= tol if tol else bool(torch.equal(g, r))
                names = kernel_names(fn)
                calls = 100 if n <= 8192 else 30
                ms, oc, dev = timed(fn, 20 if n <= 8192 else 5, calls, cooldown_s=5.0)
                for i, (m, o) in enumerate(zip(ms, oc)):
                    samples_out.append({"proc": proc, "kernel": "gemm", "dtype": name, "shape": f"{n}^3",
                                        "repeat": 0, "iter": i, "ms": m, "oc_cum": o})
                rows.append(summarise(proc, "gemm", name, f"{n}^3", ms, oc, dev, flops=fl, correct=ok,
                                      rel_err=f"{rel:.3e}", engine="SM kernel", kernel_names=" | ".join(names),
                                      note=("record only (no model uses FP8)" if name == "fp8_e4m3" else "")
                                      + (f"; {calls} timed calls (n > 8192)" if n > 8192 else "")))
                print(f"{proc} gemm {name} {n}: {rows[-1]['TFLOPps']} TFLOP/s (burst {rows[-1]['burst_TFLOPps']}, "
                      f"sustained {rows[-1]['sustained_TFLOPps']}) ok={ok} rel={rel:.2e} {names[:1]}", flush=True)
                del fn
            except Exception as exc:
                rows.append({"proc": proc, "kernel": "gemm", "dtype": name, "shape": f"{n}^3", "n": 0,
                             "note": f"not measured: {type(exc).__name__}: {str(exc)[:300]}"})
                print(f"{proc} gemm {name} {n}: FAILED {exc}", flush=True)
            torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return rows, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["bw", "gemm"], required=True)
    ap.add_argument("--proc-tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--bw-gib", default="1,2,4")
    ap.add_argument("--gemm-n", default="1024,2048,4096,8192,16384")
    ap.add_argument("--dtypes", default="fp32,tf32,bf16,fp16,int8,fp8_e4m3")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    g0 = gate.run_gate()
    if not g0["gate_pass"]:
        raise SystemExit(f"gate failed: {g0['checks']}")
    samples = []
    t0 = time.time()
    if a.part == "bw":
        rows = bandwidth(a.proc_tag, a.repeats, [float(x) for x in a.bw_gib.split(",")], samples)
        extra = {}
    else:
        rows, extra = gemm(a.proc_tag, [int(x) for x in a.gemm_n.split(",")], a.dtypes.split(","), samples)
    g1 = gate.run_gate(apply_clocks=False, check_idle=False)
    tag = f"{a.part}_{a.proc_tag}"
    with open(os.path.join(a.out, f"latency_summary_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "raw", f"latency_samples_{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["proc", "kernel", "dtype", "shape", "repeat", "iter", "ms", "oc_cum"])
        w.writeheader()
        w.writerows(samples)
    json.dump({"gate_before": g0, "gate_after": g1, "wall_s": time.time() - t0, "matmul_flags": extra,
               "torch": torch.__version__, "pid": os.getpid()},
              open(os.path.join(a.out, "raw", f"run_{tag}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
