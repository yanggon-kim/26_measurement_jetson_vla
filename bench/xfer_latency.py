"""T010 / E11b — what a transfer and a synchronisation actually cost (protocol §13.3).

The research side's hybrid model prices every hand-over between the GPU and the memory-side
engine as `bytes / 143 GB/s` with **no fixed cost**, and treats synchronisation as free
(`docs/00_context.md`). Both are assumptions. This measures the fixed term.

`t = a + b*bytes` is fitted by least squares on the per-size **medians**, per (method,
direction). **`a` is the point of the experiment**: at the sizes the mapping actually moves
(a 1.4 MB score matrix, a 0.1 MB activation tile) a fixed cost of a few tens of microseconds
is the whole story, and a peak GB/s alone cannot show it.

`zero_copy` means one allocation addressed by both sides with no copy at all — on this Tegra
the CPU and GPU share physical LPDDR5, so managed memory is genuinely that. PyTorch reaches
foreign memory through `CUDAPluggableAllocator`, which has to be installed before the process
touches CUDA, so those points are measured in a dedicated subprocess
(`--zero-copy-worker`) backed by `bench/managed_alloc.cu`.

    python bench/xfer_latency.py --outdir results/<date>_E11b_xfer
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
from common import (REPO, collect_raw, compress_and_record, sh_out,  # noqa: E402
                    software_versions, summarize, write_json)

KB, MB_BIN = 1024, 1024 * 1024
MB_DEC = 10 ** 6                      # §13.3: MB = 10^6 B for fit_b
STD_SIZES = [4 * KB, 16 * KB, 64 * KB, 256 * KB, MB_BIN, 2 * MB_BIN, 4 * MB_BIN, 16 * MB_BIN]


def pi0_sizes(S=51, T=867, D=1024, I=4096, P_tok=816, layers=18, kv_heads=1, head_dim=256):
    """The sizes the pi0 mapping actually moves, from the measured token counts."""
    return {
        "scores_fp32": 8 * S * T * 4,
        "scores_fp16": 8 * S * T * 2,
        "activation_tile": S * D * 2,
        "ffn_tile": S * I * 2,
        "kv_bytes_total": layers * 2 * P_tok * kv_heads * head_dim * 2,
    }


def timed_wall(fn, warmup: int, n: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def measure_transfers(sizes: dict[str, int], warmup: int, base_n: int) -> list[dict]:
    rows = []
    for label, nbytes in sizes.items():
        n = max(base_n, 1000) if nbytes <= 64 * KB else base_n
        for method in ("pageable", "pinned"):
            cpu = torch.empty(nbytes, dtype=torch.uint8,
                              pin_memory=(method == "pinned"))
            gpu = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            nb = (method == "pinned")
            for direction, fn in (("h2d", lambda: gpu.copy_(cpu, non_blocking=nb)),
                                  ("d2h", lambda: cpu.copy_(gpu, non_blocking=nb))):
                ms = timed_wall(fn, warmup, n)
                s = summarize(ms)
                rows.append({"method": method, "direction": direction, "size_label": label,
                             "size_bytes": nbytes, "median_ms": round(s["median_ms"], 6),
                             "p5_ms": round(s["p5_ms"], 6), "p95_ms": round(s["p95_ms"], 6),
                             "min_ms": round(s["min_ms"], 6), "n": s["n"],
                             "GBps": round(nbytes / (s["median_ms"] / 1e3) / 1e9, 3),
                             "samples": ms})
                print(f"  {method:9s} {direction} {label:16s} {nbytes:>10,d} B  "
                      f"{s['median_ms']:8.4f} ms  {rows[-1]['GBps']:7.2f} GB/s", flush=True)
            del cpu, gpu
            torch.cuda.empty_cache()
    return rows


def fit(rows: list[dict]) -> list[dict]:
    """Least squares t = a + b*bytes on the per-size medians, per (method, direction)."""
    out = []
    keys = sorted({(r["method"], r["direction"]) for r in rows})
    for method, direction in keys:
        pts = sorted(((r["size_bytes"], r["median_ms"]) for r in rows
                      if r["method"] == method and r["direction"] == direction))
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        nn = len(xs)
        mx, my = sum(xs) / nn, sum(ys) / nn
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        a = my - b * mx
        ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
        b_per_mb = b * MB_DEC
        out.append({"method": method, "direction": direction,
                    "fit_a_ms": round(a, 6), "fit_b_ms_per_MB": round(b_per_mb, 6),
                    "fit_r2": round(r2, 6),
                    "GBps_from_b": round(1.0 / b_per_mb, 3) if b_per_mb > 0 else "",
                    "n_points": nn})
        print(f"  fit {method:9s} {direction}: a={a*1e3:7.1f} us  b={b_per_mb:.4f} ms/MB  "
              f"({out[-1]['GBps_from_b']} GB/s)  r2={r2:.4f}", flush=True)
    return out


def measure_sync(warmup: int, n: int) -> tuple[list[dict], dict]:
    """The five synchronisation costs of §13.3, plus the spin-wait check."""
    a = torch.zeros(1, device="cuda")
    s_b = torch.cuda.Stream()
    done = torch.cuda.Event()
    a.add_(1.0)
    done.record()
    torch.cuda.synchronize()

    def tiny():
        a.add_(1.0)
        torch.cuda.synchronize()

    def handoff():
        with torch.cuda.stream(s_b):
            pass
        a.add_(1.0)
        ev = torch.cuda.Event()
        ev.record()
        s_b.wait_event(ev)
        with torch.cuda.stream(s_b):
            a.add_(1.0)
        torch.cuda.synchronize()

    def two_kernels_one_stream():
        a.add_(1.0)
        a.add_(1.0)
        torch.cuda.synchronize()

    cases = [
        ("synchronize_idle", lambda: torch.cuda.synchronize()),
        ("event_synchronize_done", lambda: done.synchronize()),
        ("event_query", lambda: done.query()),
        ("tiny_kernel_roundtrip", tiny),
        ("_two_kernels_one_stream", two_kernels_one_stream),
        ("stream_handoff", handoff),
    ]
    rows, med = [], {}
    for name, fn in cases:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            out.append((time.perf_counter() - t0) * 1e3)
        s = summarize(out)
        med[name] = s["median_ms"]
        rows.append({"sync": name, "median_ms": round(s["median_ms"], 6),
                     "p5_ms": round(s["p5_ms"], 6), "p95_ms": round(s["p95_ms"], 6),
                     "min_ms": round(s["min_ms"], 6), "n": s["n"],
                     "net_of_control_ms": "",
                     "note": "control, subtracted from stream_handoff"
                             if name.startswith("_") else ""})
        print(f"  {name:26s} {s['median_ms']*1e3:8.2f} us", flush=True)
    # §13.3: stream_handoff is reported net of the same two kernels on one stream
    net = med["stream_handoff"] - med["_two_kernels_one_stream"]
    for r in rows:
        if r["sync"] == "stream_handoff":
            r["net_of_control_ms"] = round(net, 6)
            r["note"] = "reported net of _two_kernels_one_stream"
    print(f"  stream_handoff net of control: {net*1e3:.2f} us")

    # spin-wait check: does the CPU burn a core while it waits?
    big = torch.empty(64 * MB_BIN // 4, device="cuda", dtype=torch.float32).normal_()
    for _ in range(3):
        big.mul_(1.0001)
    torch.cuda.synchronize()
    for _ in range(60):
        big.mul_(1.0001)
    c0, w0 = time.process_time(), time.perf_counter()
    torch.cuda.synchronize()
    cpu_s, wall_s = time.process_time() - c0, time.perf_counter() - w0
    spin = {"wall_s": round(wall_s, 6), "process_cpu_s": round(cpu_s, 6),
            "cpu_over_wall": round(cpu_s / wall_s, 4) if wall_s > 0 else None,
            "verdict": "spins (busy-wait)" if wall_s > 0 and cpu_s / wall_s > 0.5
                       else "blocks (yields the core)"}
    print(f"  spin-wait: {spin['process_cpu_s']:.4f} s CPU / {spin['wall_s']:.4f} s wall "
          f"= {spin['cpu_over_wall']} -> {spin['verdict']}")
    del big
    torch.cuda.empty_cache()
    return rows, spin


# --------------------------------------------------------------------------- zero copy

ZC_SO = REPO / "bench" / "managed_alloc.so"


def zero_copy_worker(sizes_json: str, warmup: int, n: int) -> int:
    """Runs in its own process: installs the managed-memory allocator BEFORE touching CUDA,
    then times the four parts of a zero-copy round trip on one shared allocation."""
    import ctypes
    import numpy as np

    alloc = torch.cuda.memory.CUDAPluggableAllocator(
        str(ZC_SO), "managed_malloc", "managed_free")
    torch.cuda.memory.change_current_allocator(alloc)

    sizes = json.loads(sizes_json)
    out = []
    for label, nbytes in sizes.items():
        nel = nbytes // 4
        if nel < 1:
            continue
        t = torch.empty(nel, dtype=torch.float32, device="cuda")
        buf = (ctypes.c_float * nel).from_address(t.data_ptr())
        arr = np.frombuffer(buf, dtype=np.float32, count=nel)
        src = np.ones(nel, dtype=np.float32)

        def cpu_write():
            np.copyto(arr, src)

        def gpu_read():
            return t.sum()

        parts = {}
        for name, fn, sync in (("cpu_write", cpu_write, False),
                               ("gpu_read", gpu_read, True),
                               ("roundtrip", None, True)):
            if name == "roundtrip":
                def rt():
                    np.copyto(arr, src)
                    r = t.sum()
                    torch.cuda.synchronize()
                    return float(r.item())
                fn = rt
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            ms = []
            for _ in range(n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                if sync:
                    torch.cuda.synchronize()
                ms.append((time.perf_counter() - t0) * 1e3)
            ms.sort()
            parts[name] = {"median_ms": ms[len(ms) // 2], "p5_ms": ms[int(0.05 * len(ms))],
                           "p95_ms": ms[min(len(ms) - 1, int(0.95 * len(ms)))], "n": len(ms)}
        out.append({"size_label": label, "size_bytes": nel * 4, "parts": parts})
        del t, buf, arr
        torch.cuda.empty_cache()
    print("ZERO_COPY_JSON " + json.dumps(out))
    return 0


def control_gpu_read(sizes: dict[str, int], warmup: int, n: int) -> list[dict]:
    """The same GPU kernel on an ORDINARY device tensor, for comparison with zero_copy.

    It has to be a separate process from the zero-copy one: `change_current_allocator` is
    global and irreversible, so a managed-memory process cannot also allocate ordinary memory.
    """
    rows = []
    for label, nbytes in sizes.items():
        nel = nbytes // 4
        if nel < 1:
            continue
        t = torch.empty(nel, dtype=torch.float32, device="cuda").fill_(1.0)
        ms = timed_wall(lambda: t.sum(), warmup, n)
        s = summarize(ms)
        rows.append({"method": "device_control", "direction": "gpu_read",
                     "size_label": label, "size_bytes": nel * 4,
                     "median_ms": round(s["median_ms"], 6), "p5_ms": round(s["p5_ms"], 6),
                     "p95_ms": round(s["p95_ms"], 6), "min_ms": round(s["min_ms"], 6),
                     "n": s["n"],
                     "GBps": round(nel * 4 / (s["median_ms"] / 1e3) / 1e9, 3), "samples": ms})
        del t
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=200)
    ap.add_argument("--sync-timed", type=int, default=1000)
    ap.add_argument("--zero-copy-worker", default=None,
                    help="internal: JSON size map; installs the managed allocator and exits")
    args = ap.parse_args()
    if args.zero_copy_worker:
        return zero_copy_worker(args.zero_copy_worker, args.warmup, max(50, args.timed // 4))

    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    gate = P.gate()
    print("gate:", json.dumps(gate), flush=True)

    sizes = {f"{s // KB}KiB" if s < MB_BIN else f"{s // MB_BIN}MiB": s for s in STD_SIZES}
    exact = pi0_sizes()
    sizes.update(exact)
    print(f"\n=== transfers ({len(sizes)} sizes) ===", flush=True)
    rows = measure_transfers(sizes, args.warmup, args.timed)

    print("\n=== zero copy (separate process, managed allocator) ===", flush=True)
    zc_rows, zc_err = [], None
    if not ZC_SO.exists():
        zc_err = f"{ZC_SO} not built; see env/README.md for the nvcc command"
    else:
        cmd = [sys.executable, __file__, "--zero-copy-worker", json.dumps(exact),
               "--warmup", str(args.warmup), "--timed", str(args.timed)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                           env={**os.environ})
        (out / "raw" / "zero_copy_worker.log").write_text(r.stdout + "\n" + r.stderr)
        line = next((ln for ln in r.stdout.splitlines()
                     if ln.startswith("ZERO_COPY_JSON ")), None)
        if r.returncode != 0 or line is None:
            tail = (r.stderr or r.stdout).strip().splitlines()
            zc_err = f"worker exit={r.returncode}; {tail[-1][:200] if tail else 'no output'}"
        else:
            for e in json.loads(line[len("ZERO_COPY_JSON "):]):
                for part, st in e["parts"].items():
                    zc_rows.append({"method": "zero_copy", "direction": part,
                                    "size_label": e["size_label"],
                                    "size_bytes": e["size_bytes"],
                                    "median_ms": round(st["median_ms"], 6),
                                    "p5_ms": round(st["p5_ms"], 6),
                                    "p95_ms": round(st["p95_ms"], 6), "min_ms": "",
                                    "n": st["n"],
                                    "GBps": round(e["size_bytes"] / (st["median_ms"] / 1e3)
                                                  / 1e9, 3) if st["median_ms"] > 0 else "",
                                    "samples": []})
                    print(f"  zero_copy {part:10s} {e['size_label']:16s} "
                          f"{st['median_ms']:8.4f} ms", flush=True)
    if zc_err:
        print(f"  NOT MEASURED: {zc_err}")
    ctrl = control_gpu_read(exact, args.warmup, args.timed)
    for c in ctrl:
        print(f"  control   gpu_read   {c['size_label']:16s} {c['median_ms']:8.4f} ms")

    print("\n=== fits ===", flush=True)
    fits = fit(rows)
    print("\n=== synchronisation ===", flush=True)
    sync_rows, spin = measure_sync(args.warmup, args.sync_timed)

    # --- files
    samples = [{"method": r["method"], "direction": r["direction"],
                "size_bytes": r["size_bytes"], "iter": i, "ms": v}
               for r in rows + ctrl for i, v in enumerate(r.get("samples", []))]
    for r in rows + zc_rows + ctrl:
        r.pop("samples", None)
    all_x = rows + zc_rows + ctrl
    with (out / "xfer_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_x[0].keys()))
        w.writeheader()
        w.writerows(all_x)
    with (out / "xfer_fit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fits[0].keys()))
        w.writeheader()
        w.writerows(fits)
    with (out / "sync_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sync_rows[0].keys()))
        w.writeheader()
        w.writerows(sync_rows)
    sample_csv = out / "raw" / "xfer_samples.csv"
    with sample_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "direction", "size_bytes", "iter", "ms"])
        w.writeheader()
        w.writerows(samples)
    entry = compress_and_record(sample_csv)
    write_json(out / "raw" / "ORIGINALS.json", {"files": [entry]})

    # --- model_inputs.csv (model = none: this part is model-independent, §13.8)
    from host_segments import ModelInputs
    mi = ModelInputs("none", "", sh_out(f"git -C {REPO} rev-parse HEAD"),
                     "MAXN+jetson_clocks", "", "", "")
    for r in all_x:
        key = f"{r['method']}:{r['direction']}"
        for stat, col in (("median", "median_ms"), ("p5", "p5_ms"), ("p95", "p95_ms")):
            if r[col] != "":
                mi.add("xfer_latency", r[col], "ms", stat, r["n"], "xfer_summary.csv",
                       key=key, variant=f"size_bytes={r['size_bytes']}",
                       timing="wall_isolated")
    for f_ in fits:
        k = f"{f_['method']}:{f_['direction']}"
        mi.add("xfer_latency_fit", f_["fit_a_ms"], "ms", "fit_a", f_["n_points"],
               "xfer_fit.csv", key=k, timing="wall_isolated")
        mi.add("xfer_latency_fit", f_["fit_b_ms_per_MB"], "ms_per_MB", "fit_b",
               f_["n_points"], "xfer_fit.csv", key=k, timing="wall_isolated")
        mi.add("xfer_latency_fit", f_["fit_r2"], "ratio", "fit_r2", f_["n_points"],
               "xfer_fit.csv", key=k, timing="wall_isolated")
    for r in sync_rows:
        if r["sync"].startswith("_"):
            continue
        v = r.get("net_of_control_ms") or r["median_ms"]
        for stat, col in (("median", "median_ms"), ("p5", "p5_ms"), ("p95", "p95_ms")):
            mi.add("t_sync", v if stat == "median" else r[col], "ms", stat, r["n"],
                   "sync_summary.csv", key=r["sync"], timing="wall_isolated")
    n_mi = mi.write(out / "model_inputs.csv")

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E11b", "task": "T010",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": None, "model_repo": None, "model_commit": None, "checkpoint": None,
        "precision": "uint8 buffers for transfers; float32 for zero-copy and the control",
        "compile_mode": None, "batch": 1, "num_denoise_steps": None, "action_horizon": None,
        "prefix_tokens": None, "suffix_tokens": None,
        "n_warmup": args.warmup, "n_timed": args.timed, "n_timed_sync": args.sync_timed,
        "power_mode": gate["power_mode"], "sm_count": gate["sm_count"],
        "gpu_hz": gate["gpu_cur_freq_hz"], "gate": gate,
        "emc_hz": int(sh_out("sudo -n sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'") or 0),
        "jetpack": "6.2.3+b81", "l4t": "R36.5.0",
        "cuda": sw.get("torch_cuda"), "torch": sw.get("torch"),
        "torchvision": sw.get("torchvision"), "triton": sw.get("triton"),
        "python": sw.get("python"), "venv": sw.get("executable"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "command": f"python bench/xfer_latency.py --outdir {args.outdir}",
        "zero_copy_route": "CUDAPluggableAllocator over cudaMallocManaged "
                           "(bench/managed_alloc.cu); build command in env/README.md",
        "zero_copy_error": zc_err,
        "spin_wait_check": spin,
        "sizes": sizes, "pi0_exact_sizes": exact,
        "throttled": False,
        "throttled_evidence": "GPU clock pinned min==max, recorded in `gate`.",
        "model_inputs_rows": n_mi, "raw_files": [],
        "notes": [
            "MB = 10^6 B for fit_b (§13.3); KiB/MiB labels are binary.",
            "The fixed term fit_a is the quantity this experiment exists for: the hybrid "
            "model assumes a = 0 and prices transfers purely as bytes / 143 GB/s.",
            "zero_copy is one managed allocation addressed by both sides, so there is no "
            "copy to time; the parts are cpu_write, gpu_read and the full roundtrip. Its "
            "control (the same reduction on an ordinary device tensor) has to run in a "
            "different process because change_current_allocator is global and irreversible.",
            "stream_handoff is reported net of the same two kernels on one stream, as §13.3 "
            "requires; the control row is kept with a leading underscore.",
        ],
    })
    meta = json.loads((out / "meta.json").read_text())
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out} ({n_mi} model_inputs rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
