"""T001 step 3d / protocol §8 — derive this device's own hardware constants.

Every hardware constant in a published profiling guide belongs to another GPU.
This writes results/<run>/hw_constants.json from the device itself. Measured peaks
(GB/s, TFLOP/s, ridge point) are added later by T005/E1; the fields are present and
explicitly null here so nothing downstream mistakes an expectation for a measurement.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import sh, sh_out, write_json  # noqa: E402


def cublas_library(torch) -> dict:
    """Which cuBLAS this torch actually loads, and which SASS architectures it holds."""
    import ctypes  # noqa: F401  (loading happens through torch below)
    out = {}
    # force cuBLAS to load, then read the process map
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    _ = a @ a
    torch.cuda.synchronize()
    maps = Path("/proc/self/maps").read_text()
    libs = sorted({m for m in re.findall(r"\S+/libcublas\S*\.so\S*", maps)})
    out["loaded_cublas_libraries"] = libs
    if not libs:
        out["note"] = "no libcublas found in /proc/self/maps"
        return out
    lib = libs[0]
    out["analysed"] = lib
    st = Path(lib).stat()
    out["bytes"] = st.st_size
    # cuobjdump --list-text prints one line per embedded cubin, with its sm_ arch
    r = sh(f"cuobjdump --list-text {lib}", timeout=900)
    out["cuobjdump_returncode"] = r["returncode"]
    archs = sorted(set(re.findall(r"sm_(\d+)", r["stdout"])))
    out["sm_architectures_in_library"] = [f"sm_{a}" for a in archs]
    out["cuobjdump_list_text_lines"] = len(r["stdout"].splitlines())
    if r["returncode"] != 0:
        out["cuobjdump_stderr"] = r["stderr"][:2000]
    return out


def ncu_per_sm_limits(ncu_raw_csv: Path | None) -> dict:
    """Per-SM limits as ncu reported them for the toy kernel (protocol §8)."""
    if not ncu_raw_csv or not ncu_raw_csv.exists():
        return {"source": "not available"}
    import csv
    rows = list(csv.reader(ncu_raw_csv.open(newline="")))
    if len(rows) < 3:
        return {"source": str(ncu_raw_csv), "note": "no kernel rows"}
    rec = dict(zip(rows[0], rows[2]))

    def get(suffix: str):
        for c, v in rec.items():
            if c.endswith(suffix) and v.strip():
                try:
                    return float(v.replace(",", ""))
                except ValueError:
                    return v
        return None

    return {
        "source": str(ncu_raw_csv),
        "note": "from the T001 toy kernel; device limits, not kernel-specific ones, "
                "are the *_peak_sustained / max_warps entries",
        "sm__maximum_warps_per_active_cycle_pct": get("sm__maximum_warps_per_active_cycle_pct"),
        "sm__maximum_warps_avg_per_active_cycle": get("sm__maximum_warps_avg_per_active_cycle"),
        "launch__occupancy_limit_warps": get("launch__occupancy_limit_warps"),
        "launch__occupancy_limit_blocks": get("launch__occupancy_limit_blocks"),
        "launch__occupancy_limit_registers": get("launch__occupancy_limit_registers"),
        "launch__occupancy_limit_shared_mem": get("launch__occupancy_limit_shared_mem"),
        "sm__warps_active.avg.peak_sustained": get("sm__warps_active.avg.peak_sustained"),
        "dram__cycles_elapsed.sum": get("dram__cycles_elapsed.sum"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ncu-raw-csv", default=None)
    args = ap.parse_args()

    import torch
    p = torch.cuda.get_device_properties(0)
    hw = {
        "_comment": "protocol §8 — constants derived on THIS device. Peaks marked null "
                    "are measured in T005/E1 and must not be filled from a spec sheet.",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "board_model": Path("/proc/device-tree/model").read_text().strip("\x00 \n"),
        "gpu_name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "gpu_arch": f"sm_{p.major}{p.minor}",
        "multi_processor_count": p.multi_processor_count,
        "total_memory_bytes": p.total_memory,
        "warp_size": getattr(p, "warp_size", 32),
        "max_threads_per_multi_processor": getattr(
            p, "max_threads_per_multi_processor", None),
        "regs_per_multiprocessor": getattr(p, "regs_per_multiprocessor", None),
        "shared_memory_per_multiprocessor": getattr(
            p, "shared_memory_per_multiprocessor", None),
        "l2_cache_size_bytes": getattr(p, "L2_cache_size", None),
        "device_state_when_recorded": {
            "power_mode": sh_out("nvpmodel -q | tail -2 | head -1"),
            "power_mode_id": sh_out("nvpmodel -q | tail -1"),
            "tpc_pg_mask": sh_out("cat /sys/devices/platform/gpu.0/tpc_pg_mask"),
            "gpu_max_freq_hz": sh_out(
                "cat /sys/devices/platform/17000000.gpu/devfreq_dev/max_freq"),
            "gpu_cur_freq_hz": sh_out(
                "cat /sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq"),
            "cpu_online": sh_out("cat /sys/devices/system/cpu/online"),
            "warning": "multi_processor_count and the peaks depend on the power mode: "
                       "TPC power gating removes SMs. Record the mode with every use.",
        },
        "max_warps_per_sm": (getattr(p, "max_threads_per_multi_processor", 0) or 0) // 32,
        "measured_peaks": {
            "_status": "not measured yet — T005 / E1 fills these",
            "peak_memory_bandwidth_GBps": None,
            "peak_bf16_TFLOPps": None,
            "peak_fp16_TFLOPps": None,
            "peak_fp32_TFLOPps": None,
            "peak_int8_TOPps": None,
            "ridge_point_FLOP_per_byte": None,
        },
        "expected_from_spec_do_not_use_as_measurement": {
            "peak_memory_bandwidth_GBps": 204.8,
            "peak_bf16_TFLOPps": 42.0,
            "ridge_point_FLOP_per_byte": 205.1,
            "source": "docs/00_context.md / NVIDIA AGX Orin 64GB spec; LPDDR5-6400, 256-bit",
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "arch_list": torch.cuda.get_arch_list(),
        },
    }
    hw["cublas"] = cublas_library(torch)
    hw["ncu_per_sm_limits"] = ncu_per_sm_limits(
        Path(args.ncu_raw_csv) if args.ncu_raw_csv else None)
    hw["gpu_arch_confirmed_by"] = {
        "torch_get_arch_list": torch.cuda.get_arch_list(),
        "compute_capability": f"{p.major}.{p.minor}",
        "cuobjdump_on_cublas": hw["cublas"].get("sm_architectures_in_library"),
    }
    write_json(Path(args.out), hw)
    print(f"wrote {args.out}")
    print(f"  gpu_arch={hw['gpu_arch']} SMs={hw['multi_processor_count']} "
          f"max_warps_per_sm={hw['max_warps_per_sm']} "
          f"power_mode={hw['device_state_when_recorded']['power_mode']}")
    print(f"  cublas={hw['cublas'].get('analysed')}")
    print(f"  cublas archs={hw['cublas'].get('sm_architectures_in_library')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
