"""Shared helpers for the on-device measurement harness (docs/02_protocol.md).

Nothing here measures by itself; it collects device state, software versions and
provides the CUDA-event timing and NVTX region helpers the protocol mandates (§2).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- shell


def sh(cmd: str, timeout: int = 120) -> dict:
    """Run a shell command, capture everything, never raise."""
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "command": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "seconds": round(time.time() - t0, 3),
        }
    except subprocess.TimeoutExpired:
        return {
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": f"timeout after {timeout}s",
            "seconds": round(time.time() - t0, 3),
        }


def sh_out(cmd: str, timeout: int = 120) -> str:
    return sh(cmd, timeout)["stdout"].strip()


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def raw_file_entry(path: Path, base: Path, kind: str, produced_by: str,
                   secrets_scan: str = "not_run") -> dict:
    """One entry of meta.json.raw_files (protocol §5)."""
    return {
        "path": str(path.relative_to(base)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "kind": kind,
        "produced_by": produced_by,
        "secrets_scan": secrets_scan,
        "committed": True,
    }


# --------------------------------------------------------------------------- device state


_TEGRASTATS_RE = {
    "ram_used_mb": re.compile(r"RAM (\d+)/(\d+)MB"),
    "gr3d_pct": re.compile(r"GR3D_FREQ (\d+)%"),
    "tj_c": re.compile(r"tj@([\d.]+)C"),
}
# rails printed by tegrastats on this carrier board, plus any extra ones
_RAIL_RE = re.compile(r"\b(VDD_[A-Z0-9_]+|VIN_[A-Z0-9_]+|VDDQ_[A-Z0-9_]+) (\d+)mW/(\d+)mW")


def tegrastats_sample(n: int = 3, interval_ms: int = 1000) -> list[str]:
    """n raw tegrastats lines. Needs no sudo on this board."""
    out = sh(f"timeout {max(3, n * interval_ms // 1000 + 3)} tegrastats --interval {interval_ms}",
             timeout=n * interval_ms // 1000 + 10)["stdout"]
    return [ln for ln in out.splitlines() if ln.strip()][:n]


def parse_tegrastats(line: str) -> dict:
    d = {}
    for k, rx in _TEGRASTATS_RE.items():
        m = rx.search(line)
        if m:
            d[k] = float(m.group(1))
    d["rails_mW"] = {m.group(1): int(m.group(2)) for m in _RAIL_RE.finditer(line)}
    return d


def ina3221_rails() -> dict:
    """Every INA3221 rail exposed in sysfs, including ones tegrastats does not print.

    Returns {rail_name: {"path": <hwmon input path>, "mW": <reading>}}.
    """
    rails = {}
    root = Path("/sys/bus/i2c/drivers/ina3221")
    if not root.exists():
        return rails
    for label in sorted(root.glob("*/hwmon/hwmon*/in*_label")):
        name = label.read_text().strip()
        if name.startswith("sum of"):
            continue
        idx = re.search(r"in(\d+)_label", label.name).group(1)
        power = label.parent / f"curr{idx}_input"      # mA on some kernels
        volt = label.parent / f"in{idx}_input"          # mV
        entry = {"label_path": str(label)}
        try:
            mv = int(volt.read_text().strip())
            ma = int(power.read_text().strip())
            entry.update({"mV": mv, "mA": ma, "mW": round(mv * ma / 1000)})
        except Exception as e:  # pragma: no cover - depends on kernel driver
            entry["error"] = str(e)
        rails[name] = entry
    return rails


def device_state(sudo: str = "sudo -n") -> dict:
    """Everything protocol §1 asks to record for a run."""
    st = {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "board_model": Path("/proc/device-tree/model").read_text().strip("\x00 \n")
        if Path("/proc/device-tree/model").exists() else None,
        "nv_tegra_release": Path("/etc/nv_tegra_release").read_text().strip()
        if Path("/etc/nv_tegra_release").exists() else None,
    }
    st["nvpmodel_q"] = sh("nvpmodel -q")            # no sudo needed for query
    st["nvpmodel_q_verbose"] = sh(f"{sudo} nvpmodel -q --verbose")
    st["jetson_clocks_show"] = sh(f"{sudo} jetson_clocks --show")
    st["emc_rate_hz"] = sh(f"{sudo} cat /sys/kernel/debug/bpmp/debug/clk/emc/rate")
    st["dpkg_jetpack"] = sh("dpkg -l | grep -E 'nvidia-jetpack|cuda-toolkit|tensorrt|cudnn'")
    st["nvcc_version"] = sh("nvcc --version")
    st["free_h"] = sh("free -h")
    st["swap"] = sh("swapon --show; cat /proc/swaps; zramctl 2>/dev/null || true")
    st["df_h"] = sh("df -h")
    st["lsblk"] = sh("lsblk -o NAME,SIZE,TYPE,MOUNTPOINT")
    st["cpu_online"] = sh("cat /sys/devices/system/cpu/online")
    st["tpc_pg_mask"] = sh("cat /sys/devices/platform/gpu.0/tpc_pg_mask")
    st["gpu_min_freq"] = sh("cat /sys/devices/platform/17000000.gpu/devfreq_dev/min_freq")
    st["gpu_max_freq"] = sh("cat /sys/devices/platform/17000000.gpu/devfreq_dev/max_freq")
    st["gpu_cur_freq"] = sh("cat /sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq")
    st["nvidia_smi"] = sh("nvidia-smi 2>&1 || echo 'nvidia-smi not present (expected on Tegra)'")
    st["perf_event_paranoid"] = sh("cat /proc/sys/kernel/perf_event_paranoid")
    st["thermal_zones"] = sh(
        "for z in /sys/class/thermal/thermal_zone*; do "
        "echo \"$(cat $z/type)=$(cat $z/temp)\"; done")
    st["fan"] = sh("cat /sys/class/hwmon/hwmon*/pwm1 2>/dev/null; "
                   "cat /sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1 2>/dev/null")
    st["tegrastats_idle_lines"] = tegrastats_sample(3, 1000)
    st["tegrastats_idle_parsed"] = [parse_tegrastats(l) for l in st["tegrastats_idle_lines"]]
    st["ina3221_rails"] = ina3221_rails()
    st["uptime"] = sh("uptime")
    st["date_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return st


def software_versions() -> dict:
    sw = {"python": platform.python_version(), "executable": os.sys.executable}
    try:
        import torch
        sw["torch"] = torch.__version__
        sw["torch_cuda"] = torch.version.cuda
        sw["torch_cudnn"] = torch.backends.cudnn.version()
        sw["torch_arch_list"] = torch.cuda.get_arch_list()
        sw["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            sw["gpu_name"] = p.name
            sw["sm_count"] = p.multi_processor_count
            sw["compute_capability"] = f"{p.major}.{p.minor}"
            sw["gpu_arch"] = f"sm_{p.major}{p.minor}"
            sw["total_memory_bytes"] = p.total_memory
    except Exception as e:
        sw["torch_error"] = repr(e)
    for mod in ("torchvision", "triton", "numpy"):
        try:
            sw[mod] = __import__(mod).__version__
        except Exception as e:
            sw[mod] = f"absent ({e.__class__.__name__})"
    for tool in ("ncu", "nsys", "cuobjdump", "nvdisasm", "c++filt"):
        sw[f"{tool}_path"] = shutil.which(tool)
    sw["ncu_version"] = sh_out("ncu --version | head -4 | tail -1")
    sw["nsys_version"] = sh_out("nsys --version")
    sw["cuobjdump_version"] = sh_out("cuobjdump --version | tail -2 | head -1")
    sw["nvdisasm_version"] = sh_out("nvdisasm --version | tail -2 | head -1")
    sw["bench_commit"] = sh_out(f"git -C {REPO} rev-parse HEAD")
    sw["bench_dirty"] = bool(sh_out(f"git -C {REPO} status --porcelain bench/"))
    # bench_commit alone does not identify the code that ran: HEAD can predate the
    # commit that first contains bench/, and the tree can be dirty. Hash every script
    # as it was executed, so a run is reproducible from meta.json even then.
    sw["bench_files_sha256"] = {
        f.name: sha256(f) for f in sorted((REPO / "bench").glob("*.py"))
    }
    return sw


# --------------------------------------------------------------------------- timing / NVTX


@contextmanager
def nvtx(name: str):
    import torch
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def cuda_time_ms(fn, n_warmup: int = 20, n_timed: int = 100) -> list[float]:
    """Per-iteration GPU time in ms via CUDA events (protocol §2).

    Returns every sample; the caller writes them to latency_samples.csv.
    """
    import torch
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_timed)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_timed)]
    for i in range(n_timed):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [starts[i].elapsed_time(ends[i]) for i in range(n_timed)]


def summarize(samples: list[float]) -> dict:
    s = sorted(samples)
    n = len(s)

    def pct(q: float) -> float:
        if n == 1:
            return s[0]
        i = q * (n - 1)
        lo, hi = int(i), min(int(i) + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (i - lo)

    return {
        "median_ms": statistics.median(s),
        "mean_ms": statistics.fmean(s),
        "p5_ms": pct(0.05),
        "p95_ms": pct(0.95),
        "min_ms": s[0],
        "max_ms": s[-1],
        "n": n,
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")
