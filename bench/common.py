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
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
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


_SECRET_RE = re.compile(
    rb"(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}"
    rb"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    rb"|(?i:authorization|bearer|api[-_]?key|password|passwd|secret|token)\s*[:=]\s*\S{8,})")


def scan_for_secrets(path: Path, limit: int = 64 << 20) -> str:
    """Protocol §3.5: every raw file is scanned before it is committed.

    Profiler reports and captured stdout can contain the process environment. Returns
    "clean", "MATCH: ..." or a reason the scan could not run.
    """
    try:
        if path.stat().st_size > limit:
            return f"skipped (>{limit >> 20} MB)"
        blob = path.read_bytes()
    except OSError as e:
        return f"unreadable: {e}"
    m = _SECRET_RE.search(blob)
    if m:
        # NEVER echo the match: this string is written into meta.json, which IS committed.
        # Report the pattern class and the offset only.
        head = m.group(0).split(b"=")[0][:40].decode("ascii", "replace")
        return (f"MATCH at byte {m.start()}: a value assigned to {head!r} matches the "
                f"secret pattern. NOT COMMITTED.")
    return "clean"


KIND_ENUM = {"nsys-rep", "nsys-sqlite", "ncu-rep", "ncu-csv", "sass", "ptx", "cubin",
             "res-usage", "tegrastats", "torch-trace", "stdout", "other"}


def classify_kind(f: Path) -> str:
    """Map a file to protocol §5's `kind` enum. The enum is closed: anything unrecognised is
    `other`, never the file extension."""
    n = f.name
    base = n[:-4] if n.endswith(".zst") else n
    if base.endswith(".nsys-rep"):
        return "nsys-rep"
    if base.endswith(".sqlite"):
        return "nsys-sqlite"
    if base.endswith(".ncu-rep"):
        return "ncu-rep"
    if f.parent.name == "ncu" and base.endswith(".csv"):
        return "ncu-csv"
    if base.endswith((".sass", ".ptx", ".cubin")):
        return base.rsplit(".", 1)[1]
    if "tegrastats" in base:
        return "tegrastats"
    if "trace" in base and base.endswith(".json"):
        return "torch-trace"
    if base.endswith((".txt", ".log", ".out")) or "stdout" in base:
        return "stdout"
    return "other"


def compress_and_record(path: Path, level: int = 19) -> dict:
    """zstd a raw file and return its ORIGINALS.json entry.

    Protocol §3.5 says to scan for secrets and hash *before* compressing: a scan of the
    compressed bytes finds nothing by construction and is worthless. Both happen here, on the
    original, and the scan result travels with the entry.
    """
    import subprocess

    scan = scan_for_secrets(path)
    if scan.startswith("MATCH"):
        raise SystemExit(f"{path}: {scan} — not compressing or committing it")
    entry = {"original_name": path.name, "original_sha256": sha256(path),
             "original_bytes": path.stat().st_size,
             "original_secrets_scan": scan,
             "compressed_name": path.name + ".zst", "tool": "zstd", "level": level,
             "command": f"zstd -{level} --rm {path.name}"}
    subprocess.run(["zstd", f"-{level}", "-q", "-f", "--rm", str(path)], check=True)
    z = path.with_name(path.name + ".zst")
    entry["compressed_bytes"] = z.stat().st_size
    entry["compressed_sha256"] = sha256(z)
    entry["ratio"] = round(entry["original_bytes"] / entry["compressed_bytes"], 1)
    return entry


def collect_raw(outdir: Path, produced_by: str, kind_of=None,
                dirs=("raw", "ncu", "nsys"), produced_by_map=None,
                originals: dict | None = None) -> list[dict]:
    """Every file under `<outdir>/raw/`, hashed, sized and scanned, as meta.json wants it.

    Written because `"raw_files": []` was hard-coded and shipped three times (found in the
    T002 review, reintroduced verbatim in run_e4.py, found again in the T003 review). Scripts
    call this instead of writing the list by hand, so the failure cannot recur silently.
    """
    base = Path(outdir)
    out = []
    for d in dirs:
        root = base / d
        if not root.exists():
            continue
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            kind = kind_of(f) if kind_of else classify_kind(f)
            rel = str(f.relative_to(base))
            by = (produced_by_map or {}).get(rel, produced_by)
            scan = scan_for_secrets(f)
            e = raw_file_entry(f, base, kind, by, secrets_scan=scan)
            if scan.startswith("MATCH"):
                # §3.5: "A raw file that was produced but could not be committed (... or
                # failed the secrets scan) is still listed, with committed: false, the
                # reason, and the command that regenerates it."
                e["committed"] = False
                e["not_committed_reason"] = (
                    "failed the secrets scan. nsys snapshots system-wide process metadata "
                    "into its report, which on this machine includes a session token from "
                    "the agent's own environment. Scrubbing the profiled process's "
                    "environment does not remove it (verified with `env -i`), so the binary "
                    "report cannot be committed. The exported CSVs under nsys/ carry the "
                    "same measurements and are clean.")
                e["regenerate_with"] = by
            # protocol §5: a compressed file carries its compression command and the
            # sha256 + bytes of the uncompressed original IN THE ENTRY, not only in a
            # sidecar file.
            o = (originals or {}).get(f.name)
            if o:
                # scanning compressed bytes proves nothing; report the original's scan
                e["secrets_scan"] = o.get("original_secrets_scan",
                                          "not scanned before compression")
                e["compression"] = {"tool": o["tool"], "level": o["level"],
                                    "command": o.get("command",
                                                     f"zstd -{o['level']} --rm "
                                                     f"{o['original_name']}")}
                e["original"] = {"name": o["original_name"],
                                 "sha256": o["original_sha256"],
                                 "bytes": o["original_bytes"]}
            out.append(e)
    return out


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
    # /sys/kernel/debug is root-only. The narrow sudoers rule the owner installed permits
    # `sh` but not `cat`, so the read goes through `sh -c` -- `sudo cat` fails with
    # "a password is required" and would silently record an empty EMC rate.
    st["emc_rate_hz"] = sh(
        f"{sudo} sh -c 'cat /sys/kernel/debug/bpmp/debug/clk/emc/rate'")
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


# --------------------------------------------------------------------------- CPU utilisation


def _proc_stat() -> dict[str, tuple[int, int]]:
    """{'cpu'|'cpu0'|...: (busy_jiffies, total_jiffies)} from /proc/stat.

    busy excludes idle and iowait; both are time the core was not executing.
    """
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if not line.startswith("cpu"):
            break
        parts = line.split()
        vals = [int(v) for v in parts[1:]]
        total = sum(vals)
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)   # idle + iowait
        out[parts[0]] = (total - idle, total)
    return out


class CpuSampler:
    """Per-core utilisation from /proc/stat, sampled on a thread.

    §13.6 asks for CPU utilisation during each phase. `time.thread_time()` sees only
    the calling thread, and `time.process_time()` only this process: neither sees the
    CUDA driver's worker threads or anything else on the other 11 cores. This reads the
    kernel's own accounting, so the numbers cover the whole machine.
    """

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self._rows: list[dict] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _run(self, phase: str):
        prev, t_prev = _proc_stat(), time.perf_counter()
        t0 = t_prev
        while not self._stop.wait(self.interval_s):
            cur, t_cur = _proc_stat(), time.perf_counter()
            for cpu, (busy, total) in cur.items():
                db, dt = busy - prev[cpu][0], total - prev[cpu][1]
                if dt > 0:
                    self._rows.append({"phase": phase, "t_s": round(t_cur - t0, 3),
                                       "cpu": cpu, "busy_pct": round(100 * db / dt, 2)})
            prev, t_prev = cur, t_cur

    def start(self, phase: str):
        self._stop.clear()
        self._t = threading.Thread(target=self._run, args=(phase,), daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)
            self._t = None

    @property
    def rows(self) -> list[dict]:
        return self._rows

    def summary(self, phase: str) -> list[dict]:
        """One row per scope: the aggregate 'cpu' line, the busiest core, and each core."""
        by_cpu: dict[str, list[float]] = {}
        for r in self._rows:
            if r["phase"] == phase:
                by_cpu.setdefault(r["cpu"], []).append(r["busy_pct"])
        if not by_cpu:
            return []
        out = []
        n_cores = len(by_cpu) - 1                       # minus the aggregate 'cpu' line
        for cpu, vals in sorted(by_cpu.items()):
            scope = "system_all_cores" if cpu == "cpu" else cpu
            s = summarize(vals)
            row = {"phase": phase, "scope": scope,
                   "pct": round(s["median_ms"], 2), "median_pct": round(s["median_ms"], 2),
                   "p5_pct": round(s["p5_ms"], 2), "p95_pct": round(s["p95_ms"], 2),
                   "n": s["n"]}
            if cpu == "cpu":
                # 'cpu' is the average over cores; x n_cores is "cores fully busy".
                row["cores_busy"] = round(n_cores * s["median_ms"] / 100.0, 2)
            else:
                row["cores_busy"] = ""
            out.append(row)
        # busiest core, per sample, so the peak is not hidden by the average
        per_t: dict[float, float] = {}
        for r in self._rows:
            if r["phase"] == phase and r["cpu"] != "cpu":
                per_t[r["t_s"]] = max(per_t.get(r["t_s"], 0.0), r["busy_pct"])
        if per_t:
            s = summarize(list(per_t.values()))
            out.insert(1, {"phase": phase, "scope": "busiest_core",
                           "pct": round(s["median_ms"], 2),
                           "median_pct": round(s["median_ms"], 2),
                           "p5_pct": round(s["p5_ms"], 2), "p95_pct": round(s["p95_ms"], 2),
                           "n": s["n"], "cores_busy": ""})
        return out


CPU_UTIL_FIELDS = ["phase", "scope", "pct", "median_pct", "p5_pct", "p95_pct", "n",
                   "cores_busy"]


# --------------------------------------------------------------------------- summary.md


def write_summary(outdir: Path, title: str, sections: list[tuple[str, str]],
                  not_measured: list[str] | None = None) -> Path:
    """The per-directory summary.md protocol §5 requires.

    `sections` is [(heading, body)]. `not_measured` lists the §13.8 quantities that are
    absent from model_inputs.csv, which the checker forbids carrying a non-finite value:
    the protocol says they must be named here instead.
    """
    lines = [f"# {title}", "",
             f"Run directory: `{outdir.name}`  ",
             f"Written by `bench/{Path(sys.argv[0]).name}` at "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}.", ""]
    for heading, body in sections:
        lines += [f"## {heading}", "", body.strip(), ""]
    lines += ["## Not measured", ""]
    if not_measured:
        lines += [f"- {x}" for x in not_measured]
    else:
        lines.append("Every quantity this part is responsible for is present in "
                     "`model_inputs.csv`.")
    lines.append("")
    p = outdir / "summary.md"
    p.write_text("\n".join(lines))
    return p
