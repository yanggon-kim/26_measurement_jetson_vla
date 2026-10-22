"""meta.json writer (docs/02_protocol.md §5) and raw-file bookkeeping (§3.5, §6).

Library:
  device_state()                 power mode, clocks, SM count, versions -- every §1 field, read from this device
  raw_entry(path, kind, ...)      one raw_files entry (sha256, bytes, secrets_scan)
  compress(path, scratch_dir)     zstd -19 for files > 1 MB: hash the original first, move it to scratch/
CLI (finalise a result directory):
  python bench/meta.py --run results/<run> --fields fields.json [--scan-clean]
    fields.json holds the experiment-specific fields (experiment, task, model, command, notes, ...);
    every file under <run>/raw/ is compressed if > 1 MB, hashed and listed in raw_files.
Root-only reads (EMC rate, jetson_clocks --show) go through env/rootdo.sh.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTDO = os.path.join(REPO, "env", "rootdo.sh")
GPU_DEVFREQ = "/sys/class/devfreq/gpu-gpc-0"
KIND_BY_EXT = [(".ncu-rep", "ncu-rep"), (".nsys-rep", "nsys-rep"), (".sqlite", "nsys-sqlite"), (".sass", "sass"),
               (".ptx", "ptx"), (".cubin", "cubin"), (".stdout.log", "stdout"), ("tegrastats", "tegrastats"),
               ("power", "power-log")]


def sh(cmd, root=False):
    try:
        args = [ROOTDO] + cmd if root else cmd
        return subprocess.run(args, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as exc:
        return f"n/a: {type(exc).__name__}: {exc}"


def rd(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception as exc:
        return f"n/a: {exc}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def device_state():
    import torch  # noqa: local import keeps the CLI usable for hashing without CUDA

    q = sh(["/usr/sbin/nvpmodel", "-q"])
    m = re.search(r"NV Power Mode:\s*(\S+)\s*\n\s*(\d+)", q)
    jc = sh(["jetson_clocks", "--show"], root=True)
    gpu_line = next((l for l in jc.splitlines() if l.startswith("gpu-gpc-0")), "")
    locked = "n/a: no gpu-gpc-0 line"
    if gpu_line:
        mn = re.search(r"MinFreq=(\d+)", gpu_line).group(1)
        mx = re.search(r"MaxFreq=(\d+)", gpu_line).group(1)
        locked = mn == mx
    emc = sh(["cat", "/sys/kernel/debug/bpmp/debug/clk/emc/rate"], root=True)
    dpkg = sh(["bash", "-c", "dpkg -l | grep -E 'nvidia-l4t-core|nvidia-jetpack|cuda-toolkit|tensorrt|cudnn' "
                             "| awk '{print $2\"=\"$3}'"])
    p = torch.cuda.get_device_properties(0)
    try:
        import torchvision
        tv = torchvision.__version__
    except Exception as exc:
        tv = f"n/a: {exc}"
    try:
        import triton
        tr = triton.__version__
    except Exception as exc:
        tr = f"n/a: {exc}"
    temps = {rd(f"/sys/class/thermal/{z}/type"): rd(f"/sys/class/thermal/{z}/temp")
             for z in sorted(os.listdir("/sys/class/thermal")) if z.startswith("thermal_zone")}
    return {
        "power_mode": {"id": m.group(2) if m else "n/a", "name": m.group(1) if m else "n/a", "raw": q},
        "jetson_clocks": {"applied_gpu_locked": locked, "show": jc},
        "sm_count": p.multi_processor_count,
        "gpu_hz": int(rd(f"{GPU_DEVFREQ}/cur_freq")),
        "gpu_hz_source": f"{GPU_DEVFREQ}/cur_freq",
        "emc_hz": int(emc) if emc.isdigit() else f"n/a: {emc}",
        "emc_hz_source": "/sys/kernel/debug/bpmp/debug/clk/emc/rate (via env/rootdo.sh)",
        "gpu_arch": "sm_%d%d" % torch.cuda.get_device_capability(),
        "l4t": rd("/etc/nv_tegra_release").splitlines()[0],
        "jetpack": dpkg,
        "cuda": f"torch.version.cuda={torch.version.cuda}; toolkit={sh(['bash', '-c', 'nvcc --version | tail -1'])}",
        "driver": rd("/proc/driver/nvidia/version").splitlines()[0],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": tv,
        "triton": tr,
        "kernel": os.uname().release,
        "thermal_zones_mC": temps,
        "swap": sh(["bash", "-c", "free -b | grep -i swap"]),
    }


def kind_of(path):
    for ext, k in KIND_BY_EXT:
        if ext in os.path.basename(path):
            return k
    return "other"


def compress(path, scratch_dir):
    """zstd -19 a raw file > 1 MB; returns (compressed path, original info). The original goes to scratch."""
    orig = {"name": os.path.basename(path), "sha256": sha256(path), "bytes": os.path.getsize(path)}
    long = ["--long=27"] if orig["bytes"] > 100e6 else []
    cmd = ["zstd", "-19", "-T0", "-q", "-f"] + long + [path, "-o", path + ".zst"]
    subprocess.run(cmd, check=True)
    os.makedirs(scratch_dir, exist_ok=True)
    shutil.move(path, os.path.join(scratch_dir, os.path.basename(path)))
    return path + ".zst", orig, " ".join(cmd)


def raw_entry(path, run_dir, kind=None, produced_by="", secrets_scan="n/a: text produced by the agent's own code"):
    return {"path": os.path.relpath(path, run_dir), "sha256": sha256(path), "bytes": os.path.getsize(path),
            "kind": kind or kind_of(path), "produced_by": produced_by, "secrets_scan": secrets_scan}


def finalise(run_dir, fields, scan_status):
    run_dir = os.path.abspath(run_dir)
    scratch = os.path.join(REPO, "scratch", os.path.basename(run_dir), "raw_originals")
    raw_dir = os.path.join(run_dir, "raw")
    entries = []
    produced = fields.pop("produced_by", {})
    extra_uncommitted = fields.pop("uncommitted_raw", [])
    for root, _, files in os.walk(raw_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            pb = next((v for k, v in produced.items() if k in fn), "")
            if os.path.getsize(p) > 1_000_000 and not fn.endswith(".zst"):
                zp, orig, cmd = compress(p, scratch)
                e = raw_entry(zp, run_dir, kind_of(fn), pb, scan_status.get(fn, "n/a"))
                e["compression"] = {"tool": "zstd", "level": 19, "command": cmd}
                e["original"] = orig
            else:
                e = raw_entry(p, run_dir, kind_of(fn), pb, scan_status.get(fn, "n/a"))
            entries.append(e)
    entries += extra_uncommitted
    meta = {"date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    meta.update(fields)
    meta["raw_files"] = entries
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    nulls = [k for k, v in meta.items() if v is None]
    if nulls:
        raise SystemExit(f"meta.json has null fields: {nulls}")
    print(f"meta.json: {len(entries)} raw files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fields", required=True, help="JSON file with the experiment-specific fields")
    ap.add_argument("--device-state", action="store_true", help="add device_state() fields (needs CUDA)")
    ap.add_argument("--scan", help="secrets_scan.py CSV: files without hits are marked clean")
    a = ap.parse_args()
    fields = json.load(open(a.fields))
    if a.device_state:
        st = device_state()
        for k, v in st.items():
            fields.setdefault(k, v)
    scan_status = {}
    if a.scan:
        import csv
        hits = {}
        for r in csv.DictReader(open(a.scan)):
            fn = os.path.basename(r["file"])
            hits[fn] = hits.get(fn, False) or r["found"] == "True"
        scan_status = {fn: ("NOT CLEAN" if h else "clean") for fn, h in hits.items()}
    finalise(a.run, fields, scan_status)


if __name__ == "__main__":
    main()
