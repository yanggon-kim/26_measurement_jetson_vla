"""T004 step 1 — turn an nsys capture into the timeline tables the task asks for.

From the exports `bench/run_nsys.sh` produces (`cuda_gpu_trace`, `nvtx_gpu_proj_trace`,
`cuda_gpu_kern_sum`) this writes:

  `kernels_per_region.csv`  region, kernel, launches, launches_per_step, total_ms, share
  `gaps.csv`                the gap histogram between consecutive kernels inside each region
  `timeline_summary.csv`    per region: wall time, kernel time (union), GPU idle and its share,
                            kernels, kernels per denoising step

"GPU idle" is region wall time minus the **union** of kernel intervals, so kernels that overlap
are not counted twice. That is the quantity T004's report has to answer (a) with — "kernels
executing vs GPU waiting" — and the same definition E11f uses for `gpu_idle_ae_total`.

Protocol §2: ncu's `Duration` is never a latency. Everything here comes from nsys.

    python bench/parse_nsys.py <run-dir> --tag eager_N4 --steps 4
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO, collect_raw, sh_out,  # noqa: E402
                    software_versions, write_json)


def num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def load(nsys_dir: Path, tag: str, kind: str):
    """Large exports are committed zstd-compressed (§6); read either form."""
    import io
    import subprocess as _sp
    pat = str(nsys_dir / f"{tag}*{kind}*.csv")
    hits = sorted(glob.glob(pat)) or sorted(glob.glob(pat + ".zst"))
    if not hits:
        return None, None
    f = hits[0]
    if f.endswith(".zst"):
        out = _sp.run(["zstd", "-dc", f], capture_output=True, timeout=600)
        return list(csv.DictReader(
            io.StringIO(out.stdout.decode("utf-8", "replace")))), Path(f).name
    return list(csv.DictReader(open(f))), Path(f).name


def span(r):
    """nsys names these columns differently in cuda_gpu_trace and nvtx_gpu_proj_trace."""
    st = num(r.get("Start (ns)") or r.get("Projected Start (ns)") or r.get("Start"))
    du = num(r.get("Duration (ns)") or r.get("Projected Duration (ns)") or r.get("Duration"))
    return (st, st + du) if st is not None and du is not None else (None, None)


def union_ms(intervals):
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    busy, cs, ce = 0.0, ivs[0][0], ivs[0][1]
    for a, b in ivs[1:]:
        if a <= ce:
            ce = max(ce, b)
        else:
            busy += ce - cs
            cs, ce = a, b
    busy += ce - cs
    return busy / 1e6


_MODELS = ("groot_n17", "groot_n16", "groot_n15", "smolvla", "pi05", "pi0")


def _model_from_dirname(name: str) -> str:
    """<date>_E5_<model>_<mode>[_N<n>] -> <model>. Longest match first: 'pi05' contains
    'pi0', and 'groot_n17' contains neither, so ordering decides correctness."""
    for m in _MODELS:
        if f"_{m}_" in name or name.endswith(f"_{m}"):
            return m
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--inferences", type=int, default=2)
    args = ap.parse_args()
    d = Path(args.rundir)
    nsys = d / "nsys"
    trace, tname = load(nsys, args.tag, "cuda_gpu_trace")
    proj, pname = load(nsys, args.tag, "nvtx_gpu_proj_trace")
    if trace is None:
        print(f"no cuda_gpu_trace export for {args.tag}")
        return 1

    kern = []
    for r in trace:
        a, b = span(r)
        nm = r.get("Name") or ""
        if a is None or not nm:
            continue
        kern.append((a, b, nm))
    kern.sort()
    print(f"  {len(kern)} kernels from {tname}")

    regions = collections.defaultdict(list)
    if proj:
        for r in proj:
            # projected ranges carry their NVTX domain as a ":" prefix
            nm = (r.get("Name") or r.get("Range") or "").lstrip(":").strip()
            a, b = span(r)
            if a is None or not nm:
                continue
            if nm.startswith(("ae_step_", "ae_layer_")):
                continue                   # bracket-free alias: same span, counted once
            base = nm.split("[")[0] if "[" in nm else nm
            regions[base].append((a, b))
        print(f"  NVTX ranges from {pname}: "
              + ", ".join(f"{k}x{len(v)}" for k, v in sorted(regions.items())))
    else:
        print("  no nvtx_gpu_proj_trace export — region tables will be empty "
              "(compiled captures run with --no-regions)")

    krows, trows, grows = [], [], []

    def union_and_gaps(ivs, lo, hi):
        inside = [(max(a, lo), min(b, hi), nm) for a, b, nm in ivs if b > lo and a < hi]
        busy = union_ms([(a, b) for a, b, _ in inside])
        seq = sorted((a, b) for a, b, _ in inside)
        gaps = [(seq[i + 1][0] - seq[i][1]) / 1e3 for i in range(len(seq) - 1)
                if seq[i + 1][0] > seq[i][1]]
        return inside, busy, gaps

    # A region occurs once per inference (vlm_prefix), several times per inference (vision:
    # one per camera; ae_step: one per denoising step). Taking min(start)..max(end) across ALL
    # instances would span the whole capture and count other regions' kernels as this one's —
    # vision came out at 1072 ms that way, against its real 241 ms. So the instances are
    # grouped into `inferences` consecutive chunks and each chunk is measured on its own.
    for region, spans in sorted(regions.items()):
        spans = sorted(spans)
        per_inf = max(1, len(spans) // max(1, args.inferences))
        chunks = [spans[i:i + per_inf] for i in range(0, len(spans), per_inf)]
        walls, busies, allgaps, nk = [], [], [], 0
        for ch in chunks:
            lo, hi = ch[0][0], ch[-1][1]
            inside, busy, gaps = union_and_gaps(kern, lo, hi)
            walls.append((hi - lo) / 1e6)
            busies.append(busy)
            allgaps += gaps
            nk += len(inside)
        wall = sum(walls) / len(walls)
        busy = sum(busies) / len(busies)
        idle = wall - busy
        unit = "denoising step" if region.startswith("ae_step") else "inference"
        div = args.steps if region.startswith("ae_step") else 1
        trows.append({"region": region, "instances": len(spans),
                      "instances_per_inference": per_inf,
                      "wall_ms": round(wall, 4), "kernel_union_ms": round(busy, 4),
                      "gpu_idle_ms": round(idle, 4),
                      "gpu_idle_frac": round(idle / wall, 4) if wall else "",
                      "n_kernels": round(nk / len(chunks), 1),
                      "kernels_per_unit": round(nk / len(chunks) / div, 2), "unit": unit,
                      "note": "per inference, averaged over the captured inferences"})
        if allgaps:
            allgaps.sort()
            for lo_us, hi_us in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 50),
                                 (50, 100), (100, 1e9)]:
                c = sum(1 for g in allgaps if lo_us <= g < hi_us)
                if c:
                    grows.append({"region": region,
                                  "gap_bucket_us": f"{lo_us}-{'inf' if hi_us > 1e8 else hi_us}",
                                  "count": c,
                                  "total_us": round(sum(g for g in allgaps
                                                        if lo_us <= g < hi_us), 2)})
            trows[-1]["gap_total_ms"] = round(sum(allgaps) / 1e3 / len(chunks), 4)
            trows[-1]["gap_median_us"] = round(allgaps[len(allgaps) // 2], 3)
            trows[-1]["n_gaps"] = len(allgaps)
        by, tot = collections.Counter(), collections.Counter()
        for ch in chunks:
            inside, _, _ = union_and_gaps(kern, ch[0][0], ch[-1][1])
            for a, b, nm in inside:
                by[nm] += 1
                tot[nm] += (b - a) / 1e6
        s_tot = sum(tot.values()) or 1.0
        for nm, n in by.most_common(20):
            krows.append({"region": region, "kernel": nm[:120], "launches": n,
                          "launches_per_unit": round(n / len(chunks) / div, 2),
                          "total_ms": round(tot[nm] / len(chunks), 5),
                          "share_of_region_kernel_time": round(tot[nm] / s_tot, 5)})

    if kern:
        lo, hi = kern[0][0], max(b for _, b, _ in kern)
        inside, busy, gaps = union_and_gaps(kern, lo, hi)
        wall = (hi - lo) / 1e6
        trows.append({"region": "whole_capture", "instances": 1,
                      "instances_per_inference": 1, "wall_ms": round(wall, 4),
                      "kernel_union_ms": round(busy, 4),
                      "gpu_idle_ms": round(wall - busy, 4),
                      "gpu_idle_frac": round((wall - busy) / wall, 4) if wall else "",
                      "n_kernels": len(inside),
                      "kernels_per_unit": round(len(inside) / max(1, args.inferences), 2),
                      "unit": "inference",
                      "note": f"the whole capture window, {args.inferences} inferences"})

    for name, rows_ in (("kernels_per_region.csv", krows), ("timeline_summary.csv", trows),
                        ("gaps.csv", grows)):
        if not rows_:
            continue
        keys = sorted({k for r in rows_ for k in r})
        with (d / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows([{k: r.get(k, "") for k in keys} for r in rows_])

    write_json(d / "nsys" / f"{args.tag}_summary.json", {
        "tag": args.tag, "steps": args.steps, "inferences": args.inferences,
        "n_kernels": len(kern), "regions": {k: len(v) for k, v in regions.items()},
        "timeline": trows,
        "notes": [
            "GPU idle is region wall time minus the UNION of kernel intervals, so overlapping "
            "kernels are not double-counted.",
            "All durations are nsys durations. Protocol §2: ncu Duration is never a latency.",
            "Compiled captures run with --no-regions because regions.py's context managers "
            "graph-break the model, so only whole_capture is available for those.",
        ]})
    # ---- meta.json (protocol §5) --------------------------------------------------
    # The E5 runs had none. bench/nsys_target.py prints a JSON line with the gate, the
    # model, N and the rung, and run_nsys.sh saves its stdout, so the run describes
    # itself -- it just was not being collected. GR00T's target (groot_phases
    # --ncu-region) prints its gate too. Without this a capture has a timeline and no
    # record of the device state it was taken in, which protocol §5 does not allow and
    # which "a latency without its power mode is not a result" covers directly.
    target = {}
    for log in sorted((d / "raw").glob("*stdout*")):
        for line in log.read_text(errors="replace").splitlines():
            line = line.strip()
            if line.startswith("{") and '"gate"' in line:
                try:
                    target = json.loads(line)
                except ValueError:
                    pass
            elif line.startswith("gate: {"):
                try:
                    target = {"gate": json.loads(line[6:])}
                except ValueError:
                    pass
    sw = software_versions()
    write_json(d / "meta.json", {
        "experiment": "E5 (nsys timeline)", "task": "T004/T008/T009",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tag": args.tag,
        # The stdout JSON carries the model for runs taken after nsys_target gained
        # --model; older captures predate it, so the directory name is the fallback
        # rather than leaving the field blank in the artefact protocol §5 governs.
        "model": target.get("model") or _model_from_dirname(d.name),
        "compile_mode": target.get("compile_mode"),
        "rung": target.get("rung") or "none (eager, no ladder rung)", "rung_detail": target.get("rung_detail", ""),
        "num_denoise_steps": target.get("steps", args.steps),
        "inferences_captured": target.get("inferences", args.inferences),
        "regions_present": sorted(regions),
        "gate": target.get("gate", {}),
        "power_mode": (target.get("gate") or {}).get("power_mode", ""),
        "torch": sw.get("torch"), "python": sw.get("python"),
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "command": " ".join(sys.argv),
        "raw_files": [],
        "notes": [
            "The .nsys-rep and .sqlite are NOT committed on this device: they embed the "
            "system-wide process environment including a session token, verified to "
            "survive `env -i`. They stay in scratch/ and are listed committed:false "
            "(protocol §3.5). The exported CSVs carry everything the analysis reads.",
            "nsys's own tracing inflates region wall time by about x1.53 on this device, "
            "so gpu_idle_frac computed from traced wall time OVERSTATES idleness. Where "
            "gpu_idle_frac_corrected is present, that is the number to trust.",
        ],
    })
    meta = json.loads((d / "meta.json").read_text())
    meta["raw_files"] = collect_raw(d, meta["command"])
    write_json(d / "meta.json", meta)

    for t in trows:
        print(f"  {t['region']:14s} wall {t['wall_ms']:9.3f} ms  busy "
              f"{t['kernel_union_ms']:9.3f}  idle {t['gpu_idle_ms']:8.3f} "
              f"({100*(t['gpu_idle_frac'] or 0):5.1f} %)  {t['n_kernels']:7.1f} kernels")
    print(f"wrote {d}/kernels_per_region.csv, timeline_summary.csv, gaps.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
