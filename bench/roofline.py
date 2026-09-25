"""T004 step 3 / E6 — the roofline, against this Orin's own peaks.

Plan E6 fixes the columns: `model, region_or_kernel, flops, bytes, ms, intensity,
achieved_TFLOPps, achieved_GBps, pct_peak_compute, pct_peak_bw, source` with
`source in {analytic, ncu}`, plus the floor and gap ratio of protocol §9.

Three rules this enforces rather than leaves to the writer:

  * **Times are nsys region or kernel times, never ncu `Duration`** (protocol §2, §3.3(3)).
  * **`ncu` rows take bytes from the L2 sysmem-miss sectors**, never from `Memory Throughput`,
    which is L2 throughput on this device (§3.3 [Orin]).
  * **The roof is the Orin's own** — E1's measured peaks, with the spec row alongside and
    labelled, per protocol §8. Not any other GPU's.

    python bench/roofline.py --outdir results/<date>_E6_pi0 \\
        --floor results/20260921_E3_pi0_eager_N10/floor.csv \\
        --regions results/20260921_E3_pi0_eager_N4/latency_summary.csv \\
        --ncu results/<date>_E5_pi0_eager_N4
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, collect_raw, sh_out, software_versions, write_json  # noqa: E402

E1 = REPO / "results" / "20260922_E1_peak" / "hw_constants.json"
SPEC_BW, SPEC_TF = 204.8, 42.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--floor", required=True, help="a floor.csv with the analytic bytes/FLOPs")
    ap.add_argument("--regions", action="append", default=[],
                    help="latency_summary.csv giving each region's measured ms (repeatable)")
    ap.add_argument("--ncu", action="append", default=[],
                    help="an E5 run dir with ncu/kernels.csv (repeatable)")
    ap.add_argument("--model", default="pi0")
    ap.add_argument("--vision-flops", default="",
                    help="JSON from bench/vision_flops.py. floor.csv carries flops=0 for the "
                         "`vision` region with the note 'FLOPs not counted analytically', "
                         "which leaves vision without an operational intensity and therefore "
                         "out of the roofline entirely -- the gap vision_flops.py was written "
                         "to close. Its `bytes_as_loaded_fp32` is cross-checked against "
                         "floor.csv's bytes_read and the run refuses if they disagree, since "
                         "two independent derivations of the same quantity agreeing is what "
                         "makes it safe to take the FLOPs from one and the bytes from the "
                         "other.")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    hw = json.loads(E1.read_text())["measured_peaks"]
    peak_bw = hw["peak_memory_bandwidth_GBps"]
    peak_tf = hw["peak_bf16_TFLOPps"]
    print(f"peaks: measured {peak_bw} GB/s / {peak_tf} TFLOP/s; spec {SPEC_BW} / {SPEC_TF}")

    analytic = {}
    for r in csv.DictReader(Path(args.floor).open()):
        if r.get("peak_source") != "measured":
            continue
        analytic[r["region"]] = {"bytes": float(r["bytes_read"] or 0),
                                 "flops": float(r["flops"] or 0),
                                 "floor_ms": float(r["floor_ms"] or 0),
                                 "bound": r.get("bound", "")}
    times, step_times = {}, []
    for f in args.regions:
        for r in csv.DictReader(Path(f).open()):
            reg = r.get("region")
            if reg and r.get("median_ms"):
                times.setdefault(reg, float(r["median_ms"]))
                # floor.csv names the region `ae_step` (ONE denoising step); the measured
                # summary names them `ae_step[0]`..`ae_step[N-1]`. Without this the ae_step
                # row silently got no `ms` and was dropped from the roofline -- the region
                # the whole PIM study is about.
                if re.fullmatch(r"ae_step\[\d+\]", reg):
                    step_times.append((reg, float(r["median_ms"])))
    if step_times and "ae_step" not in times:
        vals = sorted(v for _, v in step_times)
        med = vals[len(vals) // 2] if len(vals) % 2 else (
            (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2)
        times["ae_step"] = med
        first = dict(step_times).get("ae_step[0]")
        print(f"  ae_step: median over {len(vals)} measured steps = {med:.4f} ms"
              + (f"  (step 0 alone is {first:.4f} ms, {100*(first/med-1):+.1f} % -- a "
                 f"first-step effect, which is why the MEDIAN across steps is used rather "
                 f"than step 0)" if first else ""))
    if args.vision_flops:
        vf = json.loads(Path(args.vision_flops).read_text())
        v = analytic.get("vision")
        if v is not None:
            declared = float(vf["bytes_as_loaded_fp32"])
            if abs(declared - v["bytes"]) > 1:
                raise SystemExit(
                    f"vision bytes disagree: floor.csv {v['bytes']:.0f} vs "
                    f"vision_flops.py {declared:.0f}. Refusing to combine FLOPs from one "
                    f"derivation with bytes from another that does not agree with it.")
            v["flops"] = float(vf["flops_vision_region"])
            print(f"  vision FLOPs {v['flops']:.0f} from {args.vision_flops}; bytes agree "
                  f"with floor.csv at {v['bytes']:.0f} B")

    missing = [r for r in analytic if r not in times]
    if missing:
        print(f"  *** analytic regions with NO measured time, dropped from the roofline: "
              f"{missing}")
    print(f"  analytic regions: {sorted(analytic)}; measured regions: {len(times)}")

    rows = []

    def add(name, flops, nbytes, ms, source, note="", floor_ms=None):
        if not ms:
            return
        sec = ms / 1e3
        tf = flops / sec / 1e12 if flops else 0.0
        gb = nbytes / sec / 1e9 if nbytes else 0.0
        rows.append({
            "model": args.model, "region_or_kernel": name,
            "flops": int(flops), "bytes": int(nbytes), "ms": round(ms, 6),
            "intensity": round(flops / nbytes, 4) if nbytes else "",
            "achieved_TFLOPps": round(tf, 5), "achieved_GBps": round(gb, 3),
            "pct_peak_compute": round(100 * tf / peak_tf, 3),
            "pct_peak_bw": round(100 * gb / peak_bw, 3),
            "pct_spec_compute": round(100 * tf / SPEC_TF, 3),
            "pct_spec_bw": round(100 * gb / SPEC_BW, 3),
            "floor_ms": round(floor_ms, 6) if floor_ms else "",
            "gap_ratio": round(ms / floor_ms, 3) if floor_ms else "",
            "roof_source": "measured (T005/E1)", "source": source, "note": note})

    for reg, a in analytic.items():
        ms = times.get(reg) or times.get(f"{reg}_wall")
        add(reg, a["flops"], a["bytes"], ms, "analytic",
            note=f"analytic bytes and FLOPs from floor.csv; ms is the §2 region time. "
                 f"floor bound={a['bound']}", floor_ms=a["floor_ms"])

    for nd in args.ncu:
        kf = Path(nd) / "ncu" / "kernels.csv"
        if not kf.exists():
            print(f"  {kf} missing")
            continue
        for r in csv.DictReader(kf.open()):
            rb = float(r.get("offchip_read_bytes") or 0)
            wb = float(r.get("offchip_write_bytes") or 0)
            if not (rb or wb):
                continue
            # protocol §3.3(3): achieved GB/s must use an nsys duration, never ncu Duration.
            # Without one we emit the bytes and leave the rates blank rather than invent them.
            rows.append({
                "model": args.model,
                "region_or_kernel": (r.get("kernel_name") or "")[:110],
                "flops": "", "bytes": int(rb + wb), "ms": "", "intensity": "",
                "achieved_TFLOPps": "", "achieved_GBps": "", "pct_peak_compute": "",
                "pct_peak_bw": "", "pct_spec_compute": "", "pct_spec_bw": "",
                "floor_ms": "", "gap_ratio": "", "roof_source": "measured (T005/E1)",
                "source": "ncu",
                "note": f"off-chip bytes from L2 sysmem-miss sectors x 32 B (read) + "
                        f"op_write.sum x 32 B (write), region {r.get('region', '')}. "
                        f"Rates blank: §3.3(3) requires an nsys duration, never ncu Duration."})

    if not rows:
        print("nothing to write — need a floor.csv and at least one latency_summary.csv")
        return 1
    keys = list(rows[0].keys())
    with (out / "roofline.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in keys} for r in rows])

    print(f"\n{'region':14s} {'ms':>10s} {'intensity':>10s} {'%peak_bw':>9s} "
          f"{'%peak_cmp':>10s} {'xfloor':>7s}")
    for r in rows:
        if r["source"] != "analytic":
            continue
        print(f"{r['region_or_kernel']:14s} {r['ms']:>10} {r['intensity']:>10} "
              f"{r['pct_peak_bw']:>9} {r['pct_peak_compute']:>10} {r['gap_ratio']:>7}")

    sw = software_versions()
    write_json(out / "meta.json", {
        "experiment": "E6", "task": "T004",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model, "precision": "bfloat16",
        "peaks_measured": {"GBps": peak_bw, "TFLOPps": peak_tf, "source": str(E1)},
        "peaks_spec": {"GBps": SPEC_BW, "TFLOPps": SPEC_TF},
        "inputs": {"floor": args.floor, "regions": args.regions, "ncu": args.ncu},
        "bench_commit": sh_out(f"git -C {REPO} rev-parse HEAD"),
        "bench_files_sha256": sw.get("bench_files_sha256"),
        "torch": sw.get("torch"), "python": sw.get("python"),
        "command": f"python bench/roofline.py --outdir {args.outdir}",
        "throttled": False,
        "throttled_evidence": "derived file; each source run carries its own gate record.",
        "raw_files": [],
        "notes": [
            "The roof is this Orin's own measured peaks (protocol §8); the spec percentages "
            "are given alongside and labelled, never used as the roof.",
            "[Orin] ncu rows take bytes from the L2 sysmem-miss sectors, never from Memory "
            "Throughput, which is L2 throughput here.",
            "Times are §2 region times or nsys kernel times; ncu Duration is never used "
            "(protocol §2, §3.3(3)). Where no nsys duration exists the ncu rows carry bytes "
            "with blank rates rather than a rate derived from ncu Duration.",
            "Bandwidth on this device varies run to run by up to 1.19x (T005/E1), so every "
            "pct_peak_bw carries roughly +/-8 %.",
        ],
    })
    meta = json.loads((out / "meta.json").read_text())
    meta["raw_files"] = collect_raw(out, meta["command"])
    write_json(out / "meta.json", meta)
    print(f"\nwrote {out}/roofline.csv ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
