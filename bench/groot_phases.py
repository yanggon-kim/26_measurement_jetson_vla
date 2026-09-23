"""GR00T phase split: backbone vs action head, at the model's own N.

GR00T cannot use bench/paper_repro.py. It lives in ~/venvs/groot (torch 2.10.0) while the
LeRobot harness lives in ~/venvs/pi0 (torch 2.8.0), and paper_repro imports lerobot. This
is the same measurement — same regions, same CSV schema, same warm-up and timing rules —
without the import.

## The two phases, read from the source (gr00t_n1d7.py:603)

    backbone_inputs, action_inputs = self.prepare_input(inputs)
    backbone_outputs = self.backbone(backbone_inputs)                    # <- vlm_prefix
    action_outputs  = self.action_head.get_action(backbone_outputs, ...) # <- ae_total

and the denoising loop is inside the action head,
`for t in range(self.num_inference_timesteps)` (:366+).

## Why vision is not split out here, stated rather than quietly omitted

In the LeRobot family `embed_image` is a distinct method, so vision separates cleanly from
the VLM. GR00T's backbone is a whole VLM invoked as one unit — Cosmos-Reason2-2B
(Qwen3-VL) for N1.7, Eagle for N1.6 and N1.5 — and isolating its vision encoder means
reaching into internals that differ across exactly the versions we are comparing. For the
action-head-share question, backbone vs action_head is the split that matters and it is
exact. `vision` is reported as "inside vlm_prefix" and the comparison table says so.

    ~/venvs/groot/bin/python bench/groot_phases.py --outdir results/<date>_E4d_groot_n17 \
        --checkpoint nvidia/GR00T-N1.7-3B
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def summarize(xs):
    s = sorted(xs)
    n = len(s)
    q = lambda p: s[0] if n == 1 else s[int(p*(n-1))] + (
        s[min(int(p*(n-1))+1, n-1)] - s[int(p*(n-1))]) * (p*(n-1) - int(p*(n-1)))
    return {"median_ms": statistics.median(s), "mean_ms": statistics.fmean(s),
            "p5_ms": q(0.05), "p95_ms": q(0.95), "min_ms": s[0], "max_ms": s[-1], "n": n}


def gate() -> dict:
    def rd(p, d="?"):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return d
    return {"gpu_cur_freq_hz": rd("/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq"),
            "gpu_min_freq_hz": rd("/sys/devices/platform/17000000.gpu/devfreq_dev/min_freq"),
            "gpu_max_freq_hz": rd("/sys/devices/platform/17000000.gpu/devfreq_dev/max_freq"),
            "cpu_online": rd("/sys/devices/system/cpu/online"),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
            "capability": str(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available() else "-"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--checkpoint", default="nvidia/GR00T-N1.7-3B")
    ap.add_argument("--version", default="n17")
    ap.add_argument("--steps", type=int, default=None, help="default: the config's own")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--tf32", choices=("on", "off"), default="on")
    ap.add_argument("--compile", default=None,
                    help="torch.compile mode for the action head, e.g. reduce-overhead")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    g = gate()
    print("gate:", json.dumps(g), flush=True)
    if g["gpu_cur_freq_hz"] != g["gpu_max_freq_hz"]:
        raise SystemExit(f"clocks not locked: {g}; run sudo jetson_clocks first")
    torch.set_float32_matmul_precision("high" if args.tf32 == "on" else "highest")

    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7          # noqa: E402
    print(f"loading {args.checkpoint} ...", flush=True)
    model = Gr00tN1d7.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    n = args.steps or int(getattr(model.action_head.config, "num_inference_timesteps", 4))
    model.action_head.num_inference_timesteps = n
    print(f"N (num_inference_timesteps) = {n}", flush=True)

    # timing hooks around exactly the two phases, installed on the instance
    spans: dict[str, list] = {"vlm_prefix": [], "ae_total": []}

    def timed(obj, attr, label):
        orig = getattr(obj, attr)

        def wrapper(*a, **k):
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            r = orig(*a, **k)
            e.record()
            spans[label].append((s, e))
            return r
        setattr(obj, attr, wrapper)
        return lambda: setattr(obj, attr, orig)

    if args.compile:
        t0 = time.time()
        model.action_head.get_action = torch.compile(model.action_head.get_action,
                                                     mode=args.compile)
        print(f"compiled action_head.get_action @{args.compile} in {time.time()-t0:.1f} s",
              flush=True)

    undo = [timed(model, "backbone", "vlm_prefix"),
            timed(model.action_head, "get_action", "ae_total")]

    inputs = build_inputs(model, n)
    for _ in range(args.warmup):
        with torch.no_grad():
            model.get_action(inputs)
    torch.cuda.synchronize()

    e2e, per = [], {k: [] for k in spans}
    for _ in range(args.timed):
        for k in spans:
            spans[k].clear()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.get_action(inputs)
        torch.cuda.synchronize()
        e2e.append((time.perf_counter() - t0) * 1e3)
        for k, evs in spans.items():
            per[k].append(sum(s.elapsed_time(e) for s, e in evs))
    for u in undo:
        u()

    st = summarize(e2e)
    rows = []
    for k in ("vlm_prefix", "ae_total"):
        s = summarize(per[k])
        rows.append({"region": k, **{kk: round(vv, 4) for kk, vv in s.items()},
                     "share_of_e2e": round(s["median_ms"] / st["median_ms"], 4)})
    rows.append({"region": "e2e", **{kk: round(vv, 4) for kk, vv in st.items()},
                 "share_of_e2e": 1.0})
    with (out / "phase_split.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'region':12s} {'median ms':>10s} {'share':>7s}")
    for r in rows:
        print(f"{r['region']:12s} {r['median_ms']:>10.2f} {100*r['share_of_e2e']:>6.1f}%")

    (out / "meta.json").write_text(json.dumps({
        "experiment": "E4d (GR00T phase split)", "model": f"groot_{args.version}",
        "checkpoint": args.checkpoint, "precision": "bfloat16",
        "matmul_precision": "high" if args.tf32 == "on" else "highest",
        "compile_mode": args.compile, "num_inference_timesteps": n,
        "n_warmup": args.warmup, "n_timed": args.timed, "gate": g,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "vision_note": "GR00T's backbone is a whole VLM called as one unit, so `vision` is "
                       "INSIDE vlm_prefix here and is not separated. The LeRobot models "
                       "split it because embed_image is a distinct method.",
        "regions": {"vlm_prefix": "Gr00tN1d7.backbone (gr00t_n1d7.py:603)",
                    "ae_total": "action_head.get_action, containing the "
                                "num_inference_timesteps loop (:366+)"},
    }, indent=1))
    print(f"\nwrote {out}")
    return 0


def build_inputs(model, n):
    """A fixture observation in GR00T's own input format.

    Filled in once the checkpoint's expected modality keys are read from its own config —
    guessing them would be the same mistake that made SmolVLA see no images at all.
    """
    raise SystemExit(
        "build_inputs is not written yet: read the checkpoint's modality config and "
        "construct its observation, rather than guessing key names. See "
        "scratch/T012/groot_segments_from_code.md.")


if __name__ == "__main__":
    raise SystemExit(main())
