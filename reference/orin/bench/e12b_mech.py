"""E12b optional mechanisms (spec §6 "Optional, only if cheap"), each a labelled row.

Runs the M2 configuration (rung prefix + owngraph expert, full GPU) with ONE change per variant,
each with its own pipeline equivalence check and its own same-process isolated A/B:
  --variant prio_prefix   prefix stream at priority -3
  --variant prio_expert   expert stream at priority -3
  --variant maxconn1      CUDA_DEVICE_MAX_CONNECTIONS=1 (must be set BEFORE CUDA initialises, so
                          the queue sets it in this process's environment)
Writes one JSON per variant into --outdir.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import corun as CR               # noqa: E402
import e12b_common as C          # noqa: E402
import e12b_corun as H           # noqa: E402
import e12b_m2 as M2             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=("prio_prefix", "prio_expert", "maxconn1"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--frames", type=int, default=100)
    args = ap.parse_args()
    if args.variant == "maxconn1" and os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1":
        raise SystemExit("maxconn1 needs CUDA_DEVICE_MAX_CONNECTIONS=1 in the environment")
    out = REPO / args.outdir
    out.mkdir(parents=True, exist_ok=True)
    g0 = C.gate_full()
    policy, spec, cfg, note = C.load_rung(10)
    H.recompile_default(policy, spec, "denoise_step")
    frames, _ = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    prio = {"prio_prefix": (-3, 0), "prio_expert": (0, -3), "maxconn1": (0, 0)}[args.variant]
    h = H.Harness(policy, spec, frames, 10, config="rung_prefix", prio=prio)
    fin = C.Finite()
    eq, ok, _ = M2.equivalence(h, fin)
    res = {}
    with torch.no_grad():
        for cond in ("prefix_alone", "expert_alone", "serial", "pipeline"):
            h.run(cond, 8, fin)
            r = h.run(cond, args.frames, fin)
            g = lambda k: statistics.median([x[k] for x in r["records"][5:] if x.get(k) is not None]) \
                if any(x.get(k) is not None for x in r["records"]) else None      # noqa: E731
            res[cond] = {"prefix": g("prefix_ms"), "expert": g("expert_ms"),
                         "frame": g("frame_ms"), "period": r["period_ms"]}
    A, B = res["prefix_alone"]["prefix"], res["expert_alone"]["expert"]
    Ts, Tp = res["serial"]["frame"], res["pipeline"]["period"]
    nf = fin.read()
    row = {"mechanism": args.variant, "variant": h.label(),
           "priority_requested": prio, "priority_granted": list(h.prio_granted),
           "priority_range": list(torch.cuda.Stream.priority_range()),
           "equivalence_pass": ok, "n_nonfinite": nf, "valid": ok and nf == 0,
           "A_iso_ms": A, "B_iso_ms": B, "T_serial_ms": Ts, "T_pipe_ms": Tp,
           "A_co_ms": res["pipeline"]["prefix"], "B_co_ms": res["pipeline"]["expert"],
           "O": Tp / max(A, B), "phi": (Ts - Tp) / (Ts - max(A, B)),
           "p": res["pipeline"]["prefix"] / A, "s": res["pipeline"]["expert"] / B,
           "frames": args.frames, "gate": g0, **C.throttle_record(g0)}
    (out / f"mechanism_{args.variant}.json").write_text(json.dumps(row, indent=2, default=str) + "\n")
    print(json.dumps({k: row[k] for k in ("variant", "priority_granted", "equivalence_pass", "O",
                                          "phi", "p", "s", "T_pipe_ms")}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
