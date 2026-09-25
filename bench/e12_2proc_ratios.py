"""E12 Step 3: turn the two-process JSONs into corun_ratios.csv, with the caveats attached.

Reads the six files bench/queues/queue_e12_2proc.sh writes and derives `s`, `p` and the
bracket drift. Deliberately does NOT emit `T_pipe`: two processes each run ONE stage from
their own cache, so there is no pipeline period to report and no frame handoff. Claiming a
`T_pipe` from these numbers would be inventing one.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(d: Path, name: str) -> dict | None:
    f = d / name
    if not f.exists():
        return None
    return json.loads(f.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    if not d.is_absolute():
        d = REPO / d

    got = {k: load(d, f"{k}.json") for k in
           ("prefix_iso_pre", "expert_iso_pre", "prefix_corun", "expert_corun",
            "prefix_iso_post", "expert_iso_post")}
    missing = [k for k, v in got.items() if v is None]
    if missing:
        print(f"  MISSING {missing} -- cannot derive ratios from an incomplete matrix")
        return 1

    # Refuse if either co-run process never saw its peer: the window would not be a co-run,
    # and a median from it would describe an isolated run wearing a co-run label.
    for k in ("prefix_corun", "expert_corun"):
        if not got[k].get("peer_ready_seen", False):
            print(f"  {k}: peer_ready_seen=False -- that window was NOT a co-run. Refusing "
                  f"to derive s or p from it.")
            return 1

    def med(k):
        return got[k]["median_ms"]

    A_iso = (med("prefix_iso_pre") + med("prefix_iso_post")) / 2
    B_iso = (med("expert_iso_pre") + med("expert_iso_post")) / 2
    dA = abs(med("prefix_iso_post") - med("prefix_iso_pre")) / med("prefix_iso_pre") * 100
    dB = abs(med("expert_iso_post") - med("expert_iso_pre")) / med("expert_iso_pre") * 100
    A_co, B_co = med("prefix_corun"), med("expert_corun")
    s, p, r = B_co / B_iso, A_co / A_iso, A_iso / B_iso

    # Bootstrap over the co-run samples with the isolated median fixed, same rule the
    # in-process harness uses so the two intervals mean the same thing.
    import random
    rng = random.Random(7)
    v = got["expert_corun"]["samples_ms"]
    meds = sorted(statistics.median(rng.choices(v, k=len(v))) for _ in range(2000))
    lo, hi = meds[49] / B_iso, meds[-50] / B_iso

    g = got["prefix_corun"]
    print()
    print(f"  A_iso {A_iso:8.2f} ms (drift {dA:.1f} %)   B_iso {B_iso:8.2f} ms (drift {dB:.1f} %)")
    print(f"  A_co  {A_co:8.2f} ms                  B_co  {B_co:8.2f} ms")
    print(f"  s = B_co/B_iso = {s:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
    print(f"  p = A_co/A_iso = {p:.4f}")
    print(f"  r = A_iso/B_iso = {r:.4f}   <- break-even, this rung and session")
    print()
    print(f"  (1) contention on the expert: {'YES' if s > 1.05 else 'no measurable slowdown'}"
          f"  (s = {s:.4f})")
    print(f"  (2) contention on the prefix: {'YES' if p > 1.05 else 'no measurable slowdown'}"
          f"  (p = {p:.4f})")
    print(f"  (3) if the pipeline period were max(A_co, B_co): {max(A_co, B_co):.2f} ms vs "
          f"serial A_co+B_co {A_co + B_co:.2f} ms")
    print("      NOTE: (3) is arithmetic, NOT a measured T_pipe. These are two processes each "
          "running ONE stage from its own cache; there is no frame handoff to time.")

    out = d / "corun_ratios.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "mechanism", "power_mode", "sm_count", "N", "rung", "A_iso_ms", "B_iso_ms",
            "T_serial_ms", "A_co_ms", "B_co_ms", "T_pipe_ms", "s", "s_ci95_lo", "s_ci95_hi",
            "p", "r", "T_pipe_over_A_iso", "speedup_vs_serial", "reference", "notes"])
        w.writeheader()
        w.writerow({
            "mechanism": "two_processes_no_mps", "power_mode": g["power_mode"],
            "sm_count": g["sm_count"], "N": g["steps"], "rung": g["rung"],
            "A_iso_ms": round(A_iso, 4), "B_iso_ms": round(B_iso, 4),
            "T_serial_ms": "", "A_co_ms": round(A_co, 4), "B_co_ms": round(B_co, 4),
            "T_pipe_ms": "", "s": round(s, 6), "s_ci95_lo": round(lo, 6),
            "s_ci95_hi": round(hi, 6), "p": round(p, 6), "r": round(r, 6),
            "T_pipe_over_A_iso": "", "speedup_vs_serial": "",
            "reference": "own_isolated",
            "notes": (
                "TWO PROCESSES, no MPS. Each process runs ONE stage back to back from its own "
                "KV cache, so there is NO frame handoff and therefore no T_pipe and no "
                "fresh-KV question -- nothing crosses a process boundary. T_serial, T_pipe and "
                "the speed-up columns are deliberately EMPTY rather than filled with an "
                "arithmetic max(A_co,B_co), which would be an invented measurement. "
                "This is the first E12 configuration in which GPU contention CAN appear: the "
                "in-process pipeline is CPU-issue-bound (issue/wall = 1.0000) so nothing "
                "overlaps there, and the compiled rung corrupts across two streams. Two "
                "interpreters issue on two cores and each has one sequential call stream, so "
                f"the compiled {g['rung']} rung is usable. "
                f"Isolated bracket drift: prefix {dA:.1f} %, expert {dB:.1f} %. "
                "OPEN: two CUDA contexts on one GPU may TIME-SLICE rather than share "
                "spatially; E11f found two processes left host segments unaffected at 1.002x. "
                "If they time-slice, s would rise for a reason that is not CTA contention. "
                "nsys is what distinguishes them and has not been run for this mechanism.")})
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
