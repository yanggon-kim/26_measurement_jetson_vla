"""E12b: is `owngraph_expert` numerically the rung's expert? (its own equivalence check)

owngraph_expert = denoise_step compiled mode="default" (no cudagraph trees) + the whole N-step
loop captured as one CUDA graph by the harness. It runs faster than M1's fair expert (the rung's
per-step cudagraph-tree replays), which makes it a candidate "exact, GPU-agnostic inefficiency"
of the kind spec §6 M1 says to measure as its own labelled variant and NOT fold into the fair
expert. Before any number of it is quoted, this checks it computes the same actions:
  rung expert (M1's fair: bf16_both + hoist, reduce-overhead trees) vs owngraph expert, same
  prefix cache, noise and state, on the 5 LIBERO observations; bitwise yes/no, max abs/rel.
Two processes are not needed: the rung expert is evaluated first, then denoise_step is
recompiled and the owngraph harness built in the same process.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import corun as CR               # noqa: E402
import e12b_common as C          # noqa: E402
import e12b_corun as H           # noqa: E402

N = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to write owngraph_equivalence.*")
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    policy, spec, cfg, note = C.load_rung(N)
    frames, _ = CR.load_frames(policy, spec, REPO / "bench/fixtures/e12", 5, seed0=0)
    ref, caches = {}, {}
    with torch.no_grad():
        for fr in frames:
            pkv, pad = CR.prefix_stage(fr["ctx"])
            caches[fr["k"]] = (pkv, pad)
            ref[fr["k"]] = CR.expert_stage(fr["ctx"], pkv, pad, fr["noise"], N).clone()
        torch.cuda.synchronize()
        H.recompile_default(policy, spec, "denoise_step")
        h = H.Harness(policy, spec, frames, N, config="rung_prefix")
        rows = []
        for fr in frames:
            pkv, pad = caches[fr["k"]]
            H.copy_tree(h.kv[0], pkv)
            h.pad[0].copy_(pad)
            h.noise.copy_(fr["noise"])
            h.state.copy_(fr["ctx"]["state"])
            h.eg[0].replay()
            torch.cuda.synchronize()
            got = h.eout[0].clone()
            a, b = ref[fr["k"]].float(), got.float()
            d = (a - b).abs()
            rows.append({"variant": "bf16_both+hoist+owngraph_expert", "path": "compiled",
                         "check": "owngraph_vs_rung_expert", "split": "full_gpu",
                         "frame": fr["k"], "obs_id": fr["obs_id"], "seed": fr["seed"],
                         "max_abs_diff": d.max().item(),
                         "max_rel_diff": (d / a.abs().clamp_min(1e-6)).max().item(),
                         "bitwise_identical": bool(torch.equal(ref[fr["k"]], got)),
                         "n_nonfinite": int((~torch.isfinite(got)).sum().item()),
                         "verdict": "PASS" if torch.equal(ref[fr["k"]], got) else "DIFFERS"})
    with (out / "owngraph_equivalence.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"  frame {r['frame']} bitwise {r['bitwise_identical']} max|d| {r['max_abs_diff']:.3e}"
              f" rel {r['max_rel_diff']:.3e}")
    (out / "owngraph_equivalence.json").write_text(json.dumps(
        {"all_bitwise": all(r["bitwise_identical"] for r in rows),
         "max_abs": max(r["max_abs_diff"] for r in rows)}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
