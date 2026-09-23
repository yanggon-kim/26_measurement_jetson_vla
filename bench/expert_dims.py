"""C3 — the action expert's GEMM shapes, per model, from the checkpoint's own tensors.

pi0's expert runs 126 projections on S = 51 suffix tokens: M = 51 against K, N of
1024-4096. That is extremely skinny, and T005 showed every one of them is launch-bound
rather than compute- or bandwidth-bound at that size. This quantifies "skinny" for each
model instead of asserting it, by reading the weight shapes out of model.safetensors and
pairing them with the suffix length the model actually runs.

Arithmetic intensity per GEMM is 2*M*K*N / (bytes of A + B + C), which for M << K,N is
dominated by reading B: I -> M for a weight-stationary bf16 GEMM, so the suffix length IS
the arithmetic intensity to first order. Against our measured ridge of 180.57 FLOP/B, any
S below ~180 is memory-bound by construction, whatever the model.

    python bench/expert_dims.py --json results/<date>_E4f_expert_dims/expert_dims.json
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_json  # noqa: E402

RIDGE = 180.57          # measured, T005/E1: 33.85 TFLOP/s / 187.47 GB/s

MODELS = {
    "pi0":     {"repo": "lerobot/pi0_libero_finetuned_v044",
                "rev": "45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f",
                "expert_re": r"gemma_expert\.model\.layers\.(\d+)\.(\S+)\.weight",
                "S": 51, "S_src": "T010 measured: 50 action tokens + 1 state token"},
    "pi05":    {"repo": "lerobot/pi05_libero_base", "rev": None,
                "expert_re": r"gemma_expert\.model\.layers\.(\d+)\.(\S+)\.weight",
                "S": None, "S_src": "to be measured"},
    "smolvla": {"repo": "lerobot/smolvla_base", "rev": None,
                "expert_re": r"lm_expert\.layers\.(\d+)\.(\S+)\.weight",
                "S": None, "S_src": "to be measured"},
}


def header(repo: str, rev: str | None) -> dict:
    from huggingface_hub import hf_hub_download
    kw = {"revision": rev} if rev else {}
    p = Path(hf_hub_download(repo, "model.safetensors", **kw))
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--models", default="pi0,pi05,smolvla")
    args = ap.parse_args()
    out = {"_ridge_FLOP_per_byte": RIDGE,
           "_note": "I ~= S for a weight-stationary bf16 GEMM with S << K,N, so the suffix "
                    "length is the arithmetic intensity to first order. S below the ridge "
                    "means memory-bound by construction.",
           "models": {}}
    for name in args.models.split(","):
        cfg = MODELS[name]
        try:
            h = header(cfg["repo"], cfg["rev"])
        except Exception as e:                                      # noqa: BLE001
            out["models"][name] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
            print(f"{name}: {type(e).__name__}: {str(e)[:120]}")
            continue
        shapes: dict[str, list] = {}
        layers = set()
        for k, v in h.items():
            m = re.search(cfg["expert_re"], k)
            if not m:
                continue
            layers.add(int(m.group(1)))
            shapes.setdefault(m.group(2), v["shape"])
        S = cfg["S"]
        rows = []
        for proj, shp in sorted(shapes.items()):
            if len(shp) != 2:
                continue
            N, K = shp                       # torch Linear stores [out, in]
            rows.append({"projection": proj, "K_in": K, "N_out": N,
                         "gemm": f"{S or 'S'}x{K}x{N}",
                         "flops_per_call": (2 * S * K * N) if S else None,
                         "weight_bytes_bf16": 2 * K * N})
        n_layers = len(layers)
        per_layer_bytes = sum(r["weight_bytes_bf16"] for r in rows)
        out["models"][name] = {
            "checkpoint": cfg["repo"], "suffix_tokens_S": S, "S_source": cfg["S_src"],
            "expert_layers": n_layers, "projections_per_layer": len(rows),
            "projections_total": n_layers * len(rows),
            "expert_weight_bytes_bf16": n_layers * per_layer_bytes,
            "arithmetic_intensity_approx": S,
            "verdict": (None if S is None else
                        ("MEMORY-bound by construction (S < ridge)" if S < RIDGE
                         else "compute-bound")),
            "projections": rows,
        }
        print(f"\n=== {name}: {n_layers} expert layers x {len(rows)} projections "
              f"= {n_layers*len(rows)} GEMMs, S={S}")
        for r in rows:
            print(f"    {r['projection']:28s} {r['gemm']:>18s}  "
                  f"weights {r['weight_bytes_bf16']/2**20:7.2f} MiB")
    if args.json:
        write_json(Path(args.json), out)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
