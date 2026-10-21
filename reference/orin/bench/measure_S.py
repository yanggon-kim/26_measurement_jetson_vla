"""Measure the suffix token count S per model, rather than inferring it from chunk_size.

C3 needs S: for a weight-stationary bf16 GEMM with S << K,N the arithmetic intensity is
~S, so S decides whether the expert is memory-bound by construction. pi0's is 51 (50
action tokens plus one state token), measured in T010. pi0.5's and SmolVLA's are unknown
because pi0_common.token_counts is pi0-specific — embed_suffix takes three arguments for
pi0 and two for the other two (modeling_pi05.py:684, modeling_smolvla.py:719).

This runs one real inference with a hook on the expert's own forward and reads the
sequence length off the tensor the model actually passes it. No signature assumptions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as MR  # noqa: E402
import pi0_common as P  # noqa: E402
from common import write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="pi0,pi05,smolvla")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = {}
    for name in args.models.split(","):
        spec = MR.get(name)
        try:
            policy, _, _ = MR.load_policy(spec, n=spec.n_default)
            fx = P.load_fixture()
            batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                        revision=spec.revision)
            seen = {}
            container = getattr(policy.model, spec.container)
            orig = container.forward

            def fwd(*a, _o=orig, **k):
                emb = k.get("inputs_embeds") or (a[3] if len(a) > 3 else None)
                if emb is not None and len(emb) > 1 and emb[1] is not None:
                    seen.setdefault("suffix_tokens_S", int(emb[1].shape[1]))
                    seen.setdefault("suffix_hidden", int(emb[1].shape[-1]))
                if emb is not None and emb[0] is not None:
                    seen.setdefault("prefix_tokens_P", int(emb[0].shape[1]))
                return _o(*a, **k)

            container.forward = fwd
            with torch.no_grad():
                policy.predict_action_chunk(batch)
            container.forward = orig
            seen["chunk_size"] = int(getattr(policy.config, "chunk_size", -1))
            out[name] = seen
            print(f"{name:9s} {json.dumps(seen)}", flush=True)
            del policy
            torch.cuda.empty_cache()
        except Exception as e:                                      # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"{name:9s} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
    if args.json:
        write_json(Path(args.json), out)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
