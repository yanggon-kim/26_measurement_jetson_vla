"""Does a model modification change the actions? Run this BEFORE reporting its speed-up.

A faster model that computes something different is not a result. Compares `sample_actions`
with and without a modification, in eager (so nothing but the modification differs), with
fixed noise so both runs are deterministic.

    python bench/check_equivalence.py --variant cropcache --steps 10 --seed 1234
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
from run_e4 import crop_instead_of_deepcopy, no_per_step_deepcopy  # noqa: E402

VARIANTS = {"cropcache": crop_instead_of_deepcopy, "nodeepcopy": no_per_step_deepcopy}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="cropcache")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    P.gate()
    policy, cfg, _ = P.load_policy(num_inference_steps=args.steps, compile_mode=None)
    P.assert_real_weights(policy)
    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx)
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    def run():
        images, img_masks = policy._preprocess_images(batch)
        state = policy.prepare_state(batch)
        g = torch.Generator(device="cuda").manual_seed(args.seed)
        noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device="cuda",
                            dtype=torch.float32, generator=g)
        with torch.inference_mode():
            return policy.model.sample_actions(
                images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK], state, noise=noise,
                num_steps=args.steps)

    ref = run().clone()
    ref2 = run().clone()                      # control: is the harness deterministic at all?
    with VARIANTS[args.variant](policy):
        mod = run().clone()
    torch.cuda.synchronize()

    d = (ref - mod).abs()
    print(f"  variant                : {args.variant} (eager, seed {args.seed}, N={args.steps})")
    print(f"  shape / dtype          : {tuple(ref.shape)} {ref.dtype}")
    print(f"  control ref vs ref     : identical={torch.equal(ref, ref2)} "
          f"maxdiff={(ref-ref2).abs().max().item():.3e}")
    print(f"  BITWISE IDENTICAL      : {torch.equal(ref, mod)}")
    print(f"  max abs diff           : {d.max().item():.3e}")
    print(f"  max rel diff           : {(d.max()/ref.abs().max().clamp(min=1e-9)).item():.3e}")
    print(f"  both finite            : {bool(torch.isfinite(ref).all() and torch.isfinite(mod).all())}")
    return 0 if torch.equal(ref, mod) else 1


if __name__ == "__main__":
    raise SystemExit(main())
