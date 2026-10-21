"""Find the exact module that feeds fp32 activations into a bf16 GEMM, for pi0.5's B3.

pi0.5's bf16_norms and bf16_both both die on

    mm(arg6_1, reinterpret_tensor(arg7_1, (1024, 3072), ...))
    RuntimeError: expected mat1 and mat2 to have the same dtype, but got:
                  float != c10::BFloat16

1024 is pi0.5's action-expert width and 3072 is 3x1024, so the weight is a fused QKV
projection inside the expert. What is NOT known from the traceback is which module produced
the fp32 activation, and guessing it is how three earlier "fixes" in this project were
wrong. So: run one real inference in the rung's exact configuration, with a forward pre-hook
on every Linear and every norm, and print the first place an input dtype disagrees with the
weight dtype it is about to meet.

Run with the same prep chain the failing rung uses:

    python bench/pi05_dtype_probe.py --prep base,norms
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_registry as MR  # noqa: E402
import opt_ladder as OL  # noqa: E402
import pi0_common as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pi05")
    ap.add_argument("--prep", default="base,norms",
                    help="comma list of base,vision,norms — applied in that order")
    ap.add_argument("--max-report", type=int, default=25)
    args = ap.parse_args()

    spec = MR.get(args.model)
    print("gate:", P.gate(), flush=True)
    policy, _, _ = MR.load_policy(spec, n=10, compile_mode=None)
    torch.set_float32_matmul_precision("high")

    print(f"\ndominant dtype before prep: {OL._dominant_dtype(policy)}")
    for name in args.prep.split(","):
        fn = {"base": OL.prep_bf16_base,
              "vision": OL.prep_bf16_vision,
              "norms": OL.prep_bf16_norms}[name.strip()]
        print(f"  prep {name}: {fn(policy)}")
    print(f"dominant dtype after prep:  {OL._dominant_dtype(policy)}\n")

    # Where does each parameter tensor still sit, by module, so the fp32 survivors are
    # visible before any hook fires.
    surv = {}
    for n, prm in policy.model.named_parameters():
        if prm.dtype == torch.float32:
            surv.setdefault(n.rsplit(".", 1)[0], 0)
            surv[n.rsplit(".", 1)[0]] += prm.numel()
    print("fp32 parameters that survived the prep (module -> elements):")
    for k, v in sorted(surv.items(), key=lambda kv: -kv[1])[:args.max_report]:
        print(f"   {v:12,d}  {k}")
    if not surv:
        print("   (none)")

    # Now the dynamic half: the first input/weight dtype disagreement in a real forward.
    mismatches, seen = [], set()
    handles = []

    def pre(mod, inputs, mod_name=""):
        w = getattr(mod, "weight", None)
        if w is None or not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        if x.dtype != w.dtype and mod_name not in seen:
            seen.add(mod_name)
            mismatches.append((mod_name, type(mod).__name__, str(x.dtype), str(w.dtype),
                               tuple(w.shape)))

    for n, mod in policy.model.named_modules():
        if isinstance(mod, torch.nn.Linear) or "orm" in type(mod).__name__:
            handles.append(mod.register_forward_pre_hook(
                lambda m, i, _n=n: pre(m, i, _n)))

    fx = P.load_fixture()
    batch, _, _ = P.build_batch(policy, fx, checkpoint=spec.checkpoint,
                                revision=spec.revision)
    print("\nrunning one inference with hooks ...", flush=True)
    try:
        with torch.no_grad():
            policy.predict_action_chunk(batch)
        print("inference SUCCEEDED (no hard failure in this configuration)")
    except Exception as e:  # noqa: BLE001
        print(f"inference FAILED, as the rung does: {type(e).__name__}: {str(e)[:160]}")
    for h in handles:
        h.remove()

    print(f"\nmodules whose input dtype disagreed with their weight ({len(mismatches)}):")
    for name, kind, xd, wd, shape in mismatches[:args.max_report]:
        print(f"   {name}\n      {kind}  input={xd}  weight={wd}  weight_shape={shape}")
    if not mismatches:
        print("   (none) — the mismatch is then inside a fused/compiled region, not at a "
              "module boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
