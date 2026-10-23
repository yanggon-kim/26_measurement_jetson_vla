"""T006 steps 1-2: C3 operand dimensions and the analytic FLOPs / bytes per part, for one model in one mode.

The model gets the mode's preparations (precision flags, bf16 casts, bf16_islands, linear_mpad) but NOTHING is compiled:
FlopCounterMode and forward hooks need eager dispatch. The arithmetic and the parameter dtypes are those of the mode, so
FLOPs and weight bytes are the mode's. linear_mpad's padded rows are counted, because the device executes them.
  dims.csv        one row per distinct (region, module, M, K, N, dtype): calls per inference, FLOPs = 2·M·K·N per call,
                  unique bytes per call (weight K·N, activation M·K, output M·N, x element size), intensity.
                  M = rows actually multiplied, measured on a real inference: S for the expert, tokens for the prefix /
                  vision. The expert is recorded for step 0 (every step has the same shapes).
  accounting.json run_e3.accounting_pass: FLOPs per region (FlopCounterMode, attention formulas included) and parameter
                  bytes touched per region, by dtype.

python bench/t006_analytic.py --model pi0 --mode best_bf16 --out results/<date>_E6_dims/raw/pi0_best_bf16
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import opt_ladder as OL
    import regions as R
    import rungs as RG
    import run_e3 as E3
    os.makedirs(a.out, exist_ok=True)
    rung, table = a.mode, None
    if a.mode == "best_bf16":
        rung, src = OL.resolve_best(a.model)
        table = OL.mpad_table_of(a.model, src)
    r = dict(RG.get(a.model, rung))
    ad = E3.make_adapter(a.model)
    ad.name, ad.family = a.model, RG.family_of(a.model)
    if table:
        ad.mpad_fixed = table
    if ad.family == "groot":
        ad.capture_tokens()
    RG.set_precision(r["precision"])
    preps = []
    for p in r["preps"]:
        if p.startswith("attn:"):
            import e4i_candidates as EC
            _, pre, exp = p.split(":")
            preps.append(EC.prep_attn(ad, pre, exp))
        elif p in ("no_attn_fusion", "capture_scalar"):
            preps.append(f"{p}: compile-time only, not applied (nothing is compiled here)")
        else:
            preps.append(RG.PREPS[p](ad) if p in RG.ADAPTER_PREPS else RG.PREPS[p](ad.policy))
    ad.set_n(ad.n_default)
    rec = R.Recorder(account=True)
    rows = {}

    def hook(mod, args):
        if not rec._active or not rec.stack or not args or not torch.is_tensor(args[0]):
            return
        reg = rec.stack[-1]
        if reg.startswith("ae_step[") and reg != "ae_step[0]":
            return
        x = args[0]
        m = x.numel() // x.shape[-1]
        if type(mod).__name__ == "CategorySpecificLinear":
            k, n = mod.W.shape[1], mod.W.shape[2]
            wdt = mod.W.dtype
        else:
            k, n, wdt = mod.in_features, mod.out_features, mod.weight.dtype
        key = (reg, mod._t006_name, m, k, n, str(wdt))
        rows[key] = rows.get(key, 0) + 1
    for name, mod in ad.policy.model.named_modules():
        if isinstance(mod, torch.nn.Linear) or type(mod).__name__ == "CategorySpecificLinear":
            mod._t006_name = name
    acc = E3.accounting_pass(ad, ad.n_default)          # its own Recorder; installs and removes the region wrappers
    # one more inference with our Recorder open, to attribute every GEMM call to its region
    undo = ad.instrument(rec)
    names = {n: m.register_forward_pre_hook(hook) for n, m in ad.policy.model.named_modules()
             if isinstance(m, torch.nn.Linear) or type(m).__name__ == "CategorySpecificLinear"}
    rec.reset()
    rec.start()
    ad.iteration()
    rec.stop()
    for h in names.values():
        h.remove()
    undo()
    es = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4}
    out = []
    for (reg, mod, m, k, n, wdt), c in sorted(rows.items()):
        e = es.get(wdt, 2)
        fl = 2 * m * k * n
        wb, xb, yb = k * n * e, m * k * e, m * n * e
        out.append({"model": a.model, "mode": a.mode, "rung": rung, "region": reg, "module": mod, "M": m, "K": k, "N": n,
                    "weight_dtype": wdt, "calls_per_inference": c, "flops_per_call": fl, "weight_bytes": wb,
                    "act_bytes": xb, "out_bytes": yb, "intensity_flop_per_byte": round(fl / (wb + xb + yb), 3)})
    with open(os.path.join(a.out, "dims.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    acc.update({"model": a.model, "mode": a.mode, "rung": rung, "preps": preps, "tokens": ad.tokens})
    json.dump(acc, open(os.path.join(a.out, "accounting.json"), "w"), indent=1, default=str)
    print(f"{a.model} {a.mode} ({rung}): {len(out)} GEMM shapes; regions {sorted(acc['flops'])}")


if __name__ == "__main__":
    main()
