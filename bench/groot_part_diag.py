"""T005 diagnostic: where does a compiled GR00T candidate diverge from eager? Runs one rung (bench/rungs.py), warms up
(graphs captured, compilation done), then performs K inferences with torch.manual_seed(5000 + i). It records the
output of each part for each: the vision callable, the LLM decoder's last hidden state, backbone.forward's
`backbone_features`, and the action head's actions. Saves <out>/parts_<rung>.npz. Compare two files with --compare.
Compiled heads draw seeded noise as eager does (fallback_random, set by rungs.apply_rung).

python bench/groot_part_diag.py --model groot_n15 --rung eager_fp32 --out results/<dir>/raw
python bench/groot_part_diag.py --compare results/<dir>/raw/parts_eager_fp32.npz results/<dir>/raw/parts_<rung>.npz
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def flat(x):
    import torch
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    if hasattr(x, "last_hidden_state"):
        return flat(x.last_hidden_state)
    if isinstance(x, (tuple, list)):
        return flat(x[0])
    if hasattr(x, "keys"):
        for k in ("backbone_features", "action_pred", "last_hidden_state"):
            if k in x:
                return flat(x[k])
        return flat(x[sorted(x.keys())[0]])
    raise TypeError(type(x))


def run(a):
    import torch
    import gate as G
    import regions as R
    import rungs as RG
    import run_e3 as E3
    G.wait_for_gate(apply_clocks=True)
    ad = E3.make_adapter(a.model)
    ad.name, ad.family = a.model, "groot"
    desc = RG.apply_rung(ad, a.rung)
    model = ad.policy.model
    vis_owner, vis_attr, lm, _, _ = R.groot_parts(model)
    store, on = {}, {"v": False}

    def wrap(obj, attr, key):
        orig = getattr(obj, attr)

        def w(*x, **k):
            out = orig(*x, **k)
            if on["v"]:
                store.setdefault(key, []).append(flat(out))
            return out
        setattr(obj, attr, w)
    wrap(vis_owner, vis_attr, "vision")
    wrap(lm, "forward", "llm_hidden")
    wrap(model.backbone, "forward", "backbone_features")
    wrap(model.action_head, "get_action", "actions")
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    for _ in range(a.warmup):
        mark()
        ad.iteration()
    on["v"] = True
    for i in range(a.k):
        torch.manual_seed(5000 + i)
        mark()
        ad.iteration()
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, f"parts_{a.rung}.npz"),
             **{k: np.stack([x.reshape(-1) for x in v]) for k, v in store.items()},
             rung_desc=np.array(str(desc)))
    print({k: (len(v), v[0].shape) for k, v in store.items()})


def compare(ref, cand):
    r, c = np.load(ref), np.load(cand)
    print(f"{os.path.basename(cand)} vs {os.path.basename(ref)}")
    for k in ("vision", "llm_hidden", "backbone_features", "actions"):
        if k not in r or k not in c:
            continue
        a, b = r[k].astype(np.float64), c[k].astype(np.float64)
        if a.shape != b.shape:
            print(f"  {k:18s} shape differs {a.shape} vs {b.shape}")
            continue
        d = b - a
        per = [np.linalg.norm(d[i]) / max(np.linalg.norm(a[i]), 1e-30) for i in range(a.shape[0])]
        print(f"  {k:18s} rel L2 {np.linalg.norm(d) / np.linalg.norm(a):.3e}  per call {['%.2e' % p for p in per]}  "
              f"finite {np.isfinite(b).all()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="groot_n15")
    ap.add_argument("--rung", default="eager_fp32")
    ap.add_argument("--out", default="")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run(a)


if __name__ == "__main__":
    main()
