"""T005/T007 diagnostic: where does the host time of a configuration go, and does skipping dynamo's guard evaluation
remove it?

The T007 timeline (results/20261024_E5_<m>_best_bf16_timeline) shows π0 / π0.5 best_bf16 expert steps idle ~25 % of
their CUDA-event time with the GPU waiting for the CPU (≈ 2.7 ms of CPU-starved gap per step under nsys). This script,
in one process, at the configuration the timed run used (best_bf16 resolved and its linear_mpad table replayed, the timed
run's Inductor caches when TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR are set):
  1. warm up until no new graph (as profile_target.py); time --n inferences (CUDA-event regions + wall) -> baseline;
  2. cProfile one inference -> raw/cprofile_<tag>.txt (the 60 largest cumulative entries);
  3. seeded actions (seeds 5000..5000+k) -> baseline actions;
  4. torch.compiler.set_stance("default", skip_guard_eval_unsafe=True); time --n inferences again, seeded actions again,
     relative L2 and max |diff| vs the baseline actions, and graphs compiled meanwhile (must be 0).
Writes <out>/<model>_<tag>.json. Diagnostic only: a configuration that uses it is a new rung, timed by opt_ladder.py.

python bench/cpu_overhead_diag.py --model pi0 --rung best_bf16 --out results/<date>_E4_cpu_overhead_diag
"""
import argparse
import cProfile
import io
import json
import os
import pstats
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rung", default="best_bf16")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import torch
    from torch._dynamo.utils import counters
    import opt_ladder as OL
    import regions as R
    import rungs as RG
    import run_e3 as E3
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    rung, src, table = a.rung, "", None
    if a.rung == "best_bf16":
        rung, src = OL.resolve_best(a.model)
        table = OL.mpad_table_of(a.model, src)
    ad = E3.make_adapter(a.model)
    ad.name, ad.family = a.model, RG.family_of(a.model)
    if table:
        ad.mpad_fixed = table
    if ad.family == "groot":
        ad.capture_tokens()
    desc = RG.apply_rung(ad, rung)
    ad.set_n(ad.n_default)
    rec = R.Recorder()
    if desc["regions"]:
        if ad.family == "groot":
            R.instrument_groot(ad.policy, rec, inner=not desc["ae_compiled"])
        else:
            R.instrument_lerobot(ad.policy, rec, ad.spec.container)
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    ug = lambda: int(counters["stats"].get("unique_graphs", 0))  # noqa: E731
    hist, it = [ug()], 0
    while it < 20 or len(hist) <= 5 or hist[-1] != hist[-6]:
        rec.reset()
        mark()
        ad.iteration()
        hist.append(ug())
        it += 1
        if it >= 80:
            break

    def timed(tag):
        g0 = ug()
        walls, regs = [], {}
        for _ in range(a.n):
            rec.reset()
            rec.start()
            mark()
            w, _ = ad.iteration()
            rec.stop()
            walls.append(w["e2e_wall"])
            r = rec.results_ms(ad.derive_ae_total) if desc["regions"] else {}
            for j, v in enumerate(rec.step_times_ms() if desc["regions"] else []):
                r[f"ae_step_individual[{j}]"] = v
            for k, v in r.items():
                regs.setdefault(k, []).append(v)
        return {"tag": tag, "e2e_wall_median_ms": statistics.median(walls), "e2e_wall_ms": walls,
                "regions_median_ms": {k: statistics.median(v) for k, v in regs.items()}, "graphs_compiled": ug() - g0}

    def seeded():
        acts = []
        for i in range(a.seeds):
            torch.manual_seed(5000 + i)
            mark()
            _, act = ad.iteration()
            acts.append(OL.flatten_actions(act))
        return np.stack(acts)

    def warm():
        h, n = [ug()], 0
        while n < 10 or len(h) <= 5 or h[-1] != h[-6]:
            mark()
            ad.iteration()
            h.append(ug())
            n += 1
            if n >= 60:
                break
        return n

    def profile(tag):
        pr = cProfile.Profile()
        mark()
        pr.enable()
        ad.iteration()
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(60)
        open(os.path.join(a.out, "raw", f"cprofile_{a.model}_{tag}.txt"), "w").write(s.getvalue())

    def compare(arm, act, ref):
        d = act - ref
        arm["rel_l2_vs_baseline"] = float(np.linalg.norm(d) / max(np.linalg.norm(ref), 1e-30))
        arm["max_abs_diff_vs_baseline"] = float(np.abs(d).max())
        arm["bitwise_equal_to_baseline"] = bool((act == ref).all())
        arm["actions_finite"] = bool(np.isfinite(act).all())

    res = {"model": a.model, "rung": rung, "best_bf16_source": src, "warmup_iters": it, "rung_desc": desc,
           "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR", "default (not the timed run's)"),
           "note": "diagnostic arms in ONE process, in this order; none of them is the baseline (protocol §16)"}
    res["baseline"] = timed("baseline")
    profile("baseline")
    base_act = seeded()
    arms = ["baseline"]
    torch.compiler.set_stance("default", skip_guard_eval_unsafe=True)
    res["skip_guards"] = timed("skip_guard_eval_unsafe")
    profile("skip_guards")
    compare(res["skip_guards"], seeded(), base_act)
    torch.compiler.set_stance("default", skip_guard_eval_unsafe=False)
    arms.append("skip_guards")
    if a.model in ("pi0", "pi05"):
        # kv_shallow: the per-step copy.deepcopy(past_key_values) of denoise_step replaced by a shallow copy (new cache and
        # layer objects that point at the same prefix K/V tensors). Exact if the expert never writes the prefix tensors in
        # place: DynamicLayer.update rebinds self.keys = torch.cat(...), so the bitwise comparison below decides.
        import copy as _copy
        import importlib
        mod = importlib.import_module(f"lerobot.policies.{a.model}.modeling_{a.model}")

        class _ShallowCopy:
            @staticmethod
            def deepcopy(x, memo=None):
                if hasattr(x, "layers"):
                    n = _copy.copy(x)
                    n.layers = [_copy.copy(layer) for layer in x.layers]
                    return n
                return _copy.deepcopy(x, memo)

            def __getattr__(self, k):
                return getattr(_copy, k)

        mod.copy = _ShallowCopy()
        res["kv_shallow_warmup_iters"] = warm()
        res["kv_shallow"] = timed("kv_shallow")
        profile("kv_shallow")
        compare(res["kv_shallow"], seeded(), base_act)
        torch.compiler.set_stance("default", skip_guard_eval_unsafe=True)
        res["kv_shallow_skip_guards"] = timed("kv_shallow + skip_guard_eval_unsafe")
        compare(res["kv_shallow_skip_guards"], seeded(), base_act)
        torch.compiler.set_stance("default", skip_guard_eval_unsafe=False)
        arms += ["kv_shallow", "kv_shallow_skip_guards"]
    res["baseline_repeat"] = timed("baseline repeat (bracket): kv_shallow still installed" if a.model in ("pi0", "pi05")
                                   else "baseline repeat (bracket)")
    json.dump(res, open(os.path.join(a.out, f"{a.model}_{a.rung}.json"), "w"), indent=1, default=str)
    for k in arms:
        g = res[k]["regions_median_ms"]
        print(k, round(res[k]["e2e_wall_median_ms"], 2), {kk: round(g[kk], 2) for kk in ("vision", "vlm_prefix", "ae_total") if kk in g},
              "graphs", res[k]["graphs_compiled"], "rel_l2", res[k].get("rel_l2_vs_baseline"), "bitwise", res[k].get("bitwise_equal_to_baseline"))


if __name__ == "__main__":
    main()
