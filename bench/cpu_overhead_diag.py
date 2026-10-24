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

    res = {"model": a.model, "rung": rung, "best_bf16_source": src, "warmup_iters": it, "rung_desc": desc,
           "inductor_cache": os.environ.get("TORCHINDUCTOR_CACHE_DIR", "default (not the timed run's)")}
    res["baseline"] = timed("baseline")
    pr = cProfile.Profile()
    mark()
    pr.enable()
    ad.iteration()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(60)
    open(os.path.join(a.out, "raw", f"cprofile_{a.model}_baseline.txt"), "w").write(s.getvalue())
    base_act = seeded()
    torch.compiler.set_stance("default", skip_guard_eval_unsafe=True)
    res["skip_guards"] = timed("skip_guard_eval_unsafe")
    pr = cProfile.Profile()
    mark()
    pr.enable()
    ad.iteration()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(60)
    open(os.path.join(a.out, "raw", f"cprofile_{a.model}_skip_guards.txt"), "w").write(s.getvalue())
    sk_act = seeded()
    d = sk_act - base_act
    res["skip_guards"]["rel_l2_vs_baseline"] = float(np.linalg.norm(d) / max(np.linalg.norm(base_act), 1e-30))
    res["skip_guards"]["max_abs_diff_vs_baseline"] = float(np.abs(d).max())
    res["skip_guards"]["actions_finite"] = bool(np.isfinite(sk_act).all())
    json.dump(res, open(os.path.join(a.out, f"{a.model}_{a.rung}.json"), "w"), indent=1, default=str)
    print(json.dumps({k: (res[k]["e2e_wall_median_ms"], res[k]["regions_median_ms"], res[k]["graphs_compiled"])
                      for k in ("baseline", "skip_guards")}, default=str))
    print("skip_guards vs baseline actions: rel L2", res["skip_guards"]["rel_l2_vs_baseline"])


if __name__ == "__main__":
    main()
