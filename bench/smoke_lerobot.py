"""T001 step 8 (A16): one inference per LeRobot model, smoke test -- NOT a measurement.

Per model: load (weights verified real), the fixture through the checkpoint's processors, then
  eager    : predict_action_chunk at the model's default N, 3 warm-up + 5 timed calls (wall clock, synchronised)
  compiled : model.denoise_step wrapped in torch.compile (mode "default"), same calls
Actions finite, their shape, P and S as processed, peak memory, and eager-vs-compiled action difference at the
same seed. One JSON per model under --out.

python bench/smoke_lerobot.py --model pi0 --out results/<run>
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lerobot_common as lc  # noqa: E402
import model_registry as mr  # noqa: E402


def calls(policy, batch, n_warm=3, n_timed=5, seed=0):
    out = None
    wall = []
    for i in range(n_warm + n_timed):
        policy.reset()
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        if i >= n_warm:
            wall.append((time.perf_counter() - t0) * 1e3)
    return out, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(mr.REGISTRY))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = mr.get(a.model)
    res = {"model": spec.name, "label": "smoke test, not a measurement", "checkpoint": spec.checkpoint,
           "checkpoint_revision": spec.revision, "n_field": spec.n_field, "N": spec.n_default}
    try:
        t0 = time.perf_counter()
        policy, shipped = lc.load_policy(spec)
        res["load_s"] = time.perf_counter() - t0
        res["shipped_config"] = {k: str(v) for k, v in shipped.items()}
        res["param_dtypes"] = sorted({str(p.dtype) for p in policy.parameters()})
        res["weights"] = lc.assert_real_weights(policy, spec)
        batch, _, _, binfo = lc.build_batch(policy, spec)
        res["batch"] = binfo
        res["tokens"] = lc.token_counts(policy, spec, batch)
        torch.cuda.reset_peak_memory_stats()
        act_e, wall_e = calls(policy, batch)
        res["eager"] = {"ok": True, "action_shape": list(act_e.shape), "finite": bool(torch.isfinite(act_e).all()),
                        "wall_ms_5_calls": wall_e, "peak_mem_bytes": torch.cuda.max_memory_allocated()}
        try:
            policy.model.denoise_step = torch.compile(policy.model.denoise_step, mode="default")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            act_c, wall_c = calls(policy, batch)
            d = (act_c.float() - act_e.float()).norm() / act_e.float().norm()
            res["compiled_denoise_step"] = {
                "ok": True, "mode": "default", "warmup_plus_timed_s": time.perf_counter() - t0,
                "action_shape": list(act_c.shape), "finite": bool(torch.isfinite(act_c).all()),
                "wall_ms_5_calls": wall_c, "peak_mem_bytes": torch.cuda.max_memory_allocated(),
                "rel_l2_diff_vs_eager_same_seed": d.item()}
        except Exception as exc:
            res["compiled_denoise_step"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:3000],
                                            "traceback": traceback.format_exc()[-4000:]}
    except BaseException as exc:  # a model that does not run is recorded, not skipped
        res["error"] = f"{type(exc).__name__}: {exc}"[:3000]
        res["traceback"] = traceback.format_exc()[-6000:]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"smoke_{spec.name}.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("traceback",)}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
