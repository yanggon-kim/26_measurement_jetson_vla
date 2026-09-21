"""T002 / E3 — cost of the `preprocess` and `postprocess` regions (protocol §2).

`bench/run_e3.py` times `predict_action_chunk`, i.e. the policy forward. The LeRobot
processor pipeline (rename -> batch -> newline -> tokenize -> to-device -> normalize) runs
*outside* it, once per observation, and in a real control loop it is part of the per-frame
cost. It is measured separately here rather than folded in, because it is almost entirely
CPU work plus one host-to-device copy and would otherwise hide inside a GPU region.

    python bench/measure_preprocess.py --timed 100 --out results/<run>/preprocess.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pi0_common as P  # noqa: E402
from common import REPO, summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    P.gate()
    policy, cfg, _ = P.load_policy(num_inference_steps=10, compile_mode=None)
    P.assert_real_weights(policy)
    fx = P.load_fixture()
    fixtures = REPO / "bench" / "fixtures"

    # the raw observation as a control loop would hand it over, rebuilt each iteration
    raw = {}
    for key, spec in fx["images"].items():
        im = Image.open(fixtures / spec["file"]).convert("RGB")
        raw[key] = np.asarray(im, dtype=np.uint8)
    state = np.asarray(fx["state"], dtype=np.float32)

    from lerobot.policies.factory import make_pre_post_processors
    preprocess, postprocess = make_pre_post_processors(
        policy.config, P.CHECKPOINT, pretrained_revision=P.REVISION,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    def build_obs():
        obs = {}
        for k, arr in raw.items():
            a = arr.astype(np.float32) / 255.0
            obs[k] = torch.from_numpy(a).permute(2, 0, 1).contiguous()
        obs["observation.state"] = torch.from_numpy(state.copy())
        obs["task"] = fx["task"]
        return obs

    # one chunk to postprocess
    batch = preprocess(build_obs())
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()

    rows, pre_ms, post_ms, host_ms = [], [], [], []
    for i in range(args.warmup + args.timed):
        t0 = time.perf_counter()
        obs = build_obs()                       # uint8 -> float32 CHW, host side
        t1 = time.perf_counter()
        b = preprocess(obs)                     # tokenize + normalize + H2D
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        _ = postprocess(chunk)                  # unnormalize + D2H
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if i >= args.warmup:
            j = i - args.warmup
            host_ms.append((t1 - t0) * 1e3)
            pre_ms.append((t2 - t1) * 1e3)
            post_ms.append((t3 - t2) * 1e3)
            rows += [{"region": "host_image_to_tensor", "iter": j, "ms": host_ms[-1]},
                     {"region": "preprocess", "iter": j, "ms": pre_ms[-1]},
                     {"region": "postprocess", "iter": j, "ms": post_ms[-1]}]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["region", "iter", "ms"])
        w.writeheader()
        w.writerows(rows)
    for name, v in (("host_image_to_tensor", host_ms), ("preprocess", pre_ms),
                    ("postprocess", post_ms)):
        s = summarize(v)
        print(f"  {name:22s} median {s['median_ms']:7.3f} ms  "
              f"p5-p95 {s['p5_ms']:.3f}-{s['p95_ms']:.3f}  n={s['n']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
