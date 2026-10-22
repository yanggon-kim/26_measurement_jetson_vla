"""nsys target: one model with the protocol §2 regions installed, warm-up outside the capture window, then
--inferences inferences between cudaProfilerStart / Stop (T004 step 2 split check; reused by T006 for E5 captures).

nsys profile -t cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi --capture-range-end=stop -f true \
     -o scratch/<run>/<name> python bench/nsys_regions_target.py --model groot_n17 --inferences 2
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regions as R  # noqa: E402
import run_e3 as E3  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inferences", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--precision", choices=["eager_fp32", "shipped"], default="eager_fp32")
    a = ap.parse_args()
    E3.set_precision(a.precision)
    ad = E3.make_adapter(a.model)
    if isinstance(ad, E3.GrootAdapter):
        ad.capture_tokens()
    ad.set_n(ad.n_default)
    rec = R.Recorder()
    undo = ad.instrument(rec)
    for _ in range(a.warmup):
        rec.reset()
        ad.iteration()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for i in range(a.inferences):
        rec.reset()
        rec.start()
        with torch.cuda.nvtx.range(f"inference_{i}"):
            ad.iteration()
        rec.stop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    undo()
    print("calls per region:", rec.counts, flush=True)


if __name__ == "__main__":
    main()
