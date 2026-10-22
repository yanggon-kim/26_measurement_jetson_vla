"""T003 follow-up: do the E2 projection GEMMs really move more than E1's 258 GB/s from DRAM?

E2 timed gate/up_proj at S = 256 at ≈ 286 GB/s of weight + activation bytes (kernel ÷ floor < 1). This runs ONE
F.linear of a π0 expert projection inside NVTX "e2chk" for ncu (default cache control: L2 flushed before the
launch, i.e. cold), with the real layer-0 weight or a random weight of the same shape (to rule out data-dependent
effects such as compression). ncu gives DRAM bytes (L2 sysmem fill / write) and duration.

~/venvs/lerobot/bin/python bench/ncu_e2_check.py --proj gate_proj --S 256 --weights real
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--weights", choices=["real", "random"], default="real")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    ck = snapshot_download("lerobot/pi0_libero_finetuned_v044", revision="45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f",
                           local_files_only=True)
    f = safe_open(os.path.join(ck, "model.safetensors"), "pt", device="cpu")
    key = next(k for k in f.keys() if "gemma_expert" in k and ".layers.0." in k and k.endswith(f"{a.proj}.weight"))
    w = f.get_tensor(key).to("cuda", torch.bfloat16)
    if a.weights == "random":
        w = torch.randn_like(w) * w.float().std().to(torch.bfloat16)
    torch.manual_seed(0)
    x = torch.randn(a.S, w.shape[1], device="cuda", dtype=torch.bfloat16)
    for _ in range(10):
        F.linear(x, w)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    with torch.cuda.nvtx.range("e2chk"):
        F.linear(x, w)
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    N, K = w.shape
    print("ANALYTIC " + json.dumps({"proj": a.proj, "S": a.S, "weights": a.weights, "key": key,
                                    "read": N * K * 2 + a.S * K * 2, "write": a.S * N * 2, "flops": 2 * a.S * N * K}))


if __name__ == "__main__":
    main()
