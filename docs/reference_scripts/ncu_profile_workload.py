#!/usr/bin/env python3
"""
NCU-Targeted Profiling Script (Generalized Template)
=====================================================
Designed for NSight Compute profiling — minimal overhead, single pass,
with NVTX markers for targeted kernel capture and cudaProfiler markers.

Usage with NCU:
  # Profile a specific phase:
  ncu --set full --nvtx --nvtx-include "phase_name/" \
      --profile-from-start off -o profiles/workload_ncu_phase -f \
      python scripts/ncu_profile_workload.py --phase <phase_name>

  # Profile all phases:
  ncu --set full --profile-from-start off \
      -o profiles/workload_ncu_all -f \
      python scripts/ncu_profile_workload.py --phase all

  # Run without NCU to verify correctness:
  python scripts/ncu_profile_workload.py --phase all

ADAPT THIS SCRIPT:
  1. Modify load_model_and_inputs() — same as profile_workload.py
  2. Modify run_profiled_inference() — add NVTX markers for YOUR model's phases
  3. Add your phase names to the --phase argument choices
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent


# ============================================================================
# ADAPT: Model Loading (same as profile_workload.py)
# ============================================================================

def load_model_and_inputs(batch_size=1):
    """Load model and create inputs. ADAPT for your workload."""
    # === PLACEHOLDER — REPLACE WITH YOUR MODEL ===
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 512),
    ).cuda().to(dtype=torch.bfloat16).eval()

    dummy_input = torch.randn(batch_size, 128, 512, device="cuda", dtype=torch.bfloat16)
    # === END PLACEHOLDER ===

    return model, dummy_input


# ============================================================================
# ADAPT: Inference with NVTX Markers for NCU Capture
# ============================================================================

def run_profiled_inference(model, inputs, phase="all"):
    """
    Run inference with NVTX markers for NCU capture.

    ADAPT THIS: Add NVTX markers for each phase of YOUR model.
    NCU's --nvtx-include flag will use these names to filter kernels.

    Example phases for a transformer:
    - "encoder_forward"
    - "decoder_forward"
    - "attention_layer_0"

    Example phases for a diffusion model:
    - "backbone_forward"
    - "dit_denoising_loop"
    - "denoise_step_0", "denoise_step_1", ...
    """
    with torch.inference_mode():
        # === ADAPT: Add your model's phases with NVTX markers ===
        torch.cuda.nvtx.range_push("inference")
        torch.cuda.nvtx.range_push("forward")
        output = model(inputs)
        torch.cuda.nvtx.range_pop()  # forward
        torch.cuda.nvtx.range_pop()  # inference
        # === END ADAPT ===

    return output


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="NCU-Targeted Profiling")
    parser.add_argument("--phase", default="all",
                        help="Phase to profile (matches NVTX range name)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup iterations")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading model...")
    model, inputs = load_model_and_inputs(batch_size=args.batch_size)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    print(f"Warming up ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        with torch.inference_mode():
            _ = model(inputs)
    torch.cuda.synchronize()

    print(f"Running profiled inference (phase={args.phase})...")
    torch.cuda.synchronize()

    # cudaProfilerStart/Stop tells NCU when to capture
    # (use with --profile-from-start off)
    torch.cuda.cudart().cudaProfilerStart()
    output = run_profiled_inference(model, inputs, phase=args.phase)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print("Done.")


if __name__ == "__main__":
    main()
