# 00 — Research context: what your measurements are for

## The research in one page
Flow-matching VLA models (π0, π0.5, SmolVLA, GR00T) produce robot actions in two parts. A large
vision-language **prefix** (image encoder + VLM) runs once per control cycle. A small **action expert**
(AE) then runs N = 4–10 *serially dependent* Euler denoising steps; each step pushes only ~50 action
tokens through all of the expert's layers. For π0 the expert is a Gemma-300M-class decoder: 18 layers,
hidden 1024, 8 query heads × 256 with one KV head, GeGLU FFN 4096; S = 51 suffix tokens attend to a
cached prefix of ≈ 816 tokens. One denoising step costs **38.7 GFLOPs but re-reads ≈ 623 MB of weights**:
about **60 FLOPs per byte**. The Jetson AGX Orin's GPU (≈ 42 TFLOP/s bf16, 204.8 GB/s LPDDR5) balances
at ≈ 208 FLOPs per byte, so the action expert should sit deep in the **memory-bound** region, and with
only ~50 tokens its matrix multiplies are skinny, launch-dominated kernels.

Our project moves those skinny matrix multiplies into the LPDDR5 memory itself (processing-in-memory,
modelled on SK hynix's AiM) and proposes instruction-set extensions. Every comparison is against "the
same Orin-class SoC running the action expert on its GPU". That GPU side started as numbers taken from one
published characterization paper (Zhou et al., arXiv 2604.24447) plus our analytic model; your measurements have
replaced them. The cross-model picture is `reports/SYNTHESIS_all_measurements.md`. Keep telling us where the
research is wrong.

## The ladder (π0 on AGX Orin, per denoising step, MAXN)
| rung | value | source |
|---|---|---|
| bandwidth floor, spec | 3.115 ms | 638 MB (weights + KV, T002 `floor.csv`) ÷ 204.8 GB/s |
| bandwidth floor, measured peak | 3.40 ms | 638 MB ÷ 187.47 GB/s (T005 / E1) |
| "optimistic bound (2× floor)" | 6.30 ms | 2 × floor, a stand-in for a perfect software stack — **assumed** |
| crop instead of cloning the KV cache | 10.37 ms | T003 / E4 §3.3, byte-identical actions — a variant, not the baseline |
| compiled stack, best stock mode | 16.30 ms | T003 / E4 `reports/E4_pi0_compile.md` (`reduce-overhead`, slope over N) |
| eager stack | 52.94 ms | T002 / E3 `reports/E3_pi0_eager.md` (slope over N, LeRobot π0) |

## Inputs of the hybrid model (what E11 replaces)
The system we will model next is a **hybrid**: the vision encoder and the VLM prefix run on the Jetson GPU; the
action expert's projections and its attention products against the cached prefix run inside the LPDDR5 memory
(a Ramulator-based PIM simulator on the research side gives that time); the GPU keeps, in every expert layer,
small **host segments** — for π0 six of them: `rmsnorm1`, `rope`, `softmax`, `residual_rmsnorm2`,
`gelu_gate_mul_up`, `partials_residual` — and per denoising step the suffix / time embedding, the output
projection and the Euler update. Between segments the CPU synchronises with the memory side; per layer about
2.1 MB of attention scores go to the GPU as fp32 and come back as fp16; the prefix KV cache (≈ 15 MB for π0) is
handed to the memory once per inference; and in the pipelined scheme the GPU computes the next frame's vision +
VLM prefix while the memory denoises. The research side composes

`T_step = layers × (T_pim_layer + T_host_layer) + T_extra`, `T_AE(N) = N × T_step + T_kv_load`
(in the pipelined scheme the GPU stage `T_vision + T_vlm_prefix` of the next frame overlaps `T_AE(N)`),

and **every GPU-side term of it is analytic today**: `host_op_time = max(FLOPs ÷ (42 TFLOP/s × 0.4), bytes ÷
(204.8 GB/s × 0.7)) + kernels × 20 µs`, `transfer_time = bytes ÷ (204.8 GB/s × 0.7)`. E11 (`01_measurement_plan.md`)
measures them; its `model_inputs.csv` (protocol §13.8) is what the research side's model will read.

| `model_inputs.csv` quantity | term of the research-side model it replaces | value used today |
|---|---|---|
| `t_host_segment`, `t_host_layer` | `T_host_layer` = Σ `host_op_time` over the layer's host segments | ≈ 0.16 ms per π0 layer **[ASSUMPTION-based]** |
| `n_kernels_host_segment`, `t_launch_per_kernel` | kernels per segment (1 each; 2 for `attention_gpu`) × `launch_overhead_s` | 20 µs, swept 5–50 µs **[ASSUMPTION]** |
| `achieved_tflops` (per segment) | `peak_tflops × eff_flops_frac` | 42 TFLOP/s × 0.4 = 16.8 TFLOP/s **[ASSUMPTION]** |
| `achieved_bw_GBps` (per segment) | `ext_bw × eff_bw_frac` | 204.8 GB/s × 0.7 = 143 GB/s **[ASSUMPTION]** |
| `t_host_layer` key `whole_layer_graph` | the variant in which one controller macro replaces per-segment launches (launches per layer = 1) | 1 × 20 µs |
| `t_step_extra` | `T_extra` = `host_op_time(0.41 GFLOP, 1.2 MB, 3 kernels)` | ≈ 0.08 ms per step |
| `xfer_latency_fit` (`fit_a`, `fit_b`) | `transfer_time = bytes ÷ eff_bw` — no fixed cost per transfer today | a = 0, 1/b = 143 GB/s |
| `t_sync` | the synchronisation points between memory-side work and each host segment | 0 (free) |
| `t_host_segment` key `cast_scores_fp32_fp16`, `softmax` with `dtype=fp32_in_fp16_out` | score traffic per layer, 8 × 51 × 867 × (4 + 2) B ≈ 2.1 MB | bytes ÷ 143 GB/s |
| `kv_bytes_per_layer`, `kv_bytes_total` | prefix KV load: 18 × 2 × 816 × 256 × 2 B | 15–16 MB |
| `t_kv_materialise`, `kv_copied_bytes_per_step` | GPU-side part of `T_kv_load` (the simulator gives only the memory-side write, 74 µs at 16 channels) | 0 |
| `boundary_bytes_per_step` | tensors entering and leaving the expert per step | 1.2 MB inside `T_extra` |
| `t_vision`, `t_vlm_prefix`, `prefix_latency_fit` | the GPU stage of the pipelined frame time, and its dependence on P | 110 ms + 258 ms (characterization paper), P = 816 fixed |
| `t_ae_step`, `t_ae_total`, `t_e2e` | the all-GPU baseline the hybrid is compared with | 136 ms per step, 920 ms (N = 4) |
| `prefix_throughput`, `e2e_throughput`, `ae_throughput` | frames per second of the pipelined vs the serial scheme | derived from the latencies above |
| `gpu_idle_ae_total`, `gpu_idle_frac_ae_total` | GPU time free for the next frame's prefix even in the ordinary all-GPU run | unknown |
| `prefix_slowdown`, `bg_bw_GBps` | the assumption that the prefix runs at full speed while the memory is busy denoising | slowdown = 1.0 **[ASSUMPTION]** |
| `host_layer_slowdown`, `prefix_slowdown` key `host_segments` | the assumption that host segments and the next frame's prefix do not delay each other on one GPU | 1.0 **[ASSUMPTION]** |
| `cpu_util` | the assumption that the CPU is free to drive the memory-side command stream | not modelled |
| `p_mean_W` (`<phase>/<rail>`) | GPU / module power charged to the host segments and to the prefix in the energy model | 20 W |

## Models (pins used by the research side)
| model | repository | commit | notes |
|---|---|---|---|
| π0, π0.5 | github.com/Physical-Intelligence/openpi | `15a9616a00943ada6c20a0f158e3adb39df2ccac` | JAX is the reference; **use the PyTorch port** `src/openpi/models_pytorch/`; `pytorch_compile_mode` in `pi0_config.py` (default `"max-autotune"`, wraps `sample_actions`); LIBERO client in `examples/libero/` |
| SmolVLA (and a second π0 implementation) | github.com/huggingface/lerobot | `a16f34c085c9597fcbdb9fde395a3334d78df716` | LIBERO env `src/lerobot/envs/libero.py`; compile flags in `examples/rtc/eval_dataset.py` (`use_torch_compile`, `torch_compile_mode`, …) |
| GR00T N1.7 (optional) | github.com/NVIDIA/Isaac-GR00T | `376ba890cff8c9de64d71d982772a9c36185fdd7` | has Orin support: `scripts/activate_orin.sh`, `scripts/deployment/orin/`, and `scripts/deployment/benchmark_inference.py` (eager / compile / TensorRT, per-component timing) |
If a pinned commit does not run on the Jetson, use the nearest one that does and record the hash and why.
π0 is measured with **LeRobot's port** (`lerobot @ f0d2b37`, `lerobot/pi0_libero_finetuned_v044`; owner
decision, T002), not openpi's — a different fine-tune, so paper comparisons are across implementations.
All four models go through the full set of experiments (E3, E4, E11, E5, E5b, E6, E7); π0 first, then π0.5 and
SmolVLA, then GR00T N1.7. "Optional" for GR00T means "last", not "skipped if it runs".

**Environment reality (plan for it).** The three repositories pin different stacks: openpi wants Python ≥ 3.11 with
a specific torch and also JAX; lerobot wants a newer Python again; GR00T's Orin path uses the system Python 3.10
with NVIDIA's Jetson wheels. On the Jetson, GPU-enabled PyTorch comes from NVIDIA's JetPack wheels / containers
(and the community Jetson wheel index for JetPack 6 + CUDA 12.6), not from PyPI. Use **one environment per
model**, relax the repositories' torch / Python pins to what the Jetson wheels provide, and record every deviation
in `env/`. The measured software stack is part of the result.

Expert dimensions for cross-checking what you observe: π0 / π0.5 — 18 layers, hidden 1024, S = 51 / 50,
311 M expert parameters, 623 MB per step. SmolVLA — 16 expert layers (8 self + 8 cross), hidden 720,
S = 50, N = 10, 98 M parameters. GR00T N1.7 — 16 DiT blocks, hidden 1536, S = 41, N = 4, 692 M parameters.

**Hardware constants are the Orin's own.** 204.8 GB/s, ≈ 42 TFLOP/s bf16 and the ≈ 208 FLOP/B ridge are our
expectations for this SoC, to be verified in E0/E1; constants you may know from profiling guides for desktop
GPUs (bandwidth, TFLOP/s, SM count, ridge, warps per SM) do not apply here (protocol §8).
