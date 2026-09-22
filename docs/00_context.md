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
same Orin-class SoC running the action expert on its GPU". **Until now that GPU side was never
measured by us.** It is a ladder of numbers taken or derived from one published characterization paper
(Zhou et al., *Characterizing Vision-Language-Action Models across XPUs*, arXiv 2604.24447) plus our own
analytic model. You are here to put measured numbers on that ladder, and to tell us where it is wrong.

## The ladder we use today (π0 on AGX Orin, per denoising step)
| rung | value | where it comes from | status |
|---|---|---|---|
| bandwidth floor | 3.15 ms | 629 MB of weights (+KV) ÷ 204.8 GB/s | analytic |
| "optimistic bound" | 6.30 ms | 2 × floor, a stand-in for a perfect software stack | **assumed** |
| ~~compiled stack~~ | ~~≈ 90 ms~~ | 136 ms ÷ 1.51, the paper's `torch.compile` gain on *Jetson Thor* | **superseded**: 5.5× too pessimistic |
| **compiled stack (best stock mode)** | **16.30 ms** | **T003 / E4, `reports/E4_pi0_compile.md`** — `reduce-overhead`, slope over N ∈ {1,2,4,6,8,10} | **measured**, 5.2× the floor |
| *(one-line fix: crop instead of cloning the KV cache)* | *10.37 ms* | *T003 / E4 §3.3, byte-identical actions* | *measured, 3.3× the floor — **1.67× the "optimistic bound" that was assumed unreachable*** |
| ~~measured stack~~ | ~~136 ms~~ | the paper's 543 ms action-expert phase ÷ N = 4 | **superseded**: derived, and 2.6× too slow |
| **measured stack (eager)** | **52.94 ms** | **T002 / E3, `reports/E3_pi0_eager.md`** — slope fit over N ∈ {1,2,4,6,8,10}, r² = 0.99999, MAXN, LeRobot π0 | **measured on this Orin**, 17.0× the floor |

## Numbers you replace or validate
| quantity | what we use today | source | your measurement |
|---|---|---|---|
| π0 end-to-end latency on Orin | ~~920 ms (N = 4)~~ → **measured 694.1 ms** | paper Table 1 | **E3 done** |
| phase split vision / VLM / action expert | ~~110 / 258 / 543 ms = 12 / 28 / 59 %~~ → **measured 239.4 / 233.1 / 215.5 ms = 34.5 / 33.6 / 31.1 % at N = 4** | paper Fig. 5 | **E3 done — the split is the big disagreement, see below** |
| action-expert time **per denoising step** | ~~136 ms~~ → **measured 52.94 ms** | derived | **E3 done** (the most important single number) |
| action-expert share of latency vs N | ~~59.1 % (N = 4), 78.3 % (N = 10)~~ → **measured 31.1 % (N = 4), 52.7 % (N = 10)** | derived | **E3 done** |
| `torch.compile` gain on Orin | ~~1.51× (Thor's) → ≈ 90 ms/step~~ → **measured 3.22× per step** (1.39× e2e at N = 4, 1.69× at N = 10) | transplanted | **E4 done** — a single e2e factor is not transferable across N; the slope is |
| GPU utilisation in the AE phase — the paper's "SM utilisation": streaming-multiprocessor (SM) occupancy as Nsight Compute (`ncu`) reports it; you report ncu SM occupancy / activity per region, with tegrastats `GR3D_FREQ` only as a coarse cross-check | 22 % | paper Fig. 5 | E5 |
| per-pipe utilisation (LSU / FMA / ALU / …) and warp-stall reasons in `vision`, `vlm_prefix`, `ae_step` | unknown | — | E5 |
| what the dominant action-expert kernels stall on, at instruction level (memory loads vs compute vs launch), vs S and under `torch.compile` | asserted "memory-bound", never shown | — | **E5b** |
| where the ≈ 43× over the floor goes: kernel time vs launch gaps vs idle | unknown | — | E5 |
| per-kernel launch overhead | 20 µs (we sweep 5–50) | **assumed** | E2, E5 |
| achieved fraction of peak bandwidth / FLOPs on small kernels | 0.7 / 0.4 | **assumed** | E1, E2 |
| peak memory bandwidth and bf16 throughput | 204.8 GB/s, 42 TFLOP/s | NVIDIA spec | E1 |
| arithmetic intensity of an AE layer | 60 (ours) vs 64.5 (paper) FLOPs/B | analytic | E6 |
| token count where the GPU stops being launch-bound (our "S ≈ 16 crossover") | analytic | — | E2 sweep over S |
| real prefix length P | 816 | ~~assumed~~ → **measured 816** = 3 images × 256 + 48 language tokens (only 533 valid) | **E3 done — the assumption was right** |
| module power during the AE phase (sum of the rails in protocol §4) | 20 W (we sweep 15–30) | **assumed** | E7 |
| energy per denoising step, measured stack | 3,335 mJ | 136 ms × 20 W + DRAM model | E7 |
| closed-loop quality vs number of denoising steps and vs *stale* steps | paper Table 7 | paper | E8 |
| is 2 × floor really out of reach for the best engineered stack? | assumed yes | — | E10 |

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

## What E3 (T002) changed, 2026-09-21

**The action expert does not dominate π0's latency on this Orin.** At the paper's N = 4 it is
the *smallest* of the three phases (31.1 %) and `vision` is the largest (34.5 %); even at the
model's default N = 10 the expert reaches only 52.7 %. The hypothesis that it is
**memory-bound** is confirmed independently (floor 3.115 ms memory vs 0.912 ms compute), but
"dominates latency" is not supported. The PIM case should be argued from the expert's
**17× gap to its own floor**, and from N = 10 where it is half the frame, not from a claimed
59–78 % share.

Three costs worth knowing before modelling anything.

**80.1 ms of every inference encodes a dummy image** — `empty_cameras: 1` pushes an all-−1
third camera through SigLIP (33.3 % of `vision`), and its attention mask is zero, so the
result is discarded.

**`kv_copied_bytes_per_step` is not 0.** `modeling_pi0.py:919` runs
`copy.deepcopy(past_key_values)` inside `denoise_step`: all 36 KV tensors, **15,040,512 B**,
cloned **every** step, measured at **5.68 ms/step** — 10.4 % of a step, and 1.82× the step's
whole 3.115 ms memory floor. It is Python/allocator cost (2.65 GB/s effective), not
bandwidth. The hybrid model's `t_kv_materialise` / `kv_copied_bytes_per_step` assumptions of
zero should be revised. (LeRobot-specific; openpi's loop does not do this.)

*Corrected by T003/E4.* An earlier version of this paragraph suggested the copy might be
unnecessary because the expert passes `use_cache=False`. **That was wrong, and the experiment
was run:** deleting it makes the cache grow to P + 2S = 918 against an 867 mask and the model
fails on step 2 — the expert appends its suffix K/V regardless. The copy is *load-bearing*.
What is **not** needed is the 15 MB **clone**: `DynamicCache.crop(P)` after the forward gives
byte-identical actions (verified bitwise, max abs diff 0.000e+00) and takes the compiled step
from 16.30 to **10.37 ms** while removing **all four** of the model's graph breaks — every one
of which is this line, since dynamo cannot trace `copy.deepcopy`. See
`reports/E4_pi0_compile.md` §3.

**Denoising step 0 costs +0.83 ms** over the rest. An earlier version of this section said
+3.13 ms; that figure was ~74 % measurement artefact and is withdrawn.

**`T_extra` ≈ 0.08 ms per step is out by two orders of magnitude.** Of a 54.63 ms step, the
18 expert layers account for ~44.9 ms (itself carrying ~2.6 ms of event overhead); the rest —
KV deep copy, `embed_suffix`, mask construction, final norm, `action_out_proj`, Python — is
**~10 ms**.

**Implementation used.** E3 measured **LeRobot's π0** (`lerobot @ f0d2b37beb`, checkpoint
`lerobot/pi0_libero_finetuned_v044`), not openpi's PyTorch port — the openpi route costs a
12 GB orbax conversion, a mandatory JAX import, a `transformers == "4.53.2"` string-equality
check and a Python ≥ 3.11 gate, none of which measures anything. LeRobot's is a direct port of
the same architecture, but **the weights are a different fine-tune**, so the paper comparison
is across implementations *and* checkpoints. Running openpi later is the obvious way to test
whether the phase-split disagreement above is implementation or measurement.

## Models (pins used by the research side)
| model | repository | commit | notes |
|---|---|---|---|
| π0, π0.5 | github.com/Physical-Intelligence/openpi | `15a9616a00943ada6c20a0f158e3adb39df2ccac` | JAX is the reference; **use the PyTorch port** `src/openpi/models_pytorch/`; `pytorch_compile_mode` in `pi0_config.py` (default `"max-autotune"`, wraps `sample_actions`); LIBERO client in `examples/libero/` |
| SmolVLA (and a second π0 implementation) | github.com/huggingface/lerobot | `a16f34c085c9597fcbdb9fde395a3334d78df716` | LIBERO env `src/lerobot/envs/libero.py`; compile flags in `examples/rtc/eval_dataset.py` (`use_torch_compile`, `torch_compile_mode`, …) |
| GR00T N1.7 (optional) | github.com/NVIDIA/Isaac-GR00T | `376ba890cff8c9de64d71d982772a9c36185fdd7` | has Orin support: `scripts/activate_orin.sh`, `scripts/deployment/orin/`, and `scripts/deployment/benchmark_inference.py` (eager / compile / TensorRT, per-component timing) |
If a pinned commit does not run on the Jetson, use the nearest one that does and record the hash and why.
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
