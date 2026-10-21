# 00 — Research context: what your measurements are for

## The research in one page
Flow-matching VLA models (π0, π0.5, SmolVLA, GR00T) produce robot actions in three parts. A **vision** encoder
turns the camera images into tokens; a vision-language model (**VLM prefix**, also "language" or "prefill") runs
once per control cycle over image and prompt tokens and builds a KV cache; then a small **action expert** (AE)
runs N = 4–10 *serially dependent* denoising steps, each pushing only ~40–50 action tokens through all of the
expert's layers. For π0 the expert is a Gemma-300M-class decoder: 18 layers, hidden 1024, 8 query heads × 256 with
one KV head, GeGLU FFN 4096, S = 51 suffix tokens attending to a cached prefix of ≈ 816 tokens. One denoising step
costs ≈ 38 GFLOP but re-reads ≈ 623 MB of weights, about 60 FLOP per byte. Its matrix multiplies are skinny
(M ≈ 50 against K, N = 1024–4096).

Our project moves those skinny matrix multiplies into the LPDDR memory itself (processing-in-memory) and proposes
instruction-set extensions. The comparison is always against "the same SoC running the action expert on its GPU",
so the GPU side must be measured, not assumed.

## Why Thor
On the Orin, after optimisation, **no model's action expert dominated its frame** (the largest share was 48.6 %,
SmolVLA; `reference/orin/reports/E4i_best_combinations.md`). Thor has a much larger GPU but only somewhat more
memory bandwidth (LPDDR5X; NVIDIA quotes ≈ 273 GB/s, a hint until T003 measures it). The hypothesis this branch
tests: **on Thor the compute-heavy vision and VLM prefix shrink a lot, the memory-bound action expert shrinks
less, and the expert becomes the dominant phase again.** If so, VLA is a PIM workload again. If not, that is the
result. Nothing is decided; report what the device does.

## The three deliverables (per model, for the baseline and the optimised configuration)
| # | deliverable | what it is for | where |
|---|---|---|---|
| 1 | **latency breakdown** into vision, VLM prefix and action expert (+ per denoising step, + "other") | the phase shares: does the expert dominate on Thor? | T004, T005 |
| 2 | **roofline inputs**: the device's measured ceilings, and per part **and per kernel** the FLOPs, the DRAM bytes and the achieved rate | where each part and each kernel sits against Thor's own roof; the hybrid model's GPU-side terms | T003, T006 |
| 3 | **bottleneck analysis** of every significant kernel of every part: what it is bound by (counters, stalls, tensor-pipe use, instruction mix) | whether the expert's kernels are memory-bound on Thor, and what limits vision and prefix | T007, T008 |

**Baseline** = the rung `eager_fp32` (eager PyTorch, `matmul_precision="highest"`, the checkpoint's shipped dtypes).
**Optimised** = the fastest **BF16** configuration T005's search finds on Thor for that model (`best_bf16`); it may
differ from the Orin's best rung, and that difference is a result. The rung names are the Orin's
(`reference/orin/reports/E4_PARTS_ABC_GUIDE.md` §4), so the two devices compare rung by rung.

## Models (pins; one environment per model)
| model | code | code commit | checkpoint (Hugging Face) | checkpoint revision | N (its own default; field) |
|---|---|---|---|---|---|
| π0 | LeRobot (`PI0Policy`) | `f0d2b37bebddf7e8852cb624712da0ad564601b6` | `lerobot/pi0_libero_finetuned_v044` | `45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f` | 10 (`num_inference_steps`) |
| π0.5 | LeRobot (`PI05Policy`) | same | `lerobot/pi05_libero_base` | `a217bfd3b14673cf2ce597e69997ab21866438dd` | 10 (`num_inference_steps`) |
| SmolVLA | LeRobot | same | `lerobot/smolvla_base` | `d9f33c94a60fb382c90dea2164c96845bd955e28` | 10 (`num_steps`) |
| GR00T N1.7 | Isaac-GR00T | `376ba890cff8c9de64d71d982772a9c36185fdd7` | `nvidia/GR00T-N1.7-3B` | `2fc962b973bccdd5d8ce4f67cc63b264d6886495` | 4 (`num_inference_timesteps`) |
| GR00T N1.6 | Isaac-GR00T, branch `n1d6` | `9b37aa1ce69c73c6d165233fa88128283bba4508` | `nvidia/GR00T-N1.6-3B` | `d0814e7ecb19202e7c8468b46098b0b7ef3a6d61` | 4 |
| GR00T N1.5 | Isaac-GR00T, branch `n1d5` | `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` | `nvidia/GR00T-N1.5-3B` | `869830fc749c35f34771aa5209f923ac57e4564e` | 4 |
| LIBERO (simulator, E8) | `Lifelong-Robot-Learning/LIBERO` | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | — | — | — |

If a pinned commit does not run on this device's stack, use the nearest one that does, record the hash and why,
and say so in the report: the measured software stack is part of the result. Inputs are the Orin's own fixtures
(`reference/orin/bench/fixtures/`: the LIBERO observation for the LeRobot models; the GR00T harness's DROID input,
2 cameras × 2 frames, P = 273, S = 41, `reference/orin/bench/groot_*.py`), so that every Thor number has an Orin
partner at the same input.

## The Orin numbers to compare against (Orin, MAXN, ms, median; `reference/orin/reports/E4i_best_combinations.md`)
| model | N | Orin best rung | best e2e | vision | prefix | expert | AE % | eager e2e | vision | prefix | expert | AE % |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| π0 | 10 | `bf16_both` | 424.46 | 56.74 | 192.86 | 168.68 | 39.7 | 1069.04 | 241.01 | 234.57 | 586.91 | 54.9 |
| π0.5 | 10 | `bf16_both_graphs` | 487.81 | 56.57 | 239.47 | 185.60 | 38.0 | 3028.56 | 239.45 | 2162.47 | 617.74 | 20.4 |
| SmolVLA | 10 | `vlm_graphs` | 92.30 | 31.26 | 7.93 | 44.86 | 48.6 | 727.46 | 33.30 | 65.77 | 620.56 | 85.3 |
| GR00T N1.7 | 4 | `vlm_compile` | 240.49 | 70.82 | 87.59 | 79.57 | 33.1 | 372.48 | 70.62 | 88.56 | 211.12 | 56.7 |
| GR00T N1.6 | 4 | `all_compile_autotune` | 174.59 | 48.80 | 52.61 | 72.83 | 41.7 | 327.22 | 61.94 | 61.70 | 201.33 | 61.5 |
| GR00T N1.5 | 4 | `all_compile_autotune` | 168.75 | 32.22 | 70.73 | 60.55 | 35.9 | 255.22 | 50.20 | 100.62 | 98.45 | 38.6 |

Other Orin results each task compares with are named in that task (its "Orin counterpart" line).

## The research side's projection, recorded before any Thor measurement (2026-10-08; not acceptance criteria)
`reference/orin/analysis/thor_projection.md` scaled the Orin numbers by factors from NVIDIA's own GR00T N1.7
timings (Orin ÷ Thor, same harness and input). It projects, for the best configuration: π0 198.5 ms (AE share
39.7 → 50.4 %), π0.5 226.0 (38.0 → 48.7 %), SmolVLA 46.9 (48.6 → 56.7 %), GR00T N1.7 108.7 (33.1 → 43.4 %),
N1.6 81.7 (41.7 → 52.8 %), N1.5 78.3 (35.9 → 45.8 %); front (vision + prefix) ÷ 2.65 and expert ÷ 1.69. It is a
projection to be tested, not a target. **Do not tune anything toward it**; T009 compares it with what you measured.

## Hardware and software hints (unverified — T001 and T003 detect or measure each one)
| item | hint | source of the hint |
|---|---|---|
| GPU | Blackwell, 2,560 CUDA cores, 96 tensor cores; ≈ 20 SMs (derived at 128 cores per SM); up to ≈ 1.575 GHz | NVIDIA product pages; RidgeRun JetPack 7.0 wiki |
| compute capability | 11.0 → `sm_110` under CUDA 13 (some CUDA 12.x tools called it `sm_101`) | community reports |
| memory | 128 GB LPDDR5X, 256-bit, ≈ 273 GB/s spec, EMC ≈ 4,266 MHz; L2 size unconfirmed (third parties say 50 MB) | NVIDIA; `thor_projection.md` |
| CPU | 14-core Arm Neoverse-V3AE | NVIDIA |
| software | JetPack 7.x (Jetson Linux r38 / r39), CUDA 13.x, Ubuntu 24.04 → Python 3.12; Jetson aligned with the Arm SBSA standard, so CUDA-enabled PyTorch may come from PyTorch's own `cu130` aarch64 (SBSA) wheels, NVIDIA's NGC PyTorch container or the Jetson AI Lab index | NVIDIA JetPack 7 announcement; community reports |
| power modes | `MAXN` and `120W` at least; `120W` may be the default | RidgeRun JetPack 7.0 wiki |
| profiling | whether `ncu` needs root, whether `dram__*` counters exist, and the tensor-pipe metric names are unknown | — |
| GR00T | NVIDIA's Isaac-GR00T `scripts/deployment/README.md` @ `376ba890` has a same-harness table for Orin and Thor | `reference/orin/analysis/thor_projection.md` |

**Environment reality (plan for it).** The three code bases pin different stacks. On Orin, GPU PyTorch came from
NVIDIA's Jetson wheels, pip silently replaced them with CPU wheels, and one environment per model was needed
(`reference/orin/env/README.md` §3). Expect the same class of problems with a newer Python and CUDA: install the
GPU wheels first, install model repositories with `--no-deps`, and re-check `torch.cuda.is_available()` after every
install step.
