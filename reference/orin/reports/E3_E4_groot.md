# E3 / E4 — GR00T N1.7: regions and per-step time over N (T009 step 1)

Device: Jetson AGX Orin 64 GB, MAXN, GPU pinned 1300.5 MHz, 16 SMs, EMC 3199 MHz. Model
`nvidia/GR00T-N1.7-3B` through Isaac-GR00T @ `376ba890`, embodiment
`oxe_droid_relative_eef_relative_joint`, bf16, TF32 on (`matmul_precision=high`), 20 warm-up +
50 timed frames per point, one process per point. Harness `bench/groot_phases.py`; queue
`bench/queues/queue_prereboot.sh` stage `e3e4_sweep`.

## 1. Measured — per-region median (ms) at every N

`vision` = Qwen3-VL image encoder (`backbone.model.visual`), `vlm_prefix` = backbone minus vision
(derived as backbone − vision, both measured in the same frame), `ae_total` =
`action_head.get_action` (contains the N-step loop), `data_processing_cpu` = the checkpoint's own
processor + collator, timed separately and **not** inside `e2e`.

| N | rung | vision | vlm_prefix | ae_total | e2e (GPU path) | data proc (CPU) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | eager_tf32 | 69.86 | 86.28 | 48.52 | 206.89 | 9.40 |
| 2 | eager_tf32 | 69.83 | 86.81 | 102.24 | 261.20 | 9.43 |
| 4 | eager_tf32 | 70.32 | 87.08 | 205.72 | 365.25 | 9.36 |
| 6 | eager_tf32 | 70.04 | 85.35 | 304.97 | 462.59 | 9.41 |
| 8 | eager_tf32 | 70.25 | 88.22 | 420.04 | 580.63 | 9.40 |
| 10 | eager_tf32 | 69.80 | 85.03 | 511.01 | 667.97 | 9.47 |
| 1 | vlm_compile | 70.10 | 84.16 | 25.41 | 182.07 | 9.62 |
| 2 | vlm_compile | 70.09 | 84.11 | 43.55 | 200.03 | 9.27 |
| 4 | vlm_compile | 70.81 | 85.11 | 79.93 | 238.07 | 9.29 |
| 6 | vlm_compile | 70.28 | 84.52 | 116.48 | 273.37 | 9.22 |
| 8 | vlm_compile | 70.35 | 84.30 | 152.05 | 309.10 | 9.33 |
| 10 | vlm_compile | 70.27 | 84.48 | 188.71 | 345.80 | 9.36 |

Every cell is in `results/20260925_E3E4_groot_n17_N{N}_{rung}/phase_split.csv` (median, mean,
p5, p95, min, max, n).

`vlm_compile` = `action_head.get_action@reduce-overhead` (+clone) and
`backbone.model.language_model.forward@max-autotune-no-cudagraphs` — the best rung of the
committed E4d ladder (`E4i_best_combinations.md`).

## 2. Derived — per-step time (protocol §2 definition (b): least-squares slope over the six N)

| rung | region | slope ms/step | intercept ms | r² |
|---|---|---:|---:|---:|
| eager_tf32 | ae_total | **51.75** | −1.96 | 0.99939 |
| eager_tf32 | e2e | 51.68 | 157.08 | 0.99899 |
| vlm_compile | ae_total | **18.13** | 7.35 | 0.99999 |
| vlm_compile | e2e | 18.17 | 164.21 | 0.99990 |
| either | vision | ≤ 0.014 | ~70.0 | ≤ 0.03 (flat) |
| either | vlm_prefix | ≤ 0.064 | ~85 | ≤ 0.04 (flat) |

Compilation takes the denoising step from 51.75 to 18.13 ms (**2.85×**); vision and prefix are
unchanged by this rung (only the language model is autotuned without graphs, and it gained
< 1 ms). The compiled `ae_total` has a 7.35 ms N-independent part (graph launch + the
pre/post-loop work inside `get_action`); the eager intercept is ≈ 0.

## 3. Floor and gap (protocol §9) — derived

Floor per step from `results/20260925_E6_groot_n17/floor.csv`: **17.29 ms** at the measured
187.47 GB/s (bytes = DiT weights read by one step, 3.241 GB; this is a **lower bound** on bytes —
the cross-attended VLM context is not counted, and FLOPs are **not counted** because the DiT's
alternating self/cross-attention does not fit the π0 formula, so there is no compute floor).

| rung | ms/step (slope) | floor | gap ratio |
|---|---:|---:|---:|
| eager_tf32 | 51.75 | 17.29 | **2.99×** |
| vlm_compile | 18.13 | 17.29 | **1.05×** |

The compiled DiT step is within 5 % of the weight-streaming floor: at this rung GR00T's action
head is memory-bound and there is essentially nothing left to take by launch or scheduling
work. (E6's `ae_step` row uses ae_total/N at N = 4, 19.89 ms → 1.15×, which folds the 7.35 ms
intercept in; the slope is the §2 primary.)

## 4. NVIDIA's own `scripts/deployment/benchmark_inference.py` — measured, compared at N = 4

First attempt (`results/20260925_E3E4_groot_n17_benchmark_inference/`) **failed** decoding the DROID demo
video: `ImportError: torchcodec is required for video decoding`. Kept as the record. Fix, a labelled
environment change (`env/INSTALL_LOG.md`): torchcodec 0.10.0a0 installed `--no-deps` from the repository's
own Orin wheel, as its `scripts/deployment/orin/install_deps.sh` does; `uv pip freeze` before/after differs
only by that line, so torch 2.10 and every earlier GR00T number are unaffected. Re-run:
`results/20260925_E3E4_groot_n17_benchmark_inference_torchcodec/` (MAXN, GPU 1300.5 MHz, 20 iterations,
5 warm-up, the checkpoint's own N = 4), 2026-09-25 22:45.

| component (N = 4, median ms) | NVIDIA eager | ours `eager_tf32` | NVIDIA `torch.compile` | ours `vlm_compile` |
|---|---:|---:|---:|---:|
| data processing (CPU) | 14.73 | 9.36 | 14.73 | 9.29 |
| backbone (vision + VLM) | 192.15 | 157.40 (70.32 + 87.08) | 192.29 | 155.92 (70.81 + 85.11) |
| action head (4 DiT steps) | **204.42** | **205.72** | 89.25 | 79.93 |
| E2E | 411.5 | 365.25 ⁵ | 296.2 | 238.07 ⁵ |

⁵ ours is the GPU path only (`model.get_action` on collated inputs); NVIDIA's E2E adds its data processing.

**The eager action head agrees to 0.6 %** (204.42 vs 205.72 ms) — the one component both scripts time
the same way. The others differ by *definition*, read from the script (`benchmark_inference.py:220-290,
525-564`), not by measurement error:

* **backbone +34–36 ms**: NVIDIA's timer wraps `policy.model.prepare_input(...)` *and* the backbone, with a
  `torch.cuda.synchronize()` + `perf_counter` pair; ours times `backbone.model.visual` and the backbone with
  CUDA events and excludes `prepare_input`. NVIDIA's input is a real DROID frame from the demo episode;
  ours is the checkpoint-shaped fixture (same keys and shapes; pixel values random). Matmul precision:
  NVIDIA leaves PyTorch's default, ours sets TF32 (`matmul_precision=high`). Which of the three accounts for
  how much is **not measured**.
* **compiled action head 89.25 vs 79.93 ms**: NVIDIA compiles only the DiT's
  `action_head.model.forward` with `mode="max-autotune"` (its own graphs per call) and leaves the rest of
  `get_action` eager; ours captures the whole `get_action` with `reduce-overhead` (one graph tree for the
  N-step loop). Its backbone is not compiled at all (192.15 → 192.29 ms).
* **data processing 14.7 vs 9.4 ms**: real video decode + processor on NVIDIA's side vs our collated fixture.

So NVIDIA's own numbers **confirm** the eager DiT cost and give a compiled DiT 12 % slower than our rung,
consistent with ours compiling more of the loop. NVIDIA's summary line — `torch.compile` 1.39× E2E,
2.29× action head — is on its own definitions above.

## 5. Not measured

* How much of the backbone's +34 ms is `prepare_input`, the real frame, or TF32 (§4).
* `ae_step[i]` / `ae_layer[j]` individual events inside the DiT loop: `groot_phases.py` times
  `ae_total` only (the loop lives inside `get_action`); per-step is the slope, which §2 makes
  primary under compilation anyway. Eager instrumented per-step (§2 (a)) is not available.
* N1.5 / N1.6 N-sweeps (N = 4 only, in `results/20260923_E4d_groot_n1{5,6}_ladder/`).

## 6. Self-review (§11)

* Every number above traces to a committed CSV/JSON; slopes recomputed from the 12 `phase_split.csv`.
* The compiled points are the r4 re-run (2026-09-25 16:20–16:43), after the root-owned Triton
  cache made the first attempt fail and a later attempt overlapped E8 (INCIDENT in
  `scratch/queues/queue_prereboot_chain.log`; the confounded run is in
  `scratch/failed_runs/`, not in `results/`). The eager points ran alone earlier the same day.
* Linearity (r² ≥ 0.999 for ae_total in both rungs) supports reading the slope as per-step time.
* Weak point: the floor's byte count is a lower bound, so 1.05× is an upper bound on how close
  to the floor the compiled step is — the true gap can only be smaller.
