# Synthesis — every measurement of the Jetson VLA study, in one place (2026-09-28)

A derived report: nothing here is newly measured. Every number is taken from a committed report or result
(source in brackets: `E4i` = `reports/E4i_best_combinations.md`, etc.; line numbers are in the per-section source
reports). Device: Jetson AGX Orin 64 GB, MAXN (16 SMs, GPU 1300.5 MHz, EMC 3199 MHz = LPDDR5-6400, 12 CPUs @ 2.2 GHz),
JetPack 6.2.3 / L4T R36.5.0 / CUDA 12.6 / torch 2.8.0, batch 1 [E0]. Measured roofs: **187.47 GB/s** (0.915 of spec),
**33.85 TFLOP/s bf16** (0.806 of spec), ridge **180.6 FLOP/B** [E1_E2]. Models: π0, π0.5, SmolVLA (LeRobot, N = 10
denoising steps), GR00T N1.5 / N1.6 / N1.7 (N = 4).

## 0. The answer in ten lines

1. **Compilation, not the GPU, is the first-order lever**: the best configurations are 1.5× (GR00T) to 7.9× (SmolVLA)
   faster than eager, almost entirely by removing per-kernel launch cost (fusion + CUDA graphs) and by bf16 / TF32.
2. **Eager experts are host-launch-bound** (π0: 969 kernels per step, GPU 84 % idle, 62 µs between kernels); every
   eager step costs 52–62 ms regardless of model size.
3. **Compiled, the LeRobot experts are still 3–5× above their memory floor** (memory-latency-bound: Long Scoreboard,
   small GEMMs); **GR00T's compiled DiT is at its floor** (1.05–1.33×, 75–87 % of peak bandwidth).
4. **The prefix is the large, bandwidth-hungry phase after compilation**: π0's prefix moves **21.6 GB/frame = 3.7× its
   analytic bytes**; ~2/3 of the GEMM excess is weight re-reads (fit, not distinguishable from "mixed").
5. **Energy: the expert is the cheap phase** (energy share / time share 0.31–0.93 in 13 of 14 MAXN configs); π0's
   prefix draws 41 W above idle against the expert's 9–19 W. Best configs cost 2.8–16.6 J per inference.
6. **Capped power modes are not an efficiency route**: energy per inference moves ≤ 1.47× across MAXN/50W/30W/15W,
   latency 2–6×; energy·delay is 1.4–5.2× worse at 15W.
7. **Co-run of prefix and expert on one GPU gives at best O = 1.247** (T_pipe 307 vs 349 ms serial): the stages
   time-slice on the whole GPU, and SM partitions still couple through memory (8–26 %). It saves no energy (1.02×).
8. **Bandwidth stolen from the prefix slows it linearly**: σ ≈ 1 + 0.0032 × GB/s; 1.247 at 77 GB/s, 1.125 at a 37 % duty
   cycle — the throughput claim of a concurrent memory-side expert breaks at full-time 16-channel sharing.
9. **Accuracy holds**: compiled bf16 π0 = shipped model on LIBERO (249 vs 252/400); **fewer denoising steps are better**
   (seeded N = 2: 300 vs N = 10: 259/400, p = 1.3 × 10⁻⁵).
10. **Free prefix savings exist**: π0 padding trim (−101 ms/frame, prefix bytes ×0.79, accuracy unchanged); GR00T
    `keep1` + `kvonce` (−17 ms, bitwise identical).

---

## 1. Latency: base vs best, and the best combination per model

### 1.1 Breakdown (MAXN, median ms; base = `eager_fp32`, the study's eager baseline in the checkpoint's dtype) [E4i, E4d]

| model (N) | config | vision | VLM prefix | action expert | per step | **e2e** | AE share | speed-up |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| π0 (10) | base | 241.01 | 234.57 | 586.91 | 58.7 | **1069.04** | 54.9 % | |
| | best `bf16_both` | 56.74 | 192.86 | 168.68 | 16.9 | **424.46** | 39.7 % | **2.52×** |
| π0.5 (10) | base | 239.45 | 2162.47 | 617.74 | 61.8 | **3028.56** | 20.4 % | |
| | best `bf16_both_graphs` | 56.57 | 239.47 | 185.60 | 18.6 | **487.81** | 38.0 % | **6.21×** |
| SmolVLA (10) | base | 33.30 | 65.77 | 620.56 | 62.1 | **727.46** | 85.3 % | |
| | best `vlm_graphs` | 31.26 | 7.93 | 44.86 | 4.5 | **92.30** | 48.6 % | **7.88×** |
| GR00T N1.7 (4) | base | 70.62 | 88.56 | 211.12 | 52.8 | **372.48** | 56.7 % | |
| | best `vlm_compile` | 70.82 | 87.59 | 79.57 | 19.9 | **240.49** | 33.1 % | **1.55×** |
| GR00T N1.6 (4) | base | 61.94 | 61.70 | 201.33 | 50.3 | **327.22** | 61.5 % | |
| | best `all_compile_autotune` | 48.80 | 52.61 | 72.83 | 18.2 | **174.59** | 41.7 % | **1.87×** |
| GR00T N1.5 (4) | base | 50.20 | 100.62 | 98.45 | 24.6 | **255.22** | 38.6 % | |
| | best `all_compile_autotune` | 32.22 | 70.73 | 60.55 | 15.1 | **168.75** | 35.9 % | **1.51×** |

e2e p5–p95 is within ±1 % for every row except GR00T N1.6 best (164.6–200.3) and N1.5 base (251.5–287.7) [ladder CSVs].
SmolVLA's `whole_graphs` (all of `sample_actions` as one graph) is faster still, 80.97 ms (8.98×), but cannot be split
into regions [E4d]. The expert's share of the frame **falls** with optimisation in five of six models (π0.5 is the
exception: its eager prefix was the pathology); no optimised model spends more than 48.6 % in the expert [E4i].

### 1.3 Component shares of the frame (derived from §1.1: component median / e2e median)

| model | config | vision | VLM prefix | action expert | other (e2e − sum) |
|---|---|---:|---:|---:|---:|
| π0 | base / best | 22.5 % / 13.4 % | 21.9 % / 45.4 % | 54.9 % / 39.7 % | 0.6 % / 1.5 % |
| π0.5 | base / best | 7.9 % / 11.6 % | 71.4 % / 49.1 % | 20.4 % / 38.0 % | 0.3 % / 1.3 % |
| SmolVLA | base / best | 4.6 % / 33.9 % | 9.0 % / 8.6 % | 85.3 % / 48.6 % | 1.1 % / 8.9 % |
| GR00T N1.7 | base / best | 19.0 % / 29.4 % | 23.8 % / 36.4 % | 56.7 % / 33.1 % | 0.6 % / 1.0 % |
| GR00T N1.6 | base / best | 18.9 % / 28.0 % | 18.9 % / 30.1 % | 61.5 % / 41.7 % | 0.7 % / 0.2 % |
| GR00T N1.5 | base / best | 19.7 % / 19.1 % | 39.4 % / 41.9 % | 38.6 % / 35.9 % | 2.3 % / 3.1 % |

"Other" = pre/post-processing between the timed regions (≤ 8.9 ms).

### 1.2 The best combination per model, and what each ingredient buys (e2e ladder, ms) [E4d, E4i, E4]

| model | ladder (each step adds one technique) |
|---|---|
| π0 | 1069.0 → **TF32** 916.5 → **compile expert** 585.8 → **+CUDA graphs on expert** 516.4 → vision compile ≈516 → **compile VLM prefix** 467.0 → **bf16 vision + layernorms** 424.5 |
| π0.5 | 3028.6 → **TF32** 1254.9 → compile expert 917.9 → +graphs 905.7 → vision 900.7 → VLM compile 863.5 → **bf16 base cast** 592.0 → bf16 vision 576.8 → bf16 layernorms 489.2 → +prefix graphs 487.8 |
| SmolVLA | 727.5 → TF32 718.6 → **compile expert** 267.4 → **+graphs** 150.3 → vision rungs ~153–161 → **compile prefix** 103.7 → **+prefix graphs** 92.3 (→ whole-graph 81.0) |
| GR00T N1.7 | 372.5 → TF32 368.5 → **compile action head (reduce-overhead)** 243.0 → graphs ≈244 → compile LLM (max-autotune-no-cudagraphs) 240.5; vision compile **fails** (flash_attn_varlen SymInt) |
| GR00T N1.6 | 327.2 → 329.5 → compile head 206.0 → 203.1 → vision / LLM compile 186.1 / 189.8 → **both** 175.4 → forced autotune 174.6 |
| GR00T N1.5 | 255.2 → 252.4 → compile head 225.1 → 221.9 → vision / LLM 206.7 / 200.9 → both 186.2 → **forced autotune** 168.8 |

**What each technology does, and why it works here:**

| technique | mechanism | where it pays (measured) |
|---|---|---|
| **torch.compile (Inductor), expert** | fuses the elementwise chains (RoPE, GeGLU, RMSNorm, casts, concatenations) into Triton kernels; every GEMM still goes to cuBLAS — Triton won 0 of 126 π0 expert shapes. Fewer, longer kernels = fewer launches | π0 expert 2.38×, SmolVLA 3.79×; π0 per step 52.94 → 22.61 ms [E4] |
| **CUDA graphs (`reduce-overhead`)** | the whole step (or loop) is recorded once and replayed with one launch, removing the ~60 µs host cost per kernel | π0 step 22.61 → 16.30 ms (1.39×); SmolVLA expert 3.69×; SmolVLA prefix 1.96× (small, launch-bound); a **net loss on π0 / π0.5 prefixes** (compute-heavy, 816 FLOP/B) |
| **TF32 / bf16 precision** | TF32 lets fp32 GEMMs use tensor cores; bf16 casts halve weight bytes and use bf16 tensor cores. LeRobot keeps fp32 "islands" (vision tower, layernorms) for a training reason, not accuracy | TF32: π0 vision 2.35×, π0.5 prefix 4.05×; bf16 on π0.5: prefix 1.69×, floor 9.25 → 5.84 ms/step; for π0.5 precision beats compilation (2.16× vs 1.04×) |
| **`max-autotune` with forced GEMM autotune** | Inductor skips GEMM autotuning on GPUs with < 68 SMs (Orin has 16); forcing `is_big_gpu` lets it pick templates | helps only where GEMMs have headroom: N1.5 +9.4 %; N1.6 +0.5 % (its GEMMs already at 85–94 % Mem); slows π0's eager expert |
| **`max-autotune-no-cudagraphs` on the LLM** | tuned kernels without graph capture where graphs conflict with a second cudagraph-managed callable | GR00T LLM; π0 `pg.forward` |
| **harness fixes (not rungs)** | hoisting the per-step timestep tensor (removes 1 + 2N host syncs), memoising constant masks, one own CUDA graph for all N steps | π0 expert 167.1 → 111.0 ms (−33.6 %), own-graph 101.2 ms, bitwise [E12, E12b] |

**Per-step cost and gap to the memory floor** (floor at 187.47 GB/s) [E3_E4_models, E3_E4_groot, E5_E5b_models]:

| model | eager ms/step | compiled ms/step | floor ms/step | gap eager → compiled |
|---|---:|---:|---:|---|
| π0 | 52.94 | 16.30 | 3.40 | 15.6× → 4.8× |
| π0.5 | 61.60 | 18.40 | 5.84 (bf16) | 6.7× → 3.2× |
| SmolVLA | 61.46 | 4.33 | 1.08 | 56.6× → 4.0× |
| GR00T N1.7 | 51.75 | 18.13 | 17.29 | 3.0× → **1.05×** |
| GR00T N1.6 / N1.5 (ae/N) | 50.65 / 24.02 | 18.21 / 15.14 | — | 3.35× → 1.20× / 2.11× → 1.33× |

Eager steps cost 52–62 ms regardless of model size — the signature of launch-bound execution. Compiled, GR00T's DiT
sits at its memory floor; the LeRobot experts remain 3–5× above theirs.

## 2. Energy per model (E7; MAXN; above-idle module energy per inference, corrected) [E7_power_energy, E7r]

Correction applied: compiled-rung phase energies in the original E7 tables were understated (the rail sampler started
before warm-up / compile); the values below are the timed-window re-analysis of the committed raw samples, which a
re-measurement with the fixed sampler reproduced within 0.7 % (prefix + expert = full within 0.49–0.71 %) [E7r].

| model (N) | config | ms prefix / expert / full | **J above idle: prefix / expert / full** | W above idle: prefix / expert | expert mJ per step |
|---|---|---|---|---|---:|
| π0 (10) | eager | 477.9 / 595.7 / 1083.2 | 19.67 / 5.52 / **25.00** | 41.2 / 9.3 | 552 |
| | `bf16_both` | 251.3 / 167.4 / 424.7 | 10.36 / 3.25 / **13.70** | 41.2 / 19.4 | 325 |
| π0.5 (10) | eager | 2425 / 624.4 / 3041.6 | 84.72 / 13.75 / **100.44** | 34.9 / 22.0 | 1375 |
| | `bf16_both_graphs` | 291.9 / 184.3 / 482.1 | 12.24 / 4.23 / **16.57** | 41.9 / 22.9 | 423 |
| SmolVLA (10) | eager | 102.9 / 641.4 / 740.7 | 2.14 / 5.22 / **5.86** | 20.8 / 8.1 | 522 |
| | `vlm_graphs` | 39.9 / 41.9 / 88.2 | 1.68 / 1.07 / **2.84** | 42.3 / 25.5 | 107 |
| GR00T N1.7 (4) | `eager_fp32` | 161.8 / 226.8 / 381.7 | 5.14 / 4.12 / **9.29** | 31.8 / 18.2 | 1031 |
| | `vlm_compile` | 158.6 / 81.1 / 239.9 | 5.09 / 3.45 / **8.59** | 32.1 / 42.6 | 863 |
| GR00T N1.6 (4) | `eager_fp32` | 126.0 / 215.2 / 333.0 | 4.92 / 3.79 / **8.70** | 39.0 / 17.6 | 948 |
| | `all_compile_autotune` | 88.2 / 73.7 / 163.1 | 4.34 / 3.08 / **7.56** | 49.2 / 41.8 | 770 |
| GR00T N1.5 (4) | `eager_fp32` | 141.7 / 108.4 / 251.3 | 7.00 / 3.13 / **10.91** | 49.4 / 28.9 | 783 |
| | `all_compile_autotune` | 98.1 / 60.2 / 159.7 | 5.54 / 3.02 / **8.71** | 56.5 / 50.1 | 755 |

(W above idle = J / s, derived. Module idle ≈ 10.5–11.0 W.)

**Energy shares (derived from the table above; prefix includes vision — no vision-only energy loop was run):**

| model | config | energy: prefix (incl. vision) / expert | time: prefix / expert | (prefix + expert) / full |
|---|---|---|---|---:|
| π0 | eager / best | 78.1 / 21.9 % · 76.1 / 23.9 % | 44.5 / 55.5 % · 60.0 / 40.0 % | 1.008 / 0.993 |
| π0.5 | eager / best | 86.0 / 14.0 % · 74.3 / 25.7 % | 79.5 / 20.5 % · 61.3 / 38.7 % | 0.980 / 0.994 |
| SmolVLA | eager / best | 29.1 / 70.9 % · 61.1 / 38.9 % | 13.8 / 86.2 % · 48.8 / 51.2 % | **1.256** / 0.968 |
| GR00T N1.7 | eager / best | 55.5 / 44.5 % · 59.6 / 40.4 % | 41.6 / 58.4 % · 66.2 / 33.8 % | 0.997 / 0.994 |
| GR00T N1.6 | eager / best | 56.5 / 43.5 % · 58.5 / 41.5 % | 36.9 / 63.1 % · 54.5 / 45.5 % | 1.001 / 0.981 |
| GR00T N1.5 | eager / best | 69.1 / 30.9 % · 64.7 / 35.3 % | 56.7 / 43.3 % · 62.0 / 38.0 % | 0.929 / 0.983 |

Shares are of prefix + expert (the full-frame loop is measured separately; the last column checks the sum).

**Readings.**
- **Optimisation saves energy, not just time**: best vs eager uses 55 % (π0), 16 % (π0.5), 48 % (SmolVLA) and 80–93 %
  (GR00T) of the eager energy per inference (derived from the table) — the frame is shorter and the GPU does the same work in fewer, busier
  kernels, while power rises only modestly.
- **The expert is the cheap phase**: its energy share divided by its time share is 0.31–0.93 in 13 of 14 MAXN
  configurations (π0 eager lowest, 0.31–0.39); the exception is GR00T N1.7 `vlm_compile` (1.20), whose compiled DiT
  draws more power than its prefix. π0's prefix draws ~41 W above idle in both configurations; its eager expert 9 W.
  **For a memory-side accelerator aimed at the expert, the energy case is weaker than the latency case** (~2.6× for
  π0 eager: 1 / 0.39, derived).
- **Per rail** (π0 eager N10, above idle, W): prefix GPU 33.3, CPU 0.4, memory rail 2.8; expert GPU 6.0, CPU 1.3, memory
  rail 0.8. The memory rail (`VDDQ_VDD2_1V8AO`, regulator input) is an **upper bound** on DRAM power; the eager
  expert's CPU rail rises most relative to its GPU rail — the host is doing the work.
- **Power modes** [E7_capped_modes]: full-frame J above idle, MAXN / 50W / 30W / 15W — π0 `bf16_both` 13.70 / 10.85 /
  12.36 / 12.74 (latency 425 → 2188 ms); π0.5 graphs 16.57 / 12.88 / 15.77 / 15.93; SmolVLA graphs 2.84 / 2.20 / 2.27 /
  2.55; GR00T N1.7 (50W → 15W) 7.71 → 7.69. Energy spans ≤ 1.47×, latency 2–6×; E·D 1.4–5.2× worse at 15W; lowest energy
  at 50W for 4 of 6 configurations. **No capped mode is an efficiency route.**
- Vision vs language energy split: now measured, §2.1. Not measured: true DRAM power; corrected per-rail values for compiled
  rungs other than π0 `bf16_both`; SmolVLA eager's prefix + expert exceeds its full frame by 26 % (unexplained).

### 2.1 Vision / language / action energy, measured separately (E7v, 2026-09-28) [`results/20260928_E7v_summary`]

Twelve new runs at MAXN with the fixed sampler: per run five 60 s loops in one session — `vision` (the image-encoder
calls of one prefix pass, recorded and replayed: SigLIP + projector per camera for π0 / π0.5, the SmolVLM vision
encoder, GR00T's `backbone.model.visual`), `vlm` = the **language** part (the unchanged prefix / backbone with the encoder
answered from the recorded outputs), `ae` (action), plus `prefix` and `full` as checks. Per phase: median ms / above-idle
J per inference / above-idle W; shares are of vision + language + action.

| model | config | vision ms / J / W | language ms / J / W | action ms / J / W | energy share V / L / A | time share V / L / A | (V+L)/prefix energy | (prefix+A)/full energy |
|---|---|---|---|---|---|---|---:|---:|
| groot_n15 | `all_compile_autotune` | 31.8 / 1.70 / 53.3 | 65.5 / 3.88 / 59.3 | 60.9 / 2.97 / 48.8 | 20 / 45 / 35 % | 20 / 41 / 38 % | 1.018 | 0.991 |
| groot_n15 | `eager_fp32` | 41.3 / 2.08 / 50.3 | 99.2 / 5.04 / 50.8 | 110.6 / 3.42 / 30.9 | 20 / 48 / 32 % | 16 / 40 / 44 % | 1.014 | 0.951 |
| groot_n16 | `all_compile_autotune` | 43.6 / 2.29 / 52.6 | 42.2 / 2.04 / 48.2 | 74.5 / 3.07 / 41.2 | 31 / 28 / 41 % | 27 / 26 / 46 % | 1.011 | 0.987 |
| groot_n16 | `eager_fp32` | 61.7 / 2.51 / 40.7 | 63.8 / 2.59 / 40.6 | 215.8 / 4.01 / 18.6 | 28 / 28 / 44 % | 18 / 19 / 63 % | 1.008 | 1.005 |
| groot_n17 | `eager_fp32` | 69.3 / 2.59 / 37.3 | 85.7 / 2.64 / 30.8 | 213.8 / 4.17 / 19.5 | 28 / 28 / 44 % | 19 / 23 / 58 % | 1.004 | 0.997 |
| groot_n17 | `vlm_compile` | 70.1 / 2.59 / 36.9 | 85.8 / 2.50 / 29.1 | 81.6 / 3.44 / 42.2 | 30 / 29 / 40 % | 30 / 36 / 34 % | 1.008 | 0.989 |
| pi05 | `bf16_both_graphs` | 56.8 / 2.40 / 42.3 | 236.3 / 9.85 / 41.7 | 183.7 / 4.23 / 23.1 | 15 / 60 / 26 % | 12 / 50 / 39 % | 1.011 | 0.991 |
| pi05 | `none (eager)` | 240.0 / 10.71 / 44.6 | 2168.1 / 75.03 / 34.6 | 609.9 / 13.88 / 22.8 | 11 / 75 / 14 % | 8 / 72 / 20 % | 1.008 | 0.998 |
| pi0 | `bf16_both` | 57.3 / 2.32 / 40.5 | 195.3 / 8.21 / 42.0 | 170.1 / 3.36 / 19.8 | 17 / 59 / 24 % | 14 / 46 / 40 % | 1.012 | 0.990 |
| pi0 | `none (eager)` | 239.8 / 10.58 / 44.1 | 235.7 / 9.61 / 40.8 | 597.1 / 6.29 / 10.5 | 40 / 36 / 24 % | 22 / 22 / 56 % | 1.016 | 1.003 |
| smolvla | `none (eager)` | 32.2 / 1.41 / 43.7 | 66.6 / 0.64 / 9.6 | 620.1 / 3.85 / 6.2 | 24 / 11 / 65 % | 4 / 9 / 86 % | 1.011 | 1.021 |
| smolvla | `vlm_graphs` | 29.8 / 1.40 / 47.0 | 9.0 / 0.30 / 32.8 | 42.1 / 1.09 / 25.8 | 50 / 11 / 39 % | 37 / 11 / 52 % | 1.015 | 0.974 |

**Closure (both measured):** vision + language vs the unsplit prefix within **0.4–1.8 %**; prefix + action vs the full
frame within **0.3–4.9 %**. SmolVLA eager now closes (1.021) — the 26 % excess in the older E7 table was a sampler artefact.

**Reading.**
- **Vision is the most power-dense phase** in every LeRobot run (40–47 W above idle; GR00T 37–53 W), but short: at the
  best rungs it is 15–17 % of π0 / π0.5 energy, 20–31 % of GR00T's, and **50 % of SmolVLA's** (its language and action
  parts are tiny once compiled).
- **Language (the VLM prefill) is the largest energy consumer after optimisation** for π0 (59 %), π0.5 (60 %) and GR00T
  N1.5 (45 %); in GR00T N1.6 / N1.7 the three parts are nearly even (28–41 %).
- **The action expert is the cheapest per unit time**: 20–26 W above idle for the compiled LeRobot experts vs 33–47 W for
  their vision / language parts; only GR00T's compiled DiT (41–49 W) is as power-dense as its prefix or more (N1.7: 42 W
  vs vision 37 W, language 29 W) — consistent with it being the one bandwidth-bound expert (§3). Action share of energy at the best rung: π0 24 %, π0.5 26 %, SmolVLA 39 %, GR00T
  35–41 %.

## 3. Root cause of the bottleneck (nsys + ncu), base vs best [E5_timeline, E4g, E4h, E5b_*, E6_*, E12b, E13]

Orin has no `dram__` counters: ncu "Mem %" is L2 throughput; off-chip bytes exist only as L2-fill counts (π0).

### 3.1 Timeline facts (nsys; LeRobot per denoising step, GR00T whole expert)

| model | expert, eager: GPU idle / kernels / launch gap | expert, best | prefix, eager → best idle |
|---|---|---|---|
| π0 | **84.2 %** (76.4 % corrected) / 969 per step / 61.6 µs | 47 % (38 %) / 373 / 0.1 µs | 1.6 % → 0.5 % (GPU-busy) |
| π0.5 | 64.3 % (47.3 %) / 1103 / 51.2 µs | 44.5 % (36.3 %) / 402 / 0.1 µs | 0.1 % → 0.1 % |
| SmolVLA | **92.7 %** (88.9 %) / 1184 / 64.5 µs | 38 % (20.6 %) / 356 / 0.1 µs | **88.5 %** → 4.9 % |
| GR00T N1.5 / N1.6 / N1.7 | 53 / 72 / 72 % / 1982–3168 / 45–70 µs | **~0 %** / 760–1631 | N1.7 LLM stays 64 % idle (no graphs) |

### 3.2 Per-kernel behaviour (ncu, 40-kernel windows, time-weighted) and roofline

| model / phase | eager | best |
|---|---|---|
| π0 expert | SM 23 %, Mem 22 %, top stall **instruction-cache miss (imc_miss) 35 %** → latency | Mem 56.5 %, **Long Scoreboard 39 %** → memory-latency; 4.96× floor, 20 % of peak BW |
| π0 VLM prefix | LSB 74 %, SM 40 %; elementwise 59.5 % of time | GEMM 90.8 % of time; top 128×128 GEMMs latency-bound (mio_throttle), next four compute-bound (SM ~86 %) |
| π0 vision | mixed; 27× floor (eager) | latency-bound (LSB 29 %) |
| π0.5 expert | SM 4.9 %, imc_miss 32 % | mixed, LSB 55 %, Mem 60 %; 3.18× floor |
| SmolVLA expert / prefix | imc_miss 33 %, SM 16 % / LSB 49 % | LSB 35 % / 51 %, every phase < 60 % SM and Mem — still latency-bound; no kernel meets a threshold |
| GR00T experts | LSB-limited, 30–48 % peak BW | **75–87 % of peak BW, 1.15–1.33× floor → bandwidth-bound (weight streaming, 2.1–3.2 GB/step)** |
| GR00T N1.5 / N1.6 / N1.7 prefix | LSB 64–74 % latency | **memory-bound** (N1.5 83.5 % Mem; N1.6 / N1.7 top GEMMs 85–94 % Mem) |

Expert GEMM arithmetic intensity is ≈ S = 41–51 FLOP/B for every model — far below the 180.6 FLOP/B ridge: the expert
is memory-side by construction [E4g]. π0's eager VLM prefix is on the compute side (816 FLOP/B, 1.93× floor, 52 % of
peak compute).

### 3.3 The root-cause chain

| model | base (eager) | best (compiled) |
|---|---|---|
| **π0** | expert **launch-bound**: 969 kernels/step, 62 µs gaps (4× the average kernel), instruction-cache misses, 6 % of peak BW, 15.5× floor. Prefix already efficient (compute side). | launches gone (0.1 µs gaps) but the expert is **memory-latency-bound** in small bf16 GEMMs (LSB), 4.96× floor with 38 % idle left unexplained. The **prefix becomes the heavy phase**: 21.64 GB/frame moved vs 5.85 GB analytic (**3.70×**, 76 % L2 hit rate); GEMMs read ~7.4 GB of excess — Gemma MLP gate/up 2.74 GB (weight re-reads), SigLIP MLP 2.45 GB (mostly weight), Gemma down 1.72 GB (activation re-reads); weight share 0.67 (fit, indistinguishable from mixed). |
| **π0.5** | as π0 for the expert; the eager prefix (2162 ms) is fp32 without tensor cores — TF32 and bf16 fix it (the S-sweep shows its bf16 gate_proj is a pure weight stream, flat from S = 1 to 50 at 1.22–1.29× floor). | expert mixed memory-latency, 3.18× floor; compiled Triton cat / GeGLU memory-latency-bound. |
| **SmolVLA** | the most launch-bound model: expert 89 % idle, prefix 88.5 % idle, 56× floor. | 4.13× floor; GEMMs too small to stream efficiently (largest 2.9 MB at 1.5–1.8× floor; cross-attention K at 8–9× floor = launch + tail). |
| **GR00T** | expert launch-bound but already 30–48 % of BW (bigger GEMMs). | **the expert is at the memory roofline** (75–87 % BW) — the only model where a faster memory system is the direct lever; N1.7's uncompiled vision and graph-less LLM stay host-idle 42–64 %. |

### 3.4 The action expert by component (E5c, derived from the committed nsys runs) [`results/20260928_E5c_expert_components`]

Kernel time per denoising step (LeRobot) or per whole action head (GR00T, N = 4), under nsys, by model component
(`bench/e5_expert_components.py`; kernels joined to their NVTX step through the CUDA runtime calls, CUDA-graph replays
included; classified by kernel name).

| model, config | weight projections (GEMM) | attention scores + values (batched GEMM / fmha) | RoPE | norms | GeGLU | K/V cat & copy kernels | K/V memcpy into graph inputs |
|---|---:|---:|---:|---:|---:|---:|---:|
| π0 eager (per step) | 4.81 ms (129) | 4.31 + 0.31 ms | — | 0.09 | 0.12 | 1.25 (204) + 2.32 elementwise (476) | — |
| **π0 best** (per step) | **4.54 ms** (129) | **4.19 + 0.24 ms** | 0.39 | 0.15 | 0.15 | 0.37 | **15.05 MB, 0.36 ms** |
| π0.5 best (per step) | 7.64 ms (206) | 2.47 + 0.22 ms | 0.59 | — | 0.12 | 0.37 | 17.85 MB, 0.40 ms |
| SmolVLA best (per step) | 1.85 ms (118) | 0.72 + 0.17 ms | 0.27 | 0.11 | 0.05 | 0.22 | 3.63 MB, 0.14 ms |
| GR00T N1.7 best (4 steps) | **72.7 ms, 90.7 %** | 2.4 + 4.0 ms | 0.11 | — | (fused) | 0.84 | — |
| GR00T N1.6 best (4 steps) | **68.9 ms, 94.4 %** | 3.9 ms | 0.13 | — | (fused) | 0.04 | — |
| GR00T N1.5 best (4 steps) | **56.3 ms, 93.7 %** | 3.4 ms | 0.07 | — | (fused) | 0.15 | — |

(kernel counts in brackets; GR00T N1.5 / N1.6 GEMMs are Inductor templates with bias / GELU / LayerNorm fused in.)

**Findings.**
- **π0's compiled expert spends as much GPU time on attention as on all its weights.** The 36 attention matrix products
  per step (QKᵀ and PV over the 867-token cache, 8 query heads against the single shared K/V head) take **4.19 ms** —
  next to 4.54 ms for all 126 weight projections. They run in `cutlass_75_wmma` (Turing-generation) kernels at
  ~110 µs each; derived from their shapes (≈ 3.5 MB and 0.18 GFLOP each) that is ~1.6 TFLOP/s and ~30 GB/s — neither
  compute- nor bandwidth-bound, a poor kernel choice for this batched shape. The cause of the selection (likely the
  layout of the one K/V head expanded to 8 heads) is **not verified**.
- The weight projections themselves are near their floor (4.54 ms in-model vs 4.41 ms replayed alone, 1.33× floor).
- **The prefix K/V cache is copied into the CUDA graph's input buffers every denoising step** (15.05 MB = the whole
  cache, 40 `cudaMemcpyAsync` per step): 0.36 ms per step, but ~150 MB of extra memory traffic per frame.
- The remaining gap between these ~10 ms of kernel time and the unprofiled 16.9 ms step is host / graph-launch
  overhead outside the kernels (derived; the per-step idle within the graph span is not measurable from these traces
  because the two graph replays per step execute asynchronously).
- **GR00T's compiled action head is 91–94 % weight GEMMs** — the weight-streaming picture of §3.2 holds at component
  level; attention is 5–6 %.

**Bandwidth sensitivity of the compiled π0 prefix** [E13]: taking bandwidth away from it (no SMs used) slows it
linearly, σ ≈ 1 + 0.0032 × GB/s, while the bus is not saturated (EMC 76 % at the maximum load) — the coupling is memory
latency / queueing, not a full bus.

Not measured: DRAM throughput % (no counters); off-chip bytes for models other than π0; FLOP counts for GR00T; a
compiled-rung roofline for vision / prefix; the cause of the remaining ~38 % idle in the compiled LeRobot expert.

## 4. Co-run: the prefix of frame k concurrent with the expert of frame k−1 (π0, N = 10) [E12, E12b, E13, E7_capped_modes]

Quantities: `s` = expert slowdown in co-run, `p` = prefix slowdown, `r` = A_iso / B_iso, `O` = T_pipe / max(A_iso,
B_iso) (1.0 = perfect overlap, T_serial / max = no overlap). Every pipeline passed a fresh-KV equivalence check
(bitwise vs serial, stale-KV negative control detected 5/5).

| configuration (MAXN) | A_iso | B_iso | T_serial | **T_pipe** | **O** | s | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| eager (host syncs removed) | 474.08 | 523.76 | 994.38 | 994.88 | 1.90 (= serial: none) | 0.996 | 0.998 |
| compiled, whole GPU, prefix first | 248.43 | 101.21 | 353.03 | 345.69 | 1.391 | 2.676 | 1.040 |
| compiled `graphs_both`, whole GPU | 246.28 | 101.36 | 348.67 | 339.69 | 1.379 | 3.351 | 1.155 |
| + expert stream priority −3 | | | | 320.58 | 1.287 | | |
| **+ green-context SM split 12 / 4** | 246.28 | 101.36 | 348.67 | **307.15** | **1.247** | 3.009 | 1.169 |
| green 10 / 6, 8 / 8, 6 / 10, 14 / 2 | | | | 321.2 / 361.3 / 430.1 / 535.5 | 1.30 / 1.47 / 1.75 / 2.18 | | |

**What happens.**
- **Eager: no overlap at all** (T_pipe = A + B). Eager π0 is issue-bound — there is nothing for two streams to overlap.
- **Compiled, whole GPU: time-slicing.** Kernels of the two streams overlap for only 3.1 % of the time; the prefix's
  1008-CTA GEMMs (63 waves) hold every SM and the expert's kernels queue behind them (head-of-line blocking).
- **SM partitions (green contexts) give the best result, O = 1.247**, but the stages still slow each other by 8–26 %
  (s_part 1.16–1.26) through the memory system: *partitioning SMs does not partition bandwidth*. The compiled expert
  also needs SMs (2.50× slower on 4 SMs), so 12 / 4 cannot beat ~268 ms.
- **Energy: co-run saves none** — 17.9 vs 17.6 J per frame (1.02×; 1.05× above idle); eager 0.998×. Frame time falls
  10 % but power rises 13 %.
- **Correctness**: the first compiled pipeline silently produced wrong actions or NaN; the 2 × 2 isolation shows a
  **single KV buffer** is the cause (98/100 wrong; with a shared graph pool the GPU hangs); double-buffered KV is
  necessary and sufficient.
- **Measurement pitfall found**: a CUDA graph obeys the SM partition it was *captured* in, not the stream it is replayed
  on — E12's first compiled partition rows were withdrawn on that basis.
- **Capped modes**: at 8 and 6 hardware SMs the eager co-run is serial again (O ≈ 2), energy unchanged (0.9996 / 1.006).
- **Mechanisms that did not help**: `CUDA_DEVICE_MAX_CONNECTIONS=1` (O 1.403, fully serial); MPS works but needs two
  processes, and two π0 processes are OOM-killed (62 GB shared, ~15 GB each); libsmctrl's TPC mask works but was not
  needed once green contexts did.

**Bandwidth steal (E13; what a concurrent memory-side expert would do to the GPU prefix):**

| stolen GB/s (no SMs used) | 0 | 20 | 40 | 60 | 77 | 85 (max reachable) |
|---|---|---|---|---|---|---|
| prefix slowdown σ | 1.000 | 1.060 | 1.120 | 1.181 | **1.247** | 1.279 |

Pre-registered rows: (1) full-time steal: σ > 1.247 → **latency and throughput claims both break** at 16-channel
sharing; (2) steal during 37 % of the frame: σ = 1.125 → latency parity lost, **throughput claim survives**; (3) the loss
is **linear in the bytes exposed** (within ±0.02); (4) confining the GPU to a channel subset is **not possible** on stock
Orin (boot-time address map); (8) the prefix's excess reads are ~2/3 weight re-reads (not separable from mixed).

**What this means for the PIM concurrency claim** (the reports' own reading): on one Orin GPU the best valid overlap is
O = 1.247; any concurrent expert shares the memory system and slows the prefix in proportion to the bytes it moves.
If the own-graph expert enters the GPU baseline, an in-memory expert's latency advantage at 16 channels is below 1.10×.
Which baseline to use is the user's open decision (OD-1).

## 5. Other results

**Closed-loop accuracy (LIBERO, 10 tasks × 10 episodes per suite)** [E8, E8s]
- Compiled bf16 π0 (`bf16_both`) vs the shipped model: **249 vs 252 / 400**, paired p = 0.87 — the speed-up costs no accuracy.
- **Fewer denoising steps succeed more often**, with paired, seeded noise: pooled N = 2 **300** vs N = 10 **259 / 400**
  (+10.2 pp [+5.8, +14.7], exact McNemar p = 1.3 × 10⁻⁵); N = 1 / 3 / 4 also better than N = 10 (p ≤ 0.003), N = 6 not
  (p = 0.26). Seeded repeat agreement 95 / 100 episodes (unseeded: 65 %). A forward costs ~16.9 ms per step + 258 ms.
- Finding: every E8 / E8s arm fed the empty third camera as an attended zero image (training masks it). Paired
  comparisons stand; absolute rates are for that input (masked vs attended, same arm: 66 vs 71, p = 0.36).

**Prefix savings (all variants, never "baseline")** [E14, E14b]
- π0 `nopad` (drop the empty camera's 256 tokens and 27 padded prompt tokens; 816 → 533): prefix 245.4 → 180.1 ms
  (0.734), **frame 347.6 → 248.5 ms**, prefix DRAM bytes ×0.792, expert 0.653; K/V handover 15.04 → 9.82 MB (exact);
  actions not bitwise but within the rung's own noise floor (PASS); LIBERO 70 vs 66 / 100 (p = 0.29). The two halves
  differ: the camera drop saves most of the prefix; trimming the prompt alone **slows the prefill** (+2.8 %, odd
  lengths) but **cuts the expert by 31 %** without moving its bytes (unexplained).
- GR00T N1.7 `keep1` (lm_head on the last token) and `kvonce` (cross-attention K/V once per frame): −8.1 / −8.8 ms,
  together −17.0 ms (237.0 → 220.0) and frame reads ×0.874, **bitwise identical** actions.

**Model inputs for the PIM study** [E11]: the measured `model_inputs.csv` replaces analytic assumptions — host time
per layer 1.529 ms eager (0.18 ms graphed) vs 0.16 ms assumed; 9–10 kernels per norm segment vs 1; pinned transfer
~36 GB/s + 39 µs vs 143 GB/s; stream handoff 68.7 µs vs free; K/V copied per step 15.04 MB vs 0.

**Platform quirks worth knowing** [E0, ncu-limits]: `max-autotune` skips GEMM templates on a 16-SM GPU; ncu needs root,
has no DRAM counters, and fails after 480 profiled kernels per process; torch.compile of new shapes stalls under ncu.

## 6. Open items and what is not measured

- User decision OD-1: whether `owngraph_expert` / the prefix trims enter the GPU reference.
- Research-side question: re-run LIBERO arms with the empty camera masked?
- Unexplained, measured: remaining ~38 % idle in the compiled LeRobot experts; π0 compiled prefix slows 2.68× at 30W
  (above the SM × clock bracket); the `trimprompt` expert speed-up; why fewer steps succeed more often; SmolVLA eager's
  prefix + expert energy exceeding its full frame.
- Not measurable here: DRAM power and DRAM throughput % (no counters / only a regulator-input upper bound), σ at
  ≥ 94 GB/s without SMs, channel-confined prefix, two-process co-run (memory).
