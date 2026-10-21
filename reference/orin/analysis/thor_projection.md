# Orin → Jetson AGX Thor latency projection

**Every Thor number in this file is projected, not measured.** No Thor exists in this study. The projection
scales our committed Orin measurements by factors taken from NVIDIA's published GR00T N1.7 timings. Nothing ran
on a GPU to make it.

- Script: `analysis/thor_projection.py`, run with `python3 analysis/thor_projection.py`. It writes
  `analysis/thor_projection/{factors,per_model,geomean,validation,pi0_aim,pi0_aim_constants}.csv`.
- Orin inputs come from committed files and are read by the script. They are the §1.1 table of
  `reports/SYNTHESIS_all_measurements.md` (MAXN medians, six models, base and best), the green 12/4 row of
  `results/20260925_E12b_pi0_green_corun_N10_MAXN/corun_ratios.csv` (`A_iso` 246.28, `B_iso` 101.36 ms,
  `O` 1.247), and `reports/E1_E2_microbench.md` (roof 187.47 GB/s; π0 expert weights 622.85 MB per step).
- External constants are hard-coded in the script, each with its URL and the access date 2026-10-08.
- PIM constants come from the workspace's `evaluation/p1/numbers.md`: AiM step 9.31 ms at 16 ch and 5.87 ms at
  32 ch, and the "optimistic bound (2× floor)" of 6.30 ms per step. The script checks them against that file when
  the file is reachable.

## 1. Method

1. **Calibration.** We use one model, GR00T N1.7, at the same step count (4) and precision (bf16). There are two
   stage classes. *Front* is vision plus the VLM prefix (NVIDIA's "backbone"). *Expert* is the action head. Base
   rows (eager) take the eager factor, and best rows (compiled) take the torch.compile factor.
2. **Applying the factors.** Each model's vision and prefix are divided by the front factor and its action expert
   by the expert factor. "Other" (e2e minus the three stages) is divided by 1.5 `[ASSUMPTION: CPU uplift]`.
3. **Scenarios.**

| scenario | front (base / best) | expert (base / best) | source of the factor |
|---|---:|---:|---|
| **central** | 2.675 / 2.646 | 3.012 / 1.687 | NVIDIA Orin (Isaac-GR00T `scripts/deployment/README.md` @ `376ba89`) ÷ NVIDIA Thor JP 7.2 (Jetson AI Lab tutorial). Both are `GR00T-N1.7-LIBERO/libero_10` with 1 camera and 4 steps. |
| low | 1.514 / 1.514 | 1.514 / 1.333 | hardware ratios. SM × clock = (20 × 1575)/(16 × 1300.5); bandwidth = 273/204.8 |
| jp71 (sensitivity) | 2.309 / 2.313 | 2.515 / 1.221 | NVIDIA Orin ÷ NVIDIA Thor JP 7.1, both from the same README table (one harness, one input) |
| ours_vs_jp72 (main's draft) | 3.337 / 3.259 | 3.096 / 1.700 | our Orin N1.7 ÷ NVIDIA Thor JP 7.2. **Input-confounded**, see §1.1 |

Thor hardware: 2,560 CUDA cores in 10 TPCs, so 20 SMs (derived at 128 cores per SM). GPU clock up to 1.575 GHz,
the top of the ASPLOS'27 sweep. Memory is 273 GB/s on 256-bit LPDDR5X, EMC 4266 MHz. NVIDIA publishes no BF16
peak, and none is derived here.

### 1.1 What changed from main's draft, and why

**The draft's calibration pair compared different inputs.** Our Orin N1.7 run (`results/20260923_E4d_groot_n17_ladder/meta.json`)
uses the DROID embodiment `oxe_droid_relative_eef_relative_joint`. That is **2 cameras (exterior + wrist) × 2
frames (`delta_indices` [−15, 0]) = 4 images of 256²**: P = 273, of which 256 are image tokens, and S = 41. NVIDIA's
Thor run is `libero_10` / `LIBERO_PANDA` with **1 camera**, 4 steps. On the expert the mismatch barely matters:
NVIDIA's own Orin compiled action head is 78.94 ms against our 79.57 ms (0.8 % apart, despite S = 17 vs 41). On
the front it matters a lot: NVIDIA's Orin backbone is 127.6 / 128.59 ms (eager / compile) against our 158.4–159.2
ms. Part of the draft's front factor of 3.26–3.34 is therefore the smaller Thor input, not Thor.

The fix keeps NVIDIA's JP 7.2 Thor numbers and divides them into **NVIDIA's own Orin numbers for the same input**,
taken from the pinned Isaac-GR00T clone. With that change the front factor falls from 3.26 to 2.65 (best) and the
expert factor barely moves (1.700 → 1.687). The draft's figures are kept as the `ours_vs_jp72` sensitivity, and the
script reproduces them exactly (π0 179.9 … N1.5 70.7 ms; AE-share geomean 54.0 %; π0 AiM 168.7 / 134.3 ms).

The same README also has NVIDIA's own Thor numbers on **JetPack 7.1** (`jp71`). There Thor's compiled action head
is 64.66 ms, against 46.8 ms on JP 7.2. So 1.38× of the central expert factor is software that shipped between
two JetPack releases. On JP 7.1 the compiled expert factor (1.221) sits *below* the bandwidth ratio (1.333). The
"low" scenario is therefore not a strict floor.

## 2. Per-model projection (MAXN, median-based; projected, not measured)

**Best configuration** (the compiled rung of SYNTHESIS §1.1):

| model (N) | Orin e2e | Thor vision | prefix | AE | **e2e** | Hz | AE share (Orin → Thor) | AE ms/step | Orin/Thor | low e2e | low AE share | jp71 e2e | draft e2e |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| π0 (10) | 424.5 | 21.4 | 72.9 | 100.0 | **198.5** | 5.0 | 39.7 → 50.4 % | 10.00 | 2.14× | 295.5 | 42.8 % | 250.2 | 179.9 |
| π0.5 (10) | 487.8 | 21.4 | 90.5 | 110.0 | **226.0** | 4.4 | 38.0 → 48.7 % | 11.00 | 2.16× | 338.9 | 41.1 % | 284.1 | 204.1 |
| SmolVLA (10) | 92.3 | 11.8 | 3.0 | 26.6 | **46.9** | 21.3 | 48.6 → 56.7 % | 2.66 | 1.97× | 65.0 | 51.7 % | 59.2 | 43.9 |
| GR00T N1.7 (4) | 240.5 | 26.8 | 33.1 | 47.2 | **108.7** | 9.2 | 33.1 → 43.4 % | 11.79 | 2.21× | 166.0 | 36.0 % | 135.3 | 97.1 |
| GR00T N1.6 (4) | 174.6 | 18.4 | 19.9 | 43.2 | **81.7** | 12.2 | 41.7 → 52.8 % | 10.79 | 2.14× | 121.9 | 44.8 % | 103.7 | 74.2 |
| GR00T N1.5 (4) | 168.8 | 12.2 | 26.7 | 35.9 | **78.3** | 12.8 | 35.9 → 45.8 % | 8.97 | 2.15× | 116.9 | 38.9 % | 97.6 | 70.7 |

The geomean AE share is **39.2 % on Orin, 49.4 % on Thor central and 42.3 % low** (jp71 54.4 %; draft 54.0 %).
On Thor the expert's share of the compiled frame rises in every scenario, because the front speeds up more than
the bandwidth-bound expert does.

**Base configuration** (eager; LeRobot models in fp32, GR00T in bf16):

| model | Orin e2e | Thor vision | prefix | AE | **e2e** | Hz | AE share (Orin → Thor) | AE ms/step | low e2e | low AE share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| π0 | 1069.0 | 90.1 | 87.7 | 194.9 | **377.0** | 2.7 | 54.9 → 51.7 % | 19.49 | 706.2 | 54.9 % |
| π0.5 | 3028.6 | 89.5 | 808.4 | 205.1 | **1109.0** | 0.9 | 20.4 → 18.5 % | 20.51 | 2000.6 | 20.4 % |
| SmolVLA | 727.5 | 12.4 | 24.6 | 206.1 | **248.3** | 4.0 | 85.3 → 83.0 % | 20.61 | 480.6 | 85.3 % |
| GR00T N1.7 | 372.5 | 26.4 | 33.1 | 70.1 | **131.1** | 7.6 | 56.7 → 53.5 % | 17.53 | 246.1 | 56.7 % |
| GR00T N1.6 | 327.2 | 23.2 | 23.1 | 66.9 | **114.6** | 8.7 | 61.5 → 58.4 % | 16.71 | 216.2 | 61.5 % |
| GR00T N1.5 | 255.2 | 18.8 | 37.6 | 32.7 | **93.0** | 10.7 | 38.6 → 35.1 % | 8.17 | 168.6 | 38.6 % |

The base-row geomean AE share is 48.4 % on Orin and 45.4 % on Thor central. The eager expert is host-launch-bound
on Orin, so its Thor factor (3.01) folds in the CPU and launch-path uplift (and JetPack 7.2). Low applies a GPU
ratio to a CPU-bound stage, which is why low leaves the base shares unchanged.

## 3. Validation (best configuration, Orin/Thor e2e ratio)

| check | reference ratio | central | low | jp71 | draft |
|---|---:|---:|---:|---:|---:|
| π0.5, ASPLOS'27 Table 7 RTTmax 457.3 / 180.6 | 2.53× | 2.16× | 1.44× | 1.72× | 2.39× |
| GR00T N1.6, ASPLOS'27 Table 7 RTTmax 206.8 / 126.7 | 1.63× | 2.14× | 1.43× | 1.68× | 2.35× |

ASPLOS'27 (Jung et al., arXiv 2610.05062, Table 7) is a different harness, with RTT covering observation
transfer, server pre/post-processing and compiled inference. Its Thor ran **JetPack 7.0**, its π0.5 is the openpi
LIBERO checkpoint, and both models are compiled with max-autotune (Table 5). The two ratios therefore **bracket
rather than match** ours. The scenarios span them: GR00T N1.6 sits between low and jp71, close to jp71, as
expected from Thor software of the 7.0/7.1 generation. π0.5 sits above central. The central row is neither
systematically optimistic nor pessimistic against this one external pair.

Context only: NVIDIA's openpi π0.5 on Thor (Jetson AI Lab `openpi_on_thor`, JP 7.2, MAXN, `pi05_libero`,
torch.compile BF16) takes **132.0 ms total / 128.3 ms model**. Our LeRobot π0.5 best projects to 226.0 ms. The
implementations differ (openpi vs the LeRobot port), so this is not a validation of the factor.

## 4. π0: GPU expert vs AiM on Thor (E12b basis; projected, not measured)

Basis: `A` = 246.28 ms (prefix including vision), `B` = 101.36 ms (fair own-graph expert), `O` = 1.247 (green
12/4), all from E12b, N = 10. AiM per step is 9.31 ms (16 ch) and 5.87 ms (32 ch), **assumed unchanged on
LPDDR5X** `[ASSUMPTION: the step is set by the 5 ns DRAM-core column period, which does not scale with the I/O
rate]`. The GPU co-run estimate is max(A, B) × 1.247 `[ASSUMPTION: Orin's measured O carries over]`. AiM latency
mode is A + 10 × step. The throughput period lower bound is max(A, 10 × step).

| scenario (front ÷, expert ÷) | Thor A | GPU expert (per step) | GPU serial | GPU co-run est. | AiM 16 ch latency (÷ GPU serial) | AiM 32 ch latency (÷ GPU serial) | period ≥ (16 / 32 ch) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **central** (2.646, 1.687) | 93.1 | 60.1 (6.01) | **153.2** | 116.1 | **186.2 (1.215×)** | **151.8 (0.991×)** | 93.1 / 93.1 |
| low (1.514, 1.333) | 162.7 | 76.0 (7.60) | 238.7 | 202.9 | 255.8 (1.071×) | 221.4 (0.927×) | 162.7 / 162.7 |
| high-expert (2.646, 1.514 × 1.333) | 93.1 | 50.2 (5.02) | 143.3 | 116.1 | 186.2 (1.299×) | 151.8 (1.059×) | 93.1 / 93.1 |
| jp71 (2.313, 1.221) | 106.5 | 83.0 (8.30) | 189.5 | 132.8 | 199.6 (1.053×) | 165.2 (0.872×) | 106.5 / 106.5 |
| draft (3.259, 1.700) | 75.6 | 59.6 (5.96) | 135.2 | 94.2 | 168.7 (1.248×) | 134.3 (0.993×) | 93.1 / 75.6 |

GPU expert references on Thor: the floor is 3.322 / 1.333 = **2.49 ms/step** `[ASSUMPTION: Thor reaches the
same 0.915 fraction of its spec bandwidth]`, and the "optimistic bound (2× floor)" goes from 6.30 to **4.73 ms/step**.
The AiM step (9.31 / 5.87 ms) stays above both, while the projected GPU expert falls to 5.0–8.3 ms per step. At
32 ch, AiM in latency mode is about at parity with the GPU serial frame (0.87–1.06× across scenarios; 0.991×
central). At 16 ch it is 1.05–1.30× slower in every scenario. In throughput mode the period at 16 ch is set by the prefix: 10 × 9.31 =
93.1 ms ≈ central A = 93.1 ms, so the two phases balance almost exactly in central.

**Prefix floor check.** If π0's 21.64 GB/frame of prefix traffic (E12b; 3.7× the analytic bytes) persists on
Thor, the prefix cannot go below 21.64 GB ÷ (187.47 × 1.333 GB/s) = **86.6 ms**. Central A (93.1 ms) sits 1.07×
above that floor. The draft's 75.6 ms sits *below* it, so the draft's prefix needed Thor's larger L2 to remove
the re-reads. That is a second, independent reason to prefer the corrected front factor.

## 5. Caveats

- **One calibration pair.** Each factor is one model (GR00T N1.7) on NVIDIA's harness, applied to five other
  models with different kernels. PaliGemma at 816 prefix tokens behaves differently from Qwen3-VL at 1 camera.
  The central Orin side is NVIDIA's harness on JetPack 6 / CUDA 12.6 / torch 2.10. Ours is JetPack 6.2.3 / CUDA
  12.6 / torch 2.8 (GR00T runs used torch 2.10). They agree to 0.8 % on the compiled N1.7 action head.
- **Software is folded into the factors.** JetPack 7.2 / CUDA 13 versus JetPack 6 is part of every central
  factor: on the compiled expert, JP 7.1 → 7.2 alone is 1.38×. The jp71 column shows how much of the projection
  is software.
- **Expert factor above the bandwidth ratio.** The compiled N1.7 expert factor (1.687) exceeds 273/204.8 = 1.333,
  although that DiT runs at its floor on Orin (SYNTHESIS §0). Thor must therefore reach a higher fraction of
  peak, or cache part of the weights in L2. π0's compiled expert is 3–5× above its floor on Orin
  (latency-bound), so its true Thor factor could sit nearer SM × clock. The high-expert row covers that side.
- **Thor's L2 size is unconfirmed.** Third-party sites say 50 MB. It decides whether π0's prefix re-reads persist
  (§4 floor check).
- **The base rows are weak for the LeRobot models.** Their eager baseline is fp32 (π0.5's prefix runs without
  tensor cores), while the factor comes from bf16 GR00T. π0.5's base projection is the weakest.
- **"Other" ÷ 1.5 is an assumption.** It is at most 8.9 ms on Orin (SmolVLA best), so it matters only for
  SmolVLA.
- **AiM on LPDDR5X is assumed unchanged.** A 3.75 ns column, if LPDDR5X allowed it, would scale the AiM step by
  at most about 0.75×. That is not simulated (for microarch-owner).
- **Orin's O = 1.247 carrying over is an assumption.** Thor's 20 SMs and different memory system may partition
  differently.
- **None of this is a Thor measurement.** A Thor device, if one becomes available, would replace every number here.
