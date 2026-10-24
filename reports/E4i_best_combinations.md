# E4i — the best BF16 configuration per model on Thor (T005 steps 3-4)

Jetson AGX Thor, MAXN, GPU 1575 MHz locked, `jetson_clocks`, gate passed before every process, 2026-10-22 … 10-24.

## 1. The answer: `best_bf16`

Each configuration is reported from its **confirmation run**: an independent repeat in a fresh process with
`bench/opt_ladder.py --rung best_bf16`, which replays the recorded configuration. Medians in ms (% of e2e), n = 100.
Source: `results/20261024_E4i_summary/` (`best_bf16.json`, `phase_balance.csv`, `candidates.csv`) and
`results/20261024_E4i_<m>_best_confirm/`.

| model (N) | `best_bf16` | what it is | e2e | vision | prefix | **expert** | other | AE % | confirm ÷ search |
|---|---|---|---|---|---|---|---|---|---|
| π0 (10) | `bf16_mpad_graphs_ae_at` | bf16 casts · `linear_mpad` · CUDA graphs on all three parts · GEMM autotuning on the expert only | **149.3** | 21.1 (14.1 %) | 63.1 (42.2 %) | **62.6 (41.9 %)** | 2.5 | 41.9 | 1.004 |
| π0.5 (10) | `bf16_graphs_ae_at_islands` | bf16 casts incl. `bf16_islands` · graphs on all three parts · expert-only autotuning | **164.4** | 21.1 (12.8 %) | 65.9 (40.1 %) | **74.7 (45.4 %)** | 2.7 | 45.4 | 1.007 |
| SmolVLA (10) | `bf16_mpad_graphs_ae_at_islands` | `bf16_islands` · `linear_mpad` · graphs on all three parts · expert-only autotuning | **48.9** | 15.2 (31.1 %) | 4.3 (8.8 %) | **26.0 (53.2 %)** | 3.4 | 53.2 | 0.983 |
| GR00T N1.7 (4) | `ae_graphs_autotune` | action head compiled with graphs and GEMM autotuning; backbone eager (its vision cannot be compiled) | **133.4** | 29.0 (21.7 %) | 25.3 (19.0 %) | **53.2 (39.9 %)** | 25.9 | 46.2 ¹ | 0.992 |
| GR00T N1.6 (4) | `all_compile_ae_vis_at` | head graphs + autotuning · vision compiled + autotuning · LLM compiled | **102.9** | 17.3 (16.8 %) | 15.0 (14.6 %) | **50.2 (48.7 %)** | 20.4 | 56.8 ¹ | 0.995 |
| GR00T N1.5 (4) | `llm_graphs_autotune` | head graphs · LLM graphs · GEMM autotuning on both | **119.5** | 19.2 (16.1 %) | 24.5 (20.5 %) | **37.3 (31.2 %)** | 38.5 | 35.9 ¹ | 1.026 |

¹ GR00T: AE share of the model call (`model.get_action`), the Orin's definition. GR00T's "other" consists of:
- CPU processing outside the model: 14–18 ms;
- the shipped code's unused vocabulary `lm_head`: 3.6 / 4.1 / 20.1 ms for N1.7 / N1.6 / N1.5.

**Precision.**
- All six have 100 % bf16 weights, except π0's five small fp32 projections (13 MB, 0.16 %).
- Activations are bf16. These stay fp32 by the model code's design: the RMSNorm variance, the attention softmax,
  the noise / Euler update, and tensor-core accumulation.
- No fp16.

**Against the baseline and the Orin.** Orin values are its best configurations, from
`reference/orin/reports/E4i_best_combinations.md`. GR00T e2e is model call against model call
(`reports/E4d_phase_balance.md` §2).

| model | Thor `eager_fp32` → `best_bf16` | Thor ÷ Orin, best e2e | AE share, Orin best → Thor best |
|---|---|---|---|
| π0 | 436.6 → 149.3 ms (2.92×) | 2.84× (Orin `bf16_both` 424.46 ms) | 39.7 → 41.9 % |
| π0.5 | 1427.4 → 164.4 ms (8.68×) | 2.97× (Orin `bf16_both_graphs` 487.81 ms) | 38.0 → 45.4 % |
| SmolVLA | 170.1 → 48.9 ms (3.48×) | 1.89× (Orin `vlm_graphs` 92.30 ms) | 48.6 → 53.2 % |
| GR00T N1.7 | 161.0 → 133.4 ms (1.21×) | 2.09× (Orin `vlm_compile` 240.49 ms) | 33.1 → 46.2 % |
| GR00T N1.6 | 135.4 → 102.9 ms (1.32×) | 1.98× (Orin `all_compile_autotune` 174.59 ms) | 41.7 → 56.8 % |
| GR00T N1.5 | 141.9 → 119.5 ms (1.19×) | 1.63× (Orin `all_compile_autotune` 168.75 ms) | 35.9 → 35.9 % |

**Does the action expert dominate the optimised frame on Thor?**
- **It is the largest single part of the optimised frame for five of the six models:**
  - π0.5: 45.4 %;
  - SmolVLA: 53.2 %;
  - GR00T N1.7: 46.2 % of the model call;
  - N1.6: 56.8 %;
  - N1.5: 35.9 %, where the prefix and the unused `lm_head` are each smaller.
- **For π0 the prefix (42.2 %) and the expert (41.9 %) are tied.**
- Its share rises against the Orin's best for five models and is equal for N1.5.

That reverses the Orin's finding that "no optimised deployment has an action expert that dominates its frame". On
Thor the expert's share no longer falls under optimisation as it did on the Orin. Vision and the prefix gain as much
or more from bf16 casts, compilation and padding, while the expert stays near what its weight streaming allows (T006
measures how near).

## 2. How `best_bf16` was chosen
- **Candidates:**
  - the ladder: 17 rungs per LeRobot model, 11 per GR00T version (`results/20261022_E4c_*`, `E4d_*`);
  - search stage 1: attention backends, full autotuning, `linear_mpad`, whole-policy compile
    (`results/20261022_E4i_<m>_best_search/`);
  - stage 2: per-part autotuning, `bf16_islands`, GR00T re-measures (`results/2026102{3,4}_E4i_<m>_best_search_s2/`).
  - Each candidate ran in its own process: ≥ 20 warm-up until no new graph, 100 timed, regions installed after
    compilation.
- **Eligibility** (`bench/e4i_summary.py`; every exclusion and its reason is in `candidates.csv`):
  1. measured at the model's default N, with the per-part split;
  2. no recompilation during timing, not throttled, normal memory-path state, stable timing (e2e p95/p5 ≤ 1.25);
  3. bf16 weights, fp32 ≤ 1 %;
  4. finite actions, with relative L2 against `eager_fp32` on 10 seeded inferences ≤ max(1e-2, 1.5 × the model's
     compile-only reference). GR00T's reference was re-measured with `fallback_random`, so compiled heads draw the
     same seeded noise as eager;
  5. **added after the first confirmations**, a stricter rule: the choice must pass 1–4 again in its confirmation run.
- `best_bf16` = the fastest eligible candidate. It is reported from its confirmation.
- **Rule 5 decided N1.6.** Autotuning picks kernels by timing, separately in each process, so the numerics differ
  slightly between runs:
  - `vlm_graphs_ae_at` (98.4 ms) passed at 2.20e-2 against a bound of 2.21e-2, then failed its repeat at 2.43e-2;
  - `all_compile_ae_at` (102.6 ms) failed its repeat at 2.95e-2;
  - `all_compile_ae_vis_at` (103.4 → 102.9 ms) passed both.
  - All three confirmation rows are committed: `results/20261024_E4i_groot_n16_best_confirm/`.

## 3. What each lever did
- **bf16 casts are the biggest single lever for the two models LeRobot runs partly in fp32:**
  - π0: vision 173.2 → 21.1 ms;
  - π0.5: e2e 1427 → 183 ms with LeRobot's own cast path (ladder `bf16_both_graphs`).
- **`bf16_islands`** casts the fp32 Linears LeRobot's own bf16 path leaves behind.
  - π0.5 keeps 466 MB of fp32 adaptive-RMSNorm `dense` weights in its expert, read every denoising step.
  - Casting them with input / output casts gave expert 92.4 → 86.7 ms (`bf16_graphs_islands`), and 74.7 ms with
    expert autotuning.
  - It also makes the weights 100 % bf16.
- **`linear_mpad`** (`reports/NOTE_gemm_shape_cliff_m_padding.md`) zero-pads a Linear's rows to the fastest
  measured M; GEMM rows are independent.
  - π0's prefix gate/up GEMM at M = 816 tokens runs 3.8× slower than at nearby M. Padding cut π0's prefix
    108.9 → 64.1 ms, bit-identical.
  - π0.5 (M = 968 is already fast) and SmolVLA gain nothing.
  - GR00T gets *slower*: compiled pad / slice ops cost more than the faster GEMM saves.
- **GEMM autotuning** (Inductor's 68-SM gate lifted):
  - It speeds up the experts: π0 79.8 → 63.1 ms, π0.5 91.9 → 85.0 ms, N1.7 62.4 → 51.7 ms.
  - It slows vision and the prefix (π0.5 vision 21 → 31 ms).
  - So per-part autotuning on the expert only won for the LeRobot models: π0 157.8 → 148.7 ms.
- **Attention backends** (the owner's request). LeRobot forces eager attention; SDPA and flash-attn 2 were switched
  in per part (`bench/e4i_candidates.py`).
  - **Every switch was slower than leaving the eager attention to Inductor's own pattern rewrite:**
    - π0: SDPA prefix 171.9, SDPA expert 170.8, both 172.5, flash-attn prefix 171.9 ms, against 167.1 ms;
    - with the rewrite switched off: 175.7 ms.
  - flash-attn on π0.5's expert needs a per-layer key selection, which breaks its CUDA graphs: 271 ms.
  - Actions stayed within 1.7–3.6e-3 of eager.
- **Whole-policy compile** (`sample_actions` as one graph) is within noise of the per-part configuration and gives
  no per-part split: π0 158.2, π0.5 186.2, SmolVLA 46.5 ms. The fastest SmolVLA run of any kind (46.5 ms) is such a
  configuration, and it keeps 1.4 % fp32 weights.
- **GR00T-specific:**
  - N1.7's Qwen3-VL vision cannot be compiled: flash-attn varlen receives a fake `max_seqlen`, also with
    `capture_scalar_outputs`.
  - Compiling N1.5's vision changes its output by 4.6 % (`results/20261023_E4i_groot_n15_part_diag/`). One of ten
    seeds then lands in another action mode, so N1.5's vision-compiled candidates fail rule 4.
  - Their per-seed median relative L2 (0.026–0.043, `candidates.csv`) is also above the compile-only reference, so
    the rejection does not hinge on that one seed.

## 4. Configurations that maximise the expert's share
Among eligible candidates (`best_bf16.json`, `max_ae_share`):

| model | configuration | AE share |
|---|---|---|
| π0 | `eager_bf16_mpad` | 52.4 % |
| π0.5 | `bf16_graphs_islands` | 49.2 % |
| SmolVLA | `bf16_mpad_graphs_nofuse_islands` | 55.8 % |
| GR00T N1.7 | `eager_fp32` (GR00T loads bf16) | 56.2 % of the model call |
| GR00T N1.6 | `eager_tf32` | 60.8 % |
| GR00T N1.5 | `eager_fp32` | 42.4 % |

For GR00T, as on the Orin, the share peaks at the eager baseline. For the LeRobot models it peaks at intermediate
configurations: the vision / prefix speed-ups arrive before the expert's.

## 5. Caveats
- **The device is shared.** One N1.7 search row was disturbed by another job (vision p5 29 / p95 159 ms). It was
  re-run (`results/20261024_E4i_groot_n17_rerun/`), and timing blocks are now re-timed when e2e p95/p5 > 1.25.
- **The equivalence check is a numerical one, not a task-success one.** LIBERO closed-loop success is T011.
- **N1.5's equivalence is sensitive to one seed (5002)**; see §3.
- **N1.7's expert time varies between processes** (T004, 72–86 ms eager). Its `best_bf16` search and confirmation
  agree within 0.8 %.

## 6. Reproduce
```bash
source env/thor_env.sh
bash bench/queues/t005_ladders.sh <date>; bash bench/queues/t005_e4i.sh <date>; bash bench/queues/t005_e4i_s2.sh <date>
~/venvs/base/bin/python bench/e4i_summary.py --out results/<date>_E4i_summary --report reports/E4d_phase_balance.md
bash bench/queues/t005_confirm.sh <date>      # then re-run e4i_summary.py
~/venvs/lerobot/bin/python bench/opt_ladder.py --model pi0 --rung best_bf16 --out <dir>   # any model, from best_bf16.json
```
