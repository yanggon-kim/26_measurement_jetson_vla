# E4i — the best BF16 configuration per model on Thor (T005 steps 3-4)

Jetson AGX Thor, MAXN, GPU 1575 MHz locked, `jetson_clocks`, gate passed before every process, 2026-10-22 … 10-24.
**Revised 2026-10-24 after the independent review.** That review re-settled GR00T N1.7 and N1.5 (§2) and corrected
several claims in §3. The corrections are listed in the T005 status block.

## 1. The answer: `best_bf16`

Each configuration is reported from its **confirmation run**: an independent repeat in a fresh process with
`bench/opt_ladder.py --rung best_bf16`, which replays the recorded configuration. Medians in ms (% of e2e), n = 100.
Source: `results/20261024_E4i_summary/` (`best_bf16.json`, `phase_balance.csv`, `candidates.csv`) and
`results/20261024_E4i_<m>_best_confirm*/`.

| model (N) | `best_bf16` | what it is | e2e | vision | prefix | **expert** | other | AE % | confirm ÷ search |
|---|---|---|---|---|---|---|---|---|---|
| π0 (10) | `bf16_mpad_graphs_ae_at` | bf16 casts · `linear_mpad` · CUDA graphs on all three parts · GEMM autotuning on the expert only | **149.3** | 21.1 (14.1 %) | 63.1 (42.2 %) | **62.6 (41.9 %)** | 2.5 | 41.9 | 1.004 |
| π0.5 (10) | `bf16_graphs_ae_at_islands` | bf16 casts incl. `bf16_islands` · graphs on all three parts · expert-only autotuning | **164.4** | 21.1 (12.8 %) | 65.9 (40.1 %) | **74.7 (45.4 %)** | 2.7 | 45.4 | 1.007 |
| SmolVLA (10) | `bf16_mpad_graphs_ae_at_islands` | `bf16_islands` · `linear_mpad` · graphs on all three parts · expert-only autotuning | **48.9** | 15.2 (31.1 %) | 4.3 (8.8 %) | **26.0 (53.2 %)** | 3.4 | 53.2 | 0.983 |
| GR00T N1.7 (4) | `ae_at` | action head compiled with graphs + GEMM autotuning on the head only; backbone eager (its vision cannot be compiled) | **133.9** | 29.5 (22.0 %) | 26.1 (19.5 %) | **51.1 (38.2 %)** | 27.2 | 44.7 ¹ | 1.050 ² |
| GR00T N1.6 (4) | `all_compile_ae_vis_at` | head graphs + autotuning · vision compiled + autotuning · LLM compiled | **102.9** | 17.3 (16.8 %) | 15.0 (14.6 %) | **50.1 (48.7 %)** | 20.4 | 56.8 ¹ | 0.995 |
| GR00T N1.5 (4) | `ae_at_mpad` | head graphs + head-only autotuning · `linear_mpad` on the backbone and head | **106.3** | 18.5 (17.5 %) | 26.5 (25.0 %) | **36.8 (34.7 %)** | 24.3 | 41.3 ¹ | 1.008 |

¹ GR00T: AE share of the model call (`model.get_action`), the Orin's definition. GR00T's "other" consists of:
- CPU processing outside the model: 14.6–19.6 ms;
- the shipped code's unused vocabulary `lm_head`: 3.3 / 4.1 / 5.3 ms for N1.7 / N1.6 / N1.5. For N1.5, `linear_mpad`
  cut it from 20.1 ms.

² N1.7's search run measured 127.5 ms and its confirmation 133.9 ms (expert 47.4 vs 51.1 ms). That is the
process-to-process variation of N1.7's expert first seen in T004.

**Precision.**
- All six have 100 % bf16 weights, except π0's five small fp32 projections (13 MB, 0.16 %).
- Activations are bf16. These stay fp32 by the model code's design: the RMSNorm variance, the attention softmax,
  the noise / Euler update, and tensor-core accumulation.
- No fp16.
- GR00T's compiled heads run with `torch._inductor.config.fallback_random = True` (recorded in `best_bf16.json`):
  seeded noise as eager, needed for the equivalence check; it affects one small `randn`.

**Against the baseline and the Orin.** Orin values are its best configurations, from
`reference/orin/reports/E4i_best_combinations.md`. GR00T e2e is model call against model call.

| model | Thor `eager_fp32` → `best_bf16` | Thor ÷ Orin, best e2e | AE share, Orin best → Thor best |
|---|---|---|---|
| π0 | 436.6 → 149.3 ms (2.92×) | 2.84× (Orin `bf16_both` 424.46 ms) | 39.7 → 41.9 % |
| π0.5 | 1427.4 → 164.4 ms (8.68×) | 2.97× (Orin `bf16_both_graphs` 487.81 ms) | 38.0 → 45.4 % |
| SmolVLA | 170.1 → 48.9 ms (3.48×) | 1.89× (Orin `vlm_graphs` 92.30 ms) | 48.6 → 53.2 % |
| GR00T N1.7 | 161.0 → 133.9 ms (1.20×) | 2.10× (Orin `vlm_compile` 240.49 ms) | 33.1 → 44.7 % |
| GR00T N1.6 | 135.4 → 102.9 ms (1.32×) | 1.98× (Orin `all_compile_autotune` 174.59 ms) | 41.7 → 56.8 % |
| GR00T N1.5 | 141.9 → 106.3 ms (1.34×) | 1.89× (Orin `all_compile_autotune` 168.75 ms) | 35.9 → 41.3 % |

**Does the action expert dominate the optimised frame on Thor?**

*The expert is the largest single part for five of the six models:*

| model | expert share | of |
|---|---|---|
| π0.5 | 45.4 % | e2e |
| SmolVLA | 53.2 % | e2e |
| GR00T N1.7 | 44.7 % | the model call |
| GR00T N1.6 | 56.8 % | the model call |
| GR00T N1.5 | 41.3 % | the model call |

- N1.5's expert (36.8 ms) also exceeds its prefix under the Orin's prefix definition, backbone − vision (32.3 ms).
- **For π0 the prefix (42.2 %) and the expert (41.9 %) are tied.**

*Compared with the Orin's best configurations, the expert's share is higher on Thor for all six models.*
- That reverses the Orin's finding that "no optimised deployment has an action expert that dominates its frame".
- Within Thor, the share falls from eager to best for SmolVLA and the three GR00T models, and rises for π0 and π0.5.
  The rise comes from LeRobot's fp32 vision and prefix paths, which bf16 removes.
- How close each optimised expert is to its weight-streaming floor is T006's measurement (deliverable 2), not
  claimed here.

## 2. How `best_bf16` was chosen
- **Candidates.** Each candidate ran in its own process: ≥ 20 warm-up until no new graph, 100 timed, regions
  installed after compilation.
  - The ladder: 17 rungs per LeRobot model, 11 per GR00T version.
  - Search stage 1: attention backends, full autotuning, `linear_mpad`, whole-policy compile.
  - Stage 2: per-part autotuning, `bf16_islands`.
  - Re-measures: `results/20261024_E4i_groot_n17_rerun*`, `…_groot_n15_rerun`.
- **Eligibility** (`bench/e4i_summary.py`; every exclusion and its reason is in `candidates.csv`):
  1. measured at the model's default N, with the per-part split;
  2. no recompilation during timing, not throttled (no over-current event), normal memory-path state, stable timing
     (e2e p95/p5 ≤ 1.25);
  3. bf16 weights, fp32 ≤ 1 %;
  4. finite actions, with relative L2 against `eager_fp32` on 10 seeded inferences ≤ max(1e-2, 1.5 × the model's
     compile-only `ae_graphs` relative L2);
  5. the choice passes 1–4 again in its confirmation run.
  - `best_bf16` = the fastest eligible candidate by its search median, reported from its confirmation.
- **When the rules were set, disclosed.**
  - Rule 4 began as a flat 1e-2.
  - It was changed to the reference-scaled bound on 2026-10-23 (commit ff56d03). That was after all six ladders'
    equivalence results were committed, but before any search result.
    - The ladder results showed that plain compilation alone already moves some models' actions past 1e-2 (SmolVLA
      `ae_graphs` 1.15e-2).
    - Under a flat 1e-2, SmolVLA, N1.6 and N1.5 would have no eligible compiled candidate at all.
  - GR00T's compile-only reference was re-measured with `fallback_random`. The ladder's compiled GR00T rows drew
    different noise from eager: their relative L2 of 0.2–0.8 is a noise mismatch, not a numerical difference, and
    those rows are not used for selection.
  - Rule 5 was added on 2026-10-24, after N1.6's first choice failed on its repeat. It only tightens the selection.
- **The review re-settled GR00T N1.7 and N1.5.**
  - **N1.7:** two stage-2 rows had been disturbed by another job on the shared GPU (e2e p95 363 / 371 ms). Re-run
    clean, `ae_at` (127.5 ms) became the fastest eligible. Its confirmation (133.9 ms) reflects N1.7's
    process-to-process variation.
  - **N1.5:** two candidates had been excluded only by single over-current events. Also, stage 2 had dropped
    `linear_mpad` for GR00T on a premise that is wrong for N1.5, where padding speeds up its unused `lm_head`.
    - The re-measure found `llm_graphs_ae_at_mpad` (102.8 ms). It failed rule 5: relative L2 0.137 on its repeat,
      one seed flipping action mode.
    - `ae_at_mpad` was confirmed at 106.3 ms with relative L2 1.79e-2. It replaces the earlier choice
      (`llm_graphs_autotune`, 119.5 ms).
- **Rule 5 decided N1.6.** Autotuning picks kernels by timing, separately in each process, so the numerics differ
  slightly between runs.
  - `vlm_graphs_ae_at` (98.4 ms) passed at 2.20e-2 against a bound of 2.21e-2, then failed at 2.43e-2.
  - `all_compile_ae_at` (102.6 ms) failed its repeat at 2.95e-2.
  - `all_compile_ae_vis_at` passed both runs.
- **The top candidates are within run-to-run spread.**
  - The gap between the two fastest eligible candidates is 0.5–1 % for several models, while confirm ÷ search runs
    0.98–1.05.
  - The choice is therefore one of several near-equal configurations. The confirmation numbers are what is reported.

## 3. What each lever did

Each claim cites its rows: ladder `results/20261022_E4c_*`, `E4d_*`; search `results/*_E4i_*`.

**bf16 casts**, in eager mode, without compilation (`eager_bf16`):
- π0's vision 173.2 → 28.8 ms;
- π0.5's e2e 1427 → 277 ms.

With compilation and CUDA graphs added, π0's vision reaches 21.1 ms and π0.5's ladder best is 183.1 ms
(`bf16_both_graphs`).

**`bf16_islands`** casts the fp32 Linears that LeRobot's own bf16 path leaves behind.
- π0.5's expert keeps 466 MB of fp32 adaptive-RMSNorm `dense` weights, read every denoising step.
- Casting them, with input / output casts, gave expert 92.4 → 86.7 ms (`bf16_graphs_islands`), and 74.7 ms with
  expert autotuning.
- It also makes the weights 100 % bf16.

**`linear_mpad`** (`reports/NOTE_gemm_shape_cliff_m_padding.md`) zero-pads a Linear's rows to the fastest measured M.
It is exact, because GEMM rows are independent.
- **π0:** its prefix gate/up GEMM at M = 816 tokens runs 3.8× slower than at nearby M. Padding cut π0's prefix
  108.9 → 64.1 ms, bit-identical.
- **π0.5 and SmolVLA:** no gain.
- **GR00T, model-dependent:**
  - *slower for N1.7 and N1.6:* compiled pad / slice ops cost more than they save, e.g. N1.6 `ae_graphs`
    126.0 → 127.3 ms;
  - *faster for N1.5:* `ae_graphs` 134.4 → 118.9 ms, `llm_graphs_autotune` 116.6 → 100.8 ms. Padding moves N1.5's
    unused vocabulary `lm_head` GEMM (814 tokens × 151,680) into a fast kernel band: 20.1 → 5.3 ms in `best_bf16`.

**GEMM autotuning** (Inductor's 68-SM gate lifted).
- It speeds up the experts:
  - π0: 79.8 → 63.1 ms;
  - π0.5: 91.9 → 85.0 ms;
  - N1.7: 62.4 → 51.7 ms.
- It slows vision and the prefix (π0.5 vision 21 → 31 ms).
- Expert-only autotuning therefore won:
  - for π0: 157.8 → 148.7 ms;
  - for π0.5 with `bf16_islands`: 163.2 ms.
- It did not win for SmolVLA without `bf16_islands`: `bf16_mpad_graphs` 49.8 vs `_ae_at` 51.0 ms, within noise.

**Attention backends** (the owner's request). LeRobot forces eager attention; SDPA and flash-attn 2 were switched in
per part (`bench/e4i_candidates.py`). **Every switch was slower than leaving the eager attention to Inductor's own
pattern rewrite.**

| π0 configuration | e2e ms |
|---|---|
| base: Inductor's rewrite of eager attention | 167.1 |
| SDPA prefix | 171.9 |
| SDPA expert | 170.8 |
| SDPA both | 172.5 |
| flash-attn prefix | 171.9 |
| rewrite switched off | 175.7 |

- flash-attn on π0.5's expert needs a per-layer key selection, which breaks its CUDA graphs: 271 ms.
- Actions stayed within 1.7–3.6e-3 of eager.

**Whole-policy compile** (`sample_actions` as one graph) gives no per-part split, so it was never eligible.

| model | whole-policy compile | per-part compiled counterpart | `best_bf16` |
|---|---|---|---|
| π0 | 158.2 ms | 157.8 ms | 148.7 ms |
| π0.5 | 186.2 ms | — | 163.2 ms |
| SmolVLA | 46.5 ms | 54.4 ms | 48.9–49.8 ms |

SmolVLA's 46.5 ms run, with full autotuning, is 5–7 % faster than `best_bf16`, and it keeps 1.4 % fp32 weights.

**GR00T-specific:**
- N1.7's Qwen3-VL vision cannot be compiled: flash-attn varlen receives a fake `max_seqlen`, also with
  `capture_scalar_outputs`.
- N1.6's vision with CUDA graphs failed with a CUDA-graph memory-pool error (`ladder vis_graphs`, traceback in its
  raw/).
- Compiling N1.5's vision changes its output by 4.6 % (`results/20261023_E4i_groot_n15_part_diag/`).
- N1.5's equivalence is dominated by one knife-edge seed (5002).
  - Many N1.5 candidates, with or without compiled vision, flip that seed to another action mode (aggregate relative
    L2 ≈ 0.137).
  - For the vision-compiled ones, the per-seed median (0.028–0.043) is also above the bound.
  - For some others the rejection hinges on that one seed: `llm_compile_ae_at` has a per-seed median of 0.025, just
    under the 0.0258 bound.

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

## 5. Measured, derived, not measured, pending
- **Measured:** every latency (CUDA events, wall clock) and every action-equivalence value; the rows are cited.
- **Derived:** shares, ratios, Thor ÷ Orin, and the per-step slopes in `reports/E4d_phase_balance.md` §3b.
- **Not measured:** task success (LIBERO closed loop is T011); how close each part is to its roofline (T006).
- **Pending in T005:**
  - Part A (`E4a_*_paper_config`, the published configuration and its fp32 control);
  - the π0 N = 4 control.

## 6. Caveats
- **The device is shared.** Disturbed rows are re-timed automatically (e2e p95/p5 > 1.25); the earlier disturbed
  ones were re-run.
- **Over-current events exclude a row.** GR00T N1.5 `ae_graphs_mpad_autotune` raised one twice, in independent runs
  (111.3 and 111.9 ms), so it stays excluded.
- **The equivalence check is numerical, not task success** (T011).
- **N1.7's expert time varies between processes** (T004: 72–86 ms eager; here search 47.4 vs confirmation 51.1 ms).

## 7. Self-review (protocol §11)
| check | result |
|---|---|
| `eager_fp32` and `best_bf16` with vision / prefix / expert / other, median, p5, p95, n ≥ 100, same harness | yes (`phase_balance.csv`, the confirmation runs) |
| `best_bf16.json` with exact flags; `bench/` reproduces each entry from one option | yes (`--rung best_bf16`; `linear_mpad` table replayed; `fallback_random` recorded) |
| every ladder rung measured or "not applicable" with a reason; every candidate's equivalence recorded | yes (`candidates.csv`, `equivalence.csv` per directory) |
| Thor vs Orin per-part table and the one-paragraph answer | yes (`reports/E4d_phase_balance.md` §2, §1 above) |
| per-step time by slope for compiled rungs | yes for rows timed at three N (E4d §3b) |
| Part A, π0 N = 4 control | pending |
| independent review (P0) | done; findings fixed (status block) |

## 8. Reproduce
```bash
source env/thor_env.sh
bash bench/queues/t005_ladders.sh <date>; bash bench/queues/t005_e4i.sh <date>; bash bench/queues/t005_e4i_s2.sh <date>
bash bench/queues/t005_regroot.sh <date>
~/venvs/base/bin/python bench/e4i_summary.py --out results/<date>_E4i_summary --report reports/E4d_phase_balance.md
bash bench/queues/t005_confirm.sh <date>      # then re-run e4i_summary.py; repeat for a choice that fails rule 5
~/venvs/lerobot/bin/python bench/opt_ladder.py --model pi0 --rung best_bf16 --out <dir>   # any model, from best_bf16.json
```
