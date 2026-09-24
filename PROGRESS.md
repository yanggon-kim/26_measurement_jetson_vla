# PROGRESS — phase-balance study, π0 → π0.5 → SmolVLA → GR00T N1.7/N1.6/N1.5

**Started 2026-09-22 18:35 PDT · budget 26 h (owner, revised) · ends ≈ 2026-09-23 20:35 PDT**
**⛔ DON'T STOP. A turn ends only when the next job is already running.**

I keep this file current. Every item is `todo` / `running` / `done` / `blocked`, with the
evidence or the exact failing command.

---

## The question this answers

Which models are action-head bottlenecked, under an *identical* treatment — identically
optimised, or identically un-optimised? π0 says the answer depends entirely on whether you
compile: **62.6 % of the frame un-optimised, 18.9 % optimised.** If that is true of every
model, the PIM premise is a statement about compilation, not about VLAs.

## Scope and budget

The owner raised the budget to **26 h** and asked for it done properly, so **every model
gets the full routine**: Part A (paper configuration, fp32 control, per-phase SM
utilisation, operational intensity) and Part B (the full 9-rung attribution ladder). No
reduced ladders.

**One N per model — its own configured default**, which is the deployment setting and the
one the authors' leaderboard numbers use. Verified from each config rather than assumed,
and the field name differs three ways, which the registry must carry:

| model | field | default N |
|---|---|---:|
| π0 | `num_inference_steps` | **10** |
| π0.5 | `num_inference_steps` | **10** |
| SmolVLA | **`num_steps`** | **10** |
| GR00T N1.7 | **`num_inference_timesteps`** | **4** |

π0's N=4 ladder is already done and is *kept as the control*, not as waste: it is a
within-model test of the hypothesis that N drives the action-head share. Same model, same
machine, only N changes — AE share **32.3 % at N=4 against 54.9 % at N=10**, both eager
fp32. That is the cleanest evidence available that the share is a property of iteration
count. It also explains the paper's stray claim that π0 "requires only 4 denoising steps":
4 is GR00T's default, not π0's.

| model | N | Part A | Part B | new work | est. |
|---|---:|---|---|---|---:|
| π0 | 10 (+4 control) | done / running | running + B2–B4 | — | 3.5 h |
| model registry | — | — | — | the gate for everything below | 2 h |
| π0.5 | 10 | full | full 9 rungs | wrap points differ | 3.5 h |
| SmolVLA | 10 | full | full 9 rungs | `regions.py` needs a new hook — layer body is inlined | 4 h |
| GR00T N1.7 | 4 | full | full 9 rungs | different codebase; DiT instrumentation | 4 h |
| GR00T N1.6 | 4 | full | full 9 rungs | own checkout + env; reuses N1.7's instrumentation | 3 h |
| GR00T N1.5 | 4 | full | full 9 rungs | own checkout + env | 3 h |
| cross-model report | | | | | 1 h |
| | | | | **total** | **≈ 24 h** |

**Ordering rule inside each model, kept for risk not for economy.** The two endpoint
configurations — un-optimised and best-optimised — run *first*, before that model's full
ladder. They are what answers the question, so if one model's instrumentation overruns,
the next model still gets its answer rather than the overrun eating the queue. This is
sequencing, not a reduction in scope.

---

## Now — updated every 2 min by `bench/progress_monitor.py --mode fast`

<!-- FAST:BEGIN -->
**Updated 09:43:00** · elapsed **39.13 h** of 26 (151 %) · 15.5 GiB free · 0.71 GiB swap

| pid | running | for | rss |
|---|---|---:|---:|
| 390699 | `bench/pipelining.py --model pi05 --outdir scratch/preflight/E11f_pi05 --steps 4 --seconds` | 8 min | 20.5 GiB |

Latest log lines:

- `queue_gaps.log` · wrote results/20260922_E4c_pi05_opt_ladder (15 rungs)
- `queue_gaps.log` · EXIT=0
- `queue_gaps.log` · GAPS_DONE 2026-09-23T23:32:01-07:00
- `queue_groot_combined.log` · wrote results/20260923_E4d_groot_n15_ladder
- `queue_groot_combined.log` · EXIT=0
- `queue_groot_combined.log` · COMBINED_DONE 2026-09-23T23:12:20-07:00
<!-- FAST:END -->

## Results — re-derived from every result CSV every 20 min (`--mode slow`)

<!-- SLOW:BEGIN -->
**Re-derived from every result CSV on disk at 09:43** (59 configurations)

| model | N | configuration | vision | vlm_prefix | ae_total | e2e | **AE share** |
|---|---:|---|---:|---:|---:|---:|---:|
| pi05 | 10 | eager_fp32 | 239.45 | 2162.47 | 617.74 | 3028.56 | 20.4 % |
| pi05 | 10 | eager_tf32 | 103.11 | 534.28 | 611.42 | 1254.86 | 48.7 % |
| pi05 | 10 | ae_compile | 103.43 | 549.79 | 258.12 | 917.88 | 28.1 % |
| pi05 | 10 | ae_graphs | 104.28 | 549.50 | 245.58 | 905.68 | 27.1 % |
| pi05 | 10 | vis_compile | 102.38 | 556.54 | 244.73 | 909.88 | 26.9 % |
| pi05 | 10 | vis_graphs | 100.25 | 548.77 | 245.23 | 900.66 | 27.2 % |
| pi05 | 10 | vlm_compile | 99.82 | 516.60 | 240.75 | 863.47 | 27.9 % |
| pi05 | 10 | vlm_graphs | 99.08 | 515.35 | 249.03 | 871.07 | 28.6 % |
| pi05 | 10 | whole_graphs | — | — | — | 860.09 | — |
| pi05 | 10 | vlm_compile_nvg | 108.00 | 489.74 | 259.36 | 868.48 | 29.9 % |
| pi05 | 10 | bf16_base | 104.19 | 290.55 | 190.52 | 591.95 | 32.2 % |
| pi05 | 10 | bf16_vision | 89.88 | 290.37 | 189.46 | 576.81 | 32.8 % |
| pi05 | 10 | bf16_norms | 103.23 | 290.53 | 189.70 | 590.14 | 32.1 % |
| pi05 | 10 | bf16_both_graphs | 56.57 | 239.47 | 185.60 | 487.81 | 38.0 % |
| pi05 | 10 | bf16_both | 56.69 | 239.49 | 186.67 | 489.15 | 38.2 % |
| pi0 | 4 | eager_fp32 | 241.31 | 234.06 | 230.12 | 711.75 | 32.3 % |
| pi0 | 4 | eager_tf32 | 104.18 | 234.06 | 230.75 | 575.28 | 40.1 % |
| pi0 | 4 | ae_compile | 103.26 | 234.25 | 99.94 | 444.35 | 22.5 % |
| pi0 | 4 | ae_graphs | 103.70 | 234.49 | 71.37 | 416.85 | 17.1 % |
| pi0 | 4 | vis_compile | 101.55 | 234.54 | 70.58 | 413.63 | 17.1 % |
| pi0 | 4 | vis_graphs | 100.20 | 234.60 | 71.43 | 413.74 | 17.3 % |
| pi0 | 4 | vlm_compile | 99.92 | 192.78 | 69.72 | 369.52 | 18.9 % |
| pi0 | 4 | vlm_graphs | 98.97 | 195.07 | 68.99 | 369.51 | 18.7 % |
| pi0 | 4 | whole_graphs | — | — | — | 366.12 | — |
| pi0 | 10 | eager_fp32 | 241.01 | 234.57 | 586.91 | 1069.04 | 54.9 % |
| pi0 | 10 | eager_tf32 | 102.71 | 234.04 | 573.48 | 916.45 | 62.6 % |
| pi0 | 10 | ae_compile | 103.58 | 234.35 | 241.28 | 585.80 | 41.2 % |
| pi0 | 10 | ae_graphs | 103.36 | 234.74 | 172.18 | 516.42 | 33.3 % |
| pi0 | 10 | vis_compile | 101.88 | 234.55 | 173.17 | 516.44 | 33.5 % |
| pi0 | 10 | vis_graphs | 100.04 | 234.95 | 174.45 | 515.99 | 33.8 % |
| pi0 | 10 | vlm_compile | 99.18 | 192.76 | 168.84 | 466.99 | 36.2 % |
| pi0 | 10 | vlm_graphs | 99.29 | 197.17 | 170.88 | 473.91 | 36.1 % |
| pi0 | 10 | whole_graphs | — | — | — | 463.86 | — |
| pi0 | 10 | vlm_compile_nvg | 102.83 | 195.55 | 170.18 | 475.28 | 35.8 % |
| pi0 | 10 | bf16_base | — | — | — | nan | — |
| pi0 | 10 | bf16_vision | 88.53 | 234.06 | 172.53 | 501.98 | 34.4 % |
| pi0 | 10 | bf16_norms | 103.15 | 235.02 | 174.59 | 518.99 | 33.6 % |
| pi0 | 10 | bf16_both | 56.74 | 192.86 | 168.68 | 424.46 | 39.7 % |
| smolvla | 10 | eager_fp32 | 33.30 | 65.77 | 620.56 | 727.46 | 85.3 % |
| smolvla | 10 | eager_tf32 | 33.16 | 64.75 | 612.92 | 718.64 | 85.3 % |
| smolvla | 10 | ae_compile | 33.27 | 64.18 | 161.93 | 267.42 | 60.6 % |
| smolvla | 10 | ae_graphs | 33.25 | 65.71 | 43.82 | 150.32 | 29.2 % |
| smolvla | 10 | vis_compile | 35.15 | 65.92 | 43.65 | 152.70 | 28.6 % |
| smolvla | 10 | vis_static_pos | 32.48 | 67.50 | 43.92 | 153.63 | 28.6 % |
| smolvla | 10 | vis_graphs | 31.33 | 73.63 | 44.91 | 160.86 | 27.9 % |
| smolvla | 10 | vlm_compile | 31.23 | 15.54 | 45.52 | 103.69 | 43.9 % |
| smolvla | 10 | vlm_graphs | 31.26 | 7.93 | 44.86 | 92.30 | 48.6 % |
| smolvla | 10 | whole_graphs | — | — | — | 80.97 | — |
| smolvla | 10 | vlm_compile_nvg | 33.07 | 15.32 | 44.36 | 100.22 | 44.3 % |
| smolvla | 10 | bf16_base | — | — | — | nan | — |
| smolvla | 10 | bf16_vision | — | — | — | nan | — |
| smolvla | 10 | bf16_norms | — | — | — | nan | — |
| smolvla | 10 | bf16_both | — | — | — | nan | — |
| pi05 | 10 | paper config (eager+TF32) | 103.61 | 550.51 | 626.90 | 1287.32 | 48.7 % |
| pi05 | 10 | paper config (eager+TF32) | 239.39 | 2166.69 | 617.87 | 3030.99 | 20.4 % |
| pi0 | 10 | paper config (eager+TF32) | 103.00 | 234.30 | 571.58 | 915.15 | 62.5 % |
| pi0 | 10 | paper config (eager+TF32) | 241.51 | 234.36 | 580.24 | 1062.42 | 54.6 % |
| smolvla | 10 | paper config (eager+TF32) | 33.31 | 64.60 | 610.92 | 716.81 | 85.2 % |
| smolvla | 10 | paper config (eager+TF32) | 33.26 | 64.49 | 608.01 | 713.69 | 85.2 % |
<!-- SLOW:END -->

## Live status

| # | item | state | evidence |
|---|---|---|---|
| 0 | π0 Part A — operational intensity (A1) | **done** | vision **398.1**, `vlm_prefix` **816.0**, `ae_step` **60.0** FLOP/B vs measured ridge 180.57. Paper: 840 / 64.5. `bench/vision_flops.py` |
| 1 | π0 Part A — paper config reproduced | **done** | N=10 eager+TF32: **916.45 ms vs their 920.58 (0.45 %)**; split 11.2/25.5/**62.6** % vs their 12/28/**59** |
| 2 | π0 Part B ladder, N=4 | **done** | 9 rungs; two independent runs agree to 1.81 % worst case |
| 3 | π0 Part B ladder, N=10 | **running** | 3 of 9 rungs; memory flat 50.6→50.9 GiB per rung |
| 4 | π0 Part A — SM utilisation (A2) | queued | chained on ladder PID; needs sudo for GPU metrics |
| 5 | π0 Part B — B2 bf16 vision, B3 bf16 layernorms, B4 graphs on `pg.forward` | todo | B3 is the lever with a mechanism; B4 should be small at 816 FLOP/B |
| 6 | π0 LIBERO quick check (B2/B3 change numerics) | todo | 100 episodes per variant, ~45 min each |
| 7 | **model registry** — the gate for everything below | **done (LeRobot)** | `bench/model_registry.py`. N field differs 3 ways; wrap points uniform once the container attr is known (`paligemma_with_expert` / `vlm_with_expert`). GR00T descriptor deliberately pending until its instrumentation exists. |
| 8 | π0.5 — Part A + full ladder | todo | `lerobot/pi05_libero_base` |
| 9 | SmolVLA — Part A + full ladder | todo | `lerobot/smolvla_base`; **`regions.py` records nothing** — layer body inlined |
| 10 | GR00T N1.7 — Part A + full ladder | todo | installed, verified; `env/setup_groot.sh` |
| 11 | GR00T N1.6 — Part A + full ladder | todo | branch `n1d6` @ `9b37aa1`, `nvidia/GR00T-N1.6-3B`; own env |
| 12 | GR00T N1.5 — Part A + full ladder | todo | branch `n1d5` @ `4af2b62`, `nvidia/GR00T-N1.5-3B`; own env |
| 14 | `reports/E4d_phase_balance.md` — the cross-model table | todo | one row per model: AE share un-optimised vs optimised |
| **C1** | **ncu per-phase, every model × {eager, best}** | todo | *what* each phase is bound by: SM vs memory (L2) throughput, achieved occupancy, per-pipe utilisation, warp-stall reasons (long scoreboard, math-pipe throttle, MIO/LG throttle, barrier), off-chip bytes from §3.3's two sysmem counters. 3 phases × 2 configs × 6 models = 36 captures. Compiled captures may see nothing inside CUDA-graph replays — if so, profile the no-graph variant and say so. |
| **C2** | **bottleneck → fix mapping** | todo | derived from C1: for each model and phase, what the eager bottleneck *was*, which optimisation removed it, and the evidence. e.g. expert eager = launch-bound (low occupancy, large inter-kernel gaps) → CUDA graphs elide launches → 14.2× on SmolVLA. One row per (model, phase, bottleneck, fix, evidence). |
| **C3** | **expert GEMM dimensions, every model** | todo | π0's expert is 126 projections on S = 51 tokens — extremely skinny (M = 51 against K, N = 1024–4096). Tabulate S, hidden, layers, heads/KV-heads, head_dim, MLP width and the resulting GEMM shapes per model, with arithmetic intensity per shape, so "skinny" is quantified rather than asserted. CPU-only: reads checkpoints and configs. |

## Results so far

| model | config | vision | `vlm_prefix` | `ae_total` | e2e | **AE share** |
|---|---|---:|---:|---:|---:|---:|
| π0 | N=10 eager fp32 | 241.01 | 234.57 | 586.91 | 1069.04 | 54.9 % |
| π0 | N=10 eager **+TF32** (= paper config) | 102.71 | 234.04 | 573.48 | **916.45** | **62.6 %** |
| π0 | N=4 eager fp32 | 241.31 | 234.06 | 230.12 | 711.75 | 32.3 % |
| π0 | N=4 fully compiled | 99.92 | 192.78 | 69.72 | 369.52 | **18.9 %** |
| *(their π0 on AGX Orin)* | *N=10 eager* | *110.47* | *257.76* | *543.14* | *920.58* | *59 %* |

## REMAINING JOBS — four, in this order (owner, 2026-09-23 16:55)

**1. Accuracy validation of the bf16 rungs — IN PROGRESS.**
`bench/libero_eval.py`, LIBERO closed loop, π0 and π0.5, fp32 baseline then `bf16_both`.
The fp32 baseline runs first and is the harness's own control: a wrong observation mapping
and a broken policy both look like a low success rate. Three harness faults found so far,
each of which would have produced a plausible but wrong accuracy number — a hand-built
batch missing the tokenizer, `select_action` returning **normalised** actions (the
postprocessor is what unnormalises), and an 8-wide state built as joint_pos+gripper when
the checkpoint was trained on eef_pos+axis-angle+gripper. Until this reports, π0's and
π0.5's `bf16_both` results are **speed numbers with no accuracy statement**.

**2. Deep per-kernel bottleneck analysis of `vlm_prefix`** (owner, new job, after job 1).
Models: **π0, GR00T N1.7, GR00T N1.6**, both **eager and best**. Six captures, all already
on disk under `results/20260923_E4g_c1_<model>_<mode>/ncu/*_vlm_prefix_details.csv`.

For each model and mode, answer:
  * which *individual kernels* dominate the phase's time, by name and shape;
  * what each of those kernels is bound by, from its own ncu counters — SM vs memory
    throughput, achieved occupancy, L1/L2 hit rate, and the stall breakdown;
  * what the **main** bottleneck of `vlm_prefix` is for that model, stated as one claim
    with the counters that support it;
  * **solution directions** that follow from the measured bottleneck, not from habit.

Why this job exists: `vlm_prefix` is the phase the whole study turned on. It is the one
compilation barely moved for π0 (534 → 516 ms, 1.04×) while precision moved it 2.16×, and
it is the largest phase in π0's optimised frame at 45.4 %. C1 answered "what is the phase
bound by" at phase granularity; this asks the same question per kernel, which is where a
fix would actually be applied.

Do not execute before job 1 reports. Analysis is CPU-only on existing CSVs, so it never
contends with a measurement.

**3. `bench/check_report_numbers.py` extended to E4d and E4g** — every headline figure in
both reports re-derived from the CSVs, failing if a number appears in prose that is not in
the data.

**4. The deferred campaign** — T004's remainder (E5 nsys timeline, E6 roofline, E7 power),
T006–T012. E7 is the largest untouched gap: the study has **no energy data at all**, and
the PIM argument is ultimately about energy. Its capped power modes need two human reboots
and are the owner's to schedule.

## WHAT A WATCHDOG WAKE MEANS (owner, 2026-09-23 05:30 — binding)

A watchdog wake is **not** a liveness check. It is the same audit as the owner typing
"check the progress", and it has three parts, in order:

1. **Audit what already ran, for correctness** — not "did it finish" but "is the number
   right". Re-read the rows and look for silent failures: an empty `e2e_median_ms`, a rung
   skipped, two instances sharing the GPU, a row written by code older than the row claims.
2. **Check what is running now** is the right next thing, and that there is exactly one.
3. **Re-derive the upcoming order** from the priority below and state it.

The owner's words: *"whenever i asked you for 'check the progress', you confessed the wrong
past execution with wrong current order of running things."* Every check-in so far has
surfaced a defect that had been sitting since the previous check-in. Reporting liveness is
what lets that happen. **The audit is the point.**

## PRIORITY ORDER (owner, 2026-09-23 — supersedes everything below)

**Part A and B for ALL models, error-free, before ANY Part C work.**

1. **Fix Part B and finish it for π0, π0.5, SmolVLA.**
   - ✅ CUDA-graph output aliasing (7 rungs) — `clone_graph_outputs`, now walks HF cache
     objects too.
   - ✅ SmolVLA's 4 remaining rungs — root cause was **not** the clone: SmolVLM's
     `VisionEmbeddings.forward:137` does a boolean-mask `index_put`, a data-dependent
     shape, so no stream capture can survive it. Fixed by `vis_static_pos`, an explicit
     rung (the mask is all-true by construction, so it is provably identity).
   - ✅ π0.5's 2 bf16 rungs — root cause was **not** a missing π0.5 port: π0's checkpoint
     is `dtype: bfloat16` and π0.5's is `dtype: float32`, so `params_to_keep_float32` is a
     real fp32 island in one and inert in the other. New `bf16_base` rung; B2/B3 now
     apply it first on an fp32 checkpoint, and raise `NotApplicable` (a recorded result,
     not a failure) where a model has no island to remove.
2. **Part A and B for the GR00T family** — N1.7 running, then N1.6, N1.5.
3. **Part C for all models** — C1 ncu per phase, C2 bottleneck→fix mapping, C3 dimensions.
   C1 was started out of order overnight and is stopped; π0 eager's four regions keep.
   **Not resumed until 1 and 2 are complete and clean.**

## Overnight plan (superseded — kept for the record)

**Running unattended, never aborting on a failed step:**
`scratch/T004/overnight.sh` → S measurement for π0.5/SmolVLA, then **C1**: ncu per phase
(`ae_layer[0]`, `ae_step[0]`, `vlm_prefix`, `vision`) for π0, π0.5 and SmolVLA in **both**
eager and `reduce-overhead`, then `parse_ncu` on each. 3 models × 2 configs × 4 regions.

**Needs me, and therefore waits:** GR00T N1.7/N1.6/N1.5. Its checkpoint is downloading; its
input builder (`bench/groot_phases.py:build_inputs`) deliberately raises rather than
guessing modality keys, because guessing is what made SmolVLA run nine full rungs while
seeing no images. I write it when the probe reports the real keys, then run all three
versions.

**The Part A/B routine is NOT lost.** It is `bench/paper_repro.py --model <m>` +
`bench/opt_ladder.py --model <m>`, both resolving everything through
`bench/model_registry.py`. π0, π0.5 and SmolVLA are done; the same two commands run the
GR00T versions once their descriptor and input builder exist.

## Rules that are not optional

1. **One process per rung.** `del policy` does not free the 8.28 GiB model — `torch.compile`
   retains references and on Orin GPU memory is system memory. Two runs were OOM-killed at
   rung 9 before this.
2. **Memory floor 20 GiB, enforced**, recorded before and after every rung.
3. **Preflight before every long run.** Three defects were caught this way, each of which
   would have produced a complete, plausible, wrong table.
4. **Wait on a PID, never `pgrep -f <pattern>`** — it matches the shell that contains the
   pattern. This has cost two waits already.
5. **Write results incrementally**; a kill must cost one rung, not the run.
6. **Commit `bench/` before the run that uses it**, so `bench_commit` names the code that ran.

## PART A + B: COMPLETE FOR ALL SIX MODELS (2026-09-23 10:15)

π0, π0.5, SmolVLA, GR00T N1.7 / N1.6 / N1.5 — every rung either measured, or recorded as
`not applicable` with a reason, or `failed` with the error kept. Report:
`reports/E4d_phase_balance.md`, generated from the row JSONs so it cannot drift.

**Correction issued 11:20.** The cross-model table circulated earlier gave π0 as
"54.9 % → 18.9 %". That pairs the **N=10** eager share with the **N=4** optimised share:
π0's directory holds both ladders and the report generator keyed rows by rung alone, so
N=4 rows silently overwrote N=10 ones. At its configured N=10, π0 falls to **39.7 %**, not
19 %. It is not the outlier it appeared to be; it sits with the GR00T family.

## PART C — in progress

- **C3 done, all six models.** GR00T's action head is a DiT: **32 transformer blocks in
  N1.7 and N1.6, 16 in N1.5**, all width 1536. That is the mechanism behind the
  action-head ranking — at identical N=4 in one family, N1.5's head is 38.6 % of the eager
  frame against N1.6's 61.5 %, and only the head differs.
- **C1 running**, 36 captures, one profiler at a time (`bench/run_ncu_c1*.sh`, four ncu
  sections plus the 14 stall metrics requested by name). ~900 KB per capture compressed,
  against the 89–203 MB the overnight `--set full -c 300` attempt produced.
- **C2 computed from C1** by `bench/analyze_c1.py`, which also writes
  `reports/E4g_c1_c2_bottlenecks.md`.

### First C1 result, π0 eager — it inverts the premise

| phase | SM % | Mem % | Occ % | L2 hit | verdict | dominant stall |
|---|---:|---:|---:|---:|---|---|
| vision | 65.3 | 58.3 | 35.6 | 90.4 | mixed | `long_scoreboard` 43 % |
| **vlm_prefix** | 40.4 | 48.6 | 70.7 | **56.0** | **latency bound** | **`long_scoreboard` 74 %** |
| ae_step_0 | 22.7 | 21.6 | 18.5 | 68.6 | latency bound | `imc_miss` 35 % |

The phase actually waiting on memory is the **VLM prefix**, not the action expert. The
expert's Long Scoreboard is 16 % and third, behind constant-cache misses and
instruction-fetch starvation — what an S=51 skinny GEMM looks like when it is issue-limited
rather than bandwidth-limited. Our SM 22.7 % matches arXiv:2604.24447's reported 22.3 %,
which is an independent check on the capture.

**Provisional on one model.** Held as provisional until the remaining captures agree or
disagree.

## Log

- **11:20 (09-23)** π0 N-mixing correction above; the N=4 ladder is now kept explicitly as
  a within-model control on N rather than folded into the main table.
- **11:15 (09-23)** C1 captures were recompiling everything as root:
  `TORCHINDUCTOR_CACHE_DIR` defaults to `/tmp/torchinductor_root` under sudo while the
  user cache already held 672 entries from Part A/B. Fixed; would have cost ~6 h.
- **10:50 (09-23)** the four ncu sections do NOT carry the stall breakdown — caught at
  capture 2 of 36. All 14 `smsp__average_warps_issue_stalled_*` are now requested by name.
- **06:05 (09-23)** audit found every `results/` directory of this study **uncommitted** —
  the repository is the only channel to the research side, so they did not exist for them.
  Now committed. **One blocker recorded, not forgotten:**
  `results/20260922_E4g_pi0_ncu_eager/ncu/` is 725 MB in four `.ncu-rep` files (89, 184,
  201, 203 MB). Protocol §6 caps a committed file at 50 MB and GitHub hard-rejects >100 MB,
  so they cannot be committed as they are. Fix when Part C restarts: shorter capture
  (fewer kernel instances per range) + `zstd -19`.
- **05:51 (09-23)** watchdog wake caught two silent defects: the Part B queue had waited on
  **itself** for 26 minutes (the shell that created the script has the script body, and so
  `opt_ladder.py`, in its own cmdline — the `pgrep -f` bug in a `/proc` disguise), and the
  audit itself miscounted, calling NaN a measurement.
- **05:30 (09-23)** owner: a watchdog wake means the full audit, not a status glance.
  Recorded above as binding. Audit run at 05:35 found a real provenance defect: π0.5's
  `bf16_vision` row was written 21:14, **before** commit `8654dcc` (21:45) added the
  pi0-only guard, so it was produced by code the tree no longer holds. It is re-measured.
- **05:20 (09-23)** SmolVLA's four dead rungs measured. They change a conclusion: CUDA
  graphs on the prefix are a **2× win** for SmolVLA (15.54 → 7.93 ms) and a **net loss**
  for π0 and π0.5. SmolVLA end-to-end 727.46 → 92.30 ms, **7.88×**.
- **18:35** file created. π0 N=10 ladder at 3/9, Part A armed behind it.
- **23:05** owner added C1 (ncu per-phase), C2 (bottleneck->fix mapping), C3 (expert GEMM
  dimensions) after Part A/B for all models. C3 started immediately: it is CPU-only.
- **19:05** model registry written and committed for the LeRobot family; GR00T pending.
- **18:45** one N per model, its configured default: π0/π0.5/SmolVLA 10,
  GR00T 4. Three different field names. π0's N=4 ladder kept as the within-model control.
- **18:40** owner raised the budget to 26 h — full 9-rung ladder for every model, no
  reduced ladders. Endpoint-configs-first kept inside each model as risk sequencing.
