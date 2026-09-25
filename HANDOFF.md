# HANDOFF — jetson

*Root: `/home/yanggon/05_VLA_LPDDR/jetson` · Owner: `jetson-owner` · Last updated: 2026-09-25 (T015 / E12b issued) by jetson-owner*

## Current status

### 2026-09-25, later: T015 / E12b issued and pushed — the follow-up that should give E12 a clear conclusion

User-approved push (via main). **E12b = compiled co-run, MAXN only**: spec `docs/E12b_compiled_corun/README.md`
(problem T014 left open, why, definitions, 5 NNs, M0–M5, **pre-registration §10 dated 2026-09-25**, outputs,
acceptance §12, relation to T014 + Step-4 note §14); task `tasks/T015_pi0_compiled_corun.md`; order-table row 6b
+ note (after the stage in progress, before other MAXN work; after capped queues if already rebooted; owner's
direct order wins); protocol **§15**; AGENTS / README read lists; E12 README §13 pointer (§9 untouched); T014 body
note (open MAXN items → T015); `docs/04_research_answers.md` 2026-09-25 entry (5 answers incl. Step-4 correction).

Design: M1 fair expert = `bf16_both` + timestep hoist (stock vs hoist, compiled equivalence, prefix/ae/full
bracketed). **User removed `cropcache` from this task mid-drafting (2026-09-25): P2 back to not approved.** M2
diagnose + repair corruption (own capture, separate pools, double-buffered I/O), whole-GPU co-run, host-issue
record replacing `issue/wall`, nsys. M3 both stages together in ≥ 3 green splits (12/4, 8/8 + one) vs alone-in-
partition → `p_part`, `s_part`; **in-process** bandwidth-hog positive control (`p_hog`, ≥ 2 rates) so a null is
meaningful; graph-replay confinement control (M0). M4 EMC_FREQ calibrated against the hog, mem rail, nsys
(GPU-metrics tried); ncu serialises, isolation only. M5 whole-frame energy: stock full, fair full, best co-run.
Decisive number moved from `s` (r ≈ 2.25 with the hoisted expert) to **`O = T_pipe / max(A_iso, B_iso)`** and `p`.
Thresholds: O ≤ 1.05 near-ideal / ≥ 1.15 poor; p_part, s_part < 1.03 none / ≥ 1.10 material; p_hog ≥ 1.03 control
has power. Rows: near-ideal (frame-rate argument gone); SM-partitioned bandwidth coupling; partition cost only;
unpartitioned HoL fixed by partitioning; s > r; corruption unfixable (time box ≈ 1 device day); **fair B ≤ 93.1 ms
beats the 16-ch in-memory expert** (≤ 58.7 also 32 ch); 93.1–102.4 < 1.10×; hoist fails equivalence; co-run energy
≤ 0.95 × serial.

**Next:** on each pull look for `results/*_E12b_pi0_*`, `reports/E12b_pi0_compiled_corun.md`, T015 status lines,
and the restated Step-4 prediction in `STATUS.md`. Check against spec §12 item by item; hand `O`, `p_part`,
`s_part`, `p_hog`, fair `B` to main as "for microarch-owner / brainstorm-owner (case 001) / evaluation-owner" with
dir, device commit, power mode, n. Still unapproved: P1, P2 (cropcache), `check_model_inputs.py` MODELS.

### 2026-09-25: E12 / T014 MAXN results reviewed — `analysis/E12_review.md`, commit `c2e45db` (`049f569` before rebasing onto device commits; pushed with T015)

Pulled to device `b97da62`. Full review, with every number and its source, in `analysis/E12_review.md`.
Headlines (MAXN, π0 LeRobot, n = 100/condition):
- **Eager only** (`corun_N10/N4_MAXN_eager_nosync`): `T_pipe` = `A_iso + B_iso` (994.88 vs 997.83; N=4 666.77 vs
  667.88) — no overlap; s 0.996 / p 0.998 / r 0.905 (N=10). **No `bf16_both` co-run exists**: the stock
  CUDA-graph-tree path corrupts silently when pipelined; two full processes OOM at load; own-`CUDAGraph`
  remedy (spec §7) not attempted.
- Fresh-KV: eager pass (6/6 bitwise, stale control 5/5 detected); compiled check has power, pipeline fails.
  No `equivalence.*`, `mechanisms.csv`, nsys timeline or `raw/` committed.
- Step 3: green contexts **work on CUDA 12.6**, measured per stage in isolation only: bf16 expert 1.049× at 4 SM,
  prefix 1.08× at 12 SM → derived 12/4 period 271.26 ms = 1.080 × A_iso (not measured). Priority −3, graphs-both,
  libsmctrl, MAX_CONNECTIONS untested; MPS probed only (log uncommitted).
- **E7 reconciled**: stock per-step timestep tensor (pageable H2D drain) = 56.07 of 167.09 ms (**33.6 %**); hoisted
  111.02. r = 1.50 correct for shipping π0. Looks cropcache-class (OD-1 baseline side, undecided). If it enters
  the baseline: PIM latency gain 1.21–1.23× → 1.03–1.07× (16 ch) [derived] — the biggest consequence of this pull.
- Rows vs report: `issue/wall` cannot discriminate host- from GPU-bound (loop-level metric); issue order is
  expert-first; "cannot be blocked" unmeasured; rows labelled `baseline_eager`, hoist unlabelled.
- Pre-registered rows landed: `T_pipe ≥ T_serial` (marginal, eager) and the N=4 row (read as harness). Nothing
  falsified by a valid co-run; "no SM-partitioning mechanism" ruled out.

**Owed / next — superseded the same day by T015 (above):** the decisive co-run (`bf16_both` via own-`CUDAGraph`, green 12/4 and
8/8 concurrently, priority −3, prefix-first issue, one nsys) needs user approval; a compiled-path equivalence
check + whole-frame `full` for the hoist before any baseline use; note to the device that its Step-4 prediction
(eager expert × GPU-clock ratio) is confounded by the CPU slowdown in MODE_30W; missing deliverables list
(review §8). Step 4 waits for the capped-mode session (device's business — do not escalate).

### 2026-09-24, later: E12 / T014 issued and pushed (`88ad91a`) — the co-run experiment, pre-registered

User-approved push. **E12 = π0 `bf16_both` prefix(k) ∥ expert(k−1) on two CUDA streams on one Orin GPU**,
fresh KV per frame. Spec `docs/E12_corun/README.md` (purpose, definitions, 5 non-negotiables, 5 steps,
hazards, outputs, acceptance, **dated pre-registered outcome table §9**); task `tasks/T014_pi0_corun.md`;
order-table row 6 (right after T004) + note "finish the task in progress, then T014 before other open
MAXN work"; protocol **§14** (binding E12 rules) and **§1** (E12 Step 1 rides the `MODE_30W` session, no
extra reboot); T004 step 4b (iii); README / AGENTS read lists. Decisive number `s = B_co / B_iso` against
the in-session `r = A_iso / B_iso` (E7: 251.28 / 167.36 = 1.50 at N = 10; 3.72 at N = 4).

Judgement calls made in the spec (report to main; user may reverse):
1. `s < r` alone is not "ideal overlap": the period also needs `p ≈ 1`, so `s`, `p`, `T_pipe` are always
   reported together and §9 has a row for "`s < r` but the prefix slows".
2. Threshold is the **same-session** `r`, not E7's 1.50; in `MODE_30W` compare with the in-mode `r`.
3. Equivalence check strengthened beyond the brief: **distinct observations + noise per frame** (with one
   fixture a stale-KV bug is invisible), **negative stale control must fail**, and the check runs through the
   **compiled** timed path too (CUDA-graph static buffers / cross-stream allocator reuse are compiled-only hazards).
4. `MODE_30W` also changes GPU clock (612 MHz, 2.1×) and CPU (8 cores, 1.73 GHz) — the spec says what does
   not cancel; optional cleaner controls: `MODE_50W` (16 SM @ 816 MHz, no reboot) and an 8-SM partition at MAXN.
5. Energy compared as whole-frame loops (pipeline vs serial `full`), not from phase loops — because of the
   open E7 30 % residual.
6. Found while writing: **E11f's "high-priority" stream was priority 0 = torch's default** (lower number =
   higher priority; range (0, −3)); no high-priority stream has been measured. Stated in the spec's Step 3.
7. libsmctrl README lists CUDA ≤ 12.1 vs device 12.6 (from brainstorm case 001) — "try and record".

**Next on this item:** on each pull look for `results/*_E12_pi0_*`, `reports/E12_pi0_corun.md`, T014 status
lines. Check first: `equivalence.*` present with the negative control failing; `s` from unprofiled runs;
isolated A/B in-session vs E7 (±3 %); nsys decomposition of `B_co − B_iso`; `mechanisms.csv` with evidence.
Then hand `s`, `p`, `T_pipe` to main as "for microarch-owner / brainstorm-owner (case 001) / evaluation-owner"
with result dir, device commit, power mode, n.

### 2026-09-24, earlier: the 315-commit pull

**2026-09-24: 315-commit pull (`5f30ea5..cf72121`), read-only. The second measurement round has LANDED
and is much larger than asked for: six models (π0, π0.5, SmolVLA, GR00T N1.5/N1.6/N1.7), a 9-rung
one-factor optimisation ladder per model in which `vision` and `vlm_prefix` are compiled / CUDA-graphed /
cast to bf16 alongside `denoise_step`, a combination search on top of it, per-phase ncu bottleneck
captures (Part C), an nsys GPU-idle study (E5) and E7 power/energy complete at 14/14 points at MAXN.**
Device commit `cf72121`, pushed 2026-09-24 13:31 PDT — the device is live (20-minute `PROGRESS` ticks)
and currently on T007/T008/T009/T011/T012, all `claimed 2026-09-24`. Nothing is `blocked`.

### The gating answer for workspace commit `2e76296`: yes, vision and vlm_prefix are optimised

π0, LeRobot port, MAXN, bf16, N = 10 (π0's *configured* default — the earlier 34/34/31 % split was
N = 4, eager). `results/20260922_E4c_pi0_opt_ladder/opt_ladder.csv`, `reports/E4d_phase_balance.md`,
`reports/E4i_best_combinations.md`. n = 100 per point.

| π0, N = 10 | e2e ms | vision | vlm_prefix | action head | vis % | pre % | ae % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `eager_fp32` | 1069.04 | 241.01 | 234.57 | 586.91 | 22.5 | 21.9 | 54.9 |
| `ae_graphs` (expert only, as before) | 516.42 | 103.36 | 234.74 | 172.18 | 20.0 | 45.5 | 33.3 |
| `vlm_compile` (prefix compiled) | 466.99 | 99.18 | 192.76 | 168.84 | 21.2 | 41.3 | 36.2 |
| **`bf16_both` — BEST** | **424.46** | **56.74** | **192.86** | **168.68** | **13.4** | **45.4** | **39.7** |
| `whole_graphs` (no split possible) | 463.86 | — | — | — | — | — | — |

**New phase split, best configuration: 13.4 / 45.4 / 39.7 % (vision / vlm_prefix / action head).**
The VLM prefix, not vision, is now the largest phase. Against the earlier 34 / 34 / 31 % (N = 4 eager),
vision has collapsed and the prefix has taken its place. π0's own N = 4 control: `vlm_graphs` 369.51 ms,
AE share **18.7 %** (eager 32.3 %).

**Which lever moved which phase** (π0, eager → best): vision 241.0 → **56.7** (4.25×) — and it is the
**bf16 cast of the vision tower + dropping `embed_image`'s fp32 cast**, not compilation, that does it
(`vis_compile`/`vis_graphs` move vision by 1 %); vlm_prefix 234.6 → **192.8** (1.22×, `vlm_compile`);
action head 586.9 → **168.7** (3.48×). CUDA graphs on the prefix are a **net loss** on π0 and π0.5 and
a 2× win on SmolVLA — not a general win (Part C: compilation already removes 92 % of the prefix's
memory-latency stall, 34.44 → 2.73 cycles/issue, so the stall graphs would target is gone).

### For the concurrency design question (frame t vision+VLM on GPU ∥ frame t−1 expert on AiM)

The measured GPU-side fixed cost and the expert cost, same run, same power mode:

| π0 best (`bf16_both`), MAXN | latency | above-idle power | above-idle energy |
|---|---:|---:|---:|
| `prefix` loop = vision **+** VLM prefix | **251.28 ms** | 25.45 W | 6395.8 mJ |
| `ae` loop = action expert, N = 10 | **167.36 ms** | 18.60 W | 3112.5 mJ |
| `full` = whole inference | 424.68 ms | 32.18 W | 13 666.2 mJ |

`reports/E7_power_energy.md`. **The pipeline is GPU-dominated, 1.50 : 1.** A perfectly overlapped
two-engine pipeline is bound by the GPU at ≈ 251 ms/frame, so the whole PIM win over the best GPU
baseline is 424.7 → 251.3 ms = **1.69× at most**, and *any* expert implementation faster than 251 ms
is free — the AiM design has 251 ms of slack against the 9.31 ms / 5.87 ms design point, i.e. it is
not the critical path at all. At N = 4 it is far worse: prefix 249.92 ms against ae 67.18 ms, **3.72 : 1**.
E7 also says the prefix is the power-hungry phase: the expert is 40.0 % of the two-phase time but
32.7 % of its energy (ratio 0.82 optimised; **0.40** eager, 0.32 at N = 4).

### Action expert per step — the best figure has NOT moved, and the ladder does not contain the KV fix

Best remains **10.37 ms/step**, `reduce-overhead` + crop-not-clone KV (`DynamicCache.crop(P)` instead of
`copy.deepcopy`), from T003 — 3.3× the 3.115 ms analytic floor, 1.67× the 6.230 ms "optimistic bound".
Ladder for the record: 52.56–52.94 eager → 16.30 `reduce-overhead` → 10.37 with the KV fix.

**Gap to raise with the device (not yet a task):** the E4c ladder and the E4i combination search never
include `cropcache`. Slopes derived from `opt_ladder.csv` (ae_total at N = 10 minus N = 4, ÷ 6):
`ae_graphs` 16.80, `vlm_compile` 16.52 ms/step — i.e. the "best" 424.46 ms frame still carries the
`copy.deepcopy` of the KV cache. Combining the two is unmeasured and worth ≈ 6 ms/step × 10 ≈ 60 ms,
which would put π0 near 365 ms and lower the AE share further (to ≈ 28 %).

### What contradicts what the research side currently believes

1. **"The action expert dominates the frame."** Across six models and ~20 configurations **no optimised
   deployment has an expert that dominates**; the share peaks at or near the *eager* baseline in all six
   (π0 54.9 → 39.7, SmolVLA 85.3 → 48.6, N1.7 56.7 → 33.1, N1.6 61.5 → 41.7, N1.5 38.6 → 35.9; π0.5 is
   the exception only because its fp32 prefix was pathological at 2162 ms). Highest optimised share is
   SmolVLA 48.6 %.
2. **"The action expert is the memory-bound phase."** Part C (`reports/E4g_c1_c2_bottlenecks.md`):
   **false eager, true compiled.** Eager `ae_step_0` is issue-limited — SM 22.7 %, Mem 21.6 %,
   `imc_miss` 35 %, `long_scoreboard` only third at 16 %. Compiled: Mem **56.5 %** and
   `long_scoreboard` triples to dominant (11.44 cycles/issue). The phase that is memory-latency-bound
   *eager* is the **VLM prefix** (`long_scoreboard` 74 % at 40 % SM). Our eager expert SM 22.7 % matches
   arXiv:2604.24447's 22.3 % — independent check.
3. **Paper Finding #4 ("the VLM is compute-bound from operational intensity, 840 FLOP/B")** is
   contradicted by the counters: the eager prefix stalls 74 % on Long Scoreboard at 40 % SM and 49 %
   memory throughput. High arithmetic intensity does not help when latency-bound at low occupancy.
4. **Whole-frame fixed cost.** Was 485 ms, then 428 under whole-model compile. Now **249.6 ms**
   (56.74 + 192.86) — a 1.90× reduction on the eager 475.6 ms, and it is the number the pipeline is
   bound by.
5. **"GPU idle 84 % of the frame"** — that is the `ae_step` *region*, not the frame, and E11's
   nsys-inflation correction (×1.53 on region wall time) puts it at **75.6 %**. E5 reports 84.3 %
   uncorrected. Whole capture is 54.7 % (N = 10) / 36.5 % (N = 4). Quote 75.6 % for `ae_step`.
6. **N is not the driver of the AE share.** The plan predicted the share tracks N; GR00T N1.6/N1.7 at
   N = 4 sit *above* π0 at N = 10, because they carry **32** DiT blocks against N1.5's 16. Architecture
   beats step count.
7. **Latency is not a proxy for energy.** Frame energy-saving ÷ time-saving ranges 1.05 (N1.5, removes
   work) to 0.45 (SmolVLA, buys speed with power). A speedup predicts the energy saving only to a
   factor of two.
8. **π0.5's 6.21× carries no accuracy statement** — `pi05_libero_base` is a base model, not a LIBERO
   finetune, fp32 baseline 0/30 on LIBERO. π0's bf16 rungs *are* validated (fp32 6/30 vs bf16 4/30,
   z = 0.70).

### ⛔ Two energy residuals — do NOT quote GPU-vs-PIM energy until these close

Found by microarch-owner in the E7 numbers I handed over (2026-09-24). Both are in
`reports/E7_power_energy.md`, π0 N = 10 `bf16_both`:

1. **6.04 ms unplaced.** The stage loops sum to **418.64 ms** against the `full` loop's
   **424.68 ms** measured. The report says the two sub-loops do not cover the whole inference
   (pre/postprocess, the action-sampling tail), but 6.04 ms is not attributed to anything.
2. **A 30 % energy gap.** The stage above-idle energies sum to **9509 mJ** against `full`'s
   **13 666 mJ** — 4157 mJ, 30 %, unaccounted. A 1.4 % time gap cannot produce a 30 % energy gap,
   so this is not the same discrepancy twice: it is either a real cost outside both loops or a
   difference in how the loops' idle baseline is subtracted.

**Consequence: GPU-versus-PIM energy is not comparable until this is resolved.** Latency figures
are unaffected. Reconciling it needs a per-stage accounting question put to the device (the
`prefix`/`ae`/`full` loop definitions and their idle-baseline subtraction), not a re-measurement —
write it as a question in `docs/04`, not a new task, unless the loops themselves must change.

### Queued measurement requests — NOT approved, do not write as tasks

From microarch-owner's design memo, priority order (2026-09-24). Recorded here so they survive;
the user has not approved either, so no task file exists and none is to be written.

- **P1, highest value: `vlm_prefix` with the GPU restricted to 8 of 16 LPDDR channels.** A PIM
  channel is *unavailable* to the GPU, not merely slower, so this decides whether a two-stage
  GPU-prefix / PIM-expert pipeline is implementable at all. If the prefix degrades materially the
  whole design collapses. Note the device already has the relevant lever characterised in one
  direction — E0 recorded MODE_30W as 8 SMs / 612 MHz, which changes SMs *and* clock, so it is not
  a clean channel-restriction control; this needs a real channel/bank-mask mechanism or a
  documented statement that none is exposed on Orin.
- **P2: `bf16_both` + `cropcache`** — the one combination never measured, worth ≈ 60 ms. It moves
  the design's end-to-end ceiling from ≈ 1.65× to ≈ 1.39×, so it changes the headline. Slopes in
  `results/20260922_E4c_pi0_opt_ladder/opt_ladder.csv` give the compiled expert at 16.5–16.8 ms/step
  against 10.37 with the KV fix.

Also held, not approved: adding `groot_n16` / `groot_n15` to `MODELS` in
`docs/reference_scripts/check_model_inputs.py` (the device's own request). Until it lands the device
records `meta.json.schema_note` and does **not** relabel rows — that is the correct behaviour.

### Capped power modes and device state

E7 is **complete, 14/14, MAXN only**. `MODE_50W` (id 3) the device can do unattended at the end of a
MAXN session; **`MODE_30W` (id 2) and `MODE_15W` (id 1) are `PENDING — ask the owner`, one human reboot
each.** The device is **not** waiting on them now: its stated order is E7 MAXN (done) → E11 (T011/T012)
→ E5 → E5b → MODE_50W → MODE_30W/15W, so the reboots come after E5b.

**User's decision 2026-09-24: the reboots are the on-device agent's business. Do not escalate
MODE_30W / MODE_15W to main or to the user again.** This standing item lives here, in my own
HANDOFF, and I simply wait for the device's queue to reach it. Carry the row forward on every pull
and note the date it clears; raise it only if the device marks it `blocked` or asks directly.

### Open items the device has raised (answers owed, no task written this round)

- **`docs/reference_scripts/check_model_inputs.py`**: may `groot_n16` and `groot_n15` be added to
  `MODELS` (currently `pi0, pi05, smolvla, groot_n17, none`)? Their E11a rows are otherwise identical.
  The device will not relabel rows to pass the checker; it records `meta.json.schema_note` instead.
  **This is ours to fix** — a one-line `docs/` edit, deliberately not made in this pull-only round.
- π0's committed E11f two-stream rows carry `bench_commit` `de325f77`/`5eeddd5c`, both **before** the
  GIL fix `773d2a9`, so π0's two-stream numbers on disk are still GIL-confounded. Re-run is queued.
- Four `.ncu-rep` (725 MB, 89–203 MB each) cannot be committed; published to the **private** HF dataset
  `yang-gon/vla-orin-ncu-raw`, hashes in `results/20260922_E4g_pi0_ncu_eager/RAW_FILES.md`.
- GR00T N1.7 `vis_compile` **failed** and cannot be fixed: Qwen3-VL passes
  `int(grid_thw[:,1:].max().item())` to `flash_attn_varlen_func`, untraceable by dynamo. Its 240.49 ms
  is a structurally weaker optimum, not a tuning failure.
- Repository: `du -sh results/` = **920 MB**; size-pack 62.49 KiB. Passwordless sudo still not set up.

### Superseded below

The 2026-09-22 status block that follows is kept for the T001–T005/T010 record. Where it and the block
above disagree, the block above is newer.

## Current status (2026-09-22, superseded)

**2026-09-22: T002 (E3), T003 (E4), T005 (E1/E2) received, verified, accepted; T010 (E11) complete on content
(status line still "claimed"); T004 claimed by the device.** 34 device commits `c1c7fe7..e6a0af7`. Checked here:
55 `meta.json` (all MAXN, `bench_commit`, 20/100 where §2 applies; E11d ran 5/30 — deviation to be stated),
`raw_files` sha256 recomputed for E11*, E4-cropcache, E0 and a sample of the rest (one stale hash:
`…cropcache_N10/equivalence.md`), `check_model_inputs.py` exit 0 on 9 files, `results/` 39 MB, largest file 6.6 MB,
size-pack < 1 MB (device reports 62 KiB), nothing > 50 MB. `analysis/ladder.py` reproduces every ladder number from
the committed CSVs. Write partition: the device edited `docs/00_context.md` and `docs/01_measurement_plan.md`
(factual corrections, disclosed, "owner's standing instruction") — accepted and now codified as a narrow exception
in AGENTS.md; user to confirm.

**SECURITY (user action):** nsys binary reports embed the system-wide process environment; the four
`results/20260921_E0_platform{,_maxn}/raw/nsys_toy.{nsys-rep,sqlite}` pushed in T001 (commit `410a535`) contain a
session token of the device agent's harness. The device untracked them (`5ec4788`), added `.gitignore` rules and
says the owner is rotating the token. History was not rewritten (our rule). Repo must be private; user decides on
rotation confirmation / rewrite.

**Key research facts (π0 LeRobot port `lerobot @ f0d2b37beb`, ckpt `lerobot/pi0_libero_finetuned_v044`, MAXN
16 SMs / 1300.5 MHz / EMC 3199, bf16, batch 1, n = 100 per point, device commits per `meta.json.bench_commit`):**
- Ladder, ms/step (slope over N): floor 3.115 spec / 3.403 measured-peak; **eager 52.56–52.94 (16.9× / 15.4×)**;
  compiled `default` 22.61, `max-autotune` 16.71, **`reduce-overhead` 16.30 (5.2× / 4.8×)**, `step_only` 16.34,
  `is_big_gpu` defeated 16.23 (Triton won 0/126 expert shapes), **crop-not-clone KV 10.37 (3.3× / 3.05×, bitwise
  identical actions in eager)**. Research assumed 136 / ≈ 90: 2.6× / 5.5× too pessimistic. Compile gain per step
  3.22× (2.32× compile, 1.39× CUDA graphs, autotune 0); e2e 1.39× at N = 4, 1.69× at N = 10.
  Files: `results/20260921_E4_pi0_summary.csv`, `results/20260921_E3_pi0_eager_N*/`, `reports/E3_pi0_eager.md`, `reports/E4_pi0_compile.md`.
- Split eager N = 4: vision 239.4 / vlm_prefix 233.1 / ae_total 215.5 / e2e 694.0 ms → AE **31.1 %** (paper 59 %);
  N = 10: 239.6 / 232.9 / 533.0 / 1011.6 → 52.7 %. Compiled (`step_only`, instrumented): AE 12.5 % (N = 4), 26.1 % (N = 10);
  e2e 551 / 652 ms. Vision includes an 80 ms dummy-camera SigLIP pass (`empty_cameras: 1`). P = 816 (533 valid), S = 51,
  623.01 MB expert weights, 38.29 GFLOP/step, peak mem 9.04 GB. Pre/postprocess 2.0 ms total.
- T005: peaks **187.47 GB/s (0.915)**, **33.85 TFLOP/s bf16 (0.806)**, ridge 180.6; bandwidth varies 1.19× between
  processes (unexplained). 126 expert projections = 622.85 MB run at **1.33× floor as one CUDA graph** (MLP ones at
  87–90 % of peak), 2.61× from Python; **969 kernels/step, 17 % GEMM**; dispatch ~62 µs per `F.linear`, ~29 µs per
  single-kernel op; launch-bound crossover S ≈ 128 (MLP), not 16. Write-traffic counter corrected (`op_write.sum`).
  Rails independent, summable. `results/20260922_E1_peak/hw_constants.json`, `results/20260922_E2_ae_kernels/`.
- T010/E11 (`results/20260922_E11_pi0_inputs/model_inputs.csv`, 1,589 rows): host layer 1.502 ms eager / 0.177 as one
  graph / 0.085 compiled sequence (assumed 0.16); ~35 launches per layer; transfers 40 µs + 36 GB/s pinned (assumed
  0 + 143); sync 10.6 µs idle, 109 µs stream handoff; KV 15.04 MB deep-copied every step (5.53 ms); prefix constant
  816 tokens for any input (no P slope); **GPU idle 84 % of `ae_total`** (vision/prefix 1–2 %); prefix slowdown under
  a bandwidth hog 2.22× of which 1.75× is time-slicing (bandwidth part ≈ 1.27×); two-process sharing 1.001×;
  CPU 99.8 % of one core during the expert. `reports/E11_pi0_hybrid_inputs.md`.

**My changes, pushed 2026-09-22:** protocol §1 (capped modes required, per owner), §3.1/§3.5/§6e/§7 (nsys binaries
not committed; CSV exports are the raw record; secrets-scan output rule), §3.3 (two-counter off-chip bytes, ncu
replay caveat), §9 (measured-peak floor, same-session peak), §12 (six pitfalls), §13.7 (two-stream rule);
AGENTS.md (nsys exception; docs/ factual-correction exception); T004 rewritten (LeRobot, E11f captures, two
counters, temperature logs, compiled per-region from nsys, step 4b capped modes); T006/T007/T011 route lines;
`docs/04` 2026-09-22 answers (6 items + 6 small fixes); `docs/00_context.md` E11/T005 table; `analysis/ladder.py`.

### Task queue
| task | content | priority | depends |
|---|---|---|---|
| T001 | bring-up, E0 — **done, accepted** (`results/20260921_E0_platform_maxn/`) | P0 | — |
| T013 | withdrawn (E0 at MAXN was done inside T001); number reserved | — | — |
| T002 | π0 eager (E3) — **done, accepted** (`results/20260921_E3_pi0_eager_N*/`) | P0 | T001 |
| T003 | π0 compiled (E4) — **done, accepted** (`results/2026092{1,2}_E4_pi0_*/`) | P0 | T002 |
| T010 | π0 E11 → `model_inputs.csv` — **complete on content, accepted; `status: done` still to be appended** | P1 | T003 |
| **T004** | **claimed 2026-09-22**; π0 E5 (done, `reports/E5_timeline.md`), E6, E7 (**complete 14/14 at MAXN**, `reports/E7_power_energy.md`); capped modes 30 W / 15 W still pending two human reboots | P1 | T003 |
| **T014** | **E12 π0 co-run (prefix ∥ expert, one GPU) — issued 2026-09-24 (`88ad91a`), order row 6; Step 4 rides MODE_30W** | P1 | T004 (E7 MAXN) |
| T005 | E1/E2 — **done, accepted** (`results/20260922_E1_peak/`, `_E2_ae_kernels/`) | P1 | T001 |
| **T007** | **claimed 2026-09-24**; π0 E5b instruction level. Targets already chosen from C1 data by `bench/pick_e5b_targets.py` — 22 captures, 1–8 kernels per model, no exploratory pass needed | P1 | T004, T005 |
| T006 | π0.5 + SmolVLA E3/E4, LIBERO E8 — **effectively delivered** by the 2026-09-22/24 six-model campaign (E4c/E4d/E4f/E4g/E4i ladders + E8q LIBERO); no `status:` line appended, so verify before closing | P2 | T003 |
| **T011** | **claimed 2026-09-24**; π0.5 + SmolVLA E11 | P2 | T006, T010 |
| **T008** | **claimed 2026-09-24**; π0.5 + SmolVLA E5/E5b/E6/E7 (E7 half already done at MAXN) | P2 | T006, T004, T007 |
| **T009** | **claimed 2026-09-24**; GR00T N1.7 full set — in practice N1.5/N1.6/N1.7 all measured for E4/E7 | P3 | T008 |
| **T012** | **claimed 2026-09-24**; GR00T N1.7 E11; driver written (`bench/groot_segments.py`), DiT blocks confirmed patchable | P3 | T009, T010 |

Execution order is the table at the top of `docs/01_measurement_plan.md` (rows above are already in that order).

### Key numbers
See "Key research facts" above; `python analysis/ladder.py` regenerates the ladder and split tables. Handed to main
2026-09-22 as "for microarch-owner / evaluation-owner / paper-owner" items (result paths, device commit `e6a0af7`,
MAXN, n = 100). Idle module power at MAXN 11,012 mW (E0). E7 power numbers not yet measured.

## Recent decisions
- **2026-09-22, after the 34-commit pull — my judgement calls, user to confirm:** (1) the device's disclosed
  `docs/` corrections are accepted and the exception codified in AGENTS.md (factual corrections to
  `docs/00`/`01` only, marked and disclosed); (2) capped power modes are now *required* for π0 E7 (the owner told the
  device directly; protocol §1 + T004 4b restate it); (3) nsys binaries are permanently uncommitted (token leak);
  full CSV exports are the raw record; (4) floors are headlined against the measured peak with the spread stated;
  (5) T006/T011 default to LeRobot ports for π0.5 and SmolVLA, openpi only as fallback; (6) no history rewrite
  for the leaked token — user's call; (7) T010's two-stream concurrency rows are not used (GIL-confounded).
- **2026-09-21, after T001 — my judgement calls, user to confirm:** (1) E7 capped modes reduced: MAXN required,
  MODE_50W only if it switches without a reboot (last item of last MAXN session), 30 W / 15 W optional and
  human-scheduled (was: full sweep required for π0); (2) sysmem-miss substitute accepted with conditions (analytic
  bytes stay primary; read-only until T005 validates writes; nsys time, never ncu Duration); (3) memory rail is
  reported as "regulator input", absolute + above idle, no rescaling — interpretation tagged `[ASSUMPTION]` (not from
  a schematic); four- and three-rail sums both reported until T005's double-count check; (4) T003 gets one labelled
  variant that overrides Inductor's SM-count check; the ladder uses unmodified modes only; (5) ncu clock rule: keep
  `base`, lock clocks, discard captures where `SM Frequency` and `cur_freq` differ > 2 %; (6) answers live in a new
  `docs/04_research_answers.md`, binding parts in the protocol; (7) a task overtaken by events is withdrawn in
  place, never deleted or renumbered.
- **E11 design (2026-09-21), my judgement calls — user to confirm:** (1) schema has 18 columns: the brief's 15 plus
  `bench_commit`, `variant` (`name=value;…`) and `timing` (`wall_isolated | gpu_event | wall_launch | wall_loop | slope |
  analytic`); the **primary** host-segment number is `wall_isolated` (sync → launch → sync from an idle GPU, with a
  0 / 2 ms pause), because in the target system every segment is SYNC-bounded; (2) `cudagraph` is measured both per
  segment (realistic) and as one whole-layer replay (lower bound = the simulator's `loop` feature, 1 launch/layer);
  (3) segment `key` names are exactly those of `microarch/03_tools/trace_gen.py` manifests (`rmsnorm1, rope, softmax,
  residual_rmsnorm2, gelu_gate_mul_up, partials_residual, residual, attention_gpu`) so `host_model.py` can join on them;
  (4) memory-pressure levels are hog **duty cycles** (25–100 %) with achieved GB/s recorded, plus two controls
  (`gpu_spin_control`, `cpu_spin_control`) — removes the dependency on E1/T005; (5) added beyond the brief: host
  segments concurrent with the next frame's prefix (two processes / two streams) → `host_layer_slowdown`, because a
  100 µs segment can queue behind ms-long prefix kernels; spin-wait check so `cpu_util` is not misread;
  `kv_copied_bytes_per_step` (does `torch.cat` re-copy the prefix KV each step?); `ae_step` vs P fit; (6) GR00T E11 is
  its own task T012, not folded into T009 (needs T010's scripts; T009 is already large); (7) AGENTS.md task selection
  changed from "lowest number" to the plan's order table so T010 runs after T003 without renumbering; (8) SmolVLA /
  GR00T segment lists are **not** established here (op_trace.py is π0-only): T011/T012 tell the device to derive and
  record them; "4 partials" for those models is tagged `[ASSUMPTION]`; (9) restated numbers (E3/E4 latencies in the
  consolidated file) use a copy rule — tidy CSV copied to `raw/restated/<run>/` — so every `source_file` exists inside
  the result directory.
- Source of the ncu / SASS material: the user's skill `/home/yanggon/99_personal_project/02_SKILL/skills/workload-analysis`
  (server only). Only flags, metric and column names it documents are used; everything unverified on Orin is
  "discover with `--help` / exported header and record" (`env/tool_help/`, `ncu/columns.txt`).
- The skill documents two `cuobjdump` spellings (`-sass -fun` / `-symbols` vs `--dump-sass -arch=sm_XX --function` /
  `--list-text`). Both are given in docs/03 §2, the agent uses whichever its version accepts. Reported to main as
  a judgement call (not treated as a stop-condition contradiction).
- **Data policy (user decision 2026-09-21, via main): commit as much raw data as possible.** `.nsys-rep`, `.sqlite`
  (if < 50 MB compressed), `.ncu-rep`, full SASS/PTX dumps, analysed `.cubin`, tegrastats logs, torch.profiler traces,
  command stdout, compile logs, per-iteration samples and full ncu CSV exports go to the tracked
  `results/<run>/raw/`; `meta.json.raw_files` = path, sha256, bytes (+ compression, original hash, split parts).
  Summaries are in addition to raw, never instead. Never committed: weights/checkpoints, datasets, caches, venvs,
  vendor `.so`, secrets. Size rules are binding in protocol §6 (small captures: nsys 1–3 inferences; `zstd -19`;
  50 MB per file; shorten first, `split -b 45M` + `REASSEMBLE.txt` last resort; no LFS; `.sqlite` rule; size report
  in `STATUS.md`; stop near 2 GB). Inductor/Triton caches and uncompressed originals live in the ignored `scratch/`.
  Added on my own judgement: a secrets scan of raw files (protocol §3.5) because profiler reports can record the
  process environment / command line. Commits `5752f2e`, `62a162f`, `c2d3ed0` (pushed). This supersedes the
  conservative policy of `d42ae1a`.
- Review depth: self-review checklist for every task; independent sub-agent pass only for P0 tasks (T001–T003).
- Region utilisation has two derived definitions: kernel-weighted and region-weighted (idle = 0, region time from
  nsys); the region-weighted one is what is compared with the paper's 22 %.
- Latency-budget cycle costs are from Ada/Blackwell; tagged `[ASSUMPTION]` for Orin.

## Open issues
- (high, user) **Leaked session token in git history** (commit `410a535`, four nsys files, now untracked). Repo
  must be private; rotation to be confirmed; no rewrite by agents.
- (medium) **Bandwidth peak varies 1.19× between processes** with clocks/temperature constant (T005) — floors carry
  ±8 %; rule: same-session peak. Unexplained.
- (medium) **π0 is the LeRobot port, different fine-tune from openpi's** — paper comparison is across
  implementations and checkpoints; the phase-split disagreement (vision 239 vs 110 ms) may be image count (3 SigLIP
  passes incl. a dummy) or implementation. openpi cross-check remains an optional later task.
- (low) T010 `status: done` missing; `…cropcache_N10/equivalence.md` stale hash; E4 report §9 independent pass
  empty; E11d 5/30 iterations; E5 partial dirs without meta (T004 WIP) — all listed in `docs/04` for the device.
- (medium) **E11 unknowns on the device:** no documented PyTorch route to a zero-copy / managed allocation (protocol
  §13.3 lists candidates only); whether `torch.cuda.Stream(priority=…)` has any effect on Tegra; whether the openpi
  input path can drop an image slot or always encodes three and pads the prompt (then processed P is constant and the
  P fit degenerates — the task says to report that as a result). The memory-pressure experiment is a proxy; the docs
  say so and ask for the limits in every report.
- (low) The device may already be past "pick a task" when it next pulls; the order-table rule only matters after T003.
- (high) **Repository size.** Raw files are committed; growth is unbounded in principle. Guards: 50 MB per file, size
  report in `STATUS.md` after each task, stop-and-ask near 2 GB `size-pack`. Watch T001's reported toy `.nsys-rep` /
  `.ncu-rep` sizes and tighten §6 (e.g. `-c`, fewer inferences) before T004/T007 if they extrapolate badly. History
  cannot be shrunk later without a rewrite, which is forbidden — an over-size commit is permanent.
- (medium) Raw files may leak: secrets (mitigated by the scan rule), hostnames / user paths inside `.nsys-rep`, and
  SASS disassembly of NVIDIA's proprietary cuBLAS kernels. User should confirm the GitHub repo is private.
- (high) Which agent framework and credentials the device will use is unknown; the device needs push access.
- (resolved by T001) ncu on Orin exposes occupancy / pipes / stalls / LSU; no DRAM counters.
- (medium) **Passwordless sudo is still not configured on the device**; every session needs the password again (ncu, jetson_clocks, perf_event_paranoid). User action.
- (medium) ncu runs the profiled Python as root: model caches / HOME under sudo may bite in T004 (protocol §3.2 gives a pattern, unverified).
- (medium) ncu report ≈ 7 MB per kernel: an `ae_step[0]/` capture may be 1–2 GB uncompressed (20–30 MB zst) — watch T004.
- (medium) Whether cuBLAS on JetPack ships SASS for `sm_87` and whether ncu sees CUDA-graph replays: unknown; docs tell the agent to record what it finds.
- (medium) openpi π0 checkpoint may need JAX → PyTorch conversion on the device (T002 step 1).
- (resolved by T001) `torch.compile` / Triton work; mode ids, wheel index, nsys report names confirmed.
- (medium) openpi may want a torch other than 2.8.0 / pip may swap in the CPU wheel (T002 header warns).

## Next steps
0. **2026-09-24 pull done** (`5f30ea5..cf72121`, 315 commits). Owed to the device, in order: (a) add
   `groot_n16` / `groot_n15` to `MODELS` in `docs/reference_scripts/check_model_inputs.py`; (b) answer its
   three Part-C questions in `docs/04` (does the conditional memory-bound result change the PIM target;
   re-examine paper Finding #4; is fp32 the intended π0.5 deployment); (c) a task asking for
   `bf16_both` **+ cropcache** on π0, the one combination never measured; (d) schedule the two reboots
   for MODE_30W / MODE_15W. None written in the 2026-09-24 pull-only round.
0b. (superseded) **Next pull:** expect T010 `status: done` + the six small fixes, then T004 results (E5 from the E11f exports,
   ncu per region, E6, E7 at MAXN, then "next: resume E7 at mode 2" and a stop for the human reboot). Check the
   E5 idle split for the compiled mode and that `parse_ncu.py` uses the two counters. Then T007 (E5b) / T006.
0a. On each pull: check `STATUS.md` "Repository size", `find results -size +50M`, and that `raw_files` hashes match
   (`sha256sum`) for a sample of files.
1. (done 2026-09-21) T001; (done 2026-09-22) T002, T003, T005, T010 received and analysed; ladder script in
   `analysis/ladder.py`; "for microarch-owner:" items handed to main 2026-09-22.
2b. (done 2026-09-22) T010 checker run on every `model_inputs.csv`; consolidated file handed to main. Still planned:
   `analysis/model_inputs_to_hostparams.py` (join on `microarch/03_tools/trace_gen.py` segment keys) if
   microarch-owner wants a derived parameter file rather than the CSV.
3. When T004/T007 arrive: `analysis/` scripts for region utilisation and the stall/LSU tables; hand to evaluation.

## Interfaces with other sub-projects
- → microarch-owner: measured latency/power/kernel data for `host_model.py`, `energy_model.py`, roofline points.
  Interface for the hybrid model: `results/<date>_E11_<model>_inputs/model_inputs.csv` (protocol §13.8; mapping table
  in `docs/00_context.md`). `transfer_time` should become `fit_a + fit_b·bytes`; SYNC points get `t_sync`.
- → paper-owner: the measured baseline ladder, SM-occupancy and instruction-level evidence, LIBERO results.
- Results flow through evaluation-owner (collected with provenance) before paper / site.
