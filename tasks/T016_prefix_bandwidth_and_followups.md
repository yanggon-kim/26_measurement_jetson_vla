# T016 — π0 prefix under stolen memory bandwidth (EV-02), seeded LIBERO N sweep, prefix software savings, and three E7 / E12b loose ends  · priority P1 · depends on: T015 (done) · next MAXN work (order table in `docs/01_measurement_plan.md`)

This file is the whole specification (no separate `docs/` spec). Binding rules: `docs/02_protocol.md` §16, plus
§15 / §14 where a step reuses the E12b co-run machinery. Everything not fixed below is your judgement — choose,
record the choice in `meta.json` and the report, and move on.

> **Amended 2026-09-26 (Amendment 1, before any T016 measurement)** — read the section "Amendment 1" just above
> Step 1 first. Where it and the original steps differ, the amendment wins. The pre-registered table (rows 1–7)
> is unchanged; the amendment adds row 8 for its new step.

## Why (the part you need of the research design)

The research design runs π0's action expert in **processing-in-memory (PIM) inside the LPDDR5 devices**. It is
pipelined: the PIM runs the expert of frame t−1 **while the GPU runs the prefix (vision + VLM prefill) of frame
t**. While the PIM computes, the LPDDR channels it uses are unavailable to the GPU — in the design, 8 of the 16
channels, i.e. half the bus (~102 GB/s spec, ~94 GB/s at achievable efficiency), for ~37 % of each frame (the
expert's ~93 of ~250 ms). The pipeline's period *is* the prefix time, so **how much the prefix slows when it
loses bandwidth decides the design**:

- Your E12b DRAM capture (`results/20260925_E12b_pi0_dram_bytes_MAXN/`, bench `9df5628`): the compiled prefix
  moves **21.64 GB per frame in 246.79 ms = 87.85 GB/s** — ~94 % of what 8 channels supply.
- The research claims break at two prefix slowdowns: **1.04×** (latency parity with the GPU at 16 channels) and
  **1.247×** (frame-rate gain over the best GPU co-run you measured, green 12 / 4).

A software load that consumes bandwidth is only an approximation of a PIM holding channels (it competes at the
memory controller instead of taking channels away, and it may take SMs); step 1c asks whether the real thing —
confining the GPU to a channel subset — is possible at all. Steps 2–5 close questions your own reports left open.

## Fixed configuration (all steps unless a step says otherwise)

LeRobot π0 `lerobot @ f0d2b37bebddf7e8852cb624712da0ad564601b6`, checkpoint `lerobot/pi0_libero_finetuned_v044` @
`45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f`, rung **`bf16_both`** exactly as in
`results/20260924_E7_pi0_N10_bf16_both_MAXN/meta.json`, batch 1, **MAXN + `jetson_clocks`**. N = 10 except in
step 2. Harness changes (own CUDA-graph capture such as E12b's `graphs_both`, green contexts, loads) are labelled
variants named in every row and `meta.json`.

## Amendment 1 (2026-09-26, research side, before any T016 measurement)

Four refinements from the research side's appraisal of the channel partition (brainstorm case 003). The original
text below is kept as issued; this section overrides it where they differ. Step order becomes **1a → 1b → 1d →
1c**, then steps 2–5 as before; push step 1's results (1a, 1b, 1d, 1c) before starting step 2.

**A1 — Step 1a: the primary load takes no SMs.** The question is the prefix's slowdown when it keeps **all 16 SMs**
and loses **only memory bandwidth**, in the regime where it is left roughly **90–130 GB/s** of the bus.
- **Primary arm:** the unmodified rung prefix (`bf16_both`, as in "Fixed configuration", not `graphs_both`) on the
  whole GPU, beside a load that uses **no SMs**: a copy-engine load (`cudaMemcpyAsync` between **pinned host** and
  device buffers — on Orin both are the same LPDDR, so it reads and writes DRAM through the copy engine; a
  device-to-device `cudaMemcpy` may run as an SM kernel, so do not rely on it unless you show it launches none), or
  a CPU-side load (multi-threaded `memcpy` / STREAM-style copy on CPU cores pinned away from the harness thread), or
  both together if one alone cannot reach the rates. Show the load launches no GPU kernel (a kernel list or trace
  CSV under `raw/`). Say which kind you used and why; name every other resource it shares with the prefix that you
  know of (e.g. host threads, copy-engine queues, the GPU's memory hub).
- **Rates.** "Left" = the bus a streaming load sustains alone in the same session (EMC-calibrated as in E12b M4;
  178.5 GB/s last time) minus the load's rate **during** the prefix; report both. Keep the original points 0, 20, 40,
  60, 80, 94 GB/s stolen, and put **at least three points with the prefix left 90–130 GB/s** (stolen ≈ 50–90 GB/s
  against 178.5). If no SM-free load reaches a rate, say the highest it reached and stop the primary arm there.
- **Reference:** the same prefix with no load (the SM layout is then identical by construction), same session,
  bracketed (reference → loaded → reference), n ≥ 20 frames per point.
- **Secondary arm (optional):** the green-context copy kernel of the original 1a, if you judge it adds information
  (e.g. to reach rates the SM-free load cannot). It keeps every original rule, including the `graphs_both` vs rung
  equivalence check first. It never replaces the primary arm.
- Step 1b runs with the primary (SM-free) load.

**A2 — New step 1d (EV-02c): per-kernel DRAM traffic of the compiled prefix.** E12b measured the prefix at 17.80 GB
of reads + ≤ 3.83 GB of writes per frame against E6's 5.85 GB analytic. Writes bound spills: everything the prefix
writes is ≤ 3.83 GB, so **≳ 8 GB of the ~11.95 GB excess reads are re-reads**. This step says which: **H1 — weight
re-reads** (a GEMM's weight tiles fetched again for each token (M) tile or wave) or **H2 — activation re-reads**
(an activation such as the MLP's 816 × 16 384 intermediate fetched again for each output (N) tile).
- **Capture:** the compiled prefix, per kernel, with ncu **metrics only** — the E12b set
  (`lts__d_sectors_fill_device.sum, lts__d_sectors_fill_sysmem.sum, lts__t_sectors_op_read.sum,
  lts__t_sectors_op_write.sum, lts__t_sectors_op_read_lookup_miss.sum, lts__t_sectors_lookup_miss.sum,
  gpu__time_duration.sum`), no `--set` /
  `--section` (those request `dram__` counters Orin lacks — the first per-node attempt died on that). Per kernel:
  `--graph-profiling node` on the rung, or the prefix compiled without CUDA graphs if node profiling fails (then
  record that the kernel choice may differ and compare the totals). `--cache-control none` as in E12b; ≥ 2 frames.
- **Output:** `results/<date>_E13d_pi0_prefix_kernel_bytes_MAXN/kernel_bytes.csv`, one row per kernel launch: launch
  order, kernel name, grid, block, fills (B), writes (B), read requests (B), duration, the op it belongs to (layer,
  projection / attention / norm / elementwise) and, for GEMMs, M, N, K and the unique operand bytes (weight, activation,
  output). Plus the per-frame sum against E12b's 17.80 / 3.83 GB (a gap > 10 % is reported with its cause, not
  reconciled silently).
- **The split:** attribute the excess reads (fills minus unique operand bytes, per kernel, summed) to H1, H2 or
  neither. The method is yours; one that works: for the few GEMM shapes that carry most of the excess, a standalone
  run of the same kernel (same Inductor / cuBLAS choice) sweeping M at fixed N, K and N at fixed M, K — weight
  re-reads grow with the M-tile count, activation re-reads with the N-tile count. Record each such kernel's tile
  shape (from its name or launch config) so the research side can model it.
- Report it as a section "1d — EV-02c" of `reports/E13_pi0_bandwidth_steal.md`, answering row 8 by number.

**A3 — Step 1c: look at FACIL first.** FACIL (Seo et al., HPCA 2025, "Flexible DRAM Address Mapping for SoC-PIM
Cooperative On-device LLM Inference") evaluated on an AGX Orin 64 GB and selects a DRAM address map per
allocation: a MapID in spare PTE bits of 2 MB huge pages. The research side's reading of the paper is that the map
switch is a **proposed** memory-controller change (multiplexers in the address-translation frontend), not a
feature of stock Orin — confirm or correct that from the paper and any released code. Report: (a) whether stock
Orin honours anything like a per-allocation or per-page map (PTE kind / attribute bits, NvMap heap or carveout
flags) that could keep an allocation off a channel subset; (b) what FACIL states or measured about Orin's
physical-to-DRAM (channel / bank) mapping and how it realised PIM layouts on the real device; (c) whether its code
is public. Its answer feeds row 4 as before; if a stock-Orin route exists without a boot change, it is the direct
measurement (prefix confined to 8 of 16 channels, same bracket). Read-only, no boot or device-tree change, no reboot.

**A4 — Step 3: DRAM bytes are required, not optional.** Every step-3 row (`nopad`, `keep1`, `kvonce`,
`keep1+kvonce`) reports **DRAM bytes per frame** (E12b method: L2 fills + writes, split into reads and writes) for
the variant **and** its same-session reference, beside the times. The partition is sized by bytes, not time; a row
with time but no bytes is incomplete. If step 1d's tooling is ready, add each variant's per-kernel bytes too.

**Row 8 (pre-registered with this amendment, before any measurement; rows 1–7 unchanged).**

| # | quantity | bands and what each means |
|---|---|---|
| 8 | share of the prefix's excess reads attributed to **weight** re-reads (1d) | ≥ 2/3: H1 dominates — GEMM tiling / rasterisation order is the lever · ≤ 1/3: H2 dominates — activation re-reads, fusion is the lever · between: mixed; name the top kernels of each |

**Acceptance, added:** step 1a has its primary arm SM-free with ≥ 3 points at 90–130 GB/s left (or the highest
reachable rate, stated); step 1d has `kernel_bytes.csv`, the sum check and row 8 answered; step 1c states the FACIL
finding (a)–(c); every step-3 row has DRAM bytes for variant and reference.

## Step 1 — EV-02: the prefix under stolen bandwidth (highest priority; push its results before starting step 2)

**1a. Bandwidth-steal sweep.** Time the compiled prefix while an **in-process** load consumes ≈ **0, 20, 40, 60,
80, 94 GB/s** (add one point above 94 if reachable — the curve's shape matters). Output: prefix slowdown vs stolen
GB/s. The slowdown must isolate **bandwidth loss from SM loss**. Define it as
`σ(x) = A(prefix | load at x GB/s) / A(prefix | reference)`, where the reference has the same SM layout with the
load replaced by nothing or by an SM-matched compute-only control (E12b style), same session, bracketed
(reference → loaded → reference). Latitude — pick what reaches the rates with the least side effect, and say why:
- a copy kernel in its own green-context partition, rate set by throttling inside a short period (the prefix then
  runs as `graphs_both` in the other partition — the rung's graph tree escapes partitions, E12b). E12b's hog gave
  65 GB/s in 4 SMs; prefer the largest prefix partition that still lets the load reach its rate (the prefix's
  own bandwidth draw — and so its sensitivity — shrinks with its partition; report its draw in the reference);
- or a load that takes **no SMs** (a DMA copy-engine `cudaMemcpyAsync`, or a pinned multi-threaded CPU copy),
  which allows the unmodified rung prefix on the whole GPU — preferred if it reaches ~94 GB/s.
Report per point: the load's rate alone and **during** the prefix (EMC-calibrated as in E12b M4, `bench/e12b_m4.py`,
and the load's own byte count), the prefix's median / p5–p95 / n, σ, and which rate is the x-axis (primary:
the rate during). If `graphs_both` is used, first run the open E12b item: `graphs_both` prefix vs rung prefix
equivalence (KV and actions, same inputs and noise; bitwise or the max |Δ|).

**1b. Duty-cycle variant.** The same load **on for ~37 % of each prefix frame** (~93 ms of ~250), aligned to the
frame start (the expert of t−1 starts with prefix t), at the ~94 GB/s point and at one lower point; plus one run
with the on-window shifted to the frame's end. Use the per-kernel traffic profile
(`results/20260925_E12b_pi0_dram_bytes_MAXN/ncu/e12b_dram_raw.csv`) to say which prefix kernels and what share of
the prefix's bytes fall in each window, and compare the measured σ with the derived `1 + f·(σ_full − 1)`, f = that
byte share (label the prediction derived).

**1c. Feasibility probe (yes / no / unknown, with evidence).** Is there **any** way on this Orin to confine the
GPU's allocations to a subset of LPDDR channels — the memory controller's address interleaving (granularity and
hash, e.g. from a stride-vs-bandwidth sweep over a large buffer), carve-outs, NvMap heaps, CUDA allocator pools,
debugfs / BPMP controls? Probe read-only; **do not change boot configuration or the device tree and never reboot**.
If the answer is "yes but it needs a boot change", stop there and write what it would take. If it is "yes" without
one, it becomes the direct measurement: time the prefix confined to 8 of 16 channels, same bracket.

## Step 2 — seeded N sweep, all four LIBERO suites (E8 follow-up; answers your STATUS question)

E8 found `libero_spatial` N = 2 at 85 / 100 vs N = 10 at 62 / 100 (p = 0.0002), with π0's flow noise on the
unseeded global torch RNG. Replicate or refute it with **paired** arms: `bf16_both`, N ∈ {1, 2, 3, 4, 6, 10}, all
four suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`), E8's harness and settings
(`bench/libero_eval.py`, 10 tasks × 10 episodes, `env.seed(seed + episode)`, 10 settle steps), and the flow noise
drawn from a **`torch.Generator` seeded per (suite, task, episode)** — the same seed in every N arm (the initial
noise has the same shape at every N, so arms see the same noise). Record each episode's seed in `episodes.csv`.
Control: N = 10 on `libero_spatial` run **twice with the same seeds**; report episode agreement (it tells us how
much GPU non-determinism remains after seeding). Order if time is short: N = 10 and N = 2 on all suites, then the
repeat control, then N = 4, 1, 3, 6. Report per suite and pooled: successes, Wilson 95 % intervals, paired McNemar
(exact) against N = 10, the paired difference with its 95 % interval, forward time per N.

## Step 3 — prefix software savings, measured (variants, never the baseline)

Three changes that remove prefix work the output never uses. For each: prefix and full-frame time (same session,
bracketed against the unmodified configuration), DRAM bytes per frame by the E12b method (L2 fills + writes), and
an equivalence check — **bitwise on actions** with the same inputs and seeded noise where the change is exact;
where it is not bitwise, the max |Δaction| over ≥ 50 inputs and a seeded, paired LIBERO comparison as in step 2
(π0; for GR00T the action-level comparison only, unless you have a closed-loop harness for it).
- **π0 `nopad`**: 283 of the 816 prefix tokens are padding (masked out) — drop them before the prefill and give the
  expert a KV cache without them. Verify the count on the fixture. Mathematically exact; bitwise is not expected
  (shapes change). How you handle compilation of prompt-dependent lengths is yours; record it.
- **GR00T N1.7 `keep1`** (`nvidia/GR00T-N1.7-3B` through Isaac-GR00T @ `376ba890`, record the checkpoint revision,
  your best committed rung and N = 4 as in `reports/E3_E4_groot.md`): the backbone's `lm_head` runs over every prefix
  token and its logits are discarded — run it on the last token only (`logits_to_keep=1` or equivalent). Exact.
- **GR00T `kvonce`**: the DiT's cross-attention projects the (per-frame constant) VLM features to K / V at every
  denoising step — compute them once per frame and reuse them. Exact. Also `keep1+kvonce` combined.
Label rows `pi0_bf16_both+nopad`, `groot_n17_<rung>+keep1`, `+kvonce`, `+keep1+kvonce`. **Whether any of these
enters the GPU baseline is the user's open decision — never call a variant "baseline" or "optimised baseline".**

## Step 4 — energy reconciliation (EV-10)

In `reports/E7_power_energy.md` (π0 `bf16_both`, N = 10) the stage loops do not add up to the full loop:
prefix 251.28 + ae 167.36 = 418.64 ms vs `full` 424.68 ms (6.04 ms), and above-idle 6395.8 + 3112.5 = 9508.3 mJ vs
13 666.2 mJ (**~30 % missing**; the full loop draws 32.2 W above idle, more than either stage). Find the cause —
candidates: loop boundaries (what `full` runs that neither stage loop does), per-run idle baselines and drift,
clocks / EMC per loop, rail coverage, sampling. Suggested method, yours to change: the three loops interleaved in
one session under one idle baseline, per rail, with clocks and temperature logged. Deliver the corrected per-stage
energies and a residual; the goal is a residual under 5 % of the full-frame energy, or the measured reason why not.

## Step 5 — diagnose the E12b compiled co-run corruption (EV-13)

E12b removed two suspects together (one shared CUDA-graph-tree pool; the expert's KV cache overwritten by the next
prefix replay) and never isolated which one produced T014's NaNs. Run the 2 × 2: {shared pool, own pool} ×
{single KV buffer, double-buffered KV}, compiled co-run, ≥ 100 frames per cell with distinct observations and noise
per frame: non-finite count, equivalence failures vs serial, stale control. Commit the script
(e.g. `bench/e12c_diagnose.py`) — the harness must not claim a script that is not in git.

## Pre-registered outcomes (research side, 2026-09-26, before any T016 measurement; not moved after results)

| # | quantity | bands and what each means |
|---|---|---|
| 1 | σ at ~94 GB/s, full-time load (1a) | < 1.04: both claims survive · 1.04–1.247: latency parity at 16 ch lost, throughput claim survives · > 1.247: throughput claim broken too |
| 2 | σ at ~94 GB/s, 37 % duty (1b) | same three bands; this is the closer approximation of the design |
| 3 | 1b vs `1 + f·(σ_full − 1)` | within ±0.02: the loss is linear in the bytes exposed · otherwise: burst position matters, name the kernels |
| 4 | 1c | yes (direct measurement follows) · yes-with-boot-change (stop, describe) · no · unknown (say what was tried) |
| 5 | seeded N = 2 vs N = 10 (step 2) | **replicated**: N = 2 ≥ N = 10 in ≥ 3 of 4 suites and pooled paired McNemar p < 0.05 in favour of N = 2 · **reversed**: pooled p < 0.05 in favour of N = 10 · **not replicated**: anything else |
| 6 | seeded repeat control | ≥ 95 % episode agreement: seeding pairs the arms · below: GPU non-determinism dominates; read step 2 with that caveat |
| 7 | step 3 exact variants (`keep1`, `kvonce`) | bitwise equal actions expected; a mismatch is a finding to report, not to tune away |

The research side's prior, stated so that it can be wrong: row 1 lands in the middle band or above (E12b's green
co-run already coupled through memory, `s_part` 1.16–1.26). Your job is the measurement, not the prior.

## Non-negotiables

1. **Honesty.** Do not tune loads, partitions, seeds or episode sets toward any band; every run you make is reported,
   including the ones that went against the prior. References are same-session and bracketed.
2. **No silent NaN.** Every action tensor of every timed, checked, energy and LIBERO run is finite-checked (without a
   host sync inside a timed loop); a run with any non-finite value is invalid and reported, never dropped.
3. **Complete provenance.** `meta.json` has no null: `bench_commit`, model and checkpoint commits (GR00T revision
   too), rung, variant label, load kind and rate, partition sizes granted, seeds, `power_mode`, `sm_count`, clocks,
   `throttled`.
4. **MAXN only, no reboots, no boot-configuration or device-tree change.**
5. **Still not approved:** `cropcache` (P2) and the `check_model_inputs.py` `MODELS` change. (EV-02 and the seeded
   E8 follow-up are approved — they are steps 1 and 2.)

## Outputs

`results/<date>_E13a_pi0_bwsteal_<load>_MAXN`, `…_E13b_pi0_bwduty_MAXN`, `…_E13c_channel_probe`,
`…_E8s_pi0_<suite>_N<n>_seeded` (one per arm; the repeat `…_rep`), `…_E14_<model>_<variant>_MAXN`,
`…_E7r_pi0_energy_reconcile_MAXN`, `…_E12c_pi0_corruption_diag_MAXN`, each with `raw/` and `meta.json`
(protocol §5). Reports: `reports/E13_pi0_bandwidth_steal.md` (1a–1c, with the table of rows 1–4 answered first),
`reports/E8s_libero_seeded.md` (rows 5–6), `reports/E14_prefix_savings.md` (row 7), and step 4 and 5 as dated
addenda to `reports/E7_power_energy.md` and `reports/E12b_pi0_compiled_corun.md`.

## Acceptance

- Step 1: σ at ≥ 5 rates incl. ~94 GB/s with n ≥ 20 frames per point and bracketed references; the SM-loss
  isolation stated and its reference measured; the duty-cycle rows and the byte-share comparison; the 1c answer
  with its evidence committed under `raw/`; rows 1–4 of the pre-registration answered by number.
- Step 2: at least N = 10 and N = 2 on all four suites plus the repeat control (rows 5–6 answered); the other N as
  time allows, the missing ones named.
- Step 3: all three variants (and the combination) with time, bytes and the equivalence result; labelled variants.
- Step 4: corrected stage energies and the residual, with the cause or the measured reason it remains.
- Step 5: the 2 × 2 table and the committed script.
- Mark done only when every item holds, or the status line names the item that does not and why. Push each step's
  results when they land (step 1 first — it is the one the research side is waiting for).

**When:** take T016 as the next MAXN work. The capped-mode reboot session stays pending as before: if the human has
rebooted into `MODE_30W` / `MODE_15W` when you read this, finish those queues first and return to T016 once MAXN is
restored — never ask for a reboot for it. If the human owner gives you a different order directly, the owner's
order wins; say in `STATUS.md` where T016 sits.

status: claimed 2026-09-26

status: done 2026-09-27
- **Step 1 (EV-02)** — `reports/E13_pi0_bandwidth_steal.md`. Rows: **1** σ = 1.247 at 77.1 GB/s and 1.278–1.281 at
  82.9–84.9 GB/s (the SM-free maximum; ~94 GB/s not reachable without SMs) → **> 1.247 band**; **2** 37 % duty σ = 1.125
  → **1.04–1.247 band**; **3** measured vs 1 + f(σ_full − 1) within ±0.02 (f = 0.433 start / 0.315 end) → **linear**;
  **4** channel confinement **no** (boot-time BCT; not attempted); **8** H1 share **0.67** (on the 2/3 line; fit
  residuals 18–55 %). Dirs `results/20260926_E13a_pi0_bwsteal_ce+cpu_MAXN`, `…_E13b_pi0_bwduty_{start,end}_MAXN`,
  `…_E13c_channel_probe`, `…_E13d_pi0_prefix_kernel_bytes_MAXN`.
  Deviation: the in-session EMC hog calibration was not reproducible (NOTE.md); "left" uses E12b's 178.5 GB/s and
  the load's own byte count is the x-axis. The optional green-context secondary arm was not run (row 1 decided).
- **Step 2** — `reports/E8s_libero_seeded.md`. **Row 5 replicated**: N = 2 ≥ N = 10 in 4 / 4 suites, pooled 300 vs
  259 / 400, exact McNemar p = 1.3 × 10⁻⁵, +10.2 pp [+5.8, +14.7]. **Row 6**: repeat agreement **95 / 100** (on the
  line). All N ∈ {1, 2, 3, 4, 6, 10} on all four suites + the repeat: 25 arms, none missing
  (`results/20260926_E8s_pi0_<suite>_N<n>_seeded`, `…_rep`, summary `…_E8s_pi0_seeded_summary`).
  **Finding**: the harness supplied the empty camera as an all-zero image (mask 1, attended) in every E8 / E8s arm;
  paired comparisons unaffected, absolute rates are for that input (question in STATUS.md; E8 / E8s reports carry a
  correction note; `libero_eval.py --empty-camera masked` added).
- **Step 3** — `reports/E14_prefix_savings.md`. **Row 7: bitwise** — GR00T `keep1`, `kvonce`, `keep1+kvonce` equal
  the reference on 50 / 50 inputs, compiled and eager, with bitwise run-to-run controls. Time: GR00T e2e 237.0 →
  228.9 / 228.2 / 220.0 ms; π0 `nopad` prefix 253.0 → 189.6 ms (0.749), frame 420.5 → 319.5 ms (0.760). DRAM reads
  per frame: GR00T 20.96 → 19.62 / 19.66 / 18.32 GB; π0 prefix 17.65 → 14.64 GB, expert 7.03 → 6.91 GB. π0 `nopad`
  not bitwise (REL ≤ 1.7 × 10⁻³ over 50 inputs); seeded paired LIBERO: **70 vs 66 / 100 with the empty camera masked, p = 0.29** (the first nopad arm, with the camera attended, measured trimprompt only).
  Dirs `results/20260926_E14_pi0_nopad_MAXN/{time,equiv,bytes}`, `…_E14_groot_n17_savings_MAXN` (time_*, bytes,
  actions_*, equiv; `invalid_unseeded_inputs/` = the first action sets, invalid, NOTE.md),
  `results/20260926_E8s_pi0_libero_spatial_N10_seeded_nopad` (= `trimprompt`, see report),
  `results/20260927_E14_pi0_nopad_libero_speed_diag`, `results/20260927_E8s_pi0_libero_spatial_N10_seeded_maskedcam{,_nopad}`.
  Deviations: GR00T time metas were written without bench commit / revision / throttled and were back-filled
  (no measured value changed; `groot_phases.py` now writes them); GR00T bytes are one inference per variant
  (1 643 workloads, chunked 450 per ncu process).
- **Step 4 (EV-10)** — correction block at the top of `reports/E7_power_energy.md`: cause = the rail sampler started
  before warm-up / compile; re-measured prefix + ae = full within **0.49–0.71 %** (`results/20260926_E7r_pi0_energy_reconcile_MAXN`);
  28 of 132 committed E7 phase rows (all compiled) move by ≥ 5 %, listed in `E7_all_dirs_timed_window.csv`; samplers fixed.
- **Step 5 (EV-13)** — addendum in `reports/E12b_pi0_compiled_corun.md`, script `bench/e12c_diagnose.py`,
  `results/20260926_E12c_pi0_corruption_diag_MAXN`: own/double 0 failures, own/single 98 / 100 wrong, shared/double
  0, shared/single HANG → the single KV buffer is the operative cause.
- bench commit at close: 3b4cea0. Repository: committed results 529 MB; size-pack 158.10 MiB + loose 84.34 MiB.
