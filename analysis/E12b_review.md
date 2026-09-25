# E12b / T015 — research-side review of the compiled co-run (2026-09-25)

*jetson-owner, on the research server. Reviews `reports/E12b_pi0_compiled_corun.md`, the T015 status lines, the
`STATUS.md` correction and every committed row under `results/20260925_E12b_pi0_*` against the spec **as
pre-registered at `5cf1129`** (`docs/E12b_compiled_corun/README.md`, §10 thresholds and rows, §12 acceptance).
Pulled at device commit `c791ec1` (T015 done at `e4b2725`; re-checked at `55e41fa`, no E12b change). Where the report and the rows disagree, the rows are
taken. **Measured** = a committed row; **derived** = arithmetic on committed rows, never measured as such. This
file is analysis, not an instruction to the device.*

All runs: MAXN + `jetson_clocks`, 16 SMs, 1300.5 MHz GPU, EMC 3199 MHz, `throttled: false` (measured, every
`meta.json`), π0 LeRobot `f0d2b37` / ckpt `45dcc8f`, rung `bf16_both` exactly as E7, batch 1, N = 10.
`bench_commit` per directory: fairexpert `2c0912b`, corun `80bf710`, green_corun `0558963`, emc_calibration
`bc9a001`, energy `714e395`. `raw_files` sha256 recomputed here: 76/76 match; no null anywhere in any E12b
`meta.json`.

## 1. Numbers

### 1.1 Co-run table (medians, ms; `corun_ratios.csv`, recomputed from `latency_samples.csv`)

Reference for every M3 row: the same session's full-GPU `graphs_both` stages, `A_iso` 246.28, `B_iso` 101.36,
`T_serial` 348.67 (n = 200 / 200 / 100). M2's own session: 248.43 / 101.21 / 353.03.

| configuration (variant tail) | A_co | B_co | T_pipe | **O** | φ | p | s | r | A_part | B_part | **p_part** | **s_part** | p_hog (max rate, hog GB/s alone) | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| M2 whole GPU, rung prefix, `prefix_first` | 258.42 | 270.85 | 345.69 | **1.391** | 0.070 | 1.040 | 2.676 | 2.455 | — | — | — | — | — | 120 |
| M2 whole GPU, `expert_first` | 288.16 | 337.96 | 338.06 | 1.361 | 0.143 | 1.160 | 3.339 | 2.455 | — | — | — | — | — | 120 |
| M2 whole GPU, priority −3 on expert | | | 320.58 | 1.287 | | | | | — | — | — | — | — | (mechanism row) |
| M3 whole GPU, `graphs_both` | 284.37 | 339.71 | 339.69 | 1.379 | 0.088 | 1.155 | 3.351 | 2.430 | — | — | — | — | — | 100 |
| **M3 green 12 / 4** | 287.88 | 305.01 | **307.15** | **1.247** | 0.406 | 1.169 | 3.009 | 2.430 | 267.71 | 254.79 | **1.075** | **1.197** | **1.043** (65.0) | 100 |
| M3 green 10 / 6 | 319.74 | 233.33 | 321.23 | 1.304 | 0.268 | 1.298 | 2.302 | 2.430 | 290.56 | 185.20 | 1.100 | 1.260 | 1.108 (94.5) | 100 |
| M3 green 8 / 8 | 360.12 | 183.14 | 361.29 | 1.467 | −0.123 | 1.462 | 1.807 | 2.430 | 326.94 | 149.05 | 1.101 | 1.229 | 1.098 (120.3) | 100 |
| M3 green 6 / 10 | 429.03 | 151.21 | 430.05 | 1.746 | −0.795 | 1.742 | 1.492 | 2.430 | 396.34 | 130.82 | 1.082 | 1.156 | 1.116 (142.7) | 100 |
| M3 green 14 / 2 | 265.54 | 533.30 | 535.54 | 2.175 | −1.825 | 1.078 | 5.261 | 2.430 | 253.46 | 496.00 | 1.048 | **1.075** | **1.024** (33.6) | 100 |

Dispersion is small everywhere: p5–p95 of every co-run stage ≤ 0.9 % of its median; bracket drift ≤ 0.48 %
(M2), ≤ 0.04 % (M3). Every row passed its pipeline check (§2).

### 1.2 M1 — the fair expert (`results/20260925_E12b_pi0_fairexpert_N10_MAXN/loops_summary.csv`)

| loop | stock (pre / post) | **hoist** | reference |
|---|---:|---:|---|
| expert `B` | 168.14 / 167.40 | **110.95** (n = 541) | E7 167.36; reconciliation 111.02 |
| prefix `A` | 251.29 / 249.25 | 249.19 | E7 251.28 |
| whole frame, two stage callables | 417.85 / 416.70 | **354.74** | — |
| whole frame, shipping `predict_action_chunk` | 425.04 / 421.36 | — | E7 424.68 |

Hoist vs stock, compiled: **bitwise identical on 5/5 observations**; stock vs itself bitwise; frame-to-frame
difference up to 5.408, so the check had power. Labelled exact variant `owngraph_expert` (the N-step loop as one
`torch.cuda.CUDAGraph`, same Inductor kernels, own pool): **101.21 ms**, bitwise identical to the rung expert on
5/5 (`corun_N10_MAXN/owngraph_equivalence.csv`). It is the expert of every M2–M5 same-harness number.

### 1.3 M4 — bandwidth (tegrastats `EMC_FREQ`, calibrated GB/s = 2.005 × EMC% + 0.34, R² 0.99987, hog 56–178.5 GB/s)

| condition | whole GPU | 12 / 4 |
|---|---:|---:|
| prefix alone | 84.6 GB/s (42 %) | 80.6 (40 %) |
| expert alone | 80.6 (40 %) | 40.4 (20 %) |
| serial | 82.6–84.6 | 56.5 |
| co-run | 84.6 rung / 88.6 `graphs_both` | **106.6 (53 %)**, p5–p95 47–56 % |
| prefix + hog (65 GB/s alone) in the expert's partition | — | 128.7 |

Bandwidth for the co-run exists only for the whole GPU and 12 / 4. Sampled in separate runs from the timings
(stated deviation). nsys `--gpu-metrics-devices` not attempted.

### 1.4 M5 — energy per frame, whole-frame loops (`results/20260925_E12b_pi0_energy_N10_MAXN/summary.json`)

| arm | ms/frame | module mW | mJ/frame | above idle |
|---|---:|---:|---:|---:|
| (a) stock `full` (E7 anchor) | 424.69 | 42 375 | 17 996 | 13 409 (E7: 13 666, −1.9 %) |
| (b) fair serial, whole GPU (`graphs_both` + `owngraph_expert` harness) | 347.94 | 50 446 | **17 552** | 13 794 |
| (c) best co-run, green 12 / 4 | 313.75 | 57 063 | **17 903** | 14 515 |

(c)/(b) = **1.020** total, **1.052** above idle. Idle = each rail's in-session 60-s idle mean before model load.
Arm (c)'s period is 2.1 % longer than M3's 307.15 (sampler load or thermal; each arm uses its own period, so
the ratio is self-consistent). A first M5 run whose arm (b) went through the partitioned harness (526 ms, ratio
0.874, row 10 falsely true) was caught by the device and superseded; it is listed in `meta.json.superseded_run`.

## 2. Compliance (spec `5cf1129` §5 non-negotiables and §12 acceptance)

| item | verdict |
|---|---|
| §12.1 gate, MAXN, `throttled` | **met** — measured (GPU clock start vs end, peak thermal zone), false everywhere |
| §12.2 M1 bracketed loops, hoist `equivalence.*` compiled | **met** — stock → hoist → stock, ≥ 100 frames / ≥ 60 s; bitwise 5/5 with power |
| §12.3 M2 compiled co-run | **met** — equivalence bitwise 6/6, stale control detected 5/5; n = 120 (+174 in the ≥ 60 s loop); ratios; host-issue record; nsys CSVs + `corun_breakdown.csv` |
| M2 step 1 **diagnose before fixing** | **not met as specified.** The report names two conditions (one shared graph-tree pool; the next prefix replay overwriting the expert's KV) and says plainly that "which of the two was the operative cause was not isolated — the fix was not built one change at a time". No buffer, replay or stream is identified with evidence, and the spec's condition for an undiagnosed fix (the diagnosis resisted a reasonable effort, with what was tried) is not shown. `bench/e12b_corun.py`'s docstring cites an evidence run `e12b_diagnose.py` that is **not committed**. The two named conditions are plausible and the fix removes both, so the numbers stand; the *diagnosis* is a suspect list (the spec's own prime suspect plus one), not a finding |
| M2 step 4 host not the bottleneck | **met for the whole GPU** by the right evidence — the nsys union of both streams is busy 347.7 ms per frame against a 345.7-ms unprofiled period, so the GPU never idles. The per-frame `host_issue` column equals the period in every condition, including `prefix_alone` (249 ms): it is the launch back-pressure metric the E12 review rejected, and the device says so |
| §12.4 M3 | **met** — five splits (12/4, 10/6, 8/8, 6/10, 14/2), every one granted as requested, each with `A_part`, `B_part`, co-run passing its check with the stale control detected, `p_part`, `s_part`, `T_pipe,part`, `O`; M0 confinement control in `raw/`. Hog at ≥ 2 rates: **met at 10/6, 8/8, 6/10; not at 12/4 and 14/2**, where a 4- or 2-SM partition cannot drive the hog harder than the matched rate (65.0 vs 62.9; 33.6 vs 33.6 GB/s) — stated by the device, but it means the best split's control has one effective rate |
| M3 harness ruled out inside partitions | **partial.** No timeline of a partitioned co-run exists; `host_issue` ≈ `T_pipe` again carries no information. For: stage start events are recorded after the cross-stream wait, so host delay *before* a stage is not counted; dispersion is tight (≤ 0.9 %); EMC rises to 106.6 GB/s, so both stages really are live together. Not excluded: a host stall *inside* a stage's issue (between its start event and the replay, e.g. around the green-context switch) would be counted in `A_co,part` / `B_co,part` |
| **In-process hog control** (`p_hog` ≥ 1.03 at the highest rate) | **passes at the best split** (12/4: 1.043) and at 10/6, 8/8, 6/10 (1.098–1.116); **fails at 14/2** (1.024, the 2-SM hog reaches only 33.6 GB/s). Placement is right: same process, captured and replayed inside the expert partition's green context, concurrent with the prefix alone in its partition (`meta.json.hog`) |
| §12.5 M4 | **met with two stated deviations**: tegrastats sampled in separate runs, not alongside the timing runs; nsys GPU-metrics not attempted. Calibration and the memory rail delivered |
| §12.6 M5 | **met** — idle + three whole-frame arms in one session; superseded run disclosed |
| §12.7 / NN2 finite checks | **met** — device-side non-finite count on every action tensor, read after the loop; 0 non-finite in every run; no invalid run |
| NN1 fresh KV in the timed compiled path, stale control failing | **met** — whole GPU and every one of the five splits, through the timing harness |
| NN3 same session, bracketed | **met**. The one > 3 % difference (`B_iso` 101.21 vs 111.02) is the labelled `owngraph_expert` harness change, explained |
| §12.8 / NN4 provenance, labels | **met in substance**; no nulls, raw hashes verified, every row carries a `+`-joined label. Minor defect: `variant_tokens` (identical in all three co-run `meta.json`) does **not define** `dbuf_kv`, `owngraph_prefix`, `hog_max` / `hog_matched`, `serial_full_gpu`; it defines `kv_copy` as "copied on the EXPERT stream (counted in the expert stage)", which no row uses and which contradicts the code (`dbuf_kv`: the copy is on the prefix stream, counted in the prefix stage — `bench/e12b_corun.py`). `memo_681` is defined under `rung_note`/`memo`, not `variant_tokens`. `graphs_both`'s prefix (`embed_image` compiled `default`, own capture) has a pipeline check but **no equivalence check against the rung prefix** (the expert has one) |
| §12.9 self-review, sizes, partition, T014 line | **met** — largest file 9.0 MB; repo size in `STATUS.md`; only status lines appended to `tasks/T014`, `tasks/T015`; research-side files untouched |
| NN5 MAXN only, nothing unapproved | **met** — no `cropcache` token anywhere; the channel-partition measurement (EV-02) not attempted; no power-mode change |
| §9 time box | **not needed** — claimed 10:16, M2's fixed co-run committed 10:46–11:06 PDT the same morning |

The device's status line says "every acceptance item of spec §12 holds". Against the rows: yes for §12's list as
written; the gaps are M2 step 1's diagnosis (a step, not a §12 item), the hog's single effective rate at 12/4,
the unexcluded in-partition harness, and the token definitions.

## 3. Where the report's narrative and the rows disagree

1. **"Material coupling in every split (`s_part` 1.16–1.26)"** (report, row-2 line) — the 14/2 split has
   `s_part` **1.075** (minor band) and a **failing** hog control (1.024). Material in **4 of 5** splits.
2. **"Memory couples them" / "partitioning SMs does not partition bandwidth"** — the rows support *coupling
   with the SMs separated*; they do **not** support *bandwidth* as its mechanism, and the report's own M4
   paragraph half-says so ("contention inside the memory system … not the bus running out of bandwidth"):
   - the bus is not near saturation during the 12/4 co-run: 106.6 GB/s against the 178.5 GB/s the hog drew;
   - the expert slows the prefix **more** than a hog of greater EMC load does: expert alone in 4 SMs 40.4 GB/s →
     `p_part` 1.075; hog alone in the same 4 SMs 65.0 GB/s → `p_hog` 1.043. At 8/8 the expert (≈ 43 GB/s by E6
     bytes; no EMC row) gives `p_part` 1.101 against `p_hog` 1.098 at 120.3 GB/s. So the expert's effect on the
     prefix is **2–3× larger per GB/s than a streaming copy's** — pointing at L2 / controller / row-conflict
     coupling specific to the expert's access pattern, or at the unexcluded harness (§2), not at bandwidth.
   The honest reading is "partitioning SMs does not isolate the stages: they couple through the shared memory
   system (or the single-thread harness), at 53 % of the bus".
3. **EMC-calibrated GB/s are occupancy, not bytes.** The prefix alone reads 84.6 GB/s; E6's matched-rung bytes
   give ≈ 24 GB/s for vision + VLM over 249 ms (spec §3) — a 3.6× gap. The expert: 80.6 vs ≈ 63 GB/s (1.28×).
   The calibration maps EMC busy % onto a *streaming* copy's GB/s; a less regular access pattern (row misses,
   read–write turnarounds) keeps the controller busier per byte. The device quotes these as "GB/s drawn".
   Either the prefix moves ~3.6× more DRAM traffic than E6 counts (activations spilling L2), or EMC busy % over-
   states it. Unresolved; it matters for the channel-partition measurement (EV-02, §6).
4. **Host issue.** The report's "host issue is not the bottleneck" is right for the whole GPU, on the nsys
   union — not on the `host_issue` column it cites alongside it (§2). One harness effect it does not mention:
   in M2 `prefix_first` with the rung prefix, `issue_prefix(k)` hits the `modeling_pi0.py:681` host sync after
   `embed_image`, so `expert(k−1)` is not even issued until prefix(k)'s vision has run. That inflates `T_pipe`
   for that row only; `expert_first` (1.361), `graphs_both` with `memo_681` (1.379) and priority −3 (1.287)
   bracket it, so the whole-GPU conclusion (poor overlap, head-of-line blocking) is robust to it.

## 4. The pre-registered rows (quoted as committed at `5cf1129`, §10)

| # | row as committed | landed? |
|---|---|---|
| 1 | "**Some valid configuration (M2, or an M3 split) reaches `O ≤ 1.05`**" → "**The frame-rate argument for moving the expert off the GPU is gone** (≤ 5 %)." | **No.** Best valid `O` = **1.247** (12/4), poor band (≥ 1.15). The lowest whole-GPU `O` is 1.287 (priority −3 on the expert) |
| 2 | "**Best `O` > 1.05, and inside partitions `p_part ≥ 1.10` (or `s_part ≥ 1.10`), with the hog control passing**" → "**'Partitioning SMs does not partition bandwidth'** on Orin … (with the caveat of §3 that a PIM also shares the LPDDR5)." | **Yes, on its committed condition** — at the best split: `O` 1.247 > 1.05; `s_part` **1.197** ≥ 1.10; `p_hog` **1.043** ≥ 1.03. `p_part` alone reaches 1.10 only at 10/6 (1.100) and 8/8 (1.101), not at the best split (1.075). The row is carried by the **expert's** slowdown, while the positive control tests the **prefix's** sensitivity — the committed wording allows that, and the row stands. **Its reading must be narrowed** (§3.2): coupling through the memory system *other than* bus bandwidth, or the harness, not "bandwidth" |
| 3 | "Best `O` > 1.05, but `p_part`, `s_part` < 1.03 and the hog control passes" | No |
| 4 | "Unpartitioned M2 has `O` > 1.05 while the best M3 split is materially lower" | **Yes, in part** — 1.391 (1.379 same-session `graphs_both`) → 1.247: partitioning removes the queueing (nsys: 3.1 % of a whole-GPU co-run frame has kernels of both streams executing; the expert waits 175 ms/frame behind the prefix's 1008-CTA GEMMs — KerColle's head-of-line blocking, observed on Orin) but stays in the poor band. Its consequence holds: green 12/4 is the GPU-side baseline a paper must compare against |
| 5 | "`s > r` (in-session) in M2 or the best split" → "the expert, stretched by contention, sets the period despite ≈ 2.25× slack" | **Yes** — M2 `s` 2.676 > `r` 2.455; 12/4 `s` 3.009 > 2.430. At 12/4 the period is set by the **expert** (`B_co,part` 305.01 ≈ `T_pipe` 307.15) |
| 6 | "The corruption proves unfixable" | No — fixed |
| 7 | "**M1: the fair expert `B` ≤ 93.1 ms at N = 10**" | **No** — 110.95 ms. `owngraph_expert` (101.21) would not either |
| 8 | "M1: 93.1 ms < fair `B` ≤ 102.4 ms" | **No for the fair expert as defined** (110.95). **Yes for the labelled exact variant `owngraph_expert`** (101.21), if it enters the baseline — the device asks (§7) |
| 9 | "`hoist` fails its compiled equivalence check" | No — bitwise |
| 10 | "Co-run energy per frame (M5 c) ≤ 0.95 × the fair serial `full`" | **No** — 1.020 (1.052 above idle) |

**Falsified.** The research side held no prediction in §10, so no row was falsified as a hypothesis. Three
things written *outside* §10 are contradicted by these rows: (i) spec §3's "average demands sum to well under half
the peak, which would suggest little average-rate coupling" — coupling is material at 53 % bus occupancy;
(ii) T014's "compiled expert SM-independent" and the derived 271.26 ms bound (§5); (iii) my E12 review §6's lean
toward framing I (§5). KerColle's head-of-line mechanism is **confirmed** on Orin (whole GPU); Jetson-PI's "both
stages slow" is **confirmed in partitions** (`p_part` 1.05–1.10, `s_part` 1.08–1.26), but not with the mechanism
it argued (bandwidth-bound phases).

**Decomposition of the best split (derived).** With zero coupling, 12/4 would run at max(`A_part`, `B_part`) =
267.71 ms → `O` 1.087. Of the measured excess (`O` − 1 = 0.247), **0.087 (35 %) is the SM partition's own cost**
— the compiled expert is SM-sensitive (496.0 / 254.8 / 185.2 / 149.1 / 130.8 ms on 2 / 4 / 6 / 8 / 10 SMs against
101.4 on 16) — and **0.160 (65 %) is coupling**, entering through the expert (`s_part` 1.197), which sets the
period. TPC granularity is 2 SMs, so no split between 12/4 and 10/6 exists.

## 5. The T014 correction — verified

- **M0** (`results/20260925_E12b_pi0_green_corun_N10_MAXN/raw/m0_graph_confinement.json`, 2048² bf16 GEMM, 20 per
  graph): eager on a green stream 3.25 / 1.64 / 1.14× at 4 / 8 / 12 SMs; a graph **captured on the primary
  context and replayed on the green stream 0.99 / 0.99 / 0.99×** (not confined); a graph captured inside the green
  context 3.24 / 1.63 / 1.13×, also when replayed with the primary context current. A graph obeys the partition it
  was captured in. (At 12 SMs the discrimination is weak — 1.13 vs 0.99, ideal 1.33 — but 4 and 8 are
  unambiguous.)
- **T014's harness** (`bench/corun_green.py`, commit `de6afa3`): `pkv, pad = CR.prefix_stage(ctx)` and the
  control-2 reference `CR.expert_stage(...)` run **on the primary context before any green context is made
  current**. Under `bf16_both` those first calls record `denoise_step@reduce-overhead`'s and
  `embed_image@reduce-overhead`'s CUDA-graph trees; every later green-stream call replays them. So T014's
  compiled expert ran on the whole GPU at every "SM count" (hence 1.049× at 4 SMs), and the compiled prefix was
  only partly confined (the VLM's Inductor kernels launch eagerly and were confined; the vision graph tree was
  not). T014's "bitwise identical on an 8-SM green stream" for `bf16_both` compared a whole-GPU replay with itself.
- **Confined properly:** the compiled expert on 4 SMs is **254.79 ms, 2.51× its 16-SM 101.36** (E12b M3).
- **Withdrawn**, as the device says (`592e2da`, `STATUS.md` top, `reports/E12_pi0_corun.md` §5): the 1.049×
  "SM-independent expert", the derived 12/4 period **271.26 ms**, and "the expert can neither block nor be blocked
  at the CTA dispatcher" (contradicted by M2's timeline). Eager results stand: eager launches are confined (M0).
- `analysis/E12_review.md` is updated accordingly (correction block at the top; §1, §6, §8 marked).

## 6. What it means for the research question (case 001 §4.3 framings)

**Measured:** on Orin at MAXN, one GPU does **not** overlap π0's prefix and the fair expert well. The best valid
co-run is green 12/4 at **307.15 ms** (`O` 1.247); whole GPU 1.287–1.403 across every mechanism tried. Case 001's
framing I (ideal overlap, PIM adds nil) is **not what Orin does**; the reviewer's two software counters in §5.1 of
the case are both answered on silicon — SM partitioning helps (1.39 → 1.247) but stays poor, because the compiled
expert needs SMs *and* the stages couple with their SMs separated; two processes still OOM at load.

**Branch table given to the user:** row 1 ("near-ideal GPU overlap → the frame-rate argument is gone") — **did not
happen**. Row 2 ("partitioning SMs does not partition bandwidth → the memory-side design becomes the argument") —
**happened on its committed condition, with a narrower mechanism**: the stages couple through the shared memory
system at ≈ 53 % bus occupancy, not by exhausting bandwidth, and the in-partition harness is not excluded.

**Derived comparison** `[derived]` (same-session M3 references; the in-memory pipeline's period is taken as the
GPU prefix alone, i.e. **assuming the in-memory expert does not slow the prefix** — exactly what the channel-partition measurement (EV-02) would test):

| configuration, N = 10 | period (ms) | vs fair serial 348.67 |
|---|---:|---:|
| fair serial, whole GPU, same harness | 348.67 (measured) | 1.00× |
| best GPU co-run, green 12/4 | 307.15 (measured) | **1.135×** |
| in-memory expert pipeline, period = `A_iso` | 246.28 | **1.416×** (1.424× on M1's stock-harness 354.74 / 249.19) |
| **PIM frame-rate increment over the best GPU co-run** | 307.15 / 246.28 | **≤ 1.247×** |

So under case 001 §5.2's amended falsifier, "the PIM's increment is the measured `period_GPU-overlap / 251.28`":
**1.22×** on the case's 251.28 ms, **1.247×** on the same-session prefix. Row II of case 001 (1.10–1.66×, a
cross-device inference from KerColle) is **replaced by a measurement**, and lands inside that range.

**The symmetric caveat, plainly.** If memory couples the two GPU stages, an in-memory expert that takes LPDDR
channels away from the GPU may couple too. That is jetson **the channel-partition measurement (EV-02)** — the prefix on 8 of 16 channels — still **not
approved**, and it is now the natural next question: the 1.247× above is an upper bound that holds only if the
prefix loses nothing to the PIM. **Does this data bound it?** No — it sets the break-even and gives two
indications that point opposite ways:
- **Break-even `[derived]`:** the in-memory pipeline keeps a frame-rate advantage over the best GPU co-run only if
  the prefix, sharing LPDDR with the PIM, slows by **less than 1.247×** (307.15 / 246.28).
- **Against (sharper channel-partition risk) `[derived]`:** the prefix alone keeps the EMC 42 % busy (84.6 GB/s streaming-
  equivalent, 47 % of the 178.5 GB/s the hog reached). Halving the channels at the same traffic would put it near
  **95 % of the remaining capacity** — the contention regime. The 3.6× gap between EMC occupancy and E6's bytes
  (§3.3) is the uncertainty in that estimate.
- **For (tolerance) `[measured, partitioned]`:** the prefix slowed only **1.043–1.116×** beside an in-process hog
  that alone drew 65–142.7 GB/s. But a hog *shares* all 16 channels by arbitration; EV-02 *removes* half of them
  outright, and the 16-SM prefix was never measured against either.
- **Mechanism:** the coupling found here looks controller / L2 / row-conflict rather than bandwidth (§3.2). A PIM
  expert streams its weights inside the DRAM and never through the GPU's L2 or the controller's read path, so the
  L2 part would not transfer; channel loss and all-bank PIM commands blocking host access would. Only the channel-partition measurement (EV-02) measures it.

## 7. Latency and energy

**Latency (RTC's currency)** `[derived]` from measured stages, assuming zero handover cost and no prefix slowdown:

| | GPU expert | in-memory 16 ch (93.1) | in-memory 32 ch (58.7) |
|---|---:|---:|---:|
| fair expert 110.95 (M1) | — | expert **1.19×** faster; frame 342.29 vs 354.74 → **1.036×** | 1.89×; 307.89 → **1.152×** |
| `owngraph_expert` 101.21 (if admitted) | — | 1.087×; frame 339.38 vs 348.67 → **1.027×** | 1.72×; 304.98 → **1.143×** |

The latency claim at 16 channels survives, barely (rows 7 and 8 do not land for the fair expert; row 8 lands for
`owngraph_expert`). And **the GPU co-run costs latency**: a pipelined frame's observation-to-action time is at
least `A_co + B_co` = 287.88 + 305.01 = **592.9 ms** at 12/4 — **1.70× the fair serial frame** — in exchange for
1.135× frame rate. The in-memory pipeline keeps `A + B_PIM` ≈ 339–342 ms latency at a period ≈ `A`. On RTC's
axis the GPU-side overlap is a regression; this sharpens case 001 §4(a) rather than weakening it.

**Energy.** M5's +2.0 % (+5.2 % above idle) means **overlapping on the GPU does not save energy on Orin**: it cuts
the period 10 % and raises module power 13 %. The energy baseline a PIM must beat is therefore the **fair serial
frame** (17 552 mJ, 13 794 above idle, 347.94 ms — `owngraph_expert` harness), not the co-run; "GPU overlap saves
energy" is not available to a reviewer as a counter. These are whole-frame loops, so E7's 30 % phase-loop residual
does not touch them — but a GPU-vs-PIM energy comparison still needs a per-phase split, which remains blocked by
that residual.

## 8. Still open

- **T014 Step 4 restatement (`STATUS.md`, committed `ed73f56`, 10:16 PDT) — sound in method, stale in two of its
  three reference values.** Right: the eager expert is dropped as the test; the cross-check uses GPU-bound
  quantities; CPU clocks are recorded; the GEMM control (MAXN 8-SM green 0.891 ms/GEMM, M0-consistent) runs first
  in `bench/queues/queue_e12_capped.sh`. Wrong: it was written **24 minutes before M0** (`395c70d`, 10:40) and not
  revised after. Its compiled rows quote **T014's 113.70 ms expert and 320.71 ms prefix at 8 green SMs — the
  unconfined values the device itself withdrew**. The confined 8-SM MAXN references are E12b M3's `B_part` **149.05**
  and `A_part` **326.94** (`graphs_both`), and the queue does not measure compiled stages at `MODE_30W` at all,
  so those rows can be neither tested nor falsified as queued. Second: scaling a memory-bound compiled stage by the
  GPU-clock ratio alone over-predicts its slowdown unless the EMC clock falls too; the prediction should bracket
  between no change (memory-bound) and the clock ratio (compute-bound), with the `MODE_30W` EMC rate recorded.
  The queue script's header comment still carries the old eager clock-ratio prediction. **Owed to the device as a
  note** (not a task; not written this round).
- **Question from the device:** should `owngraph_expert` join the hoist in the fair GPU baseline (110.95 → 101.21 ms)?
  By the test I applied to the hoist (E12 review §7): exact (bitwise), GPU-agnostic, no change to what is computed,
  what any C++ / TensorRT deployment would do — it looks **baseline-side**. The decision is the user's (OD-1 class);
  if admitted, row 8 lands and the 16-ch expert advantage is 1.087×.
- **Owed questions, not tasks:** the diagnosis evidence (`e12b_diagnose.py`, cited, uncommitted); the EMC-vs-bytes
  3.6× gap for the prefix; the `variant_tokens` definitions (`dbuf_kv`, `owngraph_prefix`, `hog_*`,
  `serial_full_gpu`, and the stale `kv_copy`); a `graphs_both`-prefix vs rung-prefix equivalence check.
- **Would decide what is left:** a partitioned co-run timeline (harness vs memory); the channel-partition measurement (EV-02, unapproved) for the
  PIM side of the coupling; P2 `bf16_both + cropcache` (unapproved) for the latency claim.
- **Unapproved, unchanged:** the channel-partition measurement (EV-02), P2, `check_model_inputs.py` `MODELS`. The capped-mode session is the device's
  business (T004 4b, T014 Step 4).
