# E12b — the compiled co-run of π0's prefix and a fair expert on one Orin GPU

**Task T015. Specification `docs/E12b_compiled_corun/README.md` (pre-registered, §10 not edited);
binding rules protocol §15 and §14. π0 LeRobot `f0d2b37`, checkpoint `45dcc8f`, `bf16_both` exactly
as E7 (rung string checked against `results/20260924_E7_pi0_N10_bf16_both_MAXN/meta.json`), batch 1,
N = 10, MAXN + `jetson_clocks`, all runs 2026-09-25. Results: `results/20260925_E12b_pi0_*`.**

## The answer

**One Orin GPU cannot overlap π0's prefix with the fair expert well: the best valid co-run reaches
`O` = 1.247 (poor overlap), and partitioning the SMs does not partition the memory system — the two
stages still slow each other by 8–26 % with their SMs fully separate.** On the whole GPU the
scheduler runs the two streams essentially one at a time (3.1 % of a co-run frame has kernels of
both executing): the prefix's grids fill all 16 SMs and the expert queues behind them.

| configuration | `A_iso` | `B_iso` | `T_serial` | `T_pipe` | **`O`** | `φ` | `p` | `s` | `r` | `A_part` | `B_part` | `p_part` | `s_part` | `p_hog` (max) | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **M2** whole GPU, rung prefix, prefix-first | 248.43 | 101.21 | 353.03 | 345.69 | **1.391** | 0.070 | 1.040 | 2.676 | 2.455 | — | — | — | — | — | 120 |
| M2 whole GPU, expert-first | 248.43 | 101.21 | 353.03 | 338.06 | 1.361 | 0.143 | 1.160 | 3.339 | 2.455 | — | — | — | — | — | 120 |
| M3 whole GPU, `graphs_both` | 246.28 | 101.36 | 348.67 | 339.69 | 1.379 | 0.088 | 1.155 | 3.351 | 2.430 | — | — | — | — | — | 100 |
| **M3 green 12 / 4** | 246.28 | 101.36 | 348.67 | **307.15** | **1.247** | 0.406 | 1.169 | 3.009 | 2.430 | 267.71 | 254.79 | 1.075 | **1.197** | 1.043 | 100 |
| M3 green 10 / 6 | 〃 | 〃 | 〃 | 321.23 | 1.304 | 0.268 | 1.298 | 2.302 | 〃 | 290.56 | 185.20 | 1.100 | 1.260 | 1.108 | 100 |
| M3 green 8 / 8 | 〃 | 〃 | 〃 | 361.29 | 1.467 | −0.123 | 1.462 | 1.807 | 〃 | 326.94 | 149.05 | 1.102 | 1.229 | 1.098 | 100 |
| M3 green 6 / 10 | 〃 | 〃 | 〃 | 430.05 | 1.746 | −0.795 | 1.742 | 1.492 | 〃 | 396.34 | 130.82 | 1.083 | 1.156 | 1.116 | 100 |
| M3 green 14 / 2 | 〃 | 〃 | 〃 | 535.54 | 2.175 | −1.825 | 1.078 | 5.261 | 〃 | 253.46 | 496.00 | 1.048 | 1.075 | 1.024 | 100 |

ms; medians; `O = T_pipe / max(A_iso, B_iso)` and `φ` against the **same session's full-GPU,
same-harness** `A_iso` / `B_iso` / `T_serial` (the M3 rows against M3's own full-GPU reference);
splits are prefix SMs / expert SMs **as the driver granted them** (all as requested); every row
passed its pipeline equivalence check (bitwise, stale control detected) with 0 non-finite actions.
Bracket drift ≤ 0.8 % (M2: 0.47 % prefix, 0.48 % expert, 0.08 % serial). `r` uses the same-harness
expert, which is the own-graph expert (101 ms, see M1 below); against M1's fair stock-harness expert
(110.95 ms) the M2 headline is `O` 1.381, `r` 2.256 (`corun_ratios.csv`, `reference = stock_harness`).

## M1 — the fair expert (`results/20260925_E12b_pi0_fairexpert_N10_MAXN`)

Compiled `bf16_both`, `stock` (shipping per-step timestep tensor) vs `hoist` (the N tensors built once),
bracketed stock → hoist → stock, every loop ≥ 60 s / ≥ 100 frames:

| | stock | **hoist (fair)** | reference |
|---|---:|---:|---|
| expert `B` | 167.77 ms | **110.95 ms** | E7 167.36 · reconciliation 111.02 |
| prefix `A` | 250.27 ms | 249.19 ms | E7 251.28 |
| whole frame, shipping `predict_action_chunk` | 423.20 ms | — | E7 424.68 |
| whole frame, the same two stage callables | 417.28 ms | **354.74 ms** | — (hoist saves 62.53 ms/frame) |

**Equivalence, compiled path:** hoist against stock on 5 LIBERO observations with their own seeds —
**bitwise identical on every frame**; stock against itself also bitwise; consecutive frames' actions
differ by up to 5.408, so the check had the power to see a change (`equivalence.*`).

**The fair GPU expert is 110.95 ms at N = 10 — not ≤ 93.1 ms (row 7) and not ≤ 102.4 ms (row 8).**

**A second exact variant, measured and labelled, NOT folded into the fair expert** (spec §6 M1: that
decision is the research side's): capturing the whole N-step expert loop as **one** CUDA graph
(`owngraph_expert`) instead of the rung's per-step graph-tree replays gives **101.2 ms**, and its
actions are **bitwise identical** to the rung's expert on all 5 observations
(`results/…_corun_N10_MAXN/owngraph_equivalence.*`). **101.2 ms falls inside row 8's band**
(93.1 < B ≤ 102.4): if the research side admits this variant into the GPU baseline, the in-memory
expert's latency advantage at 16 channels is under 1.10× (93.1 → 101.2 is 1.087×).

## M2 — the corruption: diagnosis, and the fix

**Diagnosis.** T014's compiled pipeline replayed the rung's two graph-tree callables —
`pg.embed_image@reduce-overhead` on the prefix stream and `model.denoise_step@reduce-overhead` on the
expert stream — out of **one** CUDA-graph-tree pool, which torch builds for a single sequential call
stream, and handed the expert a KV cache that the next prefix replay could overwrite. The fix below
removes both conditions and the corruption disappears (bitwise equivalence, stale control detected,
0 non-finite over every run); **which of the two was the operative cause was not isolated** — the
fix was not built one change at a time.

**Fix (labelled, `bench/e12b_corun.py`).** `owngraph_expert`: `denoise_step` compiled `mode="default"`
(the same Inductor kernels, no trees) and the whole hoisted N-step loop captured by the harness as one
`torch.cuda.CUDAGraph` in its **own** pool; `dbuf_kv`: each frame's KV and pad mask are copied on the
prefix stream into one of two expert-owned buffers (k mod 2), prefix(k) waiting for expert(k − 2), the
last reader. The prefix is exactly the rung. `graphs_both` additionally captures the prefix itself
(`embed_image` compiled `mode="default"`, and `modeling_pi0.py:681`'s att_masks conversion memoised,
since a graph cannot contain a host sync) — needed for M3, see below.

**Equivalence** (`equivalence.*` in M2 and M3): 6 frames over the 5 observations with per-frame seeds;
pipelined vs serial bitwise on every frame, stale control 5/5 detected, in the whole-GPU run and in
**every** split.

## Why it does not overlap — the timeline (M2 step 5) and the memory system (M4)

**Whole GPU (nsys, 5 steady co-run frames, `--cuda-graph-trace=node`; ratios above are from
unprofiled runs):**

| per co-run frame | prefix | expert |
|---|---:|---:|
| kernel time, isolated → co-run | 248.84 → 252.45 ms (+3.6) | 102.03 → 106.16 ms (+4.1) |
| gaps inside the stage | 99.1 ms | **177.3 ms** |
| … of which while the other stream is executing | 95.2 ms | **175.3 ms** |

**Kernels of both streams execute together for 54.7 of 1754.9 ms (3.1 %).** The kernels themselves
barely slow (+4 ms); the stages *wait* for each other. The prefix's dominant GEMM
(`ampere_bf16_s16816gemm_bf16_128x128…`, 68 % of its kernel time) launches **1008 CTAs — 63 waves at
one CTA per SM on 16 SMs** (`prefix_grids.csv`), so it holds every SM and the expert's kernels queue
behind it: **head-of-line blocking at the dispatcher, KerColle's mechanism, observed on Orin**. The GPU
is busy for 347.7 ms of each frame (union of both streams) against an unprofiled period of 345.7 —
the co-run is GPU-bound and time-sliced, so no host-side or buffering change can recover overlap the
dispatcher does not allow. Host issue is not the bottleneck: the expert is one graph launch per frame,
and the "issue" time the harness records is launch-queue back-pressure, not host work (the same
caveat as T014's `issue/wall`, which the review rightly rejected).

**Partitioned (M3, M4).** Partitioning removes the queueing — `O` falls from 1.39 to 1.247 at 12/4 —
but the stages still couple through memory. Calibrated EMC bandwidth
(`results/…_emc_calibration_MAXN`: GB/s = 2.005 × EMC% + 0.34, R² 0.99987, hog 56–178.5 GB/s):

| condition | whole GPU | 12 / 4 |
|---|---:|---:|
| prefix alone | 84.6 GB/s | 80.6 |
| expert alone | 80.6 | 40.4 |
| co-run | 84.6 (rung) / 88.6 (`graphs_both`) | **106.6** (vs 121.0 = sum alone) |
| prefix + hog in the expert's partition | — | 128.7 |

Unpartitioned, the co-run draws no more bandwidth than one stage — consistent with time-slicing.
Partitioned, both stages are live at once, they draw **less than the sum of their separate demands**
while each stretches (`s_part` 1.197, `p_part` 1.075), and the total stays well below the 178.5 GB/s
the bus delivered to the hog: **contention inside the memory system (controllers, L2, latency), not
the bus running out of bandwidth.** The positive control confirms the method sees it: the prefix
slows 1.043× beside a 65 GB/s hog in the 4-SM partition, up to 1.116× beside 142.7 GB/s in 10 SMs.

**No split reaches good overlap for a second reason: the confined compiled expert needs SMs.** Alone in
its partition it takes 496.0 / 254.8 / 185.2 / 149.1 / 130.8 ms on 2 / 4 / 6 / 8 / 10 SMs; at 12/4 the
expert (254.8) is nearly as long as the prefix (267.7), so even with zero coupling the period could not
go below ≈ 268 ms (`O` ≈ 1.09). **This corrects T014**, whose "compiled expert SM-independent (4 SMs
cost 4.9 %)" came from graph-tree replays that were not confined (see M0).

## M0 — what a green context actually confines

A 2048² bf16 GEMM as a CUDA graph of 20, in 4 / 8 / 12-SM partitions (`raw/m0_graph_confinement.json`):
eager on the green stream **3.25 / 1.64 / 1.14× slower**; a graph **captured on the primary context**
and replayed on the green stream **0.99 / 0.99 / 0.99×** — it runs on the whole GPU; a graph captured
inside the green context **3.24 / 1.63 / 1.13×**, even when replayed with the primary context current.
**A graph obeys the partition it was captured in, not the stream it is replayed on.** Every M3 graph was
therefore captured inside its own green context, and only `graphs_both` could be partitioned: the
rung's `embed_image` graph tree is recorded outside any partition. The driver granted every split tried
(12/4, 8/8, 10/6, 14/2, 4/12, 6/10, 2/14). A graph captured in a green context must be freed before that
context is destroyed (the first M0 run segfaulted).

## M5 — energy per frame (`results/20260925_E12b_pi0_energy_N10_MAXN`)

60 s idle first, then ≥ 60 s per whole-frame arm, four rails at 20 Hz + tegrastats, idle subtracted as
each rail's in-session idle mean; 0 non-finite:

| arm | ms/frame | module mW | **mJ/frame** | above idle |
|---|---:|---:|---:|---:|
| (a) stock `full` (E7 anchor) | 424.69 | 42 375 | 17 996 | 13 409 |
| (b) fair serial, whole GPU | 347.94 | 50 446 | **17 552** | 13 794 |
| (c) best co-run, green 12/4 | 313.75 | 57 063 | **17 903** | 14 515 |

**Co-running saves 10 % of frame time but draws 13 % more power: (c)/(b) = 1.020 per frame, 1.052 above
idle.** Row 10 does **not** land. (The first M5 run measured (b) through the partitioned harness —
526 ms — and would have reported 0.874 and row 10 true; caught, kept in `scratch/superseded/`, never
committed as a result.) The fair serial frame itself is 2.5 % cheaper than the stock one (17 552 vs
17 996 mJ): the hoist removes 18 % of the frame time at 19 % more power.

## Which pre-registered rows landed (quoted from §10, fixed 2026-09-25)

| # | row | landed? |
|---|---|---|
| 1 | "Some valid configuration (M2, or an M3 split) reaches `O ≤ 1.05`" | **No** — best `O` = 1.247 |
| 2 | "Best `O` > 1.05, and inside partitions `p_part ≥ 1.10` (or `s_part ≥ 1.10`), with the hog control passing" → "Partitioning SMs does not partition bandwidth" | **Yes** — best split 12/4: `s_part` 1.197, `p_hog` 1.043 ≥ 1.03; material coupling in every split (`s_part` 1.16–1.26) |
| 3 | "`p_part`, `s_part` < 1.03 and the hog control passes" | No |
| 4 | "Unpartitioned M2 has `O` > 1.05 while the best M3 split is materially lower" | **Yes, in part** — 1.39 → 1.247: partitioning is the software fix for the queueing, but it does not reach good overlap |
| 5 | "`s > r` (in-session) in M2 or the best split" | **Yes** — M2 `s` 2.676 > `r` 2.455; 12/4 `s` 3.009 |
| 6 | "The corruption proves unfixable" | No — fixed |
| 7 | "M1: the fair expert `B` ≤ 93.1 ms" | **No** — 110.95 ms (the labelled `owngraph_expert`, 101.2 ms, would not either) |
| 8 | "93.1 ms < fair `B` ≤ 102.4 ms" | **No** for the fair expert (110.95); **yes for the labelled `owngraph_expert` variant** (101.2 ms, bitwise identical) — the research side's call |
| 9 | "`hoist` fails its compiled equivalence check" | No — bitwise identical |
| 10 | "Co-run energy per frame ≤ 0.95 × the fair serial `full`" | **No** — 1.020 |

## Mechanisms (`results/20260925_E12b_pi0_corun_N10_MAXN/mechanisms.csv`)

Each a labelled row with its own pipeline equivalence check (all passed) and same-process A / B:

| mechanism | works | result |
|---|---|---|
| own-graph capture (`owngraph_expert` + `dbuf_kv`) | yes | the fix; `O` 1.391 / 1.361 (prefix- / expert-first) |
| `graphs_both` (prefix captured too) | yes | whole GPU `O` 1.379 — no better; needed for partitioning |
| green contexts, both stages co-running | yes | best 12/4 `O` 1.247; five splits |
| stream priority −3 on the prefix | yes (granted −3) | `O` 1.386 — no change |
| **stream priority −3 on the expert** | yes (granted −3) | **`O` 1.287**, `T_pipe` 320.58 ms — the best *unpartitioned* co-run: the expert's kernels dispatched ahead of queued prefix grids, as head-of-line blocking predicts; still poor, still above 12/4 |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | yes | `O` 1.403, `p` 0.997, `s` 0.996 — one hardware queue serialises the streams completely |
| `libsmctrl` (c250928, CUDA 12.6) | partial | builds; TPC-mask tests **pass** on Orin (a TPC-0 mask confined a kernel to SMs 0–1 of 16); GPC query needs the `nvdebug` kernel module (not loaded). Not used for a co-run — green contexts already give verified in-process partitions |
| MPS | no, for this | daemon, server and a client work on this Jetson, but MPS needs two processes |
| two processes | no | two full π0 processes are OOM-killed at load (62 GB unified; ~15 GB each; T014 evidence committed under `mechanisms/evidence/`) |

## Measured / derived / not measured

**Measured:** every stage, serial and co-run time (CUDA events on each stage's stream, start after the
cross-stream wait; unprofiled); equivalence and finite checks; the nsys timeline; EMC utilisation,
memory-rail power; M0 GEMM timings; M5 rail power.
**Derived:** `O`, `φ`, `p`, `s`, `r`, `p_part`, `s_part`, `p_hog` (quotients of measured medians);
calibrated GB/s (the linear fit); energy per frame (mean power × period).
**Not measured:** the exact operative cause of T014's corruption (pool sharing vs KV overwrite — fixed
together); nsys `--gpu-metrics-devices` DRAM rows (not attempted); a partitioned run with the rung prefix
(impossible: its graph tree escapes the partition); ncu per-stage DRAM bytes (E6's analytic bytes used).

## Self-review against §12

1. **Gate on every run, MAXN, `throttled` recorded** — met. Every `meta.json` carries the protocol §1
   gate plus the EMC rate and thermal zones, and `throttled` is *measured* (GPU clock at start vs end,
   peak thermal zone), false in every run.
2. **M1 loops bracketed, ≥ 60 s / ≥ 100 frames; hoist `equivalence.*`** — met.
3. **M2 compiled co-run, equivalence passed with the stale control detected; in-session `A_iso`,
   `B_iso`, `T_serial`; n ≥ 100; `s`, `p`, `r`, `O`, `φ`; host-issue record; nsys CSVs +
   `corun_breakdown.csv`** — met (n = 120, and 174 in the ≥ 60 s loop). In-session `A_iso` 248.43 is
   1.1 % under E7's 251.28. **`B_iso` 101.21 is 8.8 % under the 111.02 reference, and that is a harness
   change, not drift**: it is the own-graph expert (whole N-step loop as one graph), bitwise identical to
   the rung expert, labelled `owngraph_expert` in every row; against M1's stock-harness fair expert the
   M2 headline is `O` 1.381.
4. **M3 ≥ 3 splits with `A_part`, `B_part`, co-run (check passed), `p_part`, `s_part`, `T_pipe,part`,
   `O`; graph-confinement control; hog ≥ 2 rates** — met: five splits, M0 in `raw/`, hog `max` and
   `matched` per split (in the 4- and 2-SM partitions the two rates coincide, because a small partition
   cannot drive DRAM harder than 65 / 33.6 GB/s — stated rather than hidden).
5. **M4 `bandwidth.csv` for isolated, co-run and hog, with the calibration and which sources worked** —
   met. Worked: tegrastats `EMC_FREQ` (calibrated, R² 0.99987), the memory rail, the nsys kernel
   timeline. **Not attempted: nsys `--gpu-metrics-devices`.** M4 runs are separate from the timing runs
   (so no headline timing carries a sampler); the spec asked for tegrastats *alongside* every timed run —
   a deliberate deviation, stated here.
6. **M5 three whole-frame arms + idle in one session** — met, after a correction: the first M5 run's arm
   (b) went through the partitioned harness and would have landed row 10 falsely; it was caught,
   replaced, and never committed as a result.
7. **No non-finite action in any run reported as valid** — met: 0 in every run, every row `valid = true`.
   No run was invalid; the superseded M5 run is listed in its `meta.json`.
8. **`mechanisms.csv` covers every §11 mechanism; no null in NN4's `meta.json` fields; every row labelled**
   — met; `write_meta` refuses to write a null in those fields.
9. **§11 self-review; no file over 50 MB; repository size in `STATUS.md`; research-side files untouched;
   T014's status line** — met (largest file 9.0 MB, the nsys GPU trace CSV).

**What this report corrects in my own earlier work.** T014's compiled green-context rows were not
confined (M0), so its "compiled expert SM-independent" claim and the derived 12/4 bound of 271.26 ms are
withdrawn; its "the expert can neither block nor be blocked at the CTA dispatcher" is contradicted by
M2's timeline. Both corrections are at the top of `STATUS.md` and in `reports/E12_pi0_corun.md` §5.

**Not tuned toward an outcome.** The splits, orders and loop lengths were fixed before seeing their
results except 6/10, added after the first split showed the confined expert needs SMs — which could only
*improve* overlap, and did not.
