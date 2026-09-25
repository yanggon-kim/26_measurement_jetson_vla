# E12 / T014 — research-side review of the MAXN results (2026-09-25)

*jetson-owner, on the research server. Reviews the device's `reports/E12_pi0_corun.md`, the T014 status
lines and every committed row under `results/20260925_E12_pi0_*` against the spec as pre-registered at
`88ad91a` (`docs/E12_corun/README.md`). Pulled at device commit `b97da62`. Where the report and the rows
disagree, the rows are taken. **Measured** = a committed row; **derived** = arithmetic on committed rows,
never measured as such. This file is analysis, not an instruction to the device.*

> **⚠ CORRECTION 2026-09-25 (later the same day), from E12b / T015 — see `analysis/E12b_review.md` §5.**
> T014's **compiled (`bf16_both`) green-context rows were not confined to their partitions.** E12b M0
> (`results/20260925_E12b_pi0_green_corun_N10_MAXN/raw/m0_graph_confinement.json`) measured that a CUDA graph
> obeys the partition it was *captured* in, not the green stream it is replayed on (primary-captured graph on a
> 4-SM stream 0.99×; captured in-partition 3.24×), and T014's harness (`bench/corun_green.py` @ `de6afa3`)
> recorded the rung's CUDA-graph trees on the primary context before timing on green streams. Device
> correction: `592e2da` (`STATUS.md` top; `reports/E12_pi0_corun.md` §5). Confined properly, the compiled
> expert on 4 SMs is **254.79 ms, 2.51× its 16-SM time**, not 1.049×.
> **Withdrawn in this review:** the `bf16 prefix` / `bf16 expert` columns of the SM-sensitivity table (§1);
> "numerics on an 8-SM green stream bitwise identical" for `bf16_both` (a whole-GPU replay compared with
> itself); the derived 12/4 period **271.26 ms = 1.080 × A_iso** and everything built on it in §6 (framing I
> "nearly", ≤ 1.08× PIM frame-rate gain, "the expert's SM demand is tiny", "freed GPU ≈ 8 %"). **Reversed:** §6's
> lean toward framing I — the measured best GPU co-run is `O` = **1.247** (green 12/4, 307.15 ms), whole GPU
> 1.29–1.40, with head-of-line blocking observed unpartitioned and coupling with the SMs partitioned; the
> evidence now sits in case 001's framing II, measured on Orin. **Still valid:** everything eager (eager launches
> are confined), §3, §4, §7, and the latency arithmetic of §6 (now superseded by measured values:
> fair expert 110.95 ms, fair serial frame 354.74 ms).

## 1. Numbers (all MAXN + `jetson_clocks`, 16 SM, π0 LeRobot `f0d2b37` / ckpt `45dcc8f`)

Recomputed from `latency_samples.csv` (median per condition; isolated = mean of the two bracket medians).
They reproduce the device's `corun_ratios.csv` to the last printed digit.

| run (dir suffix) | rung / variant | N | A_iso | B_iso | T_serial | A_co | B_co | T_pipe | s | p | r | T_pipe/A_iso | T_ser/T_pipe | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `corun_N10_MAXN_eager_nosync` (bench `5dc228e`) | eager + att_masks memo + timestep hoist | 10 | 474.08 | 523.76 | 994.38 | 473.17 | 521.44 | **994.88** | 0.9956 | 0.9981 | 0.905 | 2.099 | **0.9995** | 100 |
| `corun_N4_MAXN_eager_nosync` (bench `de6afa3`) | same | 4 | 475.17 | 192.71 | 665.63 | 474.08 | 192.39 | **666.77** | 0.9983 | 0.9977 | 2.466 | 1.403 | 0.9983 | 100 |
| **`bf16_both` co-run** | — | 10 | — | — | — | — | — | **none** | — | — | — | — | — | — |

`T_pipe` = `A_iso + B_iso` within 0.3 % at both N: **no overlap**. `max(A,B)` would have been 523.76 / 475.17 ms.

`bf16_both`, isolated only (no co-run exists):

| source | A (prefix) | B (expert) | r | note |
|---|---:|---:|---:|---|
| E7 (`20260924_E7_pi0_N10_bf16_both_MAXN`) | 251.28 | 167.36 | 1.50 | shipping π0, per-step timestep tensor |
| `2proc_N10_MAXN`, isolated brackets only (run FAILED, OOM) | 251.77 | 111.08 | 2.27 | timestep hoisted; `_run_outcome` says use no number — these isolated brackets are the ones the report quotes |
| reconciliation (`scratch/NOTE_e7_ae_time_discrepancy.txt`) | — | 167.09 stock / 111.02 hoisted | — | n = 270 / 406; 56.07 ms = **33.6 %** of B |
| `green_N10_MAXN_bf16both`, 16 SM | 251.10 | 111.61 | 2.25 | timestep hoisted, unlabelled in the rows |

Green-context SM sensitivity (`sm_sensitivity.csv`, each stage **alone** in a partition; medians, n = 30–180).
**The two `bf16` columns are WITHDRAWN (correction above): the compiled graphs ran unconfined. Confined
values (E12b M3, `graphs_both` prefix / `owngraph_expert`): 4 SM expert 254.79; 8 SM 326.94 / 149.05;
12 SM prefix 267.71; 16 SM 246.28 / 101.36.** The eager columns stand.

| SMs | eager prefix | eager expert | bf16 prefix | bf16 expert |
|---:|---:|---:|---:|---:|
| 4 | 1425.07 (2.98×) | 588.64 (1.013×) | 491.60 (1.96×) | 117.05 (1.049×) |
| 8 | 779.79 (1.63×) | 587.56 (1.012×) | 320.71 (1.28×) | 113.70 (1.019×) |
| 12 | 576.50 (1.21×) | 582.99 (1.004×) | 271.26 (1.08×) | 112.30 (1.006×) |
| 16 | 478.08 | 580.79 | 251.10 | 111.61 |

Partition control: 2048² bf16 GEMM 3.245× / 1.637× slower at 4 / 8 SM (ideal 4 / 2). Numerics on an 8-SM
green stream: bitwise identical to the default stream (*eager only* — for `bf16_both` the replay was unconfined
and compared with itself; withdrawn, see the correction above).

Energy (`energy_N10_MAXN_eager_nosync`, eager only, 60 s idle + 60 s per arm, 20 Hz rails, n ≈ 2270 samples/arm,
idle subtracted as each rail's in-session 60-s idle mean): pipelined 1031.86 ms, 35 546 mW, **36 678.7 mJ/frame**;
serial `full` 1035.31 ms, 35 518 mW, **36 772.5 mJ/frame**; ratio 0.9975. No `bf16_both` energy exists.

## 2. Compliance with the spec (`88ad91a`)

| item | status |
|---|---|
| NN1 fresh KV, eager | **passed**: pipelined vs serial 6/6 bitwise (max \|Δ\| 0); stale control (expert given frame k−2's cache) detected 5/5, max \|Δ\| 2.038 vs frame-to-frame 5.47; 5 distinct LIBERO frames + per-frame seed |
| NN1 fresh KV, compiled path | check **ran** through the compiled path and **has power** there (control 5/5 detected); the compiled pipeline itself **failed** (max \|Δ\| 1.273 one run, all-NaN the next, silent under `TORCH_LOGS=cudagraphs`). Correctly, no compiled co-run number was reported |
| NN1 deliverable | **not met**: no `equivalence.md` / `equivalence.csv` / stdout in any result dir; the evidence is the report and `scratch/NOTE_e12_*` only |
| NN2 baseline `bf16_both` | **deviation**: Step 1/2/5 ran **eager**. Two harness changes: att_masks memo (labelled `pi0_nosync` in `meta.json`, but the rows say `mechanism = baseline_eager`) and the timestep hoist (unlabelled until the E7 reconciliation, still in no row or meta field, also present in the green-context runs). The spec's own remedy for the compiled corruption — own `torch.cuda.CUDAGraph` capture, separate pools (§7) — was **not attempted**; the device judged it "uninformative" |
| NN3 same session, bracketed | met; drift 0.6 % prefix / 0.7 % expert (N=10), 0.6 / 0.8 % (N=4). E7 comparison only possible for `bf16_both` isolated, where it reconciled (§4) |
| NN4 provenance | partial: corun dirs `raw_files: []`, no `raw/`, no `summary.md`; `T_pipe`'s wall clock is not recoverable from any committed row; green dirs' `meta.json` has `bench_commit`, `power_mode`, `sm_count`, `throttled` all **null** (the CSV carries the mode); MPS probe log (`scratch/mps_probe_control.log`) **not committed** |
| NN5 power modes | met at MAXN (gate block, clocks pinned, `throttled: false`); Step 4 queued for the MODE_30W session |
| nsys timeline of 3–5 co-run frames (acceptance) | **not taken** — the device argued there is no `B_co − B_iso` to decompose. But the timeline is also what would have shown *why* nothing overlaps (see §3) |
| Step 3 mechanisms | stream priority −3: **not measured**. Graphs on both stages: **not measured**. `libsmctrl`: **not attempted**. `CUDA_DEVICE_MAX_CONNECTIONS`: skipped with a reason. MPS: daemon + client probed working, not run (argued moot by memory), evidence uncommitted. Two processes: **OOM** at model load, dmesg evidence in `scratch/NOTE_e12_two_process_oom.txt` + `_run_outcome`. Green contexts: **works on CUDA 12.6**, but measured per stage in isolation only — **never two stages concurrently**. No `mechanisms.csv` |

The device's commit `03bbf11` says "T014's MAXN scope is now complete". Against §11 it is not: the `bf16_both`
co-run, the timeline, four Step-3 mechanisms and the `equivalence.*` / `mechanisms.csv` deliverables are missing.
The report's own "Not measured" list (§5, near the end) is stale and contradicts its §5 / §5b.

## 3. Where the narrative and the rows disagree

1. **`issue/wall = 1.0000` does not measure what the report says.** In `bench/corun.py::run_pipeline`,
   `issue_s` is the time to the end of the issue loop and `wall_s` adds the final `synchronize()`. Over a 99.5-s
   loop that ratio is ≈ 1 whenever the GPU tail after the last launch is short — it would read ≈ 1 for a
   GPU-bound loop too (launch-queue back-pressure is counted as "issuing"). It cannot distinguish host-bound
   from GPU-bound. The **measured** fact is `T_pipe = A_iso + B_iso`. The *why* is an inference that the rows
   support only in part: the eager expert is host-issue-bound (E5 84 % idle; SM-flat curve), and the loop issues
   **expert(k−1) first**, so prefix(k) is not even enqueued until the expert's ~B ms of issue has finished.
   Prefix-first ordering, or interleaving, was not tried. By the spec's own pre-registered N=4 row ("suspect the
   harness first"), the one-thread / issue-order harness has **not** been cleared.
2. **"The expert can neither block nor be blocked at the CTA dispatcher."** Measured: each stage alone in a
   partition. The expert needing few SMs shows it will not block the prefix much and loses ≤ 4.9 % inside a
   4-SM partition. It does **not** show that, *unpartitioned*, its small grids cannot queue behind prefix grids
   that hold every SM slot — that is KerColle's mechanism, and it concerns the waiting kernel's access to a free
   slot, not its size. The prefix grid-size-vs-capacity table §5 Step 1 asked for is absent.
3. **Row labels.** Report §6: "`pi0_nosync` has its own mechanism name in every row". The rows say
   `baseline_eager`; only `meta.json` says `pi0_nosync`. The timestep hoist is labelled nowhere in the data.
4. **Cross-validation 16-SM eager expert 580.79 vs `B_iso` 523.76** is attributed to `pi0_nosync` alone. Plausible
   (the green harness hoists timesteps but does not memoise att_masks), not separately measured.

## 4. The host-sync finding and the E7 reconciliation

- **Sync sites (measured with `set_sync_debug_mode("warn")` + a controlled swap):** `modeling_pi0.py:681`
  (att_masks list → tensor, once per prefix), `:746` (once per denoise step), `:858-859`
  (`torch.tensor(<float>, device="cuda")` timestep, once per step — pageable H2D, host blocks until the stream
  drains). 1 + 2N per frame.
- **Removed:** att_masks memoised on contents (eager-only: under `bf16_both` it raises "accessing tensor output of
  CUDAGraphs that has been overwritten"); timesteps precomputed once. Under `bf16_both` the `:746` conversion sits
  inside the compiled `denoise_step` and costs nothing measurable (2proc expert with `nosync: false` = 111 ms);
  the timestep one is the expensive site.
- **Artefact or deployment property?** The syncs are in the **shipping LeRobot π0**, so any two-stream deployment
  driven by one host thread must remove them — not a harness artefact. openpi's PyTorch port has the same class
  of per-step host sync (`while time >= -dt/2` on a CUDA tensor, `pi0_pytorch.py:407`, openpi `15a9616`); openpi's
  JAX model (`jax.lax.while_loop` under `jit`) has none. Removing them was **not sufficient** in eager: what
  remained is specific to eager Python from one thread (a CUDA-graph replay or a C++ runtime with an issue thread
  per stream would not have it), i.e. not a property of Orin.
- **E7 `ae` reconciled (measured):** 167.09 ms with the per-step timestep tensor vs E7's 167.36 (0.16 %); 111.02
  hoisted. Two alternatives rejected by measurement (sampler thread +0.19 %, prefix residency +0.29 %). E7 measured
  the shipping model correctly; `r = 1.50` is right **for shipping π0**. **The "33.6 % π0 optimisation" = hoisting
  N constant timestep tensors out of `sample_actions`' loop: 56.07 of 167.09 ms at `bf16_both`, N = 10, MAXN.** It is
  a software change to π0. The mechanism (drain, not copy) is inferred; no nsys of the stall exists. Its numeric
  equivalence was verified only inside the eager pipeline check, not on the compiled serial path.

## 5. Mapping onto the pre-registered table (quoted from `88ad91a`, §9)

| row as committed | landed? |
|---|---|
| "**`s < r` (≈ 1.50) at MAXN, N = 10**, prefix barely slowed (`T_pipe ≈ A_iso`)" → "One GPU overlaps the two stages ideally on Orin. **The frame-rate argument for moving the expert off the GPU is gone.**" | **no** — `T_pipe/A_iso` = 2.10 |
| "**`s > r` at MAXN**" → "Contention is real on Orin: the stretched expert sets the period. **The frame-rate claim stands.**" | formally `s` 0.996 > `r` 0.905 in eager, but **the meaning does not apply**: `s ≈ 1`, nothing overlapped, and `r < 1` means the expert is the longer stage without any stretching. The table assumed `r > 1` and overlap; it did not foresee zero overlap. Not claimed |
| "`T_pipe ≥ T_serial`" → "Co-running is worse than not overlapping at all on Orin." | **yes, marginally** (994.88 vs 994.38, +0.05 %; N=4 +0.17 %) — equal within noise; true of eager, one Python thread |
| "N = 4 fails to approach `A_iso`" → "Suspect the harness first (§5 Step 5); if it survives scrutiny, it is a finding about fixed costs of co-running." | **yes** (1.403). It has not survived scrutiny (§3.1): read it as a harness finding |
| "**The prefix slows under co-run (`p` materially > 1)**" | no (0.998) — uninformative without overlap |
| "**A Step-3 mechanism reaches `T_pipe ≈ max(A_iso, B_iso)`**" | **not tested**: no mechanism was run as a co-run |
| "**No SM-partitioning mechanism works on JetPack 6.2 / CUDA 12.6**, with evidence" | **ruled out**: green contexts work on 12.6 (verified confinement + numerics) |
| "**`s` larger at 8 SMs (`MODE_30W`) than at 16**" | open (Step 4) |

**Falsified:** nothing in §9 was falsified by a valid co-run — none exists at `bf16_both`. The one outcome the
evidence rules out is the "no SM-partitioning mechanism" row. Two expectations written *outside* §9 failed:
Step 5's "overlap should be easy here" at N = 4, and §2.3's "their diagnosis … should, if anything, bite
*harder* here" is weakened — the expert's grids fit in 4 SMs. The prior-work predictions ("KerColle → `s` ≈ 2,
`p` ≈ 1; Jetson-PI → `p` > 1 and `s` > 1") remain **untested**.

## 6. What it means for case 001 (§4.3 framings)

> **⚠ Superseded 2026-09-25 by `analysis/E12b_review.md` §6.** The "Derived" bullet below rests on the withdrawn
> unconfined `bf16` curve: the 271.26 ms bound, "framing I, nearly", "≤ 1.08×", "nothing measured points toward
> framing II" and "the expert's SM demand is tiny" are **withdrawn**. Measured since: best GPU co-run `O` =
> 1.247 (green 12/4), whole GPU 1.29–1.40; the PIM's frame-rate increment over the best GPU co-run is ≤ 1.247×
> `[derived]`, before any prefix slowdown from sharing LPDDR with the PIM (the channel-partition measurement, EV-02, unmeasured). The "Freed GPU"
> bullet is withdrawn with it (the confined compiled expert needs ≈ 2.5× its time on 4 SMs). The latency bullet's
> direction holds and is now measured: fair expert 110.95 ms, fair serial frame 354.74 ms, PIM latency gain
> 1.036× (16 ch) / 1.152× (32 ch) `[derived]`.

- **Measured:** GPU-side overlap on Orin at MAXN is **not achieved** with shipping software: eager from one Python
  thread gives `T_pipe = A + B`; `bf16_both` cannot be pipelined on the stock torch 2.8 CUDA-graph-tree path
  (silent corruption), and two full-policy processes do not load side by side. Neither framing I nor II is
  measured; the cell of case 001 that E12 was meant to fill is still empty for `bf16_both`.
- **[WITHDRAWN 2026-09-25 — unconfined compiled graphs; see the correction at the top]** **Derived (not measured), from the green-context curve:** a static 12 / 4 split at `bf16_both` bounds the period
  at max(271.26, 117.05) = **271.26 ms = 1.080 × A_iso**, i.e. 1.337× over the hoisted serial 362.71 ms. If a
  working concurrent issue path realises it and memory coupling is small (the expert uses 6.45 % of peak bandwidth,
  E6), a shipping driver API gets within 8 % of `max(A, B)`: **framing I, nearly** — the PIM would add ≤ 1.08×
  frame rate over it. Nothing measured points toward framing II: no contention was observed anywhere, and the
  expert's SM demand is tiny. Unfavourable direction, equally unmeasured: bandwidth coupling of the co-run (`p`),
  per-launch context switching between two green contexts from one thread, and unpartitioned HoL blocking.
- **Case 001's recommendation (a)** — lead with latency / energy / freed GPU, give frame rate as a range starting at
  1.00× — is **reinforced**; (b), leading with frame rate against KerColle, has no support from Orin.
- **Latency — at risk, if the timestep hoist enters the GPU baseline (derived):** GPU expert 167.36 → 111.02 ms,
  serial frame 424.68 → 362.30–368.34 ms (251.28 + 111.02, plus 0–6.04 ms of E7's unplaced residual; never
  measured as a whole frame). Against case 001 §4's AiM-serial rows (344.33–350.37 ms at 16 ch, 309.98–316.02 at
  32 ch) the PIM latency gain falls from **1.21–1.23× to 1.03–1.07×** (16 ch) and from 1.34–1.37× to 1.15–1.19×
  (32 ch); the overlap ceiling `(A+B)/A` falls from 1.67× to 1.44×. `cropcache` (P2, unapproved) on top could plausibly take the GPU expert below AiM's 93 ms if
  the two savings add (16.7 − 5.6 − 5.9 ≈ 5.2 ms/step; additivity unverified). `bf16_both + hoist + cropcache` is
  the measurement that decides whether the latency claim survives.
- **Energy:** E12 adds nothing either way (eager, no overlap; 0.9975). The E7 30 % residual remains open, so GPU-vs-PIM
  energy is still not quotable.
- **[WITHDRAWN 2026-09-25 — same reason]** **Freed GPU:** qualified. At `bf16_both` the expert needs ~4 SMs; taking it off the GPU buys the prefix ≈ 8 %
  (271.26 → 251.10 ms under partition, derived), not "167 ms of GPU".

## 7. OD-1 — which side does the 33.6 % look like? (not decided here)

It looks like the **baseline** side, the `cropcache` class: an implementation inefficiency in the PyTorch port
(a constant rebuilt per step through a pageable copy), GPU-agnostic, with no change to what is computed; the JAX
reference does not pay it; any deployment engineer profiling the loop would remove it. It is unlike GR00T's head
restructuring, which changes the computation. Against that: it was found by *our* co-run harness, and it is
not in any shipping π0. If it enters the baseline, §6's latency consequence follows; if not, a reviewer will find
it. Before it enters any baseline it needs its own equivalence check on the compiled `bf16_both` serial path and a
whole-frame `full` measurement — neither exists.

## 8. Open

> *2026-09-25, later:* the decisive co-run and the missing deliverables below were taken over by T015 / E12b
> and delivered (`analysis/E12b_review.md` §2); the Step-4 note was acted on by the device (`ed73f56`), but its
> restated compiled references are the withdrawn unconfined values (`E12b_review.md` §8).

- **Step 4** (MODE_30W, 8 SM): queued as `bench/queues/queue_e12_capped.sh`, waits for the capped-mode session
  (the device's business). The device's pre-registered Step-4 prediction scales the **eager** expert by the GPU
  clock ratio; the eager expert is host-bound and MODE_30W also slows the CPU (8 × 1.73 GHz vs 12 × 2.2), so a
  deviation from the GPU-clock ratio would not by itself invalidate the green-context result. Owed as a note.
- **The decisive co-run is unmeasured:** `bf16_both` through the §7 own-`CUDAGraph` harness, then green-context
  12/4 (and 8/8) as a real concurrent co-run, stream priority −3, prefix-first issue order, one nsys timeline.
  Not queued — needs the user's approval.
- **Deliverables missing** (§2): `equivalence.*`, `mechanisms.csv`, `corun_breakdown.csv`, `summary.md`, `raw/`,
  green-dir provenance, MPS log, row labels for both harness changes.
- **Asked of us by the device:** nothing new for E12. Standing: `check_model_inputs.py` `MODELS` (unapproved);
  whether E11's compiled-mode quantities may be sourced from E5.
