# E12b — The compiled co-run: does one Orin GPU overlap π0's prefix and a fair expert, and does partitioning the SMs partition the memory?

*Written by the research side (jetson-owner) on **2026-09-25**, committed and pushed **before any E12b run**.
This file is the experiment's pre-registration: the thresholds and outcome table in §10 were fixed when it was
committed (`git log --follow docs/E12b_compiled_corun/README.md` gives the date and hash). A later change to §10 is
appended under a new date, never rewritten. Task file: `tasks/T015_pi0_compiled_corun.md`. Binding rules:
`docs/02_protocol.md` §15 (this experiment) and §14 (E12, which still applies); where this file is shorter, the
protocol wins. It follows E12 (`docs/E12_corun/README.md`, task T014) and the research side's review of its MAXN
return, `analysis/E12_review.md`.*

---

## 0. In one paragraph

T014 asked whether one Orin GPU can run π0's prefix of frame *k* and its action expert of frame *k − 1* at the same
time at near `max(A, B)`. It did not answer: the only co-run that exists is **eager**, where the expert is
host-bound and nothing overlapped; the **compiled** co-run — the configuration anyone would deploy — silently
produced corrupt or all-NaN actions and was never repaired; and green contexts, which work on this device, were
only ever used to time **one stage at a time**. E12b finishes the job, at MAXN, in five measurements: **M1** the
fair GPU expert (`bf16_both` + the timestep hoist, compiled, equivalence-checked); **M2** a correct compiled
co-run on the whole GPU; **M3** — the decisive one — the compiled co-run with **both stages running together
inside green-context SM partitions**, compared with each stage alone in the same partition, which separates
**memory-side coupling** (DRAM bandwidth, L2, memory controllers) from **SM sharing**; **M4** a bandwidth
attribution of whatever slowdown M2 / M3 show; **M5** energy per frame, whole-frame loops, best co-run against
serial.

## 1. Read this first: the result may go against us, and that is fine

The research side has a stake in the outcome. **You must not.** Do not tune the harness toward contention or
away from it, do not choose splits, orders or loop lengths because they move a ratio in a direction you think
the research side wants, and do not stop searching for a better GPU-side overlap because one has been found
that looks "good enough for the PIM story". If one Orin GPU overlaps the two stages almost perfectly, that is a
**valuable** result and the research side needs it from you now rather than from a reviewer later. If the
expert's memory traffic slows the prefix even with the SMs partitioned, that is equally valuable. **Both
outcomes are publishable.**

Two more things this experiment asks of you, because T014 fell short of them:

- **A harness finding is labelled, in whichever direction it moves the numbers.** If a number turns out to be
  about the CPU, the Python thread, an allocator, a hidden synchronisation or a graph-pool race rather than about
  the GPU, say so, fix it or label it. Every harness change appears in the row label of every row it affects
  (§11), not only in prose.
- **"Done" means the acceptance list of §12, item by item.** T014 was declared "MAXN scope complete" with the
  compiled co-run, the timeline, four mechanisms and two deliverables missing. If an item cannot be met, say
  which and why in the status line; a partial result honestly labelled is worth more than a complete-looking one.

## 2. The problem we face — what T014 established and what it did not

Sources: `reports/E12_pi0_corun.md`, the T014 status blocks, `results/20260925_E12_pi0_*`, and the research
side's review `analysis/E12_review.md` (all numbers below are MAXN + `jetson_clocks`, π0 LeRobot `f0d2b37`,
checkpoint `45dcc8f`, N = 10 unless stated).

1. **No compiled co-run exists.** With the stock `bf16_both` rung pipelined across two streams, the actions were
   corrupt (max |Δ| 1.273 against serial on one run) or **all NaN** (the next run, identical code), silently —
   `TORCH_LOGS=cudagraphs` printed nothing (`scratch/NOTE_e12_compiled_two_stream.txt`). You were right to
   report no compiled number. But the remedy the E12 spec gave (§7: capture the compiled functions yourself with
   `torch.cuda.CUDAGraph`, separate memory pools) was never tried. Both graphed callables of the rung
   (`pg.embed_image@reduce-overhead` on the prefix side, `model.denoise_step@reduce-overhead` on the expert side)
   replay out of **one** CUDA-graph-tree pool that assumes a single sequential call stream; that is the prime
   suspect, not yet a diagnosis.
2. **The eager co-run is not the answer.** `T_pipe` 994.88 ms against `T_serial` 994.38 and `A_iso + B_iso`
   997.83: the two stages ran back to back (`results/20260925_E12_pi0_corun_N10_MAXN_eager_nosync/`). In eager the
   expert is **host-bound** — one Python thread issues both streams, the expert's ~968 launches per step keep it
   busy, and the loop issued expert(k − 1) *before* prefix(k). Compiled, the expert is **memory-bound** (Part C:
   Mem 56.5 %). The bottleneck differs, so "no overlap in eager" says nothing about the deployed configuration.
   The eager harness was also not cleared by E12's own pre-registered N = 4 check (`T_pipe / A_iso` 1.403 where
   the expert has 3.7× slack). And `issue/wall` ≈ 1.0000 cannot tell host-bound from GPU-bound: it is ≈ 1 for any
   loop whose GPU tail after the last launch is short (review §3.1).
3. **Green contexts work on JetPack 6.2 / CUDA 12.6**, with a passing confinement control (a 2048² bf16 GEMM 3.245×
   / 1.637× slower on 4 / 8 SMs) and bitwise-identical actions on an 8-SM green stream. But each stage was timed
   **alone** in its partition: the compiled expert on 4 of 16 SMs was only **1.049×** slower (117.05 vs 111.61 ms),
   the prefix on 12 SMs **1.08×** (271.26 vs 251.10 ms), so a 12 / 4 split *would* give a period of about
   271.26 ms = 1.08 × `A_iso` — **derived, never measured**. The two stages never ran together. **Green contexts
   partition SMs, not LPDDR5 bandwidth, the L2 or the memory controllers**, so the derived 271 ms is blind to
   exactly the effect the research needs to know about.
4. **Shipping π0 forces 1 + 2N host synchronisations per frame**: `modeling_pi0.py:681` (att_masks list → tensor,
   once per prefix), `:746` (once per denoise step), `:858-859` (`torch.tensor(<float>, device="cuda")`, the
   timestep, once per step — a pageable host-to-device copy that blocks the host until the stream drains).
   Any two-stream deployment driven by one host thread must remove them. (A captured CUDA graph cannot contain a
   host synchronisation at all, so capturing a stage removes its syncs by construction or fails loudly.)
5. **The GPU expert we compare against is not yet the fair one.** Shipping `sample_actions` rebuilds the timestep
   tensor inside the denoise loop: **56.07 of 167.09 ms** of the compiled expert (33.6 %). Hoisted (the N
   timesteps built once), the expert is ≈ **111 ms** (111.02 ms, n = 406, E7 reconciliation; 111.61 ms in the
   green-context run). That figure comes from a harness investigation: it has **never had a compiled equivalence
   check** against the stock expert, and no whole-frame (`full`) measurement of the hoisted configuration exists.

## 3. Why it matters — the context you do not have

The research side designs a processing-in-memory (PIM) unit inside LPDDR5 (SK hynix AiM-style) for the action
expert. The proposed system is a two-stage pipeline: the GPU runs the prefix of frame *k* while the expert of
frame *k − 1* runs in memory, each frame with its own KV cache, so the period is `max(A, B)` instead of `A + B`.
The obvious reviewer question is **"why not run both stages on the GPU, on two streams or two partitions, and get
`max(A, B)` for free?"** E12b answers it on real Orin silicon.

- **If one GPU overlaps the two stages at near `max(A, B)`**, the frame-rate argument for moving the expert off the
  GPU is gone. The case for the PIM then rests on **energy** and on **freeing the GPU** for the rest of the robot's
  work, and the paper must say so.
- **If the expert's memory traffic slows the prefix even when the SMs are partitioned**, the finding is
  "partitioning SMs does not partition bandwidth" — the GPU cannot isolate the two stages on a shared LPDDR5, and
  the memory-side design becomes the argument. (For honesty: a PIM inside the same LPDDR5 also occupies memory
  channels, so a contention finding does not automatically favour the PIM either; that is a separate,
  unapproved measurement — the channel-partition measurement (EV-02) — and not yours here.)

**What changed since E12, and why the decisive number moved.** With the stock expert, `r = A / B` ≈ 1.50, so the
question was whether contention stretches the expert past the prefix (`s > r`). With the fair (hoisted) expert,
`r` ≈ 2.25: the expert would have to slow more than 2.25× before it set the period. The period is now almost
certainly set by the **prefix**, so the decisive quantities are **`T_pipe / max(A, B)`** and the prefix's slowdown
**`p`** — and, inside partitions, whether `p` comes from losing SMs (visible when the prefix runs alone in its
partition) or from sharing the memory system (visible only when both run together). `s` is still reported; it
is no longer the headline.

**For the record, what prior arithmetic suggests (derived, not a prediction you should confirm).** From the
matched-rung E6 bytes (`results/20260925_E6_rungs/pi0_bf16_both/roofline.csv`): the VLM prefix is analytically
compute-bound (4.09 TFLOP, 62.7 % of measured peak compute at 192.86 ms) and averages ≈ 26 GB/s; vision ≈ 15 GB/s;
the expert moves 0.638 GB per step, ≈ 57 GB/s at ≈ 11.1 ms per hoisted step. The average demands sum to well under
half of the 187.47 GB/s measured peak, which would suggest little *average-rate* coupling — but memory-bound
expert kernels colliding with the prefix's memory-heavy kernels can still contend in bursts. KerColle (desktop GPUs
in a simulator) predicts head-of-line blocking of the action kernels behind the VLM's grids when unpartitioned;
Jetson-PI (roofline only) predicts both stages slow on the shared LPDDR5. Nobody has measured either on Orin.

## 4. Definitions (use these names in files and in the report)

Everything from protocol §14 / E12 §3 carries over (`A_iso`, `B_iso`, `A_co`, `B_co`, `T_serial`, `T_pipe`,
`s = B_co / B_iso`, `p = A_co / A_iso`, `r = A_iso / B_iso`; stage times from CUDA events on the stage's own
stream, start after any cross-stream wait; all from unprofiled runs). New here:

| symbol | meaning |
|---|---|
| **fair expert** | `bf16_both` + **timestep hoist** (the N timestep tensors built once and reused; nothing else changed). Defined by M1; used in M2–M5 once its equivalence check passes |
| `A_iso`, `B_iso`, `T_serial` | as E12, for the **fair** configuration, full GPU (16 SMs), through the **same harness and launch path** as the co-run it is divided into |
| `A_part(n)`, `B_part(n)` | prefix / expert **alone** in a green-context partition of `n` SMs (the other partition idle), same harness |
| `A_co,part`, `B_co,part` | the same stages **co-running**, each in its own partition of a split |
| `T_pipe,part` | the steady-state period of the partitioned co-run |
| `p_part` | `A_co,part / A_part` — the prefix's slowdown from the other stage **beyond** the loss of SMs: memory-side coupling (plus anything the harness adds — which is why §8 asks you to rule the harness out) |
| `s_part` | `B_co,part / B_part` — the same for the expert |
| **`O`** | `T_pipe / max(A_iso, B_iso)` (or `T_pipe,part / max(A_iso, B_iso)`) — the **overlap ratio against the full-GPU isolated stages**: 1.00 is ideal, `T_serial / max(A_iso, B_iso)` is no overlap at all |
| `φ` | `(T_serial − T_pipe) / (T_serial − max(A_iso, B_iso))` — the fraction of the available overlap gain achieved |
| `p_hog` | prefix time in its partition with a **bandwidth hog** running in the other partition ÷ `A_part` — the positive control of M3 |

Always report `O`, `p`, `s`, `r` and `φ` together, with the partition sizes the driver actually granted.

## 5. Non-negotiables

These five are the whole of the rules. Everything else is guidance, and §9 says what is yours to decide.

1. **Fresh KV per frame, verified in the compiled path that is timed.** Each frame's expert reads that frame's own
   KV. Verify it exactly as E12 §6 prescribes — distinct observations and a distinct noise seed per frame (the five
   LIBERO frames you captured for T014 are fine), pipelined against serial frame by frame, **through the timing
   harness itself** — for the compiled full-GPU co-run (M2) and for **every** green-context split whose numbers you
   report (M3), **plus a deliberately stale control that must fail** in the same path. No check, no number.
2. **No silent NaN.** Every action tensor produced in every timed, checked or energy run is tested for
   finiteness. Do it without adding a host synchronisation to the timed loop (e.g. accumulate a device-side count
   of non-finite elements per frame and read it after the loop). Any non-finite value **invalidates the run**:
   its timings are kept, marked `valid = false` with the reason, and **reported**, never dropped or silently
   re-run.
3. **Same session, same harness, bracketed.** Every ratio's denominator (`A_iso`, `B_iso`, `T_serial`,
   `A_part`, `B_part`) is re-measured in the same session and power mode as the co-run it divides, through the same
   launch path, and the co-runs are bracketed (isolated → co-run → isolated) so drift is visible. State each
   session's `A_iso` / `B_iso` next to the reference values of §2 and explain a difference above ≈ 3 %.
4. **Complete provenance, and every harness change in the row label.** Per protocol §5, §3.5, §6 and §15: one
   result directory per run; `meta.json` with **no null** in `bench_commit`, model and checkpoint commits, rung,
   variant label, `power_mode`, `sm_count`, the granted SM partition sizes, GPU and EMC clocks and `throttled`;
   per-frame raw samples; `raw/` committed and listed in `raw_files`; `equivalence.md` / `.csv` with the stdout;
   `mechanisms.csv`; nsys CSV exports (never the `.nsys-rep` / `.sqlite`). Every row carries the variant label of
   §11 naming **every** harness change behind it (T014's rows said `baseline_eager` over two modifications).
5. **MAXN only, and nothing unapproved.** Everything here runs at MAXN + `jetson_clocks` (protocol §1 gate). No
   power-mode change, no reboot request. T014 Step 4 stays its own capped-mode item (§14 below has a note on it).
   The channel-partition measurement (EV-02), P2 (`bf16_both + cropcache`) and the `check_model_inputs.py` `MODELS`
   change are **not approved** (§13).

## 6. The measurements

Order: M0 → M1 → M2 → M3 → M4 alongside M2 / M3 → M5. M1's equivalence result gates M2–M5.

### M0 — Session gate and the tools you will lean on

- Protocol §1 gate (MAXN, 16 SMs, `jetson_clocks`, `cur_freq`, EMC rate), `micro_peak.py` same-session bandwidth
  peak (answer 5 of `docs/04_research_answers.md`), free memory. A process holding the prefix and expert graphs,
  double-buffered, needs more memory than T014's; record the peak.
- **Confinement through the launch path you will time.** T014 verified green-context confinement with eager
  launches. Repeat the GEMM control **as a CUDA-graph replay on a green-context stream** (and on the primary
  context for comparison): the co-runs of M3 will replay graphs, and a graph captured on one context and launched
  on another may or may not be confined. If a replayed graph is not confined, find a launch path that is (capture
  on the green stream, `cuCtxFromGreenCtx` + `cuCtxSetCurrent` around capture and launch, …) and say which.
  **Record the partition sizes the driver actually grants** (T014 got (4,4,4,4), (6,6), (8,8), (10,), (12,) and
  the remainder); the granularity and the minimum are part of the result.

### M1 — The fair GPU expert

Compiled `bf16_both` (exactly as `results/20260924_E7_pi0_N10_bf16_both_MAXN/meta.json` defines it), N = 10, batch
1, two variants: **`stock`** (shipping `sample_actions`, per-step timestep tensor) and **`hoist`** (the N timestep
tensors built once and reused — `bench/corun.py::timestep_tensors` already does this; nothing else changes).

- **Equivalence, compiled path (first):** `hoist` against `stock`, both compiled, same observation and fixed noise,
  on ≥ 5 distinct observations × seeds; max absolute and relative difference, bitwise yes / no. Because the timestep
  values are identical, bitwise identity is the expectation; if it is not bitwise, give the `stock` path's own
  run-to-run difference with the same inputs as the floor, and explain the difference. Deliver it as
  `equivalence.*` in the M1 directory.
- **Timings:** for each variant, the E7 loops — `prefix` (A), `ae` (B), and the whole-frame `full` (T_serial) —
  back to back, ≥ 60 s each (or ≥ 100 frames after ≥ 20 warm-up, whichever is longer), bracketed
  `stock → hoist → stock` so drift is visible. Put `stock` next to E7 (251.28 / 167.36 / 424.68 ms) and `hoist`
  next to the 111.02 ms reconciliation.
- The **fair expert** of M2–M5 is `hoist`, and only if its equivalence check passes. If it does not, stop the
  co-run measurements, report the difference and put a question in `STATUS.md`; M3's hog control (which needs no
  co-run) and M5's serial arms may still run.
- If you find another exact, GPU-agnostic inefficiency of the same class while doing this, measure it as its own
  labelled variant and report it; **do not fold it into the fair expert** — that decision is the research side's.

### M2 — A correct compiled co-run on the whole GPU

1. **Diagnose the corruption before fixing it.** Two or three lines with evidence: which buffer is overwritten or
   freed, by which replay, on which stream (the CUDA-graph-tree pool shared by `embed_image` and `denoise_step` is
   the prime suspect; you already refuted "late cloning"). A fix without a diagnosis is acceptable only if the
   diagnosis resists a reasonable effort — then say what you tried.
2. **Fix it.** The E12 remedy is your own `torch.cuda.CUDAGraph` capture of the compiled functions per stage, each
   stage with its **own memory pool**, and **double-buffered stage I/O**: prefix(k) must never write the static
   buffer that expert(k − 1) is still reading (two captured prefix graphs with alternating output buffers, or a
   device-to-device copy of the KV into an expert-owned buffer that is counted in the stage time and labelled —
   your choice). You may find a better remedy; any remedy is a labelled harness change with its own isolated times
   and its own equivalence check. The **closest-to-baseline** form is preferred as the headline: the expert graph
   replayed from your own capture and the prefix exactly as the rung compiles it; capturing the prefix too
   (`graphs_both`) is a labelled variant — E4c measured `vlm_graphs` a small loss in isolation (197.17 vs 192.76 ms),
   which is exactly why it must be judged under co-run.
3. **Measure**: in-session `A_iso`, `B_iso`, `T_serial` through the new harness; the steady-state pipeline (≥ 20
   warm-up, ≥ 100 timed frames, plus one ≥ 60 s loop for `T_pipe` as throughput); `s`, `p`, `r`, `O`, `φ`,
   `T_serial / T_pipe`; and `O` also against M1's stock-harness fair `A` / `B` (the deployment comparison), labelled.
4. **Show the host is not the bottleneck** — the defect that made T014's eager co-run uninformative. Record, per
   frame, the host time spent issuing (from the start of the frame's issue to the return of its last launch call)
   next to `T_pipe`; with graph replays it should be a small fraction. Report the issue order (prefix-first or
   expert-first) and, if cheap, the other order. Do not use `issue/wall` as evidence (§2 item 2).
5. **One Nsight Systems capture** of 3–5 steady-state co-run frames and of each stage isolated (protocol §3.1;
   `--cuda-graph-trace=node`). From it: the fraction of the frame in which both streams have kernels executing;
   the decomposition of `B_co − B_iso` and of `A_co − A_iso` into kernel-duration inflation, inter-kernel gaps
   while the other stream occupies the GPU (queueing — KerColle's head-of-line blocking) and host issue delay; the
   prefix's grid sizes against what 16 SMs hold at once (E12 §5 Step 1 asked for this table; it is still owed).
   Timings for every ratio come from the unprofiled runs, never from this capture.

### M3 — The decisive one: both stages together, inside green-context partitions

For **each split** you run (prefix SMs / expert SMs; at least three, including 12 / 4 and 8 / 8, and one more
chosen from what the driver grants — 14 / 2, 10 / 6 or others if available; the choice and order are yours):

| measurement | what it gives |
|---|---|
| prefix alone in its partition → `A_part` | the cost of losing SMs, nothing else running |
| expert alone in its partition → `B_part` | the same for the expert |
| both together, each in its partition → `A_co,part`, `B_co,part`, `T_pipe,part` | the co-run |
| **prefix in its partition + a bandwidth hog in the expert's partition** → `p_hog` | the positive control (below) |

Derive per split: `p_part`, `s_part`, `O` (against the full-GPU `A_iso`, `B_iso` of the same session), `φ`, and the
best split (lowest valid `T_pipe,part`). **`p_part` and `s_part` are the heart of the question**: SMs are
partitioned, so what remains above 1 is coupling through the memory system (or the harness — rule it out: the
host-issue record of M2 step 4 applies here too, and switching between two green contexts from one thread is a
candidate cost to look for).

**The positive control is required, because a null result is only meaningful if the method can see contention.**
Run a memory-bound copy kernel as the hog, **in the same process, on the expert's green-context stream**, replayed
back to back — not a separate process (two CUDA contexts without MPS time-slice rather than co-run on this device,
so a separate-process hog would not overlap the prefix at all; E11e's `bench/bw_hog.py` process is not this
control). Run it at least at (a) a rate matched to the fair expert's average DRAM demand (≈ bytes per step ÷ step
time; M4 gives the bytes) and (b) the highest rate its partition can drive; record each rate achieved alone. If
`p_hog` stays < 1.03 even at the highest rate, the prefix is insensitive to bandwidth at this level — say so: it
makes a `p_part` ≈ 1 a meaningful null rather than an unexamined one.

Optional, if cheap: the prefix unconfined on the primary context with only the expert in a small green partition
(overlapping resources — say whether the driver allows it).

### M4 — Attributing any slowdown to the memory system

For isolated A, isolated B, the M2 co-run, the best M3 split and the hog control, log what this device will give
you of DRAM / EMC activity, and say which sources worked:

- **tegrastats `EMC_FREQ`** utilisation (the `x%@3199` field) at its fastest interval, alongside every timed run
  of M2 / M3 (tegrastats started and stopped per protocol §4; `pgrep tegrastats` empty before a timed run).
  **Calibrate it** once against the hog at known GB/s (a few rates) so a utilisation percentage converts to GB/s;
  without calibration it is only a relative number.
- The memory rail `VDDQ_VDD2_1V8AO (regulator input)` above idle, per protocol §4.
- nsys: per-kernel duration inflation of the prefix's memory-heavy kernels vs its compute-heavy ones between
  isolated and co-run (the same kernel names in both traces). Try nsys GPU-metrics sampling
  (`--gpu-metrics-devices`) on this Orin; record whether it is supported and, if so, the DRAM-throughput rows.
- ncu, if you use it: remember it **serialises kernels**, so it can give each stage's DRAM bytes and per-kernel
  throughput **in isolation** (a check on E6's bytes) but can never observe a co-run. One filtered range only
  (protocol §3.2).

Deliver `bandwidth.csv` per co-run directory (§11). The report says, for each slowdown larger than drift, whether
the evidence points to DRAM bandwidth, to SM sharing, to queueing, or to the host — or that it cannot tell.

### M5 — Energy per frame, whole-frame loops

The E7 phase-loop energies do not add up (`prefix` + `ae` 418.64 ms vs `full` 424.68; 9 509 vs 13 666 mJ above
idle — the open ~30 % accounting gap), so compare **whole-frame loops only**. In one session: 60 s idle, then
≥ 60 s each of (a) serial `full`, `stock` (the E7 anchor), (b) serial `full`, fair configuration through the co-run
harness, (c) the best **valid** co-run configuration (M2 or the best M3 split, whichever has the lower `T_pipe`),
with all four rails from sysfs at ≥ 10 Hz (E7 used 20 Hz) and tegrastats alongside. Energy per frame = mean power ×
period, total and above idle, per rail and the four-rail module sum (and the three-rail sum without the memory
rail). State how the idle baseline was subtracted. Every action tensor in these loops passes the finite check.

### Optional, only if cheap (each a labelled row in `mechanisms.csv`)

Stream priority −3 for the prefix's stream and, separately, for the expert's (read back `stream.priority`);
`CUDA_DEVICE_MAX_CONNECTIONS` (1 vs default); `libsmctrl` on CUDA 12.6 (build it; record the outcome); the N = 4
co-run as the harness sanity check E12 intended. MPS needs two processes, and two full π0 processes are
OOM-killed at load (T014), so it is recorded as unavailable with that evidence unless you find a way.

## 7. The equivalence and finite checks, in detail

- **Pipeline equivalence** (NN1): E12 §6 in full — ≥ 3 materially different observations cycled with a seed per
  frame; the serial frame-to-frame difference reported as the scale; pipelined vs serial per frame (max abs, max
  rel, bitwise yes / no); the stale control (expert(k − 1) handed another frame's KV) run through the same check
  and **detected**. Through the timing harness (a flag that saves the actions), for M2 and for each M3 split you
  report. Expectation: the same captured graphs replayed serially or pipelined give bitwise-identical actions; a
  difference is explained or the configuration is invalid.
- **Hoist equivalence** (M1): compiled `hoist` vs compiled `stock`, above.
- **Finite check** (NN2): in every timed, checked and energy run, including isolated and serial ones.
- `equivalence.csv` columns: `variant, path (compiled|eager), check (hoist|pipeline|stale_control), split, frame,
  obs_id, seed, max_abs_diff, max_rel_diff, bitwise_identical, n_nonfinite, verdict`. `equivalence.md` states
  what was compared, the power of the control, and the verdict per variant / split.

## 8. Hazards we can foresee

- **The graph-pool race** (§2 item 1). Two replays on two streams from one pool, or one static output buffer read
  by the expert while the next prefix replay writes it, give exactly the non-deterministic corruption T014 saw.
  Separate pools and double-buffered stage I/O are the defence; the stale control and the finite check are the
  alarm.
- **Cross-stream lifetimes**: a tensor produced on the prefix stream and read on the expert stream needs a
  cross-stream event and `record_stream` (or a held reference) so the caching allocator does not reuse it early.
- **cuBLAS on a green-context stream** needs the green context current (`cuCtxFromGreenCtx` + `cuCtxSetCurrent`;
  T014 found `ExternalStream` alone gives `CUBLAS_STATUS_EXECUTION_FAILED`). Switching the current context twice per
  frame from one thread is a cost to measure, not to assume away — and cuBLAS / Inductor may pick different kernels
  when capturing under a context that reports fewer SMs; if the kernels differ between the full-GPU and the
  partitioned graphs, say so (the equivalence check then matters twice).
- **Graph confinement** (M0): confirm, do not assume, that a replayed graph stays inside its partition.
- **Synchronisation added by instrumentation**: the finite check, `.item()`, printing or saving actions inside the
  timed loop would re-create the host-sync problem. Read results after the loop.
- **Memory**: one process now holds two stages' graphs, their pools and double buffers; watch for OOM and record
  the peak. Two full π0 processes did not fit (T014).
- **nsys inflation** on this launch-heavy workload (≈ 1.53× in E11f): nsys explains, the unprofiled runs measure.
- **Thermal**: co-running draws more power; watch clocks and temperatures, mark `throttled`.

## 9. What is yours to decide

Harness structure (extend `bench/corun.py`, `bench/corun_green.py`, `bench/green_ctx.py`, or write new); how to
diagnose and fix the corruption; lockstep frames or a free-running pipeline gated by events (say which and why);
steady-state detection; warm-up and iteration counts above the minimums; the issue order; which splits beyond the
required ones, and in what order; the hog kernel and its rates; which bandwidth sources work and how to calibrate
them; how deep to take the nsys decomposition; whether a short filtered ncu capture helps; which optional items
are cheap enough. If something here is wrong for this device, do the sensible thing, write down what you did and
why, and put the question under "Questions for the research side" in `STATUS.md` — do not stop for an answer.
**Time box:** if a reasonable effort (about a working day of device time) does not produce a compiled co-run
that passes the check, stop trying, commit the evidence (outcome row 6 of §10), and complete everything that does
not depend on it: M1, M3's `A_part` / `B_part` and hog control, M4's calibration, M5's serial arms.

## 10. Pre-registered thresholds and outcomes (fixed 2026-09-25, before any run)

These are what the research side will conclude from each outcome. They are **not** acceptance criteria: any
outcome, measured properly, completes the task.

**Thresholds, and why these values.** T014's brackets drifted 0.6–0.8 % within a session, and the 16-SM green
points reproduced E7 and the reconciliation within 0.07–0.5 %; so 3 % is ≈ 4× the worst drift (the smallest
effect we will call real), 5 % ≈ 6×. The 12-SM partition alone costs the prefix 8 %; a coupling of 10 % would be
larger than the whole SM cost of the split the derived bound favours.

| quantity | band | reading |
|---|---|---|
| `O = T_pipe / max(A_iso, B_iso)` | **≤ 1.05** | near-ideal overlap on the GPU |
|  | 1.05 – 1.15 | partial overlap: the GPU recovers most, not all, of the gain |
|  | **≥ 1.15** | poor overlap: the GPU gives up about a third or more of the available gain (with the fair expert, no overlap at all is `T_serial / A_iso` ≈ 1.44) |
| `p_part`, `s_part` (and `p`, `s` unpartitioned) | < 1.03 | no coupling detected |
|  | 1.03 – 1.10 | minor coupling |
|  | **≥ 1.10** | material coupling |
| `p_hog` at the highest hog rate | ≥ 1.03 | the method can see bandwidth contention (control passes) |

| # | outcome | what it means for the research |
|---|---|---|
| 1 | **Some valid configuration (M2, or an M3 split) reaches `O ≤ 1.05`** | One Orin GPU overlaps the prefix and the fair expert near-ideally. **The frame-rate argument for moving the expert off the GPU is gone** (≤ 5 %). The research leads with energy (M5) and a freed GPU, and states that configuration as the GPU-side baseline. |
| 2 | **Best `O` > 1.05, and inside partitions `p_part ≥ 1.10` (or `s_part ≥ 1.10`), with the hog control passing** | **"Partitioning SMs does not partition bandwidth"** on Orin: the stages couple through the memory system even when their SMs are separate. The memory-side argument is the finding (with the caveat of §3 that a PIM also shares the LPDDR5). |
| 3 | Best `O` > 1.05, but `p_part`, `s_part` < 1.03 and the hog control passes | The loss is the **SM partition's own cost** (a stage slower on fewer SMs), not memory coupling: GPU co-run costs what the best split costs. The frame-rate gain an off-GPU expert could add is `O` of the best split, quoted as measured. |
| 4 | Unpartitioned M2 has `O` > 1.05 while the best M3 split is materially lower | The unpartitioned scheduler (queueing, KerColle's mechanism if the timeline shows it) is the problem and partitioning is the software fix; green contexts at that split are the GPU-side baseline a paper must compare against. |
| 5 | `s > r` (in-session) in M2 or the best split | The expert, stretched by contention, sets the period despite ≈ 2.25× slack: contention on the expert is severe on Orin. |
| 6 | **The corruption proves unfixable** within §9's time box (own capture with separate pools and double-buffered I/O, plus at least one further remedy, each failing the check), with evidence | **A compiled two-stream co-run is not achievable on the shipping stack (torch 2.8, JetPack 6.2 / CUDA 12.6) without further engineering** — itself a finding, stated with the evidence; the GPU-co-run cell stays empty and the paper can quote only bounds (the derived 271.26 ms) labelled as derived. |
| 7 | **M1: the fair expert `B` ≤ 93.1 ms at N = 10** | The GPU expert is **faster than the research side's in-memory expert at 16 channels** (9.31 ms per step × 10 = 93.1 ms), and the latency claim at 16 channels is lost. This is a **real possibility**, not a formality, and must be reported as plainly as any other row. (For completeness: ≤ 58.7 ms would also beat the 32-channel design.) |
| 8 | M1: 93.1 ms < fair `B` ≤ 102.4 ms | The in-memory expert's latency advantage over a fair GPU expert at 16 channels is under 1.10×. |
| 9 | `hoist` fails its compiled equivalence check | The hoist cannot enter the GPU baseline as it stands; the research side keeps the stock expert and the co-run is re-planned (question in `STATUS.md`). |
| 10 | Co-run energy per frame (M5 c) ≤ 0.95 × the fair serial `full` | Overlap also saves energy per frame on the GPU (shorter period, idle power amortised) — the energy baseline a PIM must beat is the co-run, not serial. The research side holds no prediction here. |

Rows are not exclusive (e.g. 1 and 7 can both hold). The research side holds no prediction of its own that it wants
confirmed.

## 11. Outputs

Result directories (protocol §5 naming; the prefix `E12b` is what the research side searches for):

| directory | content |
|---|---|
| `results/<date>_E12b_pi0_fairexpert_N10_MAXN/` | M1: `stock` and `hoist` loops, hoist `equivalence.*` |
| `results/<date>_E12b_pi0_corun_N10_MAXN/` | M2: isolated, serial, pipeline; `host_issue.csv`; nsys CSV exports under `nsys/`; `corun_breakdown.csv`; `equivalence.*`; `bandwidth.csv`; `mechanisms.csv` (optional mechanisms too) |
| `results/<date>_E12b_pi0_green_corun_N10_MAXN/` | M0 graph-confinement control; M3 per split; hog control; `equivalence.*`; `bandwidth.csv` |
| `results/<date>_E12b_pi0_emc_calibration_MAXN/` | M4's hog-rate calibration of `EMC_FREQ` utilisation |
| `results/<date>_E12b_pi0_energy_N10_MAXN/` | M5: idle, three arms, `power.csv`, raw rail samples |

Split or merge directories if you prefer; keep the `E12b` prefix. Files, beyond `meta.json`, `summary.md`, `raw/`:

- `latency_samples.csv` — `variant, split, condition (isolated|alone_part|serial|corun|corun_part|hog_part), stage (prefix|expert|frame|hog), frame, ms, valid`.
- `corun_ratios.csv` — as E12 §10 plus `variant, split, prefix_sms, expert_sms, A_part_ms, B_part_ms, p_part, s_part, O, phi, p_hog, reference (fair_full_gpu_same_harness|stock_harness), valid`.
- `host_issue.csv` — per frame: issue ms, `T_pipe` sample, order.
- `bandwidth.csv` — `variant, split, condition, source (tegrastats_emc|mem_rail|nsys_gpu_metrics|ncu), metric, median, p5, p95, n, calibrated_GBps, notes`.
- `mechanisms.csv` — `mechanism, attempted, works (yes|no|partial|not_attempted), evidence_file, version_info, error, notes`, covering at least: own-graph capture, `graphs_both`, green contexts (co-run), priority −3, `CUDA_DEVICE_MAX_CONNECTIONS`, `libsmctrl`, MPS, two processes.
- `equivalence.md`, `equivalence.csv` (§7).

**Variant labels.** A `+`-joined list of tokens naming every change from E7's `bf16_both`, for example
`bf16_both+hoist+owngraph_expert+dbuf_kv+green_12_4+prefix_first`. Suggested tokens: `hoist`, `owngraph_expert`,
`owngraph_prefix` (= `graphs_both` with the former), `dbuf_kv` / `kv_copy`, `green_<p>_<e>`, `prio_prefix_-3` /
`prio_expert_-3`, `maxconn_1`, `prefix_first` / `expert_first`, `hog_<GBps>`. Define any token you add in
`meta.json` under `variant_tokens`. The same label goes in `meta.json` and in every CSV row.

Report: **`reports/E12b_pi0_compiled_corun.md`**, in the repository's reporting style: lead with one table —
`A_iso`, `B_iso`, `T_serial`, `T_pipe`, `O`, `φ`, `p`, `s`, `r` for the fair configuration on the full GPU and for
each split, with `A_part`, `B_part`, `p_part`, `s_part`, `p_hog`, n and dispersion; then M1 (`stock` vs `hoist`,
`B` against 93.1 ms as a plain statement); then the diagnosis and fix of the corruption; then the "why" from the
timeline and M4; then energy; then which rows of §10 landed, quoted; then measured / derived / not measured; then
the protocol §11 self-review against §12 below, item by item.

## 12. Acceptance

Do not mark T015 done until every item holds, or the status line names the item that does not and why.

1. Protocol §1 gate recorded for every run; all at MAXN, `throttled` recorded.
2. M1: `stock` and `hoist` `prefix` / `ae` / `full` loops in one bracketed session (≥ 100 frames or ≥ 60 s each);
   the compiled hoist equivalence check delivered as `equivalence.*`.
3. M2: a compiled co-run whose pipeline equivalence check passed with its stale control detected, with in-session
   `A_iso`, `B_iso`, `T_serial`, n ≥ 100 per condition, `s`, `p`, `r`, `O`, `φ` in `corun_ratios.csv`, the
   host-issue record, and the nsys CSV exports with `corun_breakdown.csv` — **or** outcome 6's evidence (diagnosis,
   every remedy tried, the failing checks) committed.
4. M3: at least three splits, each with `A_part`, `B_part`, the co-run (check passed), `p_part`, `s_part`,
   `T_pipe,part`, `O`; the graph-confinement control; the hog control at ≥ 2 rates — **or**, if outcome 6 holds,
   `A_part`, `B_part` and the hog control without the co-run, and the reason.
5. M4: `bandwidth.csv` for isolated, co-run and hog conditions, with the calibration, and a statement of which
   sources worked on this device.
6. M5: the three whole-frame energy arms and the idle baseline in one session.
7. No non-finite action in any run reported as valid; every invalid run listed with its reason.
8. `mechanisms.csv` covers every mechanism of §11; `meta.json` has no null in the fields of NN4; every row carries
   its variant label.
9. Protocol §11 self-review; no committed file over 50 MB; repository size in `STATUS.md`; nothing on the research
   side's files edited; T014 gets a status line saying its open MAXN items were taken over by T015 (§14).

## 13. Not part of this experiment

- **The channel-partition measurement (EV-02)** (formerly "P1": the VLM prefix with the GPU restricted to 8 of 16
  LPDDR channels), **P2** (`bf16_both + cropcache`) and
  the `check_model_inputs.py` `MODELS` change are recorded on the research side and **not approved**. Do not run
  or implement them here. (`cropcache` was measured in E4 and exists in `bench/run_e4.py`; it is **not** part of the
  fair expert and must not appear in any E12b run.)
- Capped power modes. T014 Step 4 (`bench/queues/queue_e12_capped.sh`) stays its own item for the `MODE_30W`
  session; E12b adds nothing to it.
- Other models. π0 only.

## 14. Relation to T014, and a note owed on its Step 4

- **T015 takes over T014's open MAXN items** (the compiled co-run, the nsys timeline, the Step-3 mechanisms,
  `equivalence.*`, `mechanisms.csv`, compiled co-run energy). When T015 is done, append to T014:
  `status: MAXN items superseded by T015 <UTC date>; Step 4 waits for MODE_30W`. T014's own §9 table is unchanged.
- **Your Step-4 prediction needs a correction before the `MODE_30W` session reads it** (`STATUS.md`, "MODE_30W,
  8 SMs in hardware: predicted ≈ 587.56 × (clock ratio)" for the eager expert). The eager expert is **host-issue-
  bound** (flat SM curve, E5 84 % GPU idle), and `MODE_30W` slows the **CPU** too (8 cores at ≤ 1.73 GHz against
  12 at ≈ 2.2 GHz), not only the GPU (≤ 612 against 1300.5 MHz). Scaling the eager expert by the GPU-clock ratio
  alone is therefore the wrong prediction, and a deviation from it would **not** by itself show the green-context
  result wrong. A fair cross-check between the 8-SM green partition at MAXN and the 8 hardware SMs of `MODE_30W`
  uses a **GPU-bound** quantity — the GEMM control, and the compiled (graph-replayed) stages — scaled by the GPU
  clock, with the CPU clocks recorded (`/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`) in both modes.
  Please restate the prediction that way in `STATUS.md` before the session, and add the GEMM control to the queued
  Step 4 if it is cheap. Nothing else about Step 4 changes.
