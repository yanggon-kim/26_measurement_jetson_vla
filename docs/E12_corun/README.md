# E12 — Two stages, one GPU: co-running π0's VLM prefix and action expert on Orin

*Written by the research side (jetson-owner) on **2026-09-24**, committed and pushed **before any E12 run**.
This file is the experiment's pre-registration: the predictions in §9 were fixed when it was committed
(`git log --follow docs/E12_corun/README.md` gives the date and hash). A later change to §9 is appended
under a new date, never rewritten. Task file: `tasks/T014_pi0_corun.md`. Binding rules: `docs/02_protocol.md`
(§14 is this experiment's); where this file is shorter, the protocol wins.*

---

## 0. In one paragraph

Run π0's two stages (**prefix** = vision + VLM prefix, and **expert** = the N denoising steps) **at the same
time on the one Orin GPU**, as a steady-state two-stage pipeline: the prefix of frame *k* on one CUDA stream
while the expert of frame *k − 1* runs on another, each frame using **its own** KV cache. Measure how much each
stage slows down, what frame period the pipeline achieves, why (from an nsys timeline), which software
mechanisms improve it, and how it changes when the GPU has 8 SMs instead of 16. The one number that decides the
research question is **`s` = the expert's stage time under co-run ÷ its isolated stage time**, compared with
**A / B ≈ 1.50** at MAXN.

## 1. Read this first: the result may go against us, and that is fine

The research side has a stake in one outcome. **You must not.** Do not tune the harness toward contention or
away from it. If co-running is nearly free on Orin, or if some software mechanism (stream priorities, graphs,
SM partitioning, two processes, something we did not think of) lets one GPU overlap the two stages ideally,
that is a **valuable** result: the research side needs to know it now, from you, rather than later from a
reviewer. If contention is severe, that is equally valuable. Report what the device does, with the evidence.
`AGENTS.md` already says "report what you see, not what we hope"; this experiment is where it matters most.

The same holds for your own harness: if the first design produces a number that is really about the CPU,
the GIL, an allocator or a hidden synchronisation rather than about the GPU, say so and fix it or label it —
in whichever direction that moves `s`.

## 2. Why this matters — the context you do not have

### 2.1 What the research side is designing

The research side designs a processing-in-memory (PIM) accelerator in LPDDR5 for the **action expert** of
flow-matching VLAs. The system it proposes is a **two-stage pipeline** per camera frame:

```
            frame k-1          frame k            frame k+1
GPU      : [prefix(k-1)]      [prefix(k)]        [prefix(k+1)]  ...
other    :        [expert(k-2)]     [expert(k-1)]      [expert(k)] ...
```

The VLM prefix of frame *k* runs on the GPU while the action expert of frame *k − 1* runs **concurrently**
somewhere else. **Each frame keeps its own KV cache**: expert(k − 1) reads exactly the KV that prefix(k − 1)
produced. This is **not** a stale-KV scheme (V-AEFusion, and the "stale prefix KV" stretch of E8, reuse an
older frame's cache to save work and pay in accuracy); here nothing is reused, so accuracy is unchanged. What
changes is the frame period: **max(A, B)** instead of **A + B**, where A and B are the two stage times.

### 2.2 The measured stage times (yours)

E7, `reports/E7_power_energy.md`, `results/20260924_E7_pi0_N10_bf16_both_MAXN/` (bench commit `b4f2879`),
π0, rung `bf16_both`, N = 10, MAXN + `jetson_clocks`, 60 s loops:

| loop | median ms |
|---|---:|
| `prefix` (vision + VLM prefix) = **A** | **251.28** |
| `ae` (action expert, N = 10) = **B** | **167.36** |
| `full` (serial inference) | 424.68 |

**A : B = 1.50 : 1** at N = 10. At N = 4 (`results/20260924_E7_pi0_N4_bf16_both_MAXN/`): A = 249.92,
B = 67.18, **A : B = 3.72 : 1**.

### 2.3 The open question

If the expert moves off the GPU, the GPU runs only the prefix and the period is ≈ A. But a reviewer will ask
the obvious question: **why not run both stages on the one GPU, on two streams, and get max(A, B) for free?**
Whether that works on Orin is **unmeasured by anyone**.

- **KerColle** (PACT '26, arXiv 2609.22335) studied exactly this schedule — the VLM of frame N + 1 with the
  action head of frame N, two CUDA streams, one GPU, batch 1, fresh KV per frame, π0.5 among its models. It
  found that naive two-stream overlap suffers **head-of-line blocking at the CTA dispatcher**: the VLM's grids
  exceed the GPU's spatial capacity, fill every SM, and the action phase's kernels queue behind them. They
  describe the action phase as running at roughly half its isolated rate while the VLM is unaffected (a
  qualitative Nsight observation, not a tabulated number). **But KerColle measured on desktop GPUs** — NVBit
  traces from an RTX 4090 / 5090 replayed in the Accel-Sim simulator (128 / 170 SMs). Never on a mobile SoC.
  **Orin has 16 SMs.** Their diagnosis — grids larger than the GPU can hold — should, if anything, bite
  *harder* here. Or Orin may behave differently for reasons a simulator of a discrete GPU would not show
  (shared LPDDR5, a different dispatcher, different kernels chosen for 16 SMs).
- **Jetson-PI** (arXiv 2607.12659) argues, **by roofline and without measuring it**, that on Orin both the VLM
  and the action expert are bandwidth-bound on the shared LPDDR5, so co-running them would slow **both**. That
  predicts the prefix slows too, which KerColle says does not happen on desktop.

Nobody has put the two stages on one Orin GPU at the same time and measured it. That is E12.

### 2.4 What each answer means for the research

- **`s < A/B` (≈ 1.50) and the prefix barely slows** → one GPU overlaps the two stages essentially ideally on
  Orin. There is **no frame-rate reason** to move the expert off the GPU. The research then leads with what
  still holds — latency per frame, energy, and a GPU freed for other work — and drops the frame-rate claim.
- **`s > A/B`** → the expert, stretched by contention, becomes the pipeline's bottleneck; contention is real
  on Orin and the frame-rate claim for moving the expert off the GPU stands.

Both are publishable, both are important, and the research side does not know which it will be.

## 3. Definitions (use these names in files and in the report)

All times are per frame, in ms, from **unprofiled** runs (never from an nsys or ncu run — the E11f
nsys-inflation factor for this launch-bound workload was ≈ 1.53×).

| symbol | meaning |
|---|---|
| `A_iso` | prefix stage time, **isolated**: the prefix running alone through the *same harness, stream and code path* used in the co-run, with the other stream empty |
| `B_iso` | expert stage time (all N steps of one frame), isolated, same harness |
| `A_co`, `B_co` | the same stages **while co-running** in the steady-state pipeline |
| `T_serial` | frame time with the two stages back to back on one stream (the E7 `full`-style serial frame), same session |
| `T_pipe` | achieved **period** of the pipeline: wall time of a long steady-state loop ÷ frames completed |
| **`s`** | **`B_co / B_iso`** — the expert's co-run slowdown. **The decisive number.** |
| `p` | `A_co / A_iso` — the prefix's co-run slowdown (Step 2) |
| `r` | `A_iso / B_iso` — the break-even for `s`, **from the same session** (E7 gave 1.50 at N = 10, 3.72 at N = 4; use the in-session value, report both) |

How to read them: the pipeline's period is set by the slower stage under co-run, so
`T_pipe ≈ max(A_co, B_co)` plus whatever the frame boundary costs. **Ideal overlap** means `T_pipe ≈ A_iso`;
no gain means `T_pipe ≥ T_serial`. `s < r` is the condition for the expert **not** to be the bottleneck; the
period also depends on `p`, so always report `s`, `p` and `T_pipe` together, and state
`T_pipe / A_iso` and `T_serial / T_pipe` (speed-up of the pipeline over serial).

**Stage time under co-run.** Measure it on the GPU, with CUDA events recorded **on the stage's own stream**:
the start event after any cross-stream wait that gates the stage (so waiting for the other stage is not
counted as work), the stop event after the stage's last kernel. Summarise over frames as §2 of the protocol
requires (median, mean, p5, p95, min, max, n) and keep every per-frame sample. A per-frame ratio distribution
of `s` is welcome; a bootstrap 95 % interval for the median `s` is welcome. If you find a better-defined
measurement of the same quantity, use it and explain it.

## 4. Non-negotiables

These five are the whole of the rules. Everything else in this file is guidance, and §8 says what is yours
to decide.

1. **Fresh KV, verified — the result is void without it.** Each frame's expert must read **that frame's own**
   KV cache. A pipelined harness that accidentally shares, reuses, aliases or overwrites a cache across frames
   is a different mechanism (stale KV, V-AEFusion's) and would invalidate every number. Verify it, per §6:
   distinct inputs per frame, pipelined actions against serial actions frame by frame, in eager **and** in the
   compiled `bf16_both` path you time, **plus a negative control that must fail**. Every mechanism of Step 3
   whose numbers you report passes the check through its own code path.
2. **Baseline unchanged.** π0 as in E7: LeRobot port `lerobot @ f0d2b37bebddf7e8852cb624712da0ad564601b6`,
   checkpoint `lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f`, rung
   **`bf16_both`** exactly as `results/20260924_E7_pi0_N10_bf16_both_MAXN/meta.json` defines it
   (vision tower + multi-modal projector in bf16 with `embed_image`'s fp32 cast removed; 74 layernorm modules
   in bf16; compiled `model.denoise_step@reduce-overhead;pg.embed_image@reduce-overhead+clone;pg.forward@max-autotune-no-cudagraphs`),
   batch 1, N = 10 (N = 4 in Step 5). **Do not change the baseline rungs, and do not mix in `cropcache`** —
   `bf16_both + cropcache` is a separate request that the user has **not** approved. A mechanism in Step 3 that
   needs a different compilation of a stage (e.g. CUDA graphs on the prefix) is a **labelled variant** with its
   own isolated times, never a replacement for the baseline.
3. **Isolated and co-run in the same session.** `A_iso`, `B_iso` and `T_serial` are re-measured in the same
   session, same process where possible, same power mode, same thermal state as the co-runs they are divided
   into. Bracket the co-runs (isolated → co-run → isolated) so drift is visible. Put the session's `A_iso` /
   `B_iso` next to E7's 251.28 / 167.36 ms and explain a difference above ≈ 3 %.
4. **Provenance as always** (protocol §5, §3.5, §6): one result directory per run, `meta.json` with
   `bench_commit`, model and checkpoint commits, rung string, power mode, SM count, clocks, `throttled`;
   per-frame raw samples; n and dispersion; raw files committed and listed in `raw_files`; the repository's
   size rules; **no `.nsys-rep` / `.sqlite` in git** (the §3.5 session-token exception) — commit the full nsys
   CSV exports instead; secrets scan before committing.
5. **Power modes.** Steps 1–3 and 5 run at MAXN (protocol §1 gate). Step 4 runs **inside** the `MODE_30W`
   session that T004 step 4b already needs — no extra reboot. You still **never** confirm a reboot prompt and
   never use `nvpmodel --force`. The capped-mode reboots are yours to schedule with the human as before; this
   experiment adds no new request to anyone.

## 5. The five steps

### Step 1 — Is the contention real on Orin? (the decisive step)

Build a steady-state pipeline in **one process**: prefix(k) on one CUDA stream, expert(k − 1) on another,
continuously, with each frame's KV handed from its prefix to its expert (a cross-stream event dependency and
the caching-allocator bookkeeping that cross-stream use needs — `Tensor.record_stream` or holding references
until the consuming stream is past them).

Measure, in the same session: `A_iso`, `B_iso`, `T_serial`, then the pipeline's `A_co`, `B_co`, `T_pipe`
(≥ 20 warm-up frames, ≥ 100 timed frames; plus one ≥ 60 s loop for `T_pipe` as throughput). Derive `s`, `p`,
`r`, `T_pipe / A_iso`, `T_serial / T_pipe`.

Then one **Nsight Systems** capture of the same pipeline, **3–5 steady-state frames** (protocol §3.1:
cudaProfilerApi window, `--capture-range-end=stop`, `--cuda-graph-trace=node` because the expert is
CUDA-graphed), and the same for the isolated stages. The timeline is there to show **why**:

- Do kernels of the two streams actually execute at the same time, and for what fraction of the frame?
- When they overlap, what happens to the expert's kernels: **longer kernel durations** (sharing SMs or memory
  bandwidth), or **longer gaps between consecutive expert kernels** while prefix kernels occupy the GPU
  (queued behind the prefix's grids — KerColle's head-of-line blocking), or a **host-side issue delay** (the
  CPU had not launched them yet)? Decompose `B_co − B_iso` into those three parts as well as the trace allows
  (inside a CUDA-graph replay, a gap on the expert stream while the prefix is running is queueing, not
  launching). Say which dominates.
- The prefix's kernel grids: blocks per grid, registers and shared memory per block, against what 16 SMs can
  hold at once (the `cuda_gpu_trace` export has grid and block dimensions). KerColle's mechanism requires the
  prefix's grids to exceed the GPU's capacity; say whether they do here.

Seeing action kernels queued behind VLM grids on real mobile silicon would be the first observation of
KerColle's head-of-line blocking outside a simulator. Seeing them run side by side unharmed would be equally
important.

### Step 2 — Does the expert also slow the prefix?

From the same runs as Step 1: `p = A_co / A_iso`, with its distribution. KerColle says the VLM is unaffected
on desktop; Jetson-PI argues by roofline that on Orin both slow because both are bound by the shared LPDDR5.
If the prefix does slow, say what the timeline and the counters you can get cheaply suggest the coupling is —
SM sharing or memory bandwidth (tegrastats `EMC_FREQ` utilisation during isolated vs co-run, the memory rail
of protocol §4, kernel-duration inflation of the prefix's memory-heavy kernels). A full ncu study is **not**
asked for; if you believe one short filtered ncu capture would settle the attribution, it is your call.

### Step 3 — The best GPU-side overlap achievable on Orin today

Try every software mitigation a reviewer would name, each as a labelled variant, and for each report
`A_iso`, `B_iso` (under that mechanism where it changes them, e.g. an SM partition slows an isolated stage),
`A_co`, `B_co`, `T_pipe`, `s`, `p` — with `s` and `T_pipe / A_iso` computed **against the baseline's
full-GPU isolated times of the same session** (that is what a deployment would compare with), and the
mechanism's own-isolated ratios alongside. The list:

| mechanism | notes |
|---|---|
| **stream priorities** | high priority for the expert's stream, and the reverse. **Check the sign first:** in CUDA a *lower* number is a *higher* priority; `torch.cuda.Stream.priority_range()` gave `(0, −3)` on this device. E11f's `bench/pipelining.py` requested `lo` = 0 believing it the highest, so its `high_priority_stream_accepted: true` (`reports/E11_pi0_hybrid_inputs.md`, "requested 0, got 0") recorded the **default** priority (torch's own docstring: "A lower number indicates a higher priority. By default, the priority is set to 0"); no high-priority stream has been measured yet. Use the greatest priority (−3 here) for "high", and read back `stream.priority` to confirm. KerColle says priority affects dispatch order but not the admission of CTAs already resident — check whether it changes anything on Orin |
| **CUDA graphs for both phases** | the expert is already graphed in `bf16_both`; graph the prefix too (a labelled variant — E4c's `vlm_graphs` rung, 197.17 ms prefix vs 192.76 compiled, was a small loss in isolation, which is exactly why it must be judged under co-run, not alone). Graphs change how work reaches the GPU, which may matter only when two streams compete |
| **SM partitioning — CUDA green contexts** | driver API, CUDA 12.4+ (`cuDeviceGetDevResource`, `cuDevSmResourceSplitByCount`, `cuDevResourceGenerateDesc`, `cuGreenCtxCreate`, `cuGreenCtxStreamCreate`; a green-context stream can be used from PyTorch through `torch.cuda.ExternalStream`). Try several splits of the 16 SMs (e.g. 8/8, 10/6, 12/4, and the reverse) — the granularity and minimum per partition are part of the result |
| **SM partitioning — `libsmctrl`** | Bakita & Anderson (RTAS '23), per-stream TPC masks (`libsmctrl_set_stream_mask`). Its README lists support for CUDA 8.0–12.1; this device runs **12.6**, so support is **unknown** — build it and try |
| **MPS** | if it exists on this JetPack (`nvidia-cuda-mps-control`), two processes under MPS, with and without `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` |
| **two processes, no MPS** | the naive multi-process deployment. E11f found two processes left the host segments unaffected (1.002×); for two heavy stages, say from the timeline whether the contexts **co-run or time-slice** |
| anything else you judge a reviewer would raise | e.g. `CUDA_DEVICE_MAX_CONNECTIONS` (hardware work queues), splitting the prefix into layer-sized chunks interleaved with expert steps from the issuing thread (cooperative software time-slicing). Your judgement — include what is cheap and informative, skip what is not, and say what you skipped |

**For every mechanism that does not work on this JetPack / CUDA 12.6, record the evidence**: the version
check (`cuDriverGetVersion`, `nvcc --version`, the library's version or commit), the exact call or command,
and the error code or behaviour, in `mechanisms.csv` and verbatim in `raw/`. **"No shipping software
mechanism for SM partitioning is available on Orin" is itself a result**, and it is only a result if the
evidence is committed. A mechanism that runs but produces wrong actions (fails §6) is reported as failing,
with its timing shown and marked invalid.

### Step 4 — Does contention grow as the GPU shrinks? (inside the `MODE_30W` session)

Power modes change the SM count: MAXN has **16 SMs** at 1300.5 MHz; `MODE_30W` has **8 SMs** (4 of 8 TPCs
power-gated) at ≤ 612 MHz, with 8 CPU cores at ≤ 1.73 GHz and the memory clock uncapped
(`results/20260921_E0_platform/nvpmodel_modes.csv`). Run **Step 1** (isolated `A_iso`, `B_iso`, `T_serial`,
the pipeline, one short nsys capture) in the `MODE_30W` session and compare `s`, `p`, `r` with MAXN. Because
each ratio is taken against isolated runs **at the same mode**, a uniform slowdown from the lower clock
cancels; what does not cancel is exactly what this step is after — fewer SMs for the same grids — together
with anything that does not scale uniformly (the GPU clock falls 2.1× while memory bandwidth does not, and
the CPU slows too). Say which of these the timeline points to; `r` itself will move, so compare `s` with
the in-mode `r`.

This rides the `MODE_30W` session already planned for T004 step 4b (protocol §1 lists it): no extra reboot,
nothing new to ask anyone. Run the Step-3 mechanisms again there only if they were informative at MAXN and
it is cheap; the baseline pipeline is what is required.

Optional controls, only if cheap: `MODE_50W` (16 SMs at ≤ 816 MHz, same TPC mask as MAXN, switchable
without a reboot as the last item of a MAXN session per protocol §1) separates the clock from the SM count;
an 8-SM green-context or `libsmctrl` partition **at MAXN** (if Step 3 found one that works), running both
stages inside 8 SMs, separates the SM count from the clock the other way. `MODE_15W` changes the memory clock
too; skip it unless you have a reason.

### Step 5 — N = 4, and energy, cheaply alongside

- **N = 4** at MAXN: repeat Step 1 (no nsys needed unless something is odd). A : B ≈ 3.72 : 1, so the expert
  has a lot of slack; overlap should be easy here. This is a sanity check on the harness: if N = 4 does not
  approach `T_pipe ≈ A_iso`, look for a harness cause before trusting N = 10.
- **Energy per frame under co-run**, N = 10, MAXN, with E7's method (protocol §4): 60 s idle baseline in the
  same session, then a ≥ 60 s back-to-back pipeline loop with the four rails sampled from sysfs (E7 used
  20 Hz), tegrastats alongside; energy per frame = mean module power × `T_pipe`, total and above idle, per
  rail and summed. Measure the **serial** `full` loop in the same session the same way and compare frame
  against frame. Do not build the comparison from the separate `prefix` / `ae` phase loops: in E7 those two
  do not add up to the `full` loop (418.64 vs 424.68 ms, and 9 509 vs 13 666 mJ above idle — an open
  accounting question), so a whole-frame loop on each side is the like-for-like comparison. State how the
  idle baseline was subtracted.

## 6. The fresh-KV equivalence check (non-negotiable 1, in detail)

The cropcache result was verified with `bench/check_equivalence.py` (eager, fixed noise, bitwise against the
unmodified model, a determinism control). Do the same for the pipeline, with three additions that a pipeline
needs:

1. **Distinct inputs per frame.** If every frame uses the same fixture observation, frame *k*'s KV equals
   frame *k − 1*'s and a stale-KV bug is **invisible**. Cycle at least three observations that give materially
   different actions (other frames of the fixture's LIBERO episode, or other episodes — record which; state
   and prompt may differ too) **and** a different noise seed per frame. Report the serial actions' frame-to-frame
   difference: it is the scale a stale-KV mix-up would show up at.
2. **Pipelined against serial, frame by frame**, same observation and same noise per frame: max absolute and
   relative difference per frame, bitwise-identical yes/no. Do it (a) in **eager** — nothing but the schedule
   differs, so bitwise identity is the expectation; explain any difference — and (b) through the **compiled
   `bf16_both` path you time**, against serial `bf16_both`, because static CUDA-graph buffers and cross-stream
   memory reuse are hazards only the compiled path has. Run it through the timing harness itself (a flag that
   saves the actions), not a separate script that could differ from it.
3. **A negative control that must fail**: a deliberately stale variant (expert(k − 1) given frame *k*'s KV,
   or the previous frame's) run through the same check. If the check cannot tell the stale variant from the
   correct one, it has no power and the pipeline is not verified.

Deliver `equivalence.md` and `equivalence.csv` (`mechanism, path (eager|compiled), frame, obs_id, seed,
max_abs_diff, max_rel_diff, bitwise_identical, verdict`) in each result directory whose timings depend on
them, with the stdout under `raw/`.

## 7. Hazards we can foresee (yours to handle; listed so you do not rediscover them the slow way)

- **One issuing thread, and hidden synchronisations.** Protocol §13.7: two streams are driven from **one**
  Python thread, never a second Python thread (T010's first two-stream numbers were GIL artefacts;
  `bench/pipelining.py` after `773d2a9` is the corrected pattern, and π0's E11f two-stream rows on disk
  predate it). With one thread, any host synchronisation inside either stage's issue path blocks the other
  stage's launches too, and shows up as "contention" that is really the CPU. Candidates in π0's loop: the
  per-step `torch.tensor(time, device=...)` in `sample_actions`, any `.item()`, pageable copies, allocator
  frees. `torch.cuda.set_sync_debug_mode("warn")` and the nsys CUDA-API rows find them. Removing one
  (e.g. precomputing the step times) is a harness change, allowed if the equivalence check passes and the
  isolated stages are re-measured through the changed harness. Order the launches so that a cheap stage is
  never waiting behind the issue of an expensive one (e.g. enqueue expert(k − 1)'s graph replays, then issue
  prefix(k)), and show from the trace that the GPU is not starved for work with both streams empty.
- **torch.compile's CUDA-graph trees** were designed for one sequential call stream. Interleaving the
  graphed `embed_image` and `denoise_step` across two streams may trigger re-recording, fallback to non-graph
  execution, output-overwrite errors, or hidden synchronisation (`TORCH_LOGS` for cudagraphs / perf hints
  shows re-records). If the stock path cannot run pipelined, capturing the same compiled functions yourself
  (`torch.cuda.CUDAGraph`, separate memory pools) is an acceptable labelled harness change — same kernels,
  same equivalence check, isolated stages re-measured through it.
- **Cross-stream memory.** A tensor made on the prefix stream and read on the expert stream can be freed and
  reused by the caching allocator while the expert still reads it, unless `record_stream` or held references
  prevent it. This is precisely a stale / corrupted-KV bug and the equivalence check is what catches it.
- **Two processes** without MPS may time-slice rather than co-run; nsys shows which. A time-sliced "co-run"
  is a valid measurement of the naive deployment, not of spatial sharing — say which you observed.
- **Thermal.** Co-running draws more power than either stage alone; watch temperatures and clocks, mark
  `throttled` per protocol §1.
- **tegrastats** must be stopped properly between runs (`bench/common.py:stop_tegrastats()`; `pgrep tegrastats`
  empty before a timed run).

## 8. What is yours to decide

Harness structure (reuse `bench/phase_loops.py`, `bench/pipelining.py`, `bench/pi0_common.py`,
`bench/model_registry.py` or write new); lockstep frames with a per-frame barrier or a free-running pipeline
gated only by cross-stream events (say which, and why); how you detect steady state; warm-up and iteration
counts above the protocol's minimums; which observations to cycle; which Step-3 splits and extra mechanisms to
try and in what order; how deep to take the timeline decomposition; whether a short ncu capture is worth it;
how to handle a surprise. If something here turns out to be wrong for this device, do the sensible thing,
write down what you did and why, and put the question under "Questions for the research side" in `STATUS.md`
— do not stop for an answer. Only the five non-negotiables of §4 are fixed.

## 9. Pre-registered predictions and their meaning (fixed 2026-09-24, before any run)

These are what the research side will conclude from each outcome. They are **not** acceptance criteria:
any outcome, measured properly, completes the task.

| outcome | what it means for the research |
|---|---|
| **`s < r` (≈ 1.50) at MAXN, N = 10**, prefix barely slowed (`T_pipe ≈ A_iso`) | One GPU overlaps the two stages ideally on Orin. **The frame-rate argument for moving the expert off the GPU is gone.** The research leads with latency, energy and a freed GPU. |
| **`s > r` at MAXN** | Contention is real on Orin: the stretched expert sets the period. **The frame-rate claim stands.** |
| `s < r` but the prefix slows so that `T_pipe` is well above `A_iso` | Overlap is not ideal, for the *other* reason: the bandwidth coupling of Step 2 is the finding, and it bears on any design that shares the LPDDR5 with the GPU. |
| **`s` larger at 8 SMs (`MODE_30W`) than at 16 (MAXN)** | The "worse on a small mobile GPU" argument is supported by measurement, within one device. |
| **The prefix slows under co-run (`p` materially > 1)** | Jetson-PI's roofline argument confirmed on hardware; the bandwidth coupling between the stages matters. |
| **A Step-3 mechanism reaches `T_pipe ≈ max(A_iso, B_iso)`** | The software route closes the gap on Orin today; the research must state that mechanism as the GPU-side baseline. |
| **No SM-partitioning mechanism works on JetPack 6.2 / CUDA 12.6**, with evidence | The software reply to contention is not available on this platform today — a result in itself. |
| `T_pipe ≥ T_serial` | Co-running is worse than not overlapping at all on Orin. |
| N = 4 fails to approach `A_iso` | Suspect the harness first (§5 Step 5); if it survives scrutiny, it is a finding about fixed costs of co-running. |

For the record, prior work would predict: KerColle's desktop observation → `s` ≈ 2 (> 1.50), `p` ≈ 1;
Jetson-PI's roofline → `p` > 1 and `s` > 1. Neither was measured on Orin. The research side holds no
prediction of its own that it wants confirmed.

## 10. Outputs

Result directories (protocol §5 naming, one per run; the prefix `E12` is what the research side searches for):

| directory | content |
|---|---|
| `results/<date>_E12_pi0_corun_N10_MAXN/` | Steps 1 + 2: isolated, serial, baseline pipeline; nsys CSV exports under `nsys/`; `equivalence.*` |
| `results/<date>_E12_pi0_mechanisms_N10_MAXN/` | Step 3 (one directory, one row set per mechanism; split it if you prefer, same prefix) |
| `results/<date>_E12_pi0_corun_N10_MODE_30W/` | Step 4 (and `_MODE_50W`, or an `_MAXN_8SM` partition directory, if you run those controls) |
| `results/<date>_E12_pi0_corun_N4_MAXN/` | Step 5, N = 4 |
| `results/<date>_E12_pi0_energy_N10_MAXN/` | Step 5, energy: pipeline and serial `full` loops, idle baseline, `power.csv`, raw rail samples |

Files in each, beyond the protocol's `meta.json`, `summary.md`, `raw/`:

- `latency_samples.csv` — `mechanism, condition (isolated|serial|corun), stage (prefix|expert|frame), frame, ms`
  — every sample.
- `corun_summary.csv` — `mechanism, power_mode, sm_count, N, condition, stage, median_ms, mean_ms, p5_ms, p95_ms,
  min_ms, max_ms, n`.
- `corun_ratios.csv` — `mechanism, power_mode, sm_count, N, A_iso_ms, B_iso_ms, T_serial_ms, A_co_ms, B_co_ms,
  T_pipe_ms, s, s_ci95_lo, s_ci95_hi, p, r, T_pipe_over_A_iso, speedup_vs_serial, reference (baseline_full_gpu|own_isolated), notes`.
- `mechanisms.csv` (Step 3) — `mechanism, attempted, works (yes|no|partial), evidence_file, version_info, error, notes`.
- `corun_breakdown.csv` (Step 1, from nsys) — per stage and condition: kernel-busy ms, stream-gap ms, time with
  both streams executing, and the three-way decomposition of `B_co − B_iso`.
- `equivalence.md`, `equivalence.csv` (§6).

Report: **`reports/E12_pi0_corun.md`**, in the repository's reporting style — lead with the answer: one table
of `s`, `p`, `r`, `T_pipe`, `T_serial`, `A_iso`, `B_iso` for MAXN N = 10 (and N = 4, `MODE_30W`, the best
Step-3 mechanism), with n and dispersion; then the "why" from the timeline; then Step 3's mechanism table
including the ones that failed and their evidence; then energy; then measured / derived / not measured;
then the self-review of protocol §11.

## 11. Acceptance

- The Step 1 numbers exist at MAXN N = 10 with n ≥ 100 frames per condition, from the same session, and `s`,
  `p`, `r`, `T_pipe / A_iso`, `T_serial / T_pipe` are derived in `corun_ratios.csv`.
- The equivalence check of §6 passed (eager and compiled) for the baseline pipeline and for every mechanism
  whose numbers are reported as valid, and the negative control failed.
- An nsys timeline of 3–5 steady-state co-run frames, exported as CSV and committed, and the report says from
  it which of kernel slowdown, queueing, or host issue explains `B_co − B_iso`.
- Every Step-3 mechanism in §5 is either measured or recorded as unavailable with committed evidence.
- Step 4 is done in the `MODE_30W` session, or the report says it waits for that session (it does not block
  the rest of the task — mark the MAXN part done and leave Step 4 as the open item, as T004 does with its
  capped modes).
- Step 5's N = 4 pipeline and co-run energy per frame against serial exist.
- Protocol §11 self-review; no committed file over 50 MB; repository size in `STATUS.md`; nothing on the
  research side's files edited.

## 12. Not part of this experiment

- **P1** (the VLM prefix with the GPU restricted to 8 of 16 LPDDR channels), **P2** (`bf16_both + cropcache`)
  and the `check_model_inputs.py` `MODELS` change are recorded on the research side and **not approved**.
  Do not run or implement them as part of E12.
- Other models. This is π0 only; if it is informative the research side will ask for others by task.
