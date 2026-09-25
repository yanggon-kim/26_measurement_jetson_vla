# E12 — co-running π0's VLM prefix and action expert on one Orin GPU

**Task T014. Specification `docs/E12_corun/README.md`; binding rules protocol §14, §13.7, §1,
§3.5, §5. This report covers the MAXN eager part. Three parts are open and named in §7 below.**

## The answer

**On one Orin GPU, π0's two stages do not overlap — and the reason is the host, not the GPU.**

**And the GPU is not the obstacle even in principle.** §5 partitions Orin's 16 SMs with CUDA
green contexts and finds the **action expert is SM-independent**: taking away three quarters of
the SMs costs it 1.3 % (eager) / 4.9 % (`bf16_both`), while the prefix slows 2.98×. A stage that
does not occupy the SM array can neither block nor be blocked at the CTA dispatcher, so
**KerColle's contention mechanism cannot be what limits co-running these two stages on this
device**. That is the answer to the question E12 was commissioned to settle, and it is a
*measurement*, not an inference from the no-overlap result above.

`results/20260925_E12_pi0_corun_N10_MAXN_eager_nosync/`, π0 eager, N = 10, MAXN +
`jetson_clocks`, n = 100 frames per condition, bench commit `5dc228ee4b03`:

| quantity | value | 95 % CI | n |
|---|---:|---|---:|
| `A_iso` prefix isolated | **474.08 ms** | p5–p95 472.2–479.0 | 100 |
| `B_iso` expert isolated | **523.76 ms** | p5–p95 520.6–535.2 | 100 |
| `T_serial` both stages, one stream | **994.38 ms** | p5–p95 992.6–996.7 | 100 |
| `A_co` prefix under co-run | 473.17 ms | p5–p95 472.8–473.8 | 100 |
| `B_co` expert under co-run | 521.44 ms | p5–p95 520.1–522.9 | 100 |
| **`T_pipe`** achieved period | **994.88 ms** | — | 100 |
| **`s` = `B_co`/`B_iso`** | **0.9956** | 0.9951 – 0.9959 | 100 |
| `p` = `A_co`/`A_iso` | 0.9981 | — | 100 |
| `r` = `A_iso`/`B_iso` (this session) | 0.9051 | — | — |
| `T_pipe`/`A_iso` | 2.0986 | — | — |
| speed-up over serial | **0.9995** | — | — |

**`T_pipe` equals `A_iso + B_iso` (994.88 vs 997.83 ms).** Ideal overlap would give
`max(A, B) = 523.76 ms`. The pipeline is serial.

### `s ≈ 1.00` is not "co-running is free"

This is the trap this experiment is most exposed to, so it is stated first. `s = 0.9956`
and `p = 0.9981` say the expert and the prefix each took their isolated time under co-run.
That is **what both ideal overlap and zero overlap look like** in per-stage event timings,
because a stage that has the GPU to itself measures its isolated time either way. Only
`T_pipe` distinguishes them, which is why §3 of the specification requires `s`, `p` and
`T_pipe` to be reported together — and it is the only reason this was caught here.

Read from `s` alone, these numbers would have said *one GPU overlaps the two stages
essentially ideally on Orin, so there is no frame-rate reason to move the expert off the
GPU* — which is one of the two outcomes §9 pre-registers, and it would have been false.

### Why there is no overlap: the host issues for the entire wall clock

| | |
|---|---:|
| host time spent issuing | 99 475.2 ms |
| wall time of the same loop | 99 475.3 ms |
| **`issue/wall`** | **1.0000** |

Over a 99.5-second run the CPU is issuing kernels essentially 100 % of the time. While one
stage is being issued the other stream holds no work, so there is nothing to overlap however
correctly the streams are wired. This is consistent with E5, which measured π0's eager
`ae_step` at **84.2 % GPU idle over 968 kernels per step** — the workload is launch-bound.

**Consequence for the research question.** KerColle's mechanism is GPU contention: grids that
exceed the GPU's spatial capacity, with the action phase's CTAs queued behind the VLM's. A
configuration that never puts both streams' work on the GPU cannot exhibit it. **π0 eager
therefore cannot answer that question**, and the `s` above should not be read as an answer to
it. The compiled `bf16_both` rung — 372 kernels/step and 47 % idle (E5) — is where it becomes
answerable, and §7 below explains why that path is currently blocked.

### Which stage is longer depends on the rung, and `r` flips

`r = 0.905` here: in **eager** the expert is the longer stage (523.8 vs 474.1 ms). Under
**`bf16_both`** (E7) the prefix is longer (251.28 vs 167.36 ms), giving `r = 1.50`. Compiling
speeds the expert more than the prefix and **inverts which stage sets the period**. A
break-even borrowed from another rung would invert the conclusion, which is why §3 requires
`r` from the same session and both are reported.

## 2. Fresh KV, verified — the gate on everything above

Non-negotiable 1: each frame's expert must read that frame's own KV cache. Verified per §6,
in eager, through the timing harness itself:

| check | requirement | result |
|---|---|---|
| pipelined vs serial actions, frame by frame | bitwise identity expected in eager | **6/6 PASS, max abs diff 0.000e+00** |
| negative control (expert given frame *k−2*'s cache) | **must fail** | **5/5 detected**, max abs diff 2.038 |
| observations materially different | else the check has no power | frame-to-frame action diff **5.47** |

The control's 2.038 sitting below the 5.47 frame-to-frame scale is the expected shape: a
cache two frames old is wrong but related, so it perturbs the actions by a fraction of the
full between-frame difference, while the correct path is exactly zero.

Frame 0 is excluded from the control (no earlier frame exists to be stale from), which is why
the control reports 5 of 6 rather than 6 of 6.

**Observations.** `bench/fixtures/` held exactly one observation, which would have made a
stale-KV bug invisible — frame *k*'s KV would equal frame *k−1*'s. `bench/e12_fixtures.py`
captures five real frames of the fixture's own LIBERO episode (episode 0, offsets 0/25/50/75/99,
all `task_index` 0) from the same source the existing fixture records. `obs_0` is offset 0 and
its decoded pixels are **bit-identical** to the committed fixture; its PNG bytes differ only
because the server re-encodes.

## 3. π0's shipping model cannot be pipelined across two streams

Measured with `torch.cuda.set_sync_debug_mode("warn")`, the stock model performs a host
synchronisation at:

| site | frequency | what |
|---|---|---|
| `modeling_pi0.py:681` | once per **prefix** | `att_masks` list → bool tensor |
| `modeling_pi0.py:746` | once per **denoise step** | `att_masks` list → `embs.dtype` tensor |
| `modeling_pi0.py:858-859` | once per **denoise step** | `torch.tensor(<python float>, device="cuda")` for the timestep — a **pageable** H2D copy, so the host blocks until the stream drains |

That is **1 + 2N host-blocking operations per frame** (21 at N = 10). The third site was found
last, by reconciliation against E7 rather than by the sync-debug pass, and it is the expensive
one: **33.6 % of the expert stage** (measured below). A sync in either stage's issue path
stalls the *other* stream's launches, so with one issuing thread (§13.7 — a second Python
thread is forbidden, T010's numbers were GIL artefacts) the streams cannot overlap.

**This is a property of the model implementation, not of Orin.** It holds on any GPU, and it
is the first obstacle a reviewer proposing "just use two CUDA streams" has to clear.

### The de-synced variant, and a free optimisation

`mechanism = pi0_nosync` (this run) memoises those two conversions on the list's **contents** —
keying on contents rather than call site is what keeps it correct when the prompt length
changes, since a different `att_masks` gets its own tensor. It removes 12 of 12 per-step syncs
and 2 of 3 per-prefix syncs (the remaining one is the cold memo on the first prefix). It is a
**labelled harness change** per §7, verified to keep the equivalence check bitwise identical,
and the baseline rung is untouched.

**It made the expert stage ≈ 19 % faster at N = 4 (226.81 → 184.18 ms.)** π0's per-denoise-step
`torch.tensor(att_masks, …)` costs about a fifth of the expert's runtime. That is an
optimisation to the shipping model, independent of E12, and worth taking regardless of how the
co-run question resolves.

**Removing the syncs was necessary but not sufficient**: `T_pipe` still equals `A_iso + B_iso`,
because the workload remains launch-bound.

**Caveat, and a correction to the sentence above.** "Worth taking regardless" is too broad for
this variant *as implemented*: it is **eager-only**. Under the compiled `bf16_both` rung it
raises `Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent
run` — the memoised tensor is allocated **inside** the compiled region on the first call, so it
becomes a CUDAGraph output and cannot be handed back on the next invocation. E12's MAXN run is
eager (§4), so the 19 % figure is sound for what it measures. A compile-safe version must
allocate those tensors outside the compiled region; not attempted.

### Reconciled against E7's `ae` time — and a correction to this report

E7 reports `ae` = **167.36 ms** for π0 `bf16_both`, N = 10, MAXN. Measuring the same stage in
`bench/corun_2proc.py` at the same rung, N and power mode gave **111.01 ms** — a 50.8 %
disagreement, while the **prefix agreed to +0.24 %** (251.89 vs 251.28). A rung-level number
cannot be quoted with a third of one stage unexplained, so it was chased to the end. Two
hypotheses were tested and rejected before the cause was found:

| hypothesis | measured effect | verdict |
|---|---:|---|
| E7's 20 Hz Rails **Python thread** + tegrastats subprocess steal the GIL from the issue loop | +0.19 % | rejected |
| E7 measures `ae` in a process where the **prefix's** graphs, CUDA-graph pool and fragmented allocator are resident (`power_phases.py:187` runs `prefix`→`ae`→`full` in one process) | +0.29 % | rejected |
| E7 builds the timestep tensor **inside** the denoise loop | **+50.5 %** | **confirmed** |

The residency hypothesis predicted the right *direction* — more resident state should make the
expert slower — which is why it was worth a run; direction agreement is not magnitude, and
238 resident prefix iterations moved the median from 111.02 to 111.34 ms.

The cause is one line. `phase_builders.make_ae` builds the timestep per step, and
`bench/corun.py`'s `expert_stage` precomputed it. Swapping only that line back
(`--e7-timesteps`), everything else identical:

| expert variant, `bf16_both`, N = 10, MAXN | median | p5–p95 | n |
|---|---:|---|---:|
| precomputed timestep tensors | **111.02 ms** | 110.90–111.28 | 406 |
| E7's per-step `torch.tensor(<float>, device="cuda")` | **167.09 ms** | 166.62–168.05 | 270 |
| *E7's reported `ae`, for comparison* | *167.36 ms* | — | — |

**167.09 against E7's 167.36 is 0.16 %.** The discrepancy is accounted for in full, and
`r = A_iso/B_iso = 1.50` from E7 **stands and may be quoted**.

**Whose defect this is.** Not E7's: `modeling_pi0.py:858-859` in the shipping `sample_actions`
does exactly the same thing, so `make_ae` faithfully reproduces the model and E7's 167.36 ms is
a correct measurement of it. **The defect is in this report.** §6 below stated that precomputing
the timestep tensors was "not labelled, because §7 names it as the example of a permitted
harness fix". That was written believing the change was free. It is worth **33.6 % of the
expert stage** — 56.07 ms of 167.09, or 5.6 ms per denoise step. §7 permitting the *class* of
change does not make an unlabelled 33.6 % speed-up acceptable, and it is labelled here and in
§6 now.

**Mechanism, and how strong the evidence for it is.** 5.6 ms per step is far too large for a
4-byte transfer, which is the evidence that the cost is the **drain** rather than the copy: a
pageable H2D copy blocks the host until it lands, which requires the stream to reach it, so the
host's run-ahead over a launch-bound stage is discarded once per step. The controlled swap above
is the measurement; the drain is the explanation offered for it. It was **not** separately
instrumented (no nsys capture of the stall), so the mechanism is an inference and the 56 ms is
the measurement.

**A second free optimisation to shipping π0**, independent of E12 and simpler than the
`att_masks` de-sync, which needs no memoisation and no compile-safety argument: hoist the N
constant timestep tensors out of the denoise loop. **33.6 % of the expert stage** at N = 10,
`bf16_both`, MAXN.

**What this does and does not change about E12's answer.** Nothing in direction. `T_pipe` still
equals `A_iso + B_iso` and `issue/wall` is still 1.0000; removing host-blocking work makes a
stage faster without making the pipeline overlap. It **strengthens** §3's claim: π0 hands the
GPU back to the host `1 + 2N` times per frame, which is why nothing overlaps.

## 4. The compiled `bf16_both` path corrupts when pipelined

| condition | pipelined actions vs serial |
|---|---|
| eager, pipelined | **bitwise identical** |
| compiled `bf16_both`, **serial** | correct (consecutive frames differ by ~5, as they must) |
| compiled `bf16_both`, **pipelined** | **corrupted, non-deterministically**: max abs diff **1.273** on one run, **all-NaN** on the next |

Identical code between those two runs. The non-determinism makes it a race rather than a
mis-wiring. The negative control detected a stale cache 5/5 in the compiled path too, so the
check has power there — the compiled pipeline is failing a test that works.

`TORCH_LOGS=cudagraphs` emits **no** re-record, fallback or warning: it fails silently while
producing entirely plausible per-frame milliseconds.

This is hazard §7, predicted in the specification before any run: the rung graphs
`pg.embed_image@reduce-overhead+clone` on the prefix side and
`model.denoise_step@reduce-overhead` on the expert side, so both stages replay graphs from one
shared CUDAGraphTrees pool while running on different streams.

A first hypothesis — late cloning into `denoise_step`'s static output buffers — was **refuted
by measurement**: the diagnostic reports distinct tensor objects and per-frame NaN rather than
a repeated last frame.

**No compiled co-run number is reported.** Timing that path would produce plausible
milliseconds for a pipeline computing NaN, which is worse than reporting nothing.

## 5. Step 3 answered with green contexts: **neither stage saturates Orin's 16 SMs**

The sections above establish that π0's two stages do not overlap and that the **host** is why.
They do not answer the research question, because KerColle's mechanism is GPU contention and a
configuration that never fills the GPU cannot exhibit it. Every direct route to filling it is
closed: one Python thread cannot feed two streams (§13.7), the compiled rung corrupts across
streams (§4), and two processes are OOM-killed (Orin's 62 GB is shared CPU+GPU; one π0 at
`bf16_both` is ~15 GB).

**CUDA green contexts partition the SMs inside one process**, which dodges the memory limit and
lets the question be asked from the other side:

> How much slower is ONE stage when it is given 4 SMs instead of 16?

This is a *necessary condition* argument. Contention between two stages requires that each can
actually use the SMs the other would take. A stage that runs at the same speed on a quarter of
the array was not using the array, and a co-runner could have had the rest.

`results/20260925_E12_pi0_green_N10_MAXN_{eager,bf16both}/sm_sensitivity.csv`, π0, N = 10,
MAXN + `jetson_clocks`, medians over n = 30–180 per point:

| SMs | eager prefix | eager expert | `bf16_both` prefix | `bf16_both` expert |
|---:|---:|---:|---:|---:|
| 4 | 1425.07 ms (2.981×) | **588.64 ms (1.013×)** | 491.60 ms (1.958×) | **117.05 ms (1.049×)** |
| 6 | 993.87 (2.079×) | 584.07 (1.006×) | 378.06 (1.505×) | 114.69 (1.028×) |
| 8 | 779.79 (1.631×) | 587.56 (1.012×) | 320.71 (1.277×) | 113.70 (1.019×) |
| 10 | 662.12 (1.385×) | 583.35 (1.004×) | 290.56 (1.157×) | 113.30 (1.015×) |
| 12 | 576.50 (1.206×) | 582.99 (1.004×) | 271.26 (1.080×) | 112.30 (1.006×) |
| **16** | **478.08** | **580.79** | **251.10** | **111.61** |

**The action expert is SM-independent. Removing three quarters of the GPU's SMs costs it 1.3 %
(eager) and 4.9 % (`bf16_both`).** The prefix does scale — 2.98× on 4 SMs in eager — so the
measurement has power: the same apparatus that shows the expert flat shows the prefix moving.

### Consequence for the research question

**CTA-level contention cannot be what limits co-running π0's two stages on Orin.** The expert
does not occupy the SM array, so it can neither be blocked at the CTA dispatcher nor block
anything else there. The binding constraint is host kernel issue (`issue/wall` = 1.0000, §1).

**This is not a refutation of KerColle**, and the distinction matters. They modelled 128–170 SM
desktop GPUs. What is measured here is that *π0's kernels, on Orin's 16 SMs*, do not produce
grids that compete. The grid sizes from E5b's ncu captures show why:

| expert kernel | CTAs | block | can it even fill 16 SMs? |
|---|---:|---:|---|
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2…` | 16 | 128 | exactly 1 CTA per SM |
| `ampere_bf16_s16816gemm_bf16_128x64…`, `ampere_sgemm_128x64_tn` | 32 | 128 | 2 per SM |
| `Kernel2` | 128 | 128 | 8 per SM |
| `elementwise_kernel`, `vectorized_elementwise_kernel` | **1** | 128 | **one SM, by construction** |

**Derived, not measured:** at 128 threads per CTA and Orin's 1536 threads per SM, the ceiling is
12 CTAs/SM, so 4 SMs offer ~48 CTA slots. Every kernel above except `Kernel2` therefore has its
**entire grid resident even on 4 SMs**, and runs in one wave whether it has 4 SMs or 16. That is
the mechanism behind the flat curve, and it predicts the small residual: only `Kernel2` (128
CTAs) needs extra waves on 4 SMs, which is consistent with the 4.9 % rather than 0 %. This
argument uses the device's spec occupancy ceiling, **not** a measured achieved occupancy — the
E5b exports do not carry the `launch__occupancy_limit_*` columns — so it is offered as the
explanation for a measured curve, not as an independent measurement.

### What a working co-run would buy, if the host were fixed

**Derived arithmetic, explicitly not a measurement.** Taking each stage's time at its best
partition and pairing them to 16 SMs:

| rung | expert @ 4 SMs | prefix @ 12 SMs | period = max | serial at 16 SMs | ratio |
|---|---:|---:|---:|---:|---:|
| eager | 588.64 | 576.50 | **588.64 ms** | 1058.87 | **1.799×** |
| `bf16_both` | 117.05 | 271.26 | **271.26 ms** | 362.71 | **1.337×** |

This is the upper bound on what fixing the host issue bottleneck could buy *on the GPU alone*,
conditional on two green contexts actually running concurrently — which **was not measured**,
because one Python thread still issues both and §13.7 forbids a second. It is arithmetic over
six measured numbers, and it is in this report as a bound, not a result. No `T_pipe` is claimed
for it.

### Controls, both of which had to pass before any number above was recorded

1. **Is the partition real?** A green context that does not actually confine work would return
   plausible milliseconds measuring nothing — the failure mode that made §4's compiled pipeline
   emit NaN behind sensible timings. A compute-bound bf16 GEMM was timed per SM count:
   **3.245× slower on 4 SMs, 1.637× on 8** (ideal 4.0 / 2.0). Confinement confirmed. The
   harness refuses to time and writes `REFUSED.md` if the curve is flat.
2. **Are the numerics unchanged?** Green contexts share the device address space with torch's
   primary context, so the caching allocator's tensors *should* be reachable — and *should* is
   not *is*. Actions computed on an 8-SM green stream against the default stream:
   **max |Δ| = 0.000e+00, no NaN, bitwise identical.**

**Cross-validation of the whole apparatus.** The 16-SM points are the full GPU, so they must
reproduce numbers measured by unrelated harnesses. `bf16_both` prefix **251.10 ms** against E7's
**251.28** (0.07 %); `bf16_both` expert **111.61** against §3's reconciled **111.02** (0.5 %);
eager prefix **478.08** against §1's `A_iso` **474.08** (0.8 %). The eager expert is 580.79 here
against §1's `B_iso` 523.76 — an 10.9 % gap that is **expected and accounted for**: §1's run is
the labelled `pi0_nosync` variant, which removes `1 + N` host syncs per frame, and this run is
stock π0. Four independent agreements and one difference with a known cause.

### An engineering limitation worth recording

Wrapping a green-context stream in `torch.cuda.ExternalStream` is **not sufficient**. A plain
`a @ b` runs on it, but `nn.Linear` fails: it takes cuBLASLt's fused `gemm_and_bias` path, which
returns `CUBLAS_STATUS_EXECUTION_FAILED`, after which torch's "recover by calling unfused cublas
path" raises `CUDA error: invalid device context`. `DISABLE_ADDMM_CUDA_LT=1` does **not** fix it.
What works is making the green context *current* on the thread — `cuCtxFromGreenCtx` +
`cuCtxSetCurrent` — so cuBLAS binds its handle to the context that owns the stream
(`bench/green_ctx.py::GreenCtxStream.activate`). Anyone repeating this on Jetson will hit it.

## 5b. Step 5 (energy), N = 4, and the remaining Step 3 mechanisms

### Energy: the schedule saves nothing, because it overlaps nothing

`results/20260925_E12_pi0_energy_N10_MAXN_eager_nosync`, protocol §4 as E7: in-session 60 s
idle baseline, 60 s per arm, four sysfs rails at 20 Hz plus tegrastats. Both arms run the
**same frames through the same stage functions**; only the schedule differs. The serial arm is
a whole-frame loop, **not** a prefix loop plus an expert loop — in E7 those do not sum to
`full`.

| arm | ms/frame | module power | above idle | energy/frame |
|---|---:|---:|---:|---:|
| pipelined | 1031.86 | 35 546 mW | +24 759 mW | **36 678.74 mJ** |
| serial | 1035.31 | 35 518 mW | +24 731 mW | **36 772.51 mJ** |
| **ratio** | **0.9967** | | | **0.9975** |

**No difference, which is the predicted result and the point of measuring it.** `T_pipe` equals
`A_iso + B_iso`, so a schedule that does not overlap cannot save time or energy. Had a
difference appeared it would have indicated something the timing harness missed; none did.
Energy is module power × measured frame time and carries no claim beyond those two
measurements.

### N = 4: the pre-registered sanity check failed, and that is informative

`results/20260925_E12_pi0_corun_N4_MAXN_eager_nosync`. The plan pre-registered this as a
harness check: at N = 4 the stages are more unequal, *"so overlap should be easy. If N=4 does
not approach `T_pipe ≈ A_iso`, suspect the harness before trusting N=10."*

| | N = 4 | N = 10 (§1) |
|---|---:|---:|
| `A_iso` | 475.17 ms | 474.08 ms |
| `B_iso` | 192.71 ms | 523.76 ms |
| `T_serial` | 665.63 ms | 994.38 ms |
| `T_pipe` | **666.77 ms** | **994.88 ms** |
| `r` = `A_iso`/`B_iso` | 2.466 | 0.905 |
| `T_pipe`/`A_iso` | **1.403** | 2.099 |
| `issue/wall` | **1.0000** | **1.0000** |

`T_pipe/A_iso` is 1.403, not ≈ 1.0, so **the check fails as written**. The harness is not the
cause: `issue/wall` = 1.0000 reproduces exactly at N = 4, and `T_pipe` again equals
`A_iso + B_iso` (666.77 vs 667.88). The pre-registered expectation assumed GPU-side overlap was
available; it is not, at either N. The check did its job — it confirmed the cause is systemic
rather than an N = 10 artefact, which is what a sanity check is for even when it fails.

### MPS: available on this Jetson, and moot for this experiment

Commonly assumed unsupported on Tegra. **It is not**: the control daemon started, an
`nvidia-cuda-mps-server` process spawned, and the log recorded
`NEW SERVER 501594: Ready` / `NEW CLIENT 501571 from user 1001` for a real CUDA client
(`scratch/mps_probe_control.log`). Device attributes: `COMPUTE_MODE` = 0 (DEFAULT),
`INTEGRATED` = 1, `UNIFIED_ADDRESSING` = 1.

**But it cannot help E12**, and the reason is worth stating because it is not a scheduling
fact: the two-process co-run is blocked by **memory**, not by context switching. Each process
holds its own ~15 GB π0 against Orin's 62 GB shared CPU+GPU pool, and MPS does not reduce that.
Running the two arms under MPS would reproduce the same OOM. The daemon was stopped and the GPU
verified clean before any subsequent measurement.

*Untested idea, recorded rather than pursued:* a process that loads **only** the action expert
and its KV cache, rather than the whole policy, would have a far smaller footprint and might
fit alongside a prefix process. That is a different experiment — it changes what is deployed,
not just how it is scheduled — and the green-context result in §5 already answers the question
this one was meant to answer.

### `CUDA_DEVICE_MAX_CONNECTIONS` and the own-`CUDAGraph` compiled pipeline: not done, with reasons

Both remain open, and both are now **uninformative rather than merely unfinished**, which is a
scoping judgement stated here rather than buried:

* `CUDA_DEVICE_MAX_CONNECTIONS` controls how many hardware work queues back concurrent streams.
  With `issue/wall` = 1.0000 there is never work on two streams simultaneously, so there is
  nothing for extra queues to carry. Not tested.
* The own-`CUDAGraph`, separate-memory-pool harness (§7's fallback for §4's corruption) would
  make a *compiled* in-process pipeline measurable. §5 now shows that at `bf16_both` neither
  stage saturates the SMs — the prefix loses only 27.7 % of ideal scaling at 8 SMs and the
  expert 1.9 % — so a compiled pipeline would be limited by the same host issue bottleneck and
  would measure it again. It is the right thing to build if the host bottleneck is ever fixed.

## 5. Measured / derived / not measured

**Measured.** `A_iso`, `B_iso`, `A_co`, `B_co`, `T_serial` (CUDA events on each stage's own
stream, start recorded after the cross-stream wait so waiting is not counted as work);
`T_pipe` (wall ÷ frames); host issue time; the sync sites and counts; the equivalence
differences.

**Derived.** `s`, `p`, `r`, `T_pipe/A_iso`, speed-up over serial — all quotients of the above.
`s`'s interval is a bootstrap over the co-run samples with the isolated median held fixed:
`s` is a ratio of medians from two conditions, and resampling both would blend two spreads the
report states separately.

**Not measured.**

1. **Compiled `bf16_both` co-run** — blocked by §4. Needs the own-`CUDAGraph`,
   separate-memory-pool harness §7 prescribes, with isolated stages re-measured through it and
   the equivalence check re-run.
2. **Step 3 mechanisms** — stream priority −3, CUDA graphs on both stages, green contexts,
   `libsmctrl`, MPS, two processes, `CUDA_DEVICE_MAX_CONNECTIONS`. Not attempted. Availability
   probed: driver **12.6**; `cuDeviceGetDevResource`, `cuDevSmResourceSplitByCount` and
   `cuGreenCtxCreate` all **present** in `libcuda.so.1`; `/usr/bin/nvidia-cuda-mps-control`
   **exists**. So SM partitioning and MPS are plausibly testable here rather than foregone.
3. **Step 4 (MODE_30W, 8 SMs)** — rides the capped-mode session that needs the owner's reboot,
   which §5 Step 4 and §11 both anticipate; §11 permits the MAXN part to be marked done.
4. **Step 5 energy, and N = 4 with artefacts** — N = 4 was run during harness validation
   (`B_iso` 184.18 ms after de-syncing) but not written to a result directory.
5. **nsys timeline of co-run frames** — the §11 item that would decompose `B_co − B_iso` into
   kernel slowdown, queueing and host issue delay. Not taken, because `B_co − B_iso` is
   −2.3 ms: there is no slowdown to decompose in this configuration. It becomes necessary as
   soon as a configuration shows one.

## 6. Self-review (protocol §11)

* **The number most likely to be misread is `s`.** Section 1 states why, the printed verdict
  splits contention from bottleneck from overlap, and `corun_ratios.csv`'s `notes` field
  carries the same caveat, because that file is what gets read directly.
* **A harness change is labelled, not folded into the baseline.** `pi0_nosync` has its own
  mechanism name in every row and its own equivalence pass. ~~Precomputing my own timestep
  tensors is *not* labelled, because §7 names it as the example of a permitted harness fix.~~
  **Corrected 2026-09-25 (§3).** That was written believing the change was free; it is worth
  **33.6 % of the expert stage**, and it is what made this report's expert disagree with E7's
  `ae` by 50.8 %. §7 permitting the class of change does not license leaving a third of a
  stage unlabelled. It is labelled now. **The general lesson is the one this campaign keeps
  relearning: "permitted by the protocol" was allowed to stand in for "measured to be
  small", and the two are not the same.** The cost was never measured until the disagreement
  with E7 forced it.
* **The isolated reference is the mean of both brackets** (drift 0.6 % prefix, 0.7 % expert),
  so drift is shared across the ratio rather than attributed to whichever end ran first. Both
  bracket medians are in `corun_summary.csv`.
* **In-session vs E7.** `A_iso` 474.08 ms here vs E7's 251.28 ms is the eager/`bf16_both`
  difference, not drift; this run is eager by necessity (§4). The expert side was reconciled
  separately and completely in §3: at E7's own rung the two harnesses agree to **0.16 %** once
  the unlabelled timestep change is undone. Two candidate explanations were measured and
  rejected first (sampler +0.19 %, in-process residency +0.29 %), so the agreement is not a
  coincidence of offsetting errors.
* **What would change the conclusion.** A configuration where `issue/wall` is well below 1.0
  and `T_pipe` falls below `A_iso + B_iso`. Everything in §5's "not measured" list is aimed
  there. Until one exists, the honest statement is that **E12 has not yet tested GPU
  contention on Orin** — it has established why the obvious attempt does not, which is a
  different and smaller claim.
* **Not tuned in either direction.** Three defects in this harness were found and fixed before
  any number was reported: a negative control that was not actually stale, frame 0 being judged
  by a control that cannot apply to it, and a guard on `assert_real_weights` that could never
  fire. Two were found by reading the code, one by the diagnostic added to test a hypothesis
  that then turned out to be wrong.
