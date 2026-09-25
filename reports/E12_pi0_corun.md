# E12 — co-running π0's VLM prefix and action expert on one Orin GPU

**Task T014. Specification `docs/E12_corun/README.md`; binding rules protocol §14, §13.7, §1,
§3.5, §5. This report covers the MAXN eager part. Three parts are open and named in §7 below.**

## The answer

**On one Orin GPU, π0's two stages do not overlap — and the reason is the host, not the GPU.**

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

That is **1 + N host syncs per frame** (11 at N = 10). A sync in either stage's issue path
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
  mechanism name in every row and its own equivalence pass. Precomputing my own timestep
  tensors is *not* labelled, because §7 names it as the example of a permitted harness fix.
* **The isolated reference is the mean of both brackets** (drift 0.6 % prefix, 0.7 % expert),
  so drift is shared across the ratio rather than attributed to whichever end ran first. Both
  bracket medians are in `corun_summary.csv`.
* **In-session vs E7.** `A_iso` 474.08 ms here vs E7's 251.28 ms is the eager/`bf16_both`
  difference, not drift; this run is eager by necessity (§4).
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
