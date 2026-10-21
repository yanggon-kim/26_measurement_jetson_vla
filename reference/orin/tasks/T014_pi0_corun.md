# T014 — π0: co-run the VLM prefix and the action expert on one GPU (E12)  · priority P1 · depends on: T004 (E7 at MAXN — done) · next after the task in progress (order table in `docs/01_measurement_plan.md`)

**The specification is `docs/E12_corun/README.md` — read all of it before starting; it carries the purpose,
the definitions, the five non-negotiables, the steps, the outputs and the acceptance criteria.** Binding rules:
`docs/02_protocol.md`, including §14 (E12) and §1 (the capped-mode session). This file only schedules the work.

> **Research-side note, 2026-09-25:** the open MAXN items of this task (the compiled co-run, the nsys timeline,
> the Step-3 mechanisms, `equivalence.*`, `mechanisms.csv`, compiled co-run energy) are taken over by **T015**
> (`docs/E12b_compiled_corun/README.md`; review of this task's return: `analysis/E12_review.md`). Step 4 stays
> here, with the `MODE_30W` session.

The research side considers this the most decisive measurement in the current thread. The single number that
decides the question is **`s` = the expert's stage time under co-run ÷ its isolated stage time**, against
`A_iso / B_iso` ≈ 1.50 (E7, π0 `bf16_both`, N = 10, MAXN). **The result may go against the research side;
that is fine — report what the device does, and do not tune the harness in either direction.**

When: finish the task you are executing when you read this, then take T014 **before any other open MAXN work**
(T012 if not yet under way, the remaining E5 / E5b of T007 / T008, `MODE_50W`). Steps 1–3 and 5 run at MAXN.
**Step 4 is not a separate session**: it runs inside the `MODE_30W` session of T004 step 4b (protocol §1) — no
extra reboot, and nothing new to ask anyone. Until that session happens, T014's MAXN part can be marked done
with Step 4 left as its open item (append `status: done (MAXN) <UTC date>; Step 4 waits for MODE_30W`).

Model and configuration (fixed): LeRobot π0 `lerobot @ f0d2b37bebddf7e8852cb624712da0ad564601b6`, checkpoint
`lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f`, rung `bf16_both` exactly as in
`results/20260924_E7_pi0_N10_bf16_both_MAXN/meta.json`, batch 1, N = 10 (Step 5 adds N = 4), MAXN +
`jetson_clocks` (Step 4: `MODE_30W` + `jetson_clocks`). No `cropcache`, no change to the rungs.

Steps (details, latitude and hazards in the specification):
1. Steady-state pipeline, prefix(k) ∥ expert(k − 1) on two streams in one process; `A_iso`, `B_iso`, `T_serial`
   in the same session; `A_co`, `B_co`, `T_pipe`; `s`, `p`, `r`; nsys timeline of 3–5 co-run frames and the
   "why" of `B_co − B_iso`. Fresh-KV equivalence check (spec §6) with its negative control **before** the
   numbers are reported.
2. Prefix slowdown `p` from the same runs.
3. Software mitigations: stream priorities, CUDA graphs on both stages, green contexts, `libsmctrl`, MPS, two
   processes, anything else a reviewer would name — measured, or unavailable with committed evidence.
4. Step 1 again in the `MODE_30W` session (8 SMs), compared with MAXN.
5. Step 1 at N = 4; energy per frame of the pipeline against the serial `full` loop, E7 method.

Result directories `results/<date>_E12_pi0_*` (names in spec §10); report `reports/E12_pi0_corun.md`.
Acceptance: spec §11.

status: claimed 2026-09-25

status: done (MAXN eager) 2026-09-25; compiled baseline, Step 3, Step 4 and Step 5 energy remain open

**What is measured.** `results/20260925_E12_pi0_corun_N10_MAXN_eager_nosync/`, n = 100 frames per
condition, bracketed in one session. Report: `reports/E12_pi0_corun.md`. bench commit `2685b9245576`.

**The answer for eager: the stages do not overlap, and the host is why.** `T_pipe` 994.88 ms
equals `A_iso + B_iso` 997.83 ms (ideal would be `max(A,B)` = 523.76), speed-up over serial
0.9995, and **host `issue/wall` = 1.0000 over a 99.5-second run** — the CPU issues kernels for
essentially the whole wall clock, so the two streams never both hold resident work.

**`s` = 0.9956 must not be read as "co-running is free".** It is what ideal overlap AND zero
overlap both look like in per-stage timings, since a stage alone on the GPU measures its
isolated time either way. Only `T_pipe` separates them. Reported from `s` alone these numbers
would have asserted one of §9's two pre-registered outcomes, falsely.

**π0 eager cannot answer KerColle's question.** That mechanism is GPU contention; a
launch-bound workload (E5: 84.2% idle, 968 kernels/step) never fills the GPU for contention to
arise. The compiled rung is where it becomes answerable — and that path corrupts when
pipelined (below).

**Fresh-KV verified (non-negotiable 1).** Eager pipelined actions are **bitwise identical** to
serial (6/6, max abs diff 0.000e+00); the negative control — expert given frame *k−2*'s cache
— is detected 5/5. Five real LIBERO frames were captured so the check has power at all;
`obs_0` reproduces the committed fixture bit-identically in decoded pixels.

**Two findings beyond the headline.**
1. **π0's shipping model synchronises 1 + N times per frame** — `modeling_pi0.py:681` per
   prefix, `:746` per denoise step, both materialising a constant `att_masks`. This blocks
   two-stream overlap on ANY GPU, independent of Orin. De-syncing them (labelled variant
   `pi0_nosync`, equivalence re-verified bitwise) also made the **expert ≈19% faster at N = 4**
   (226.81 → 184.18 ms) — an optimisation for the shipping model, independent of E12.
2. **The stock compiled `bf16_both` path corrupts when pipelined**: max abs diff 1.273 on one
   run, all-NaN on another, from identical code, while compiled SERIAL is correct and eager
   PIPELINED is exact. `TORCH_LOGS=cudagraphs` says nothing — it fails silently. Hazard §7,
   predicted in the spec. No compiled co-run number is reported.

**`r` is configuration-dependent.** 0.905 here (eager: the expert is longer) vs E7's 1.50
(`bf16_both`: the prefix is longer). Compiling speeds the expert more than the prefix and flips
which stage sets the period, so `r` must come from the same session — §3 already requires this.

**Open, and listed in the result's `meta.json` under `not_measured` rather than left implicit:**
the compiled co-run (needs the own-`CUDAGraph` fallback of §7); Step 3's mechanisms (green
contexts and MPS both probed as present on CUDA 12.6, so testable); Step 4 at MODE_30W (rides
the owner's capped-mode reboot, as §11 permits); Step 5 energy and an N = 4 result directory;
and the nsys timeline, which is unnecessary while `B_co − B_iso` is −2.3 ms because there is no
slowdown to decompose.

status: updated 2026-09-25 — Step 3 ANSWERED with green contexts; the co-run question is settled for Orin

**The question T014 exists to settle now has a measured answer: on Orin's 16 SMs, CTA-level
contention cannot be what limits co-running π0's prefix and expert, because the expert does
not occupy the SM array at all.**

`results/20260925_E12_pi0_green_N10_MAXN_eager` and `..._bf16both`, π0, N = 10, MAXN, medians
over n = 30–180 per point. Time at 4 SMs relative to 16:

| stage | eager | `bf16_both` |
|---|---:|---:|
| prefix | 1425.07 ms, **2.981×** | 491.60 ms, **1.958×** |
| expert | 588.64 ms, **1.013×** | 117.05 ms, **1.049×** |

The expert is flat across a 4× reduction in SMs; the prefix is not, so the apparatus has power.

**Why this was the route left.** Every direct co-run route is closed and each closure is itself
a recorded result: one Python thread cannot feed two streams (§13.7 forbids a second; T010's
numbers were GIL artefacts), the compiled rung corrupts across two streams (max |Δ| 1.273 /
all-NaN, silently), and two processes are OOM-killed because Orin's 62 GB is shared CPU+GPU and
one π0 at `bf16_both` is ~15 GB. Green contexts partition SMs *within* one process, so they
dodge the memory limit and invert the question: instead of "do the stages slow each other
down", ask "does either stage use the SMs the other would take". Necessary-condition logic —
weaker than exhibiting interference, and not confounded by the host bottleneck that defeated
every direct attempt.

**Mechanism, from E5b's grid sizes.** The expert's kernels launch 1–128 CTAs of 128 threads;
two elementwise kernels launch a **single** CTA and so occupy one SM by construction. At Orin's
1536 threads/SM every kernel but `Kernel2` has its whole grid resident even on 4 SMs, running
in one wave regardless of partition size. (Derived from the spec occupancy ceiling — the E5b
exports lack `launch__occupancy_limit_*` — so it explains the measured curve rather than
standing as its own measurement.)

**Controls, both required to pass before any timing was kept.** A compute-bound GEMM slowed
3.245× on 4 SMs and 1.637× on 8 (ideal 4.0/2.0), confirming the partition actually confines
work — the harness writes `REFUSED.md` and takes no timings if that curve is flat. Actions on
an 8-SM green stream matched the default stream **bitwise (max |Δ| = 0.000e+00)**.

**Cross-validation.** The 16-SM points are the full GPU and reproduce unrelated harnesses:
`bf16_both` prefix 251.10 vs E7's 251.28 (0.07 %), expert 111.61 vs the reconciled 111.02
(0.5 %), eager prefix 478.08 vs `A_iso` 474.08 (0.8 %). The eager expert differs by 10.9 % for
a known reason — the §1 run is the labelled `pi0_nosync` variant, this one is stock π0.

**Derived bound, not a measurement:** pairing each stage at its best partition (expert 4 SMs,
prefix 12) gives a period of 588.64 ms eager / 271.26 ms `bf16_both` against serial 1058.87 /
362.71 — **1.799× / 1.337×**. That is the ceiling on what fixing the host issue bottleneck
could buy, conditional on two green contexts running concurrently, which was NOT measured. No
`T_pipe` is claimed from it.

**Engineering limitation recorded for anyone repeating this:** wrapping a green-context stream
in `torch.cuda.ExternalStream` is not enough. `nn.Linear` takes cuBLASLt's fused `gemm_and_bias`
path, which returns `CUBLAS_STATUS_EXECUTION_FAILED` on such a stream, and torch's fallback then
raises `CUDA error: invalid device context`. `DISABLE_ADDMM_CUDA_LT=1` does not help; making the
green context current via `cuCtxFromGreenCtx` + `cuCtxSetCurrent` does.

**Still open:** MPS and `CUDA_DEVICE_MAX_CONNECTIONS`; two green contexts running concurrently
(needs the host-issue problem solved); the own-`CUDAGraph` compiled pipeline; Step 4 at
MODE_30W (owner's reboot); Step 5 energy and an N = 4 result directory.

status: MAXN items superseded by T015 2026-09-25; Step 4 waits for MODE_30W

status: done 2026-09-27 — Step 4 (the last open item) was measured in the MODE_30W session of 2026-09-26
- `results/20260926_E12_pi0_{gemm_control,compiled_stages_N10,corun_N10_*_eager_nosync,equiv,green_N10_*_eager,energy_N10_*_eager_nosync}_MODE_30W`:
  GEMM control 2.23× vs the 2.125× clock ratio; compiled expert 1.92× (inside the bracket), compiled prefix 2.68×
  (above it — the restated prediction falsified for that stage); eager co-run serial (T_pipe 2439.1 vs 2441.6 ms),
  equivalence PASS 6/6 bitwise, stale control 5/5, energy pipelined/serial 0.9996. Beyond this task the same queue
  ran at MODE_15W (6 SMs) on 2026-09-27: same picture (`…_MODE_15W`). Addendum in `reports/E12_pi0_corun.md`,
  tables in `reports/E7_capped_modes.md` §3. The MAXN items were closed by T015.
