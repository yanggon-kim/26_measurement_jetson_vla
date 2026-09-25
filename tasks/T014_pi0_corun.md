# T014 — π0: co-run the VLM prefix and the action expert on one GPU (E12)  · priority P1 · depends on: T004 (E7 at MAXN — done) · next after the task in progress (order table in `docs/01_measurement_plan.md`)

**The specification is `docs/E12_corun/README.md` — read all of it before starting; it carries the purpose,
the definitions, the five non-negotiables, the steps, the outputs and the acceptance criteria.** Binding rules:
`docs/02_protocol.md`, including §14 (E12) and §1 (the capped-mode session). This file only schedules the work.

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
