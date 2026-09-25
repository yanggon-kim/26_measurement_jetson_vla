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
