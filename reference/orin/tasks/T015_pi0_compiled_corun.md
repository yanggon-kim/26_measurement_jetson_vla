# T015 — π0: the compiled co-run — fair expert, whole GPU, and both stages together inside green-context partitions (E12b)  · priority P1 · depends on: T014 (MAXN part — its results exist) · next after the task in progress (order table in `docs/01_measurement_plan.md`)

**The specification is `docs/E12b_compiled_corun/README.md` — read all of it before starting; it carries the
problem T014 left open, the purpose, the definitions, the five non-negotiables, the measurements M0–M5, the dated
pre-registered thresholds and outcomes, the outputs and the acceptance criteria.** Binding rules:
`docs/02_protocol.md` §15 (E12b) and §14 (E12). This file only schedules the work. The research side's review of
T014's return, which this task answers, is `analysis/E12_review.md`.

**Why this task exists.** T014 did not answer its question: the only co-run is eager (host-bound, nothing
overlapped); the compiled co-run silently produced corrupt / all-NaN actions and was not repaired; green contexts
work but each stage was timed alone in its partition, so bandwidth coupling was never observable. E12b measures the
compiled co-run on the whole GPU and — the decisive part — **both stages running together inside green-context
partitions**, against each stage alone in the same partition, so memory-side coupling is separated from SM
sharing. **The result may go against the research side; that is fine — report what the device does, and do not
tune the harness in either direction.**

When: finish the queue stage or task you are executing when you read this, then take T015 **before any other open
MAXN work**. If the human has already rebooted into `MODE_30W` / `MODE_15W` when you read this, finish the capped
queues first and take T015 once MAXN is restored — **never ask for a reboot for it**. If the human owner has given
you a different order directly, the owner's order wins; say in `STATUS.md` where T015 sits.

Model and configuration (fixed): LeRobot π0 `lerobot @ f0d2b37bebddf7e8852cb624712da0ad564601b6`, checkpoint
`lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f`, rung `bf16_both` exactly as in
`results/20260924_E7_pi0_N10_bf16_both_MAXN/meta.json`, batch 1, N = 10, MAXN + `jetson_clocks`. The only model
change is the **timestep hoist** (the N timestep tensors built once), which M1 verifies; **no `cropcache`**, no other
change to the rung. Harness changes (own CUDA-graph capture, double-buffered I/O, green contexts, issue order,
priorities) are labelled variants, each in every row's variant label.

Measurements (details, latitude and hazards in the specification):
- **M0** — gate; graph-replay confinement control on a green-context stream; granted partition sizes.
- **M1** — the fair expert: `stock` vs `hoist`, compiled, `prefix` / `ae` / `full` loops bracketed; compiled
  hoist equivalence check. Gates M2–M5.
- **M2** — diagnose and fix the compiled two-stream corruption; compiled co-run on the whole GPU; `s`, `p`, `r`,
  `O`, `φ`; host-issue record; nsys timeline of 3–5 frames and the decomposition.
- **M3** — both stages together in green-context partitions, ≥ 3 splits (incl. 12 / 4 and 8 / 8): `A_part`,
  `B_part`, co-run, `p_part`, `s_part`, `O`; in-process bandwidth-hog positive control at ≥ 2 rates.
- **M4** — DRAM / EMC attribution (`EMC_FREQ` calibrated against the hog, memory rail, nsys).
- **M5** — energy per frame, whole-frame loops: stock `full`, fair `full`, best valid co-run.
- Optional if cheap: priority −3, `CUDA_DEVICE_MAX_CONNECTIONS`, `libsmctrl`, N = 4 co-run.

Non-negotiables (spec §5): fresh KV verified in the compiled path that is timed, stale control must fail; no
silent NaN — a finite check on every action tensor, a non-finite run is invalid and reported; same-session
bracketed baselines through the same harness; complete provenance with every harness change in the row label and
no null in `meta.json`; MAXN only, nothing unapproved (the channel-partition measurement (EV-02), P2 = `bf16_both + cropcache`, the
`check_model_inputs.py` `MODELS` change stay **not approved**).

Result directories `results/<date>_E12b_pi0_*` (names in spec §11); report `reports/E12b_pi0_compiled_corun.md`.
Acceptance: spec §12 — **do not mark this task done until every item holds or the status line names the one that
does not and why.** When done, also append to T014 the status line given in spec §14. Spec §14 also carries a
correction owed to your T014 Step-4 prediction in `STATUS.md` (the eager expert is host-bound and `MODE_30W` slows
the CPU too); please restate that prediction before the `MODE_30W` session.

status: claimed 2026-09-25 (owner directed: taken ahead of the pre-reboot queue, which is paused)

status: done 2026-09-25 (MAXN). Every acceptance item of spec §12 holds; the one deliberate deviation
(M4 bandwidth sampled in separate runs rather than alongside the timing runs, so no headline timing
carries a sampler) is stated in the report's §12 review. Report: `reports/E12b_pi0_compiled_corun.md`.
Results: `results/20260925_E12b_pi0_{fairexpert,corun,green_corun,emc_calibration,energy}_*`.
bench: `bench/e12b_*.py`, `bench/green_ctx.py`, `bench/queues/queue_e12b.sh`.
Answer: best valid co-run O = 1.247 (green 12/4) — poor overlap; whole GPU O = 1.39 (head-of-line
blocking: 3.1 % concurrency, expert queued 175 ms/frame behind 1008-CTA prefix grids); partitioned, the
stages still couple through memory (s_part 1.16–1.26, hog control passing). Rows landed: 2, 4 (in part),
5; not 1, 3, 6, 7, 9, 10; row 8 not for the fair expert (110.95 ms) but yes for the labelled, bitwise-
equivalent owngraph_expert (101.2 ms) — the research side's call. Fair expert: bf16_both + hoist,
bitwise equivalent to stock, 110.95 ms. Co-run energy 1.020 x fair serial per frame.
Corrections owed and made: T014's compiled green-context rows were not confined (M0) — see STATUS.
