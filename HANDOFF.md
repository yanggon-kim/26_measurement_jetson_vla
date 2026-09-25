# HANDOFF — jetson

*Root: `/home/yanggon/05_VLA_LPDDR/jetson` · Owner: `jetson-owner` · Last rewritten: 2026-09-25 by jetson-owner.*
*State only — rewritten, never appended. History (every dated section, the 2026-09-22 block, old decisions and
issues): `HANDOFF.history.md`. Full reviews: `analysis/E12_review.md`, `analysis/E12b_review.md`.*

## State now

- **Device:** live (20-min `PROGRESS` ticks; last seen 2026-09-25 15:27 PDT). Queue per its `STATUS.md`: the
  pre-reboot queue (E3/E4 N sweep → GR00T N1.5/N1.6 E6 → E8 LIBERO), then the capped-mode reboot session.
- **Closed:** T001–T003, T005, T010–T012 done; T007 done 2026-09-25; **T014** (E12, MAXN) done and superseded by
  **T015** (E12b), done 2026-09-25 and reviewed. T013 withdrawn. **T004 re-opened by the device**: E6 was never
  delivered (its own correction block, 2026-09-25); E5 and E7 (MAXN, 14/14) are complete.
- **Still claimed:** T008, T009. T006 has no status line though delivered in substance by the six-model campaign
  (verify before closing).
- **Local, not pushed:** `0a08832` (E12b review + E12 review correction), `cdd0bdc` (P1 → EV-02 relabel), and this
  rewrite. Push only on approval.
- **Standing item — capped power modes (user's decision 2026-09-24: the device's business, never escalate):**
  `MODE_30W` (id 2) and `MODE_15W` (id 1) each need one human reboot; they come last in the device's order. T014
  Step 4 and the E7 capped points ride that session (`bench/queues/queue_e7_capped.sh`,
  `queue_e12_capped.sh`). Carry this row on every pull; raise it only if the device marks it `blocked` or asks.

## Current numbers (π0 LeRobot `lerobot @ f0d2b37beb`, MAXN, bf16) — sources, not re-derived here

| quantity | value | source |
|---|---|---|
| best frame, `bf16_both`, N = 10 | 424.46 ms; split 13.4 / 45.4 / 39.7 % (vis / prefix / ae) | `results/20260922_E4c_pi0_opt_ladder/opt_ladder.csv`, `reports/E4d_phase_balance.md` |
| E7 loops: prefix / ae / full | 251.28 / 167.36 / 424.68 ms; 6395.8 / 3112.5 / 13 666.2 mJ | `reports/E7_power_energy.md` |
| expert per step, best | 10.37 ms (`reduce-overhead` + crop-not-clone KV); floor 3.115 spec / 3.403 meas. | `results/20260921_E4_pi0_summary.csv` |
| fair expert (hoist, bitwise = stock) | 110.95 ms; fair serial frame 354.74 ms, 17 552 mJ | E12b, review §1 |
| `owngraph_expert` (loop as one graph) | 101.21 ms — baseline membership is the user's call | E12b |
| best GPU co-run `O` | 1.247 (green 12/4); whole GPU 1.29–1.40 | E12b; `results/20260925_E12b_pi0_*` |
| 12/4 coupling | `p_part` 1.075, `s_part` 1.197, `p_hog` 1.043 | E12b review |
| co-run energy | +2.0 % per frame vs fair serial | E12b M5 |
| stock timestep tensor cost | 56.07 of 167.09 ms (33.6 %) of the expert | `analysis/E12_review.md` |

Verdict (case 001): framing I did not happen on Orin; PIM frame-rate gain over the best GPU co-run ≤ 1.247×
`[derived]`, and only if the prefix loses nothing to the PIM → EV-02 is the next question. Device commits,
dirs and n: `analysis/E12b_review.md` §1.

Standing reading of the campaign (details in `HANDOFF.history.md`, "What contradicts…"): the expert dominates only
the eager baseline in all six models; it is memory-bound only after compilation; latency is not an energy proxy
(saving ratio 1.05–0.45); quote GPU idle 75.6 % of `ae_step` (nsys-corrected), not 84 %.

## Open issues

- **(blocks energy claims) E7 residuals:** stage loops 418.64 vs `full` 424.68 ms (6.04 ms unplaced) and 9509 vs
  13 666 mJ (30 %). No GPU-vs-PIM energy until reconciled; latency unaffected. Fix = a question in `docs/04`
  about loop definitions / idle subtraction, not a re-measurement.
- **Undiagnosed corruption of the stock compiled co-run** (E12b M2): fixed, but the cause is a two-suspect list
  (shared graph-tree pool; next prefix replay overwriting the expert's KV); evidence script
  `e12b_diagnose.py` cited but uncommitted. Numbers stand; diagnosis owed.
- **Step-4 note owed to the device (not written):** its restated MODE_30W prediction uses the withdrawn unconfined
  compiled rows (113.70 / 320.71 ms at 8 SMs); confined references are 149.05 / 326.94; memory-bound stages must
  not be scaled by GPU clock alone.
- **E12b minor gaps:** hog has one effective rate at 12/4 and 14/2 (fails at 14/2); `variant_tokens` incomplete and
  its `kv_copy` contradicts the code; no `graphs_both`-prefix vs rung equivalence. T014's compiled green rows are
  withdrawn (correction block in `analysis/E12_review.md`).
- **Not approved — never write as tasks without the user's explicit, current approval:** EV-02 (prefix with the
  GPU on 8 of 16 LPDDR channels; formerly "P1"); P2 (`bf16_both` + `cropcache`, ≈ 60 ms); adding
  `groot_n16` / `groot_n15` to `MODELS` in `docs/reference_scripts/check_model_inputs.py`.
- **Open decisions for the user:** does `owngraph_expert` join the GPU baseline; does the timestep hoist (OD-1)
  enter it (if so the PIM latency gain falls to ≈ 1.03–1.07× at 16 ch `[derived]`).
- **Carried from 2026-09-22 (user actions, unchanged):** leaked session token in history (`410a535`, nsys files;
  no rewrite; repo must be private); passwordless sudo not configured on the device; bandwidth peak varies 1.19×
  between processes (quote floors against the same-session peak); π0 is the LeRobot port, not openpi.
- **Device-raised, answers owed:** its three Part-C questions (`docs/04`); π0's E11f two-stream rows predate the
  GIL fix `773d2a9` (re-run queued); four `.ncu-rep` (725 MB) live in private HF dataset `yang-gon/vla-orin-ncu-raw`;
  GR00T N1.7 `vis_compile` structurally unfixable.

## Next steps

1. On each pull: fetch, read `STATUS.md` / `PROGRESS.md`, new `results/` and `reports/`; check repo size
   (`find results -size +50M`, sample `raw_files` sha256).
2. Get approval to push `0a08832`, `cdd0bdc` and this rewrite.
3. When approved: write the Step-4 note and the E7-residual question into `docs/04`; answer the Part-C questions.
4. When the device reports T004 E6 or the capped-mode session: review against the task's acceptance and the
   E12 spec's Step 4; hand numbers to main as "for <owner>:" with dir, device commit, power mode, n.
5. EV-02 / P2 / `MODELS` only after explicit user approval (spec style: `docs/E##_*/README.md` + task + order row +
   protocol §).

## Interfaces with other sub-projects

- → microarch-owner: latency / power / kernel data for `host_model.py`, `energy_model.py`, roofline points; hybrid
  model input `results/<date>_E11_<model>_inputs/model_inputs.csv` (protocol §13.8; mapping in `docs/00_context.md`).
- → brainstorm-owner: case 001 (co-run framing) — `O`, `p_part`, `s_part`, `p_hog`, fair `B`.
- → paper-owner: baseline ladder, occupancy and instruction-level evidence, LIBERO results.
- Every number leaves through main as "for <owner>:" and reaches paper / site via evaluation-owner.
- Names: EV-02 = channel-partition measurement (formerly "P1"); the ISA mechanism is CoRe (not in this repo);
  priority tier "P0–P3" and SASS `@P1` unchanged.
