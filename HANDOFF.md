# HANDOFF — jetson

*Root: `/home/yanggon/05_VLA_LPDDR/jetson` · Owner: `jetson-owner` · Last rewritten: 2026-09-26 by jetson-owner.*
*State only — rewritten, never appended. History (older state, dated blocks, old decisions and issues):
`HANDOFF.history.md`. Full reviews: `analysis/E12_review.md`, `analysis/E12b_review.md`.*

## State now

- **Device:** in the capped-mode session — E7 `MODE_30W` arms (SmolVLA, GR00T) landing at `7970a1b` (last pull; not
  yet reviewed). T016 runs once MAXN is restored (its own "When" clause); it has not started.
- **Issued, pushed: T016** (`tasks/T016_prefix_bandwidth_and_followups.md`, order row 6c; protocol §16; `docs/04`
  2026-09-26 entry). Five steps: (1) EV-02 prefix under stolen bandwidth — sweep 0–94 GB/s isolating SM loss,
  37 % duty variant, channel-confinement feasibility probe; (2) seeded LIBERO N ∈ {1,2,3,4,6,10} × 4 suites +
  repeat control; (3) prefix savings as labelled variants: π0 `nopad`, GR00T `keep1`, `kvonce`; (4) E7 energy
  residual; (5) E12b corruption 2 × 2. Pre-registered bands: σ@94 GB/s < 1.04 / 1.04–1.247 / > 1.247 (full and
  37 % duty); N = 2 replicated iff ≥ 3 of 4 suites and pooled McNemar p < 0.05; repeat agreement ≥ 95 %.
- **T016 Amendment 1 (2026-09-26, pushed before any measurement; from brainstorm case 003):** 1a's primary load is
  **SM-free** (copy engine pinned-host↔device or CPU-side), prefix unmodified on all 16 SMs, ≥ 3 points with
  90–130 GB/s left; green-context load demoted to an optional secondary arm. New **1d = EV-02c**
  (`…_E13d_pi0_prefix_kernel_bytes_MAXN`): per-kernel `lts__` fills / writes split weight re-reads (H1) from
  activation re-reads (H2); row 8 added (weight share ≥ 2/3 / ≤ 1/3 / mixed). 1c reads FACIL first (our survey: its
  per-allocation map is proposed MC hardware, not stock Orin — device to confirm). Step 3 rows must carry DRAM bytes.
  Rows 1–7 unchanged. Protocol §16 and plan row 6c amended to match.
- **Closed:** T001–T003, T005–T012 done (T006 / T008 / T009 closed 2026-09-26); T014 MAXN superseded by T015
  (done, reviewed); T013 withdrawn. **T004**: re-opened by the device for missing E6; `reports/E6_pi0_roofline.md`
  and `results/20260925_E6_pi0/` now exist but the task file has no closing status line — verify and ask it to close.
- **Pushed:** everything of ours, including T016 and this rewrite.
- **Standing item — capped power modes (user's decision 2026-09-24: the device's business, never escalate):**
  `MODE_30W` (id 2) and `MODE_15W` (id 1) each need one human reboot; they are now the **only** remaining work
  (`bench/queues/queue_e7_capped.sh`, `queue_e12_capped.sh`; step 0c measures compiled stages at 30W). The device
  never confirms a reboot prompt itself (`STATUS.md` "NEXT, needs you"). Carry this row on every pull.

## Current numbers (π0 LeRobot `lerobot @ f0d2b37beb`, ckpt `45dcc8fc`, MAXN, bf16) — sources, not re-derived

| quantity | value | source |
|---|---|---|
| best frame, `bf16_both`, N = 10 | 424.46 ms; split 13.4 / 45.4 / 39.7 % (vis / prefix / ae) | `results/20260922_E4c_pi0_opt_ladder/opt_ladder.csv` |
| E7 loops: prefix / ae / full | 251.28 / 167.36 / 424.68 ms; 6395.8 / 3112.5 / 13 666.2 mJ | `reports/E7_power_energy.md` |
| expert per step, best | 10.37 ms; floor 3.115 spec / 3.403 meas. | `results/20260921_E4_pi0_summary.csv` |
| fair expert (hoist) / fair serial frame | 110.95 / 354.74 ms, 17 552 mJ | E12b, review §1 |
| compiled stages alone, whole GPU (n = 60) | prefix 246.79, expert 101.48, serial 348.73 ms | `results/20260925_E12_pi0_compiled_stages_N10_MAXN/stages.csv` (bench `352f14f`) |
| same, green 8/8 (n = 60) | prefix 327.70, expert 149.41, serial 478.61 ms | `…_MAXN_green8_8/stages.csv`; reproduces E12b M3 within 0.25 % |
| best GPU co-run `O` | 1.247 (green 12/4); whole GPU 1.29–1.40 | E12b |
| DRAM bytes per frame (L2 fills + writes ≤) | prefix 21.64 GB (3.70× E6 analytic 5.85), expert 8.21 GB (1.29× E6) | `results/20260925_E12b_pi0_dram_bytes_MAXN/dram_bytes.csv` (bench `9df5628`) |
| real DRAM draw alone | prefix 87.85 GB/s (reads only 72.3), expert 81.03 GB/s; EMC-calibrated 84.6 / 80.6 | same; EMC / measured 0.963 / 0.995 |
| prefix floors | memory ≈ 115 ms at real traffic ≈ compute 120.9 ms (both roofs) | `reports/E12b_pi0_compiled_corun.md` addendum |
| LIBERO success, stock vs `bf16_both` N = 10 | 252 vs 249 / 400 pooled (McNemar p 0.87); per suite ≤ 2 ep. apart | `reports/E8_libero.md` (bench `d51ccc6`) |
| LIBERO spatial, `bf16_both` N = 10 / 4 / 2 | 62 / 74 / 85 of 100; forward 430.7 / 329.2 / 294.7 ms | same |
| LIBERO run-to-run (stock spatial twice) | 63 vs 56 / 100, 65 % episode agreement | same §4 |

Readings: `bf16_both` is accuracy-neutral within noise (gaps 1–3 ep. vs a 7-point self-repeat; per-arm 95 % CI
±9–10 pts). N = 2 beating N = 10 by 23 pts (p = 0.0002) exceeds the noise; N = 4 (+12, p = 0.096) does not.
π0's flow noise is unseeded torch RNG, so arms are not paired in policy noise. EV-02a: the prefix alone draws
~88 GB/s ≈ 94 % of an 8-channel half (~94 GB/s), not ~26 % — the E6-bytes premise of EV-02 was wrong.
Case 001: PIM frame-rate gain over the best GPU co-run ≤ 1.247× `[derived]`, and only if the prefix loses
nothing to the PIM — which a ~90 %-of-half-bus prefix makes unlikely.

## Open issues

- **(blocks energy claims) E7 residuals:** stage loops 418.64 vs `full` 424.68 ms and 9509 vs 13 666 mJ (30 %).
  Now T016 step 4.
- **E12b M2 corruption cause** (T016 step 5) stays a two-suspect list (`e12b_diagnose.py` never existed in git); no
  one-change-at-a-time isolation; no `graphs_both`-prefix vs rung equivalence. `hog_matched` was ~22 % light
  (matched to E6 bytes).
- **Corrections to carry downstream:** E5b S-sweep CUDA-event times were +45–75 µs harness overhead
  (`ae47c79`); GR00T largest DiT projection is 1.17–1.20× its floor (was 1.6×), `to_q` 1.44–1.47× (was
  3.5–3.7×) — `reports/E5b_groot.md`. `opt_ladder.py` wrote π0's checkpoint into π0.5 / SmolVLA `meta.json`
  (fixed, data unaffected). E8 no-settle arms (18 / 24 of 100) superseded.
- **Not approved — never write as tasks without the user's explicit, current approval:** P2 (`bf16_both` +
  `cropcache`); `groot_n16` / `groot_n15` in `check_model_inputs.py` `MODELS`. (EV-02 and the seeded E8 follow-up
  were approved 2026-09-26 → T016.)
- **Open decisions for the user:** does `owngraph_expert` (101.21 ms) join the GPU baseline; does the timestep
  hoist (OD-1) enter it (PIM latency gain then ≈ 1.03–1.07× at 16 ch `[derived]`).
- **Device questions, answers owed:** `owngraph_expert` (above); back-fill `model_commit: null` in older
  `meta.json` or accept per-report citation; its three Part-C questions (`docs/04`). E8 seeding answered (T016).
- **Carried user actions (unchanged since 2026-09-22):** leaked session token in history (`410a535`; repo must stay
  private); no passwordless sudo on the device; bandwidth peak varies 1.19× between processes; π0 is the LeRobot
  port, not openpi. Tracked repo 413.9 MB (results 408.6 MB), largest file 13.6 MB.

## Next steps

1. On each pull: fetch, read `STATUS.md` head, new `results/` and `reports/`; repo size check. Watch for
   `results/*_E13*`, `reports/E13_pi0_bandwidth_steal.md` first; review against T016's pre-registration table
   (rows 1–8; check 1a's primary arm is SM-free and 1d's per-kernel sum is within 10 % of E12b's).
2. Ask the device (a `docs/04` note, on approval) to close T004 against its E6 deliverables, and put the E7
   residual question and the Part-C answers there.
3. When the capped-mode session lands: check step 0c against the STATUS bracket (prefix 326.94–326.94×r,
   expert 149.05–149.05×r, r = 1300.5 / f_GPU,30W); hand numbers to main as "for <owner>:".
4. When E13 lands: hand σ(x), the duty result and EV-02c's H1/H2 split with tile shapes to main "for
   brainstorm-owner / microarch-owner:" (cases 001, 003; EV-02b/c; microarch's EV-17 re-read estimate).
5. P2 / `MODELS` only after explicit user approval (task + order row + protocol §).

## Recent decisions

- 2026-09-26: T016 Amendment 1 — SM-free steal primary, EV-02c per-kernel bytes, FACIL in 1c, bytes required in
  step 3; dated amendment, pre-registered rows 1–7 untouched. Stray `.claude/agent-memory/` removed from this repo
  and ignored (canonical store is the workspace's).
- 2026-09-26: T016 issued (user-approved EV-02 + seeded E8 + prefix-savings variants + EV-10 + EV-13); self-contained
  task file instead of a `docs/E##` spec; prefix-savings variants are never the baseline.
- 2026-09-26: E8 read as `bf16_both` accuracy-neutral within run-to-run noise; E12b DRAM bytes settle EV-02a on
  the ~90 % side.
- 2026-09-25: E12b reviewed against the `5cf1129` pre-registration; T014 compiled green rows withdrawn.
- 2026-09-25: "P1" relabelled EV-02 (channel partition); CoRe is the ISA name elsewhere.
- 2026-09-24: capped modes are the device's business; never escalate unless it marks `blocked`.
- 2026-09-24: the GR00T head restructuring is ours, never in the GPU baseline.

## Interfaces with other sub-projects

- → microarch-owner: latency / power / kernel / DRAM-bytes data for `host_model.py`, `energy_model.py`, roofline
  points; hybrid inputs `results/<date>_E11_<model>_inputs/model_inputs.csv` (protocol §13.8).
- → brainstorm-owner: case 001 and EV-02a (prefix DRAM draw); `O`, `p_part`, `s_part`, `p_hog`, fair `B`.
- → paper-owner: baseline ladder, occupancy / instruction-level evidence, LIBERO results.
- Every number leaves through main as "for <owner>:" and reaches paper / site via evaluation-owner.
- Names: EV-02 = channel-partition measurement (formerly "P1"); CoRe = the ISA (not in this repo).
