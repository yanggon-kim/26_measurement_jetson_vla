# HANDOFF — jetson

*Root: `/home/yanggon/05_VLA_LPDDR/jetson` · Owner: `jetson-owner` · Last rewritten: 2026-10-08 by jetson-owner.*
*State only. It is rewritten, never appended. Older state, decisions and issues are in `HANDOFF.history.md`.
Full reviews: `analysis/E12_review.md`, `analysis/E12b_review.md`, `analysis/E13_E14_review.md` (T016 + T017 + the
late addenda).*

## State now

- **Device:** idle at MAXN since 2026-09-28 17:45. Its last commit is `2984671` (2026-09-29, the E12d addendum).
  Every task in `tasks/` carries a closing status line: T016 and T017 done 2026-09-27, T004 4b(ii) 2026-09-28, T014
  2026-09-27. Nothing is queued and no task is open. Both capped-mode sessions (MODE_30W, MODE_15W) are done.
- **Reviewed (2026-10-08):** T016 (E13, E8s, E14, E7r, E12c) and T017 (E14b), plus E12d, the analytic co-run
  report, E7 capped §2b / §3, E7v and E5c. Every pre-registered row was answered honestly and no band moved. The
  only deviations were the unreachable 94 GB/s point and the optional per-variant `kernel_bytes.csv`, both allowed.
- **Pushed:** everything of ours through T017 (`1c5cd12`). **Local, not pushed:** the review commit (this
  rewrite, `analysis/E13_E14_review.md`, `analysis/e13_cap_model.py`). The user approves pushes.
- **Nothing issued is outstanding.** The follow-ups below are proposals awaiting the user's approval.

## Current numbers (π0 LeRobot `f0d2b37`, ckpt `45dcc8fc`, MAXN, bf16, N = 10) — sources, not re-derived

| quantity | value | source |
|---|---|---|
| best frame, `bf16_both` | 424.46 ms; vis / prefix / ae 13.4 / 45.4 / 39.7 % | `results/20260922_E4c_pi0_opt_ladder/opt_ladder.csv` |
| energy above idle, `bf16_both` (**corrected**) | prefix 10.36 J, ae 3.25 J, full 13.70 J (was 6.40 / 3.11 / 13.67); V / L / A 17 / 59 / 24 % | `results/20260926_E7r_pi0_energy_reconcile_MAXN`, `results/20260928_E7v_summary` |
| compiled stages, `graphs_both` harness | prefix 245.41, expert 101.60, serial 347.64 ms (n 180) | `results/20260927_E14b_pi0_padtrim_stages_MAXN` |
| best GPU co-run `O` | 1.247 (green 12/4, T_pipe 307.15 ms) | E12b |
| prefix DRAM bytes / draw | 21.60–21.64 GB per frame; 85–88 GB/s; 45–49 % of kernel time above 90 GB/s | E12b, E13d |
| **σ (SM-free steal, contention)** | 1.060 / 1.120 / 1.181 / 1.247 / 1.279 at 20 / 40 / 60 / 77 / 85 GB/s; 37 % duty 1.125 | `results/20260926_E13a_pi0_bwsteal_ce+cpu_MAXN`, `…_E13b_*` |
| σ_cap, hard partition, lower bound `[derived]` | 1.044 / 1.084 / 1.19 at 12 / 10 / 8 channels left | `analysis/e13_cap_model.py` on E13d |
| excess reads, weight share | 0.67 (fit, read as mixed) | `results/20260926_E13d_pi0_prefix_kernel_bytes_MAXN` |
| padding trim (`nopad`, 816 → 533 tokens) | prefix 180.12, expert 66.32, serial 248.48 ms; bytes ×0.792; PASS | E14b |
| `trimprompt` / `nocam` | prefix 252.28 / 177.77, expert 69.81 / 88.35 ms | E14b |
| expert attention, `bf16_both` | 4.19 ms/step (`cutlass_75_wmma`) vs weights 4.54 ms/step; K/V re-copied 15.05 MB/step | `results/20260928_E5c_expert_components` |
| seeded LIBERO N sweep (pooled /400) | N 1/2/3/4/6/10: 295/300/286/284/267/259; N = 2 vs 10 p 1.3e-5; repeat 95 % | `results/20260926_E8s_pi0_seeded_summary` |
| `nopad` LIBERO, camera masked | 70 vs 66 / 100, p 0.29 | `…_E8s_pi0_libero_spatial_N10_seeded_maskedcam{,_nopad}` |
| GR00T N1.7 `keep1` + `kvonce` | 237.0 → 220.0 ms, reads ×0.874, bitwise | `results/20260926_E14_groot_n17_savings_MAXN` |

**Readings (detail in the review):**
- **Steal vs partition.** E13 is queueing contention on all 16 channels, with the bus never full. A hard partition
  lies between σ_cap and E13's value at the same bandwidth left. At k = 6 that is 1.08–1.21, giving 1.04–1.15× over
  the GPU co-run `[derived]`. A PIM that seizes channels under a shared address map blocks, and nothing measured
  covers that case.
- **The trim anomalies are probably alignment.** Prefill M ≡ 0 (mod 8) is fast. The expert's key count
  (prefix + 51) ≡ 0 (mod 8) is fast. pad aligns the first and misaligns the second, so the GPU expert at pad may
  carry a ~30 ms attention-kernel penalty. One LIBERO clue does not fit. This is untested.
- **N = 1–4 beats N = 10** on every suite (π0, this fine-tune).
- **The empty camera** was attended in every E8 / E8s arm. It moves only absolute LIBERO rates; no timing, byte or
  energy number changes.

## Open issues

- **Open decisions for the user (OD-1 family):** does `owngraph_expert` (101.21 ms) or the hoist enter the GPU
  baseline; does a prefix trim (`nopad`, GR00T `keep1` / `kvonce`); which configured N (the checkpoint default 10 vs
  the more accurate 1–4). All shrink the PIM's share of the frame.
- **The AiM vs GPU expert comparison is not like-for-like.** The 66.3 ms GPU `nopad` expert attends 584 keys, and
  the AiM 93.1 ms models attention at the padded length. microarch-owner has to confirm that length. Review §3(b).
- **Corrections to carry downstream:** E7 compiled phase energies (28 of 132 rows ≥ 5 %, e.g. π0 prefix +62 %; read
  `E7_all_dirs_timed_window.csv`). E8 / E8s absolute rates are for the attended-camera input. The directory
  `…_E8s_pi0_libero_spatial_N10_seeded_nopad` is `trimprompt` (D5; it has no NOTE.md). E5b S-sweep event times
  are +45–75 µs. GR00T projection floors are 1.17–1.20× (`to_q` 1.44–1.47×). The ANALYTIC report's "activations
  spill" contradicts E13d.
- **Unexplained, measured:** the compiled prefix at MODE_30W is 2.68× MAXN green-8 (clock ratio 2.125×). The LIBERO
  rung-path `trimprompt` gave ~0–4 ms saving vs ~25 ms in the harness. ~38 % idle remains in the compiled expert.
- **Device questions, answers owed (`docs/04`, on approval):** the camera (read as-is with the caveat?), OD-1 (the
  user's), the `model_commit: null` back-fill, and `owngraph_expert`.
- **Not approved; never written as tasks without the user's explicit, current approval:** P2 (`bf16_both` +
  `cropcache`), and `groot_n16` / `groot_n15` in `check_model_inputs.py` `MODELS`.
- **Carried user actions (unchanged since 2026-09-22):** the leaked session token in history (`410a535`; the repo
  must stay private); no passwordless sudo on the device; π0 is the LeRobot port, not openpi. The device reports
  713 MB of committed results; the local size-pack is 245 MiB.

## Next steps

1. **Push on approval:** the review commit.
2. **Proposed follow-ups (review §9; none issued):** F-c1 nsys kernel names per E14b variant plus the rung path with
   the camera masked and attended (~30 min). F-c2 standalone attention / GEMM alignment sweep (~15 min). F-c3 exact
   `pad+kvalign` / `nopad+align` variants (~1 h; decides whether the GPU expert reference drops ~30 ms). F-a1 an
   EMC-locked prefix as an upper proxy for a hard partition (runtime debugfs clock; the device first confirms it is
   feasible). F-a2 the CPU-only steal arm. F-e1 seeded N = 2 vs 10 with the camera masked. Order if approved:
   F-c1 + F-c2 → F-c3 → F-a1 → F-a2 → F-e1. Write them as one task with an order row and a protocol §, and
   pre-register the bands.
3. On each pull: fetch, read the head of `STATUS.md` and any new `results/` / `reports/`, and check repo size.
4. Hand-offs through main (listed in the 2026-10-08 report): σ, σ_cap and the partition mechanisms (brainstorm /
   microarch); the nopad fairness and the PIM run at 584 keys (microarch); the E7 energy correction (microarch /
   evaluation); the camera caveat and the mislabeled directory (evaluation); the N result (brainstorm / paper).

## Recent decisions

- 2026-10-08: T016 and T017 reviewed (`analysis/E13_E14_review.md`). E13 read as contention, not a partition, and
  a cap model added as the partition's lower bound (`analysis/e13_cap_model.py`, derived only). The trim anomalies
  are read as one alignment mechanism (a hypothesis). Follow-ups proposed, not issued.
- 2026-09-26: T017 issued (case 004). The camera drop and the prompt trim are measured separately, the camera
  dropped before SigLIP, with the pass criterion relative to the stock-vs-`bf16_both` floor.
- 2026-09-26: T016 Amendment 1: SM-free steal as the primary arm, EV-02c per-kernel bytes, FACIL in 1c, bytes
  required in step 3. Pre-registered rows untouched.
- 2026-09-26: T016 issued. It is a self-contained task file, and its prefix-savings variants are never the baseline.
- 2026-09-26: E8 read as `bf16_both` accuracy-neutral within noise. E12b's bytes settle EV-02a on the ~90 % side.
- 2026-09-25: E12b reviewed against the `5cf1129` pre-registration.
- 2026-09-25: "P1" relabelled EV-02 (channel partition); CoRe is the ISA's name elsewhere.
- 2026-09-24: capped modes are the device's business, never escalated (both sessions are now done).
- 2026-09-24: the GR00T head restructuring is ours, never in the GPU baseline.

## Interfaces with other sub-projects

- → microarch-owner: latency / power / kernel / DRAM-bytes data for `host_model.py`, `energy_model.py` and
  roofline points; hybrid inputs `results/<date>_E11_<model>_inputs/model_inputs.csv` (protocol §13.8).
- → brainstorm-owner: cases 001 / 003–006 (σ, partition, N, capped modes).
- → paper-owner: the baseline ladder, occupancy / instruction-level evidence, LIBERO results.
- Every number leaves through main as "for <owner>:" and reaches the paper and the site via evaluation-owner.
- Names: EV-02 = the channel-partition measurement (formerly "P1"); CoRe = the ISA (not in this repo).
