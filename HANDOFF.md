# HANDOFF — jetson

*Root `/home/yanggon/05_VLA_LPDDR/jetson` · owner `jetson-owner` · rewritten 2026-10-08. State only: rewritten, never
appended; superseded text is deleted (git keeps it). Full reviews: `analysis/E12_review.md`, `analysis/E12b_review.md`,
`analysis/E13_E14_review.md`.*

## State now
- **Device:** idle at MAXN since 2026-09-28 17:45; last device commit `2984671` (E12d addendum). T001–T017 closed,
  both capped-mode sessions done. **Open: T018** (repo hygiene, no GPU; issued 2026-10-08): trim `STATUS.md` /
  `PROGRESS.md`, delete superseded reports, keep everything the research side cites.
- **Everything of ours is pushed**, including the T016/T017 review and the Thor projection (`3cfe1be`).
- **Thor projection** (static, projected, not measured): `analysis/thor_projection.{py,md}` → `analysis/thor_projection/`
  (pinned by evaluation's characterization manifest). Factors NVIDIA N1.7 Orin ÷ NVIDIA Thor, same input: front
  ×2.65, expert ×1.69. Our-N1.7-based draft kept only as a sensitivity (input-confounded, front ×3.26).

## Current numbers (π0 LeRobot `f0d2b37`, ckpt `45dcc8fc`, MAXN, bf16, N = 10) — sources, not re-derived
| quantity | value | source |
|---|---|---|
| best frame `bf16_both` | 424.46 ms; vis / prefix / ae 13.4 / 45.4 / 39.7 % | `results/20260922_E4c_pi0_opt_ladder/opt_ladder.csv` |
| energy above idle `bf16_both` (corrected) | prefix 10.36, ae 3.25, full 13.70 J; V / L / A 17 / 59 / 24 % | `results/20260926_E7r_*`, `results/20260928_E7v_summary` |
| compiled stages (`graphs_both`) | prefix 245.41, expert 101.60, serial 347.64 ms (n 180) | `results/20260927_E14b_pi0_padtrim_stages_MAXN` |
| best GPU co-run `O` | 1.247 (green 12/4, T_pipe 307.15 ms) | E12b |
| prefix DRAM | 21.60–21.64 GB/frame, 85–88 GB/s | E12b, E13d |
| σ (SM-free steal) | 1.060 / 1.120 / 1.181 / 1.247 / 1.279 at 20 / 40 / 60 / 77 / 85 GB/s | `results/20260926_E13a_*`, `…_E13b_*` |
| σ_cap, hard partition, lower bound `[derived]` | 1.044 / 1.084 / 1.19 at 12 / 10 / 8 channels left | `analysis/e13_cap_model.py` |
| `nopad` trim (816 → 533 tokens) | prefix 180.12, expert 66.32, serial 248.48 ms; bytes ×0.792; PASS | E14b |
| expert attention `bf16_both` | 4.19 ms/step vs weights 4.54 ms/step | `results/20260928_E5c_expert_components` |
| seeded LIBERO N sweep (/400) | N 1/2/3/4/6/10: 295/300/286/284/267/259 | `results/20260926_E8s_pi0_seeded_summary` |
| GR00T N1.7 `keep1` + `kvonce` | 237.0 → 220.0 ms, reads ×0.874, bitwise | `results/20260926_E14_groot_n17_savings_MAXN` |

**Readings:** E13 is contention on all 16 channels, not a partition; a hard partition lies between σ_cap and σ
(k = 6: 1.08–1.21) `[derived]`. The trim anomalies are probably one mod-8 alignment mechanism (untested). N = 1–4
beats N = 10 on every suite. The empty camera was attended in every E8 / E8s arm (absolute rates only).

## Open issues
- **User decisions (OD-1 family):** does `owngraph_expert` (101.21 ms) or the hoist enter the GPU baseline; does a
  prefix trim (`nopad`, GR00T `keep1` / `kvonce`); which N (10 vs 1–4). All shrink the PIM's share.
- **AiM vs GPU expert not like-for-like:** GPU `nopad` expert attends 584 keys; AiM's 93.1 ms models the padded
  length (microarch-owner to confirm; review §3(b)).
- **Corrections to carry downstream:** E7 compiled phase energies (28 of 132 rows ≥ 5 %; `E7_all_dirs_timed_window.csv`);
  E8 / E8s rates are for the attended-camera input; `…_E8s_pi0_libero_spatial_N10_seeded_nopad` is `trimprompt`;
  E5b S-sweep event times +45–75 µs; GR00T projection floors 1.17–1.20× (`to_q` 1.44–1.47×); the ANALYTIC report's
  "activations spill" contradicts E13d.
- **Unexplained, measured:** compiled prefix at MODE_30W is 2.68× MAXN (clock ratio 2.125×); LIBERO `trimprompt`
  saves ~0–4 ms vs ~25 ms in the harness; ~38 % idle in the compiled expert.
- **Answers owed to the device (`docs/04`, on approval):** the camera, OD-1, `model_commit: null` back-fill,
  `owngraph_expert`.
- **Never written as tasks without the user's current approval:** P2 (`bf16_both` + `cropcache`); `groot_n16` /
  `groot_n15` in `check_model_inputs.py` `MODELS`.
- **Carried user actions:** leaked session token in history (`410a535`; repo stays private); no passwordless sudo
  on the device; π0 is the LeRobot port.

## Next steps
1. On each pull: fetch, read the head of `STATUS.md` and new `results/` / `reports/`; review T018's return against
   its acceptance (nothing cited deleted, `git grep` clean) and check repo size.
2. **Proposed follow-ups (review §9; none issued):** F-c1 nsys kernel names per E14b variant + rung path camera
   masked / attended (~30 min); F-c2 attention / GEMM alignment sweep (~15 min); F-c3 `pad+kvalign` / `nopad+align`
   (~1 h; may drop ~30 ms from the GPU expert); F-a1 EMC-locked prefix as a hard-partition proxy (feasibility first);
   F-a2 CPU-only steal arm; F-e1 seeded N = 2 vs 10 camera masked. If approved: one task, order row, protocol §,
   pre-registered bands; order F-c1 + F-c2 → F-c3 → F-a1 → F-a2 → F-e1.
3. Hand-offs through main (listed in the 2026-10-08 report): σ / σ_cap / partition (brainstorm, microarch); nopad
   fairness and AiM at 584 keys (microarch); E7 correction (microarch, evaluation); camera caveat and mislabeled
   directory (evaluation); N result (brainstorm, paper); Thor projection (evaluation, paper; labelled projected).

## Recent decisions
- 2026-10-08: Markdown slimmed; superseded HANDOFF text is deleted, not moved. T018 issued for the device's side.
- 2026-10-08: Thor projection calibrated on NVIDIA Orin ÷ NVIDIA Thor at the same input, not on our Orin run.
- 2026-10-08: T016 / T017 reviewed: E13 read as contention; cap model as the partition's lower bound (derived only);
  trim anomalies read as one alignment mechanism (hypothesis). Follow-ups proposed, not issued.
- 2026-09-26: T017 issued (camera drop and prompt trim measured separately; pass criterion relative to the
  stock-vs-`bf16_both` floor). T016 issued with Amendment 1 (SM-free steal primary, EV-02c bytes).
- 2026-09-25: "P1" relabelled EV-02 (channel partition); CoRe is the ISA's name elsewhere.
- 2026-09-24: capped modes are the device's business, never escalated. GR00T head restructuring is never baseline.

## Interfaces
- → microarch-owner: latency / power / kernel / DRAM-bytes data, hybrid inputs `results/<date>_E11_<model>_inputs/model_inputs.csv`.
- → brainstorm-owner: cases 001 / 003–006. → paper-owner: baseline ladder, occupancy evidence, LIBERO.
- Numbers leave only via main as "for <owner>:" and reach paper and site through evaluation-owner.
