# HANDOFF — jetson

*Root `/home/yanggon/05_VLA_LPDDR/jetson` · owner `jetson-owner` · rewritten 2026-10-10. State only: rewritten, never
appended; superseded text is deleted (git keeps it). Full reviews: `analysis/E12_review.md`, `analysis/E12b_review.md`,
`analysis/E13_E14_review.md`; prefix root causes: `analysis/prefix_root_cause.{md,py,csv}`.*

## State now
- **Device:** idle at MAXN since 2026-09-28 17:45; last device commit `2984671`. T001–T017 closed. **Open: T018** (repo
  hygiene, no GPU; issued 2026-10-08), no return yet.
- **Prefix root-cause analysis (2026-10-10, CPU-only, phase 1 of main's prefix-acceleration study):** committed locally,
  **not pushed** (main asks the user). Report at `analysis/prefix_root_cause.md`, not `reports/` (device-owned, and T018
  is trimming it). Classes from E5 nsys (full coverage), shapes / bytes from E13d, counters from E4g C1 (first 40).
- **Thor projection** (static, projected): `analysis/thor_projection.{py,md}`; front ×2.65, expert ×1.69 (NVIDIA ÷ NVIDIA).

## Current numbers (MAXN, bf16) — sources, not re-derived
| quantity | value | source |
|---|---|---|
| π0 best frame `bf16_both` | 424.46 ms; vision / prefill / expert 56.74 / 192.86 / 168.68 | `results/20260922_E4c_pi0_opt_ladder` |
| π0 compiled stages `graphs_both` | vision 54.81, vlm 190.22, expert 101.60 ms (n 180) | `results/20260927_E14b_pi0_padtrim_stages_MAXN` |
| π0 prefill down-proj (816×2048×16384) | 99.19 ms of 192.86 at 9.9 TFLOP/s; every K 16384 shape 8.9–10.4, K 2048 15–32 | `analysis/prefix_root_cause.csv` (E5, E13d sweep) |
| π0 empty camera (graphs_both) | vision −18.36, vlm −49.39 ms | E14b `nocam` |
| GR00T N1.7 `vlm_compile` GPU idle | vision 19.11 of 70.82, prefill 42.46 of 87.59 ms (derived: wall − kernel union) | `analysis/prefix_root_cause.csv` |
| GR00T `lm_head` (dead) | 11.55 ms kernel; `keep1` −8.1 ms e2e, bitwise | E5, `results/20260926_E14_groot_n17_savings_MAXN` |
| SmolVLA vision attention (mem-efficient) | 10.15 of 31.26 ms, 7.6 TFLOP/s | E5, E4g |
| prefix floors (compute, 33.85 TFLOP/s) | π0 19.51 + 98.45; GR00T 20.65 + 13.28; SmolVLA 12.72 + 1.70 ms | microarch `stage_flops_bytes.csv` @ `f751ff4` |
| energy above idle `bf16_both` | prefix 10.36, ae 3.25, full 13.70 J | `results/20260926_E7r_*`, `…_E7v_summary` |
| best GPU co-run `O` | 1.247 (green 12/4, T_pipe 307.15 ms) | E12b |
| prefix DRAM | 21.60–21.64 GB/frame, 85–88 GB/s | E12b, E13d |
| σ (SM-free steal) / σ_cap | 1.060–1.279 at 20–85 GB/s / 1.044–1.19 at 12–8 ch `[derived]` | E13a/b, `analysis/e13_cap_model.py` |
| `nopad` trim | prefix 180.12, expert 66.32, serial 248.48 ms; bytes ×0.792; PASS | E14b |

**Readings:** no prefix stage runs two kernels at once (union = sum), so SIMT time is tensor-idle time. π0 is one bad GEMM
plus per-camera vision at M 256; GR00T is host-bound; SmolVLA is vision attention. E13 is contention, not a partition.

## Open issues
- **User decisions (OD-1 family):** `owngraph_expert` / hoist in the GPU baseline; prefix trims (`nopad`, `nocam`,
  GR00T `keep1` / `kvonce`); which N. All shrink the PIM's share.
- **AiM vs GPU expert not like-for-like:** GPU `nopad` expert attends 584 keys; AiM models the padded length.
- **Corrections to carry downstream:** E7 compiled phase energies (`E7_all_dirs_timed_window.csv`); E8 / E8s rates are for
  the attended camera; `…_E8s_…_N10_seeded_nopad` is `trimprompt`; E5b event times +45–75 µs; GR00T projection floors.
- **Unexplained, measured:** the K 16384 GEMM penalty (mechanism open, J2); compiled prefix at MODE_30W 2.68× MAXN;
  LIBERO `trimprompt` ~0–4 ms vs ~25 ms in the harness; ~38 % idle in the compiled expert.
- **Not captured anywhere:** tensor-pipe utilisation; counters for π0 vision at `bf16_both` (E4g ran fp32 vision);
  GR00T vision GEMM counters; SmolVLA DRAM bytes.
- **Answers owed to the device (`docs/04`, on approval):** the camera, OD-1, `model_commit: null`, `owngraph_expert`.
- **Never written as tasks without the user's approval:** P2 (`bf16_both` + `cropcache`); `groot_n16` / `groot_n15` in `MODELS`.
- **Carried user actions:** leaked session token in history (`410a535`; repo private); no passwordless sudo; π0 is LeRobot.

## Next steps
1. Main pushes the prefix analysis once the user approves; on each pull read the head of `STATUS.md` and new results;
   review T018's return (nothing cited deleted — the prefix analysis cites E4c, E4d, E4g C1, E5, E13d, E14, E14b, E4i).
2. **Proposed prefix tasks (not issued):** J1 full-coverage ncu with tensor-pipe metric (π0 vision bf16_both, GR00T,
   SmolVLA); J2 K 16384 isolation (padded pitch, layouts, cuBLASLt algos, Triton / CUTLASS tiles); J3 SigLIP at M 256 vs
   batched 3 cameras / fused QKV; J4 SDPA at prefix shapes; J5 GR00T host trace + CUDA-graph capture; J6 π0 last layer cut,
   GR00T `lm_head` removed. Order J2 → J1 → J5 → J3 / J4 → J6. Earlier F-c1..F-e1 (review §9) still proposed.
3. Phases 2–3 of the study: brainstorm-owner turns the ranked causes into methods, microarch-owner models them; answer
   their questions from `analysis/prefix_root_cause.csv`.

## Recent decisions
- 2026-10-10: prefix analysis lives in `analysis/` (research side), not `reports/`; dead-work rows never add to efficiency rows.
- 2026-10-08: Markdown slimmed; superseded HANDOFF text deleted. T018 issued for the device's side.
- 2026-10-08: Thor projection calibrated NVIDIA Orin ÷ NVIDIA Thor at one input, not on our run.
- 2026-10-08: T016 / T017 reviewed: E13 read as contention; cap model as the partition's lower bound; trim anomalies one
  alignment mechanism (hypothesis).
- 2026-09-26: T017 issued (camera and prompt trims separately; pass criterion relative to the stock-vs-`bf16_both` floor).
- 2026-09-25: "P1" relabelled EV-02 (channel partition); CoRe is the ISA's name elsewhere.
- 2026-09-24: capped modes are the device's business, never escalated. GR00T head restructuring is never baseline.

## Interfaces
- → microarch-owner: latency / power / kernel / DRAM-bytes data, `results/<date>_E11_<model>_inputs/model_inputs.csv`,
  prefix causes `analysis/prefix_root_cause.csv`. → brainstorm-owner: same CSV, cases 001 / 003–006. → paper-owner: ladder, LIBERO.
- Numbers leave only via main as "for <owner>:" and reach paper and site through evaluation-owner.
