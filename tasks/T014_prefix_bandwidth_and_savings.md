# T014 — The prefix under stolen bandwidth (E13), prefix software savings (E14), π0's padding trim (E14b) · priority P2 · depends on: T013

## Why
While the PIM computes, the memory channels it uses are unavailable to the GPU, and the pipeline's period is the
prefix time: **how much the prefix slows when it loses bandwidth decides the design.** E13 measures that slowdown
with a load that takes no SMs, and which prefix kernels carry the DRAM traffic. E14 / E14b measure prefix work the
output never uses (padding, a dead `lm_head`, re-projected K / V), as labelled variants, never as the baseline.

## Orin counterpart
`reference/orin/tasks/T016_prefix_bandwidth_and_followups.md` (steps 1 and 3, Amendment 1 — the specification of
E13 a/b/c/d and E14), `reference/orin/tasks/T017_pi0_padding_trim.md` (the specification of E14b), both with their
status blocks; reports `reference/orin/reports/E13_pi0_bandwidth_steal.md`, `E14_prefix_savings.md`,
`E14b_pi0_padtrim.md`; scripts `reference/orin/bench/e13_bwsteal.py`, `e13c_stride_probe.py`, `e13d_*.py`,
`bwload.py`, `e14_*.py`, `e14b_*.py`, `pi0_padtrim.py`, `groot_savings.py`; fixtures
`reference/orin/bench/fixtures/e14b/`. Binding rules: protocol §16–§17.

## Configuration and the Thor adaptations of the two specifications
- π0 at **`best_bf16`** (wherever the specifications say `bf16_both`), N = 10; GR00T N1.7 at its `best_bf16`, N = 4.
  The maximum-performance mode only; no reboot, no boot or device-tree change.
- **E13a rates.** Measure the streaming bandwidth a load sustains alone in the same session (`B_alone`, against
  T003's peak). The primary load takes no SMs (protocol §16). Points: stolen bandwidth 0 and ≈ 10, 20, 30, 40, 50 %
  of `B_alone`, plus at least three points that leave the prefix 50–70 % of `B_alone`, and the highest rate the
  SM-free load reaches; report the rate alone and during the prefix. Reference: the same prefix, no load, bracketed,
  n ≥ 20 frames per point.
- **E13b duty cycle**: the load on for the share of each prefix frame that the expert of the previous frame would
  take in the research design, using T005's `best_bf16` stage times (on for `B / (A + B)` of each frame — state the
  number), aligned to the frame start and once to its end; compare with `1 + f·(σ_full − 1)` (derived).
- **E13c channel probe** (read-only): how this device interleaves physical addresses across LPDDR5X channels
  (stride-vs-bandwidth sweep), and whether any stock mechanism (carve-outs, NvMap heaps, allocator pools, PTE kind
  bits, debugfs / BPMP controls) could confine an allocation to a channel subset. The FACIL reading of the Orin
  task (Amendment 1, A3) applies; answer yes / yes-with-boot-change / no / unknown with evidence.
- **E13d per-kernel DRAM bytes** of the compiled prefix with an explicit metric list (protocol §3.2 form D; T006 may
  already have them — reuse and cite), the H1 / H2 split of excess reads (weight vs activation re-reads).
- **E14**: π0 `nopad`; GR00T N1.7 `keep1`, `kvonce`, `keep1+kvonce`; time, DRAM bytes and equivalence for each
  variant and its same-session reference.
- **E14b**: `pad`, `nocam`, `trimprompt`, `nopad` with the gate, bracket order and fixed pass criterion of protocol §17.

## Steps
Order: E13a → E13b → E13d → E13c (push E13's results before starting E14), then E14, then E14b — each as the
Orin specification describes it, with the adaptations above. The Orin steps 2, 4 and 5 of T016 are covered
elsewhere (E8s in T011, energy reconciliation in T010, the co-run corruption diagnosis in T013 if it recurs).

## Pre-registered outcomes (research side, 2026-10-21, before any Thor measurement; not acceptance criteria)
| quantity | bands and what each means |
|---|---|
| σ with half of `B_alone` stolen, full-time load | < 1.04: the prefix barely needs the bus · 1.04–1.25: moderate sensitivity (the Orin's E13a range was 1.06–1.28 at 20–85 GB/s) · > 1.25: the prefix is bandwidth-sensitive on Thor |
| share of the prefix's excess reads that are weight re-reads (E13d) | ≥ 2/3: GEMM tiling is the lever · ≤ 1/3: activation re-reads, fusion is the lever · between: mixed, name the top kernels |
| E14 exact variants (`keep1`, `kvonce`) | bitwise-equal actions expected; a mismatch is a finding, not something to tune away |

## Result directories and reports
`results/<date>_E13a_pi0_bwsteal_<load>/`, `…_E13b_pi0_bwduty/`, `…_E13c_channel_probe/`,
`…_E13d_pi0_prefix_kernel_bytes/`, `…_E14_<model>_<variant>/`, `…_E14b_pi0_padtrim_<stage>/`; reports
`reports/E13_pi0_bandwidth_steal.md`, `reports/E14_prefix_savings.md`, `reports/E14b_pi0_padtrim.md` (the
pre-registered rows answered by number first, Orin values labelled).

## Acceptance
- E13a: σ at ≥ 6 rates with n ≥ 20 per point, bracketed references, the load shown SM-free; E13b rows and the
  byte-share comparison; E13c answer with its evidence under `raw/`; E13d `kernel_bytes.csv` with the sum check.
- E14: every variant with time, DRAM bytes and its equivalence result. E14b: every stage gated, timed, with bytes
  and the PASS / FAIL verdict.
- No variant called baseline; protocol §15's finite check and provenance on every run.

## Push points
After each result directory (E13a first: it is the number the research side waits for), at the close.

## Status (appended by the on-device agent)
