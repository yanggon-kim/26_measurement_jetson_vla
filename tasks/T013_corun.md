# T013 — π0: co-running the prefix and the action expert on one GPU (E12, E12b) · priority P2 · depends on: T005

## Why
In the research design, the PIM runs frame t−1's expert while the GPU runs frame t's prefix. The all-GPU rival to
that is the same pipeline on the GPU alone: prefix and expert co-running on one GPU. On the Orin the best measured
GPU co-run reached `O = T_pipe / max(A_iso, B_iso)` = 1.247 inside green-context partitions, and the stages coupled
through memory (`reference/orin/reports/E12b_pi0_compiled_corun.md`). The Thor projection *assumed* that O carries
over (`reference/orin/analysis/thor_projection.md` §4, `[ASSUMPTION]`); this task measures it. **The result may go
against the research side; that is fine — report what the device does, and do not tune the harness either way.**

## Orin counterpart
Specifications `reference/orin/specs/E12_corun/README.md` and `reference/orin/specs/E12b_compiled_corun/README.md`
(read both in full: purpose, definitions, non-negotiables, the steps / measurements M0–M5, hazards, the equivalence
and finite checks, outputs); `reference/orin/tasks/T014_pi0_corun.md`, `T015_pi0_compiled_corun.md` with their
status blocks; scripts `reference/orin/bench/corun.py`, `corun_green.py`, `green_ctx.py`, `probe_green_ctx.py`,
`e12b_*.py`, `e12_fixtures.py`, `graph_clone.py`, `corun_energy.py`; fixtures `reference/orin/bench/fixtures/e12/`.
Binding rules: protocol §14–§15.

## Configuration and the Thor adaptations of the two specifications
- π0 (LeRobot, pins of `docs/00_context.md`) at **`best_bf16`** as `best_bf16.json` names it, wherever the
  specifications say `bf16_both`; N = 10 (N = 4 optional, cheap). The fair expert = `best_bf16` + the timestep hoist,
  after its compiled equivalence check (E12b M1). No `cropcache`; no other model change.
- The maximum-performance mode only. **Dropped:** E12 Step 4 (the co-run at a reduced SM count in a capped mode)
  and every capped-mode item: they need a reboot.
- **SM partitions from this device**: query the green-context API of the installed CUDA (driver API
  `cuDevSmResourceSplitByCount` / `cuGreenCtxCreate`, through `cuda-python` or a small extension) for the SM count
  and the split granularity it grants; measure ≥ 3 splits, including the one nearest 3 : 1 (prefix : expert, the
  Orin's 12 / 4) and 1 : 1, and record the sizes granted per run.
- **Other partition mechanisms**, read-only availability checks, each a row in `mechanisms.csv`: MIG (`nvidia-smi
  mig -lgip` or equivalent; enabling MIG may need a reset or reboot — then record it and do not enable it), MPS,
  stream priorities, `CUDA_DEVICE_MAX_CONNECTIONS`, `libsmctrl`.
- Rates and DRAM attribution (E12b M4) against T003's measured bandwidth and the EMC / power-monitor paths T001
  found.

## Steps
E12 Steps 1–3 and 5 (eager and stock compiled co-run on the whole GPU, mechanisms, energy), then E12b M0–M5 as
specified, with the adaptations above. If the compiled two-stream co-run gives non-finite or stale actions, run the
Orin's 2 × 2 diagnosis ({shared, own graph pool} × {single, double-buffered KV};
`reference/orin/tasks/T016_prefix_bandwidth_and_followups.md` step 5) before any repair is timed.

## Pre-registered outcomes (research side, 2026-10-21, before any Thor measurement; not acceptance criteria)
| quantity | bands and what each means |
|---|---|
| best valid `O` on Thor (E12b, any partition) | ≤ 1.10: the GPU co-run overlaps far better than on the Orin; the all-GPU rival is stronger than projected · 1.10–1.35: about the Orin's 1.247; the projection's assumption holds · > 1.35: worse than the Orin; the rival is weaker than projected |
| `s_part` at the best split | ≤ 1.05: little memory coupling · > 1.05: memory coupling, report its size (the Orin: 1.16–1.26) |

## Result directories and reports
As the specifications' outputs sections name them, with the prefix `results/<date>_E12_pi0_*` and
`results/<date>_E12b_pi0_*`; `mechanisms.csv`; reports `reports/E12_pi0_corun.md`, `reports/E12b_pi0_compiled_corun.md`
(the pre-registered rows answered by number first, Orin values labelled).

## Acceptance
- The specifications' acceptance lists (E12 §11, E12b §12) minus the dropped items: every item holds, or the status
  line names the one that does not and why. Fresh-KV verified in the timed compiled path with a stale control that
  fails; no silent NaN; same-session bracketed baselines; no null in `meta.json`; granted partition sizes recorded.

## Push points
After each measurement's result directory (M0 … M5), at the close.

## Status (appended by the on-device agent)
