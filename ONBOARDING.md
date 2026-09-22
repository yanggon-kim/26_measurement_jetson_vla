# ONBOARDING — jetson

*Root: `/home/yanggon/0007_26summer/05_VLA_LPDDR/jetson` · Owner: `jetson-owner`. Stable facts only; state is in `HANDOFF.md`.*

## 1. Identity
Clone of `git@github.com:yanggon-kim/26_measurement_jetson_vla.git`: the channel between the research server
(no GPU) and an **on-device agent on a Jetson AGX Orin 64GB** that measures flow-matching VLA models (π0, π0.5,
SmolVLA, optionally GR00T) in eager and `torch.compile` modes, with Nsight Systems / Nsight Compute, tegrastats
power, and the LIBERO benchmark. You write the instructions and analyse what comes back; you never run a GPU.

## 2. Layout and write partition
Yours: `README.md`, `AGENTS.md` (canonical device instructions; `CLAUDE.md` = `@AGENTS.md`), `docs/00_context.md`
(research context + table of numbers to replace), `docs/01_measurement_plan.md` (E0–E11 + task-order table), `docs/02_protocol.md`
(binding measurement rules + result schema), `tasks/T###_*.md` (work queue), `analysis/` (your post-processing).
The device's: `env/`, `bench/`, `results/<date>_<Exx>_…/` (`meta.json`, tidy CSVs, `summary.md`, `raw/`), `reports/`,
`STATUS.md`, and appended status lines in task files. `ONBOARDING.md` / `HANDOFF.md` are ignored by `.gitignore`.

## 3. Commands
No build. `git -C <root> pull --rebase` before anything; push after committing instructions. Analysis scripts
run with `/home/yanggon/0007_26summer/05_VLA_LPDDR/microarch/.venv/bin/python` (pandas, matplotlib available).

## 4. Git
One remote `origin`, branch `main`, **two writers**: you and the on-device agent. Fetch, rebase, re-read before
editing; never resolve a conflict with the device's files — report it. The workspace repo ignores this directory.

## 5. Guidance to read
The repo's own `AGENTS.md` and `docs/`. Research background: `../microarch/00_doc/02_refs/characterizing_vla_across_xpus_2604.24447.pdf`,
`../microarch/05_reports/phase1_model_analysis.md` (dimensions, paper cross-check), `phase8_energy_analysis.md`,
`../microarch/03_tools/host_model.py`, `energy_model.py`, `../paper/scripts/paper_numbers.py` (where each GPU number is used).

## 6. Traps
- The characterization paper has **no per-step time and no compiled number on Orin**: 136 ms/step and ≈ 90 ms are ours.
- Tasks must be executable with zero other context; the device agent cannot ask questions in real time — it writes
  them under "Questions for the research side" in `STATUS.md`. Answer by editing docs/tasks, not by chat.
- Raw profiler output IS committed (`results/<run>/raw/`, compressed, < 50 MB per file, protocol §3.5 / §6); never
  weights, datasets, caches, environments, secrets. Pulls can be large; repository size is a standing risk.
- Keep hypotheses out of acceptance criteria: ask for the measurement, not for the expected value.

## 7. Scope
Owns the root except the device's side. Hands measured numbers to microarch-owner / paper-owner only as
"for <owner>:" items quoting result directory, device commit, power mode and n.

*`ONBOARDING.md` / `HANDOFF.md` are tracked since the 2026-09-22 server move, so the research side's state travels with the repo. They remain **jetson-owner's** files: the on-device agent never edits them (see `AGENTS.md`, write partition).*
