# 26_measurement_jetson_vla — branch `thor`

Real measurements of flow-matching Vision-Language-Action (VLA) models on an **NVIDIA Jetson AGX Thor**, for a
computer-architecture research project on processing-in-memory (LPDDR-PIM) acceleration of the VLA *action
expert*. The same study already ran on a Jetson AGX Orin (branch `main`); this branch repeats it on Thor. It starts
empty, and the Thor agent fills it.

**Clone only this branch** (branch `main` carries ~250 MB of Orin results):
`git clone -b thor --single-branch git@github.com:yanggon-kim/26_measurement_jetson_vla.git`

Two agents work in this branch and nobody else:

| who | where it runs | writes | never writes |
|---|---|---|---|
| **on-device agent** (the Thor agent) | on the Jetson AGX Thor | `env/ bench/ results/ reports/ STATUS.md PROGRESS.md`, status blocks at the end of `tasks/*.md` | `README.md AGENTS.md CLAUDE.md docs/ analysis/ reference/ ONBOARDING.md HANDOFF.md`, task bodies |
| **research-side agent** (jetson-owner) | on the research server (no GPU) | `README.md AGENTS.md CLAUDE.md docs/ tasks/ analysis/ ONBOARDING.md HANDOFF.md` | the on-device agent's side |

Data moves only through this git remote. Both sides `git pull --rebase` before every push. The on-device agent
pushes after every task, and after every finished result directory inside a long task.

**Raw data policy.** We want as much raw data as possible: profiler reports, full SASS dumps, power logs, traces and
per-iteration samples are committed under `results/<run>/raw/`, compressed, every file under 50 MB
(`docs/02_protocol.md` §3.5 and §6). Never committed: model weights, datasets, caches, environments, secrets.

**If you are the on-device agent, read [`AGENTS.md`](AGENTS.md) now, then start [`tasks/T001_setup_and_availability.md`](tasks/T001_setup_and_availability.md).**

| path | content |
|---|---|
| `PROGRESS.md` | **the top tracker**: one row per task, the deliverables checklist, and the "Needs the human" box |
| `AGENTS.md` | the on-device agent's instructions: role, rules, workflow, push cadence, reporting |
| `docs/00_context.md` | why Thor, what the deliverables are for, and the Orin numbers they will be compared with |
| `docs/01_measurement_plan.md` | the task order, and the experiments E0–E14b: definition, sweeps, outputs |
| `docs/02_protocol.md` | binding measurement rules: device gate, timing, `nsys` / `ncu`, power, result schema, sizes, E11 hand-off format, co-run and variant rules |
| `docs/03_instruction_level_analysis.md` | E5b: the SASS-level analysis of the dominant kernels of every part |
| `docs/04_research_answers.md` | the research side's answers to the questions the on-device agent raises in `STATUS.md` |
| `docs/05_setup.md` | the requirement list, the availability checks, and the rules and templates for asking the human |
| `docs/reference_scripts/` | read-only reference scripts for E5b (copy into `bench/`) and `check_model_inputs.py` (run in place) |
| `reference/orin/` | **read-only** snapshot of the Orin harness, setup scripts, key reports and task files: Orin code and Orin numbers, to port and to compare, never to run unchanged |
| `tasks/` | one file per task, T001–T015, in priority order; claim, do, report |
| `env/` | how this Thor was set up: scripts, exact versions, pins, the availability matrix |
| `bench/` | the measurement harness you write (ported from `reference/orin/bench/`) |
| `results/` | one directory per run: `meta.json`, tidy CSVs, `summary.md`, and `raw/` |
| `reports/` | your analysis per experiment |
| `STATUS.md` | your running state: now, needs the human, questions, repository size, log |
| `analysis/` | research-side post-processing of your results |
| `scratch/` | ignored by git: caches and uncompressed originals on the device |
