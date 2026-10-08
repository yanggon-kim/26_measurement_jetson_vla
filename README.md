# 26_measurement_jetson_vla

Real measurements of flow-matching Vision-Language-Action (VLA) models on an **NVIDIA Jetson AGX Orin
64GB**, for a computer-architecture research project on processing-in-memory (LPDDR5-PIM) acceleration of
the VLA *action expert*. The research side has so far used only analytic GPU numbers; this repository
replaces them with measured ones.

Two agents work in this repository and nobody else:

| who | where it runs | writes | never writes |
|---|---|---|---|
| **on-device agent** | on the Jetson | `env/ bench/ results/ reports/ STATUS.md`, status lines at the end of `tasks/*.md` | `README.md AGENTS.md CLAUDE.md docs/ analysis/`, task bodies |
| **research-side agent** (jetson-owner) | on the research server (no GPU) | `README.md AGENTS.md CLAUDE.md docs/ tasks/ analysis/` | the on-device agent's side |

Data moves only through this git remote. Both sides `git pull --rebase` before every push.

**Raw data policy.** We want as much raw data as possible: profiler reports (`.nsys-rep`, `.ncu-rep`), full SASS
dumps, tegrastats logs, traces and per-iteration samples are committed under `results/<run>/raw/`, compressed,
every file under 50 MB (`docs/02_protocol.md` §3.5 and §6). Never committed: model weights, datasets, caches,
environments, secrets.

**If you are the on-device agent, read [`AGENTS.md`](AGENTS.md) now.**

| path | content |
|---|---|
| `AGENTS.md` | your instructions: role, rules, workflow, reporting |
| `docs/00_context.md` | what the research claims and which of its numbers you are replacing |
| `docs/01_measurement_plan.md` | the task order, and experiments E0–E12b: definition, sweeps, outputs (E11 = inputs for the hybrid GPU + PIM model) |
| `docs/02_protocol.md` | binding measurement rules: device state, timing, `nsys` / `ncu`, power, result schema, sizes; §13 E11 and `model_inputs.csv`; §14–17 co-run and variant rules |
| `docs/03_instruction_level_analysis.md` | E5b: SASS-level analysis of the dominant action-expert kernels, and the scripts to write for it |
| `docs/04_research_answers.md` | the research side's answers to the questions the on-device agent raises in `STATUS.md` |
| `docs/E12_corun/`, `docs/E12b_compiled_corun/` | specifications of E12 (T014) and E12b (T015), with their pre-registered predictions |
| `docs/reference_scripts/` | read-only reference scripts for E5b (copied into `bench/`) and `check_model_inputs.py` (run in place) |
| `tasks/` | one file per task, in priority order; claim, do, report |
| `env/` | how the device was set up: scripts, exact versions, pins |
| `bench/` | the measurement harness you write |
| `results/` | one directory per run: `meta.json`, tidy CSVs, `summary.md`, and `raw/` — the run's raw profiler output (compressed `.zst`; split files carry a `REASSEMBLE.txt`) |
| `scratch/` | ignored by git: caches and uncompressed originals on the device |
| `reports/` | your analysis per experiment |
| `STATUS.md` | your running state: what is done, what is blocked, what you need |
| `analysis/` | research-side post-processing of your results |
