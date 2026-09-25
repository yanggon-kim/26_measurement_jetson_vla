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
| `docs/01_measurement_plan.md` | experiments E0–E12, E5b and E12b: definition, sweeps, priority, task order, outputs. E11 = inputs for a hybrid GPU + PIM system model (host segments, transfer / synchronisation latency, KV hand-over, prefix vs input, memory pressure, pipelining) |
| `docs/02_protocol.md` | how to measure, tool commands (`nsys`, `ncu`, tegrastats), SM-occupancy utilisation metric, result schema, file mapping, physical floor, review; §13 = E11 methods and the `model_inputs.csv` schema |
| `docs/03_instruction_level_analysis.md` | E5b: SASS-level analysis of the dominant action-expert kernels, and the scripts to write for it |
| `docs/04_research_answers.md` | the research side's answers to the questions the on-device agent raises in `STATUS.md`, newest first |
| `docs/E12_corun/` | E12 (task T014): co-running π0's VLM prefix and action expert on one GPU — purpose, definitions, non-negotiables, steps, outputs, pre-registered predictions |
| `docs/E12b_compiled_corun/` | E12b (task T015), the follow-up to E12: the fair GPU expert, the compiled co-run on the whole GPU, and both stages together inside green-context partitions — problem, purpose, measurements, pre-registered thresholds, acceptance |
| `docs/reference_scripts/` | read-only reference implementations for E5b; the on-device agent copies and adapts them into `bench/`. Also `check_model_inputs.py`, the validator of E11's `model_inputs.csv`, which is run in place |
| `tasks/` | one file per task, in priority order; claim, do, report |
| `env/` | how the device was set up: scripts, exact versions, pins |
| `bench/` | the measurement harness you write |
| `results/` | one directory per run: `meta.json`, tidy CSVs, `summary.md`, and `raw/` — the run's raw profiler output (compressed `.zst`; split files carry a `REASSEMBLE.txt`) |
| `scratch/` | ignored by git: caches and uncompressed originals on the device |
| `reports/` | your analysis per experiment |
| `STATUS.md` | your running state: what is done, what is blocked, what you need |
| `analysis/` | research-side post-processing of your results |
