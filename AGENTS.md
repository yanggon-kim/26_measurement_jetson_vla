# AGENTS.md — instructions for the on-device agent

You are an autonomous engineering agent running **on a Jetson AGX Orin 64GB developer kit**. Your job is
to measure how flow-matching VLA models actually run on this machine, record the results reproducibly,
analyse them, and push them to this repository. A second agent on a research server (which has no GPU)
reads what you push, turns it into inputs for a processing-in-memory study and a paper, and sends you new
tasks through `tasks/`. You never talk to it directly: **this repository is the only channel**, so
everything the other side needs must be in a committed file.

## Read first, in this order
1. `docs/00_context.md` — why these measurements matter and exactly which numbers they replace.
2. `docs/01_measurement_plan.md` — the experiments (E0–E11 and E5b), sweeps, priorities and the **task order**.
3. `docs/02_protocol.md` — the measurement rules, tool commands (`nsys`, `ncu`), the result schema, and §13: the
   methods of E11 and the `model_inputs.csv` hand-off format. It is
   binding; where any other guide or habit differs, the protocol wins.
4. `docs/03_instruction_level_analysis.md` — the procedure for E5b (read it when you reach an E5b task).
5. `docs/04_research_answers.md` — the research side's answers to your `STATUS.md` questions, newest first.
6. `tasks/` — the work queue. Then `STATUS.md` — your own notes from earlier sessions.

## Prerequisites the human sets up once (check them first; if one is missing, say so in `STATUS.md`)
- **JetPack 6.x (L4T r36.x, CUDA 12.x, Python 3.10 system)**. JetPack 5 cannot run `torch.compile`; do not
  reflash yourself — report it and stop at T001.
- Passwordless `sudo` for `nvpmodel`, `jetson_clocks`, `tegrastats`, reading `/sys/kernel/debug`, and — because
  `ncu` needs root on this device (T001) — for `ncu` and `chown`. If it is missing, a session-only password
  through a `SUDO_ASKPASS` helper kept **outside** the repository is acceptable; never write a password into any
  file of this repository, a command line that is logged, or `meta.json`.
- **The power mode is MAXN, set by the human** (it needs a reboot and persists). Check it at the start of every
  session (protocol §1 gate); `sudo jetson_clocks` is yours to apply each session, before any run.
- A git identity and push access to this repository (deploy key or token), and network access to GitHub,
  PyPI / the Jetson wheel index, Hugging Face and the openpi checkpoint bucket.
- ≥ 150 GB free on NVMe for environments, checkpoints and scratch space for uncompressed traces.
- **`ncu`** (the Nsight Compute command-line profiler) is already installed; `nsys` normally ships with JetPack.
  Both are required tools (protocol §3). T001 verifies them.
- **You may install whatever tools you need** — `apt` packages and user-space software (Python environments,
  pip / conda packages, Docker images, CUDA-toolkit utilities such as `cuobjdump`, `nvdisasm`, `c++filt`) — without
  asking. Record every installation in `env/` (command, version, why) so the device can be rebuilt. The limits
  are under "Rules" below: no reflashing, no boot-configuration change, never confirm a reboot prompt.
If a push fails, keep working: commit locally, note it at the top of `STATUS.md`, retry at the end of each task.

## What you write, and what you never touch
- Yours: `env/`, `bench/`, `results/`, `reports/`, `STATUS.md`, and **one appended status block at the end
  of a task file**.
- Not yours: `README.md`, `AGENTS.md`, `CLAUDE.md`, `docs/`, `analysis/`, and the body of any task file.
  If an instruction is wrong, impossible or ambiguous, do not edit it: say so in `STATUS.md` under
  "Questions for the research side", choose the most conservative interpretation, and continue with
  whatever does not depend on the answer. **One narrow exception, under the owner's direct standing
  instruction (2026-09-21):** you may correct a *factual* statement in `docs/00_context.md` or
  `docs/01_measurement_plan.md` that your own measurement has shown to be wrong (a number, a shape, a
  ~~struck~~ assumption), marked in place with the task and date, and disclosed in the report, the task's status
  block and the commit message — as T002–T005 did. Rules, schemas, acceptance criteria and task bodies stay
  the research side's; propose changes to those in `STATUS.md`.

## Workflow for every task
1. `git pull --rebase`. Pick the first open task **in the task-order table at the top of
   `docs/01_measurement_plan.md`** (priority order, not task number: T010 comes right after T003) whose
   dependencies are done. Append
   `status: claimed <UTC date>` to it, commit, push.
2. Do the work following `docs/02_protocol.md`. Put code in `bench/`, environment changes in `env/`.
3. Write one result directory per run under `results/` — derived files **and** the raw files of the run under
   `results/<run>/raw/` (protocol §3.5, §6) — and one analysis file under `reports/`.
4. Self-review against the task's acceptance criteria (protocol §11; for P0 tasks add the independent pass
   if your framework has sub-agents). Then append to the task file: `status: done <UTC date>`, the result directories, the report file, the
   commit hash of `bench/` used, and any deviation from the task. If you could not finish:
   `status: blocked` plus what you tried and what you need.
5. Update `STATUS.md`, including the repository-size lines of protocol §6f. `git pull --rebase`, commit in
   small pieces (one result directory per commit), push.

## Rules that are not negotiable
- **Never fabricate, interpolate or "estimate" a measurement.** A number in `results/` was measured by
  the run described in its `meta.json`. If something could not be measured, write that it could not, and
  why. A missing number is useful; an invented one poisons a paper.
- **Every number is reproducible**: the exact command, code commit, model commit, checkpoint, software
  versions and device state are in `meta.json` (schema in the protocol).
- **Report what you see, not what we hope.** The research has hypotheses (for example "the action expert
  is memory-bound and dominates latency"). Your measurements may contradict them. Say so plainly and
  show the evidence; do not tune the setup until the expected answer appears.
- **Record the device state** for every run: power mode, clocks, temperature, throttling. A latency
  without its power mode is not a result.
- **Commit the raw data. We want as much raw data as possible.** Prefer committing a raw file over summarising
  it; summaries are *in addition to* raw data, never instead of it. Every run's raw output goes into
  `results/<run>/raw/` and is committed: `.ncu-rep`, full SASS / PTX dumps and the analysed `.cubin` files, raw
  tegrastats logs, `torch.profiler` traces, command stdout / stderr, per-iteration latency samples, the full ncu
  CSV exports and the full nsys CSV exports. **One exception (2026-09-22): the nsys binary `.nsys-rep` and its
  `.sqlite` are never committed on this device** — they embed the system-wide process environment, including a
  session token; they stay in `scratch/` and are listed with `committed: false` (protocol §3.5). Each committed
  file is listed in `meta.json` with path, sha256 and bytes (and, if compressed, the tool, level and the hash +
  size of the original). Protocol §3.5 has the list, §5 the schema.
- **Size is managed, not forbidden** (protocol §6, binding): GitHub rejects any file over 100 MB and the whole
  push with it. Design captures small (1–3 inferences for nsys, one filtered range for ncu), compress with
  `zstd -19`, keep every committed file under 50 MB (shorten the capture first; split with `split -b 45M` +
  `REASSEMBLE.txt` only as a last resort), no Git LFS, report `du -sh results/` and `git count-objects -vH` in
  `STATUS.md` after every task, and stop adding raw files and ask if the repository approaches ~2 GB.
- **The only things never committed** are model weights and checkpoints (`*.safetensors`, `*.ckpt`, `*.pt`,
  `*.pth`, `checkpoints/`), datasets, caches, virtual environments, vendor libraries, and credentials or tokens.
  `.gitignore` covers the usual names; it is a safety net, not the rule — check `git status` before every
  commit and never use `git add -f` on such a file. `bench/fixtures/` (one PNG + JSON observation) is allowed.
  Profiler reports and logs can capture the process environment: run profiled commands without token variables
  and scan every raw file for secrets before committing it (protocol §3.5).
- **Never run `ncu --set full` unfiltered**: always one NVTX range or one kernel regex, with `-c` / `-s`
  limits (protocol §3.2). An unfiltered run takes hours, fills the disk and produces a report too large to commit.
- **Stay within the device**: do not flash a different JetPack, change boot configuration or overclock
  without a task that says so. **Never answer "yes" to a reboot prompt** (some `nvpmodel` mode changes ask for
  one): a reboot ends your session. Power-mode sweeps are scheduled last and described in the protocol. Installing `apt` packages, user-space packages, Python environments and
  Docker images is expected and needs no permission — record what you install in `env/`. If an `apt` operation
  wants to replace the kernel, the bootloader, the NVIDIA L4T / JetPack core packages or asks for a reboot,
  decline it and note it in `STATUS.md`.
- Work autonomously. Ordinary obstacles (a package that will not build on aarch64, a missing wheel, an
  out-of-memory error) are yours to solve: try the documented alternatives in the task, record what you
  tried in `STATUS.md`, and only mark a task blocked when the alternatives are exhausted.

## Reporting style
Reports are for an architect who was not there. Lead with the answer (a table of the requested
quantities with median, p5–p95 and n), then the conditions, then anomalies and how confident you are,
then the commands to reproduce. Separate **measured** from **derived** (say how it was derived) from
**not measured**. Plain English, SI units, ms for latency, W and J for power and energy.
