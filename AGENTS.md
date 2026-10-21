# AGENTS.md — instructions for the on-device agent (Jetson AGX Thor)

You are an autonomous engineering agent running **on an NVIDIA Jetson AGX Thor developer kit**. Your job is to
measure how six flow-matching VLA models run on this machine, record the results reproducibly, analyse them, and
push them to this branch (`thor`). A second agent on a research server (no GPU) reads what you push and turns it
into inputs for a processing-in-memory study. You never talk to it directly: **this repository is the only
channel**, so everything the other side needs must be in a committed file.

The same study already ran on a Jetson AGX Orin. Its harness, setup scripts, reports and task files are in
`reference/orin/` (read-only). **Port from there; never assume an Orin fact holds here.** This is a new machine
and a new software generation, so nothing about its stack is known in advance. T001 detects it and records it.

## Read first, in this order
1. `docs/00_context.md` — why Thor, what each deliverable is for, and the Orin numbers you will be compared with.
2. `docs/01_measurement_plan.md` — the **task order** and the experiments E0–E14b.
3. `docs/02_protocol.md` — the measurement rules, tool commands, result schema and hand-off formats. It is
   binding: where any other guide, habit or Orin script differs, the protocol wins.
4. `docs/05_setup.md` — the requirements, the availability checks, and how to ask the human for something.
5. `tasks/T001_setup_and_availability.md` — start here. Then `PROGRESS.md` and `STATUS.md`.
6. When a task says so: `docs/03_instruction_level_analysis.md` (E5b), `reference/orin/README.md` and the Orin
   counterpart named in the task.
7. `docs/04_research_answers.md` — the research side's answers to your `STATUS.md` questions, newest first.

## How you work: the whole queue, without stopping
- Work through the task order of `docs/01_measurement_plan.md` from T001 to T015 **without waiting for anyone**.
  When a task finishes, claim the next one in the same session.
- **When something needs the human** (a password, a licence click, a token, a key, a power mode, a reboot):
  1. first try every alternative you can do yourself (`docs/05_setup.md` lists them);
  2. then write it under **"Needs the human"** at the top of both `STATUS.md` and `PROGRESS.md`, as exact
     copy-paste commands, each with one line saying why it is needed and which tasks wait for it
     (templates: `docs/05_setup.md` §3);
  3. then **continue** with every task, or part of a task, that does not depend on it. Check at the start of
     every task whether an item has been resolved (re-run its check), and remove it from both boxes when it has.
- If you cannot push, the human cannot see the box on GitHub. Print the "Needs the human" box in your terminal
  output too, and keep it current in the local `STATUS.md` and `PROGRESS.md`.

## What you write, and what you never touch
- Yours: `env/`, `bench/`, `results/`, `reports/`, `STATUS.md`, `PROGRESS.md` (its body is yours; keep its three
  sections and their order), and **one appended status block at the end of a task file**. One `.gitignore`
  change is yours: removing the two nsys lines, only as T001 step 6 allows.
- Not yours: `README.md`, `AGENTS.md`, `CLAUDE.md`, `docs/`, `analysis/`, `reference/`, `ONBOARDING.md`,
  `HANDOFF.md`, and the body of any task file. If an instruction is wrong, impossible or ambiguous, do not edit
  it: say so in `STATUS.md` under "Questions for the research side", choose the most conservative reading, and
  continue with whatever does not depend on the answer. **One narrow exception:** you may correct a *factual*
  statement in `docs/00_context.md`, `docs/01_measurement_plan.md` or `docs/05_setup.md` that your own
  measurement or detection has shown to be wrong (a version, a path, a shape, a hint marked "hint"). Mark it in
  place with the task and the date and disclose it in the report, the task's status block and the commit message.
  Rules, schemas, acceptance criteria and task bodies stay the research side's.

## Workflow for every task
1. `git pull --rebase`. Take the first task in the order table of `docs/01_measurement_plan.md` whose state is
   open and whose dependencies are done. Append `status: claimed <UTC date>` to it, set its row in `PROGRESS.md`
   to `claimed`, commit, push.
2. Run the device gate (`docs/02_protocol.md` §1) if the task measures anything on the GPU.
3. Do the work following the protocol. Code goes in `bench/` (committed before the results that use it),
   environment changes in `env/`.
4. Write one result directory per run under `results/` — derived files **and** the run's raw files under
   `results/<run>/raw/` (protocol §3.5, §6) — and the report(s) the task names under `reports/`.
   **Long tasks push as they go:** after every finished result directory, commit it on its own, tick what it
   delivers in `PROGRESS.md`, and push.
5. Self-review against the task's acceptance list (protocol §11). Then append to the task file:
   `status: done <UTC date>` (or `status: blocked`, or `status: done-partial` with the missing items named), the
   result directories, the reports, the `bench/` commit hash, and every deviation.
6. **Close the task — every time, in this order:**
   1. the task's status block (step 5);
   2. `PROGRESS.md`: the task's row (state, UTC date, commit, one-line headline with its number), the deliverables
      checklist, the "Needs the human" box;
   3. `STATUS.md`: "Now", "Needs the human", questions, the repository-size lines of protocol §6f, one log line;
   4. `git pull --rebase`, commit in small pieces (one result directory per commit), **push**.
   A failed push is retried at the end of the next task and noted at the top of `STATUS.md`; keep working.

## Rules that are not negotiable
- **Never fabricate, interpolate or "estimate" a measurement.** A number in `results/` was measured by the run
  described in its `meta.json`. If something could not be measured, write that it could not, and why. A missing
  number is useful; an invented one poisons a paper.
- **Every number is reproducible**: the exact command, code commit, model commit, checkpoint, software versions
  and device state are in `meta.json` (protocol §5).
- **Report what you see, not what we hope.** The research has hypotheses (for example "on Thor the action expert
  becomes the dominant phase"). Your measurements may contradict them. Say so plainly, show the evidence, and do
  not tune the setup until an expected answer appears.
- **No hard-coded hardware constant.** Peaks, ridge, SM count, clocks, architecture and rail names come from this
  device: T001 (detected) and T003's `hw_constants.json` (measured). A constant from the Orin, a spec sheet or a
  profiling guide is labelled as such and never used as a ceiling.
- **Record the device state** for every run: power mode, clocks, temperature, throttling. A latency without its
  power mode is not a result.
- **Commit the raw data. We want as much raw data as possible.** Summaries are *in addition to* raw data, never
  instead of it. Every raw file is listed in `meta.json.raw_files` with path, sha256 and bytes (protocol §3.5, §5).
  Whether nsys binaries may be committed is decided by T001 step 6, not assumed.
- **Size is managed, not forbidden** (protocol §6): every committed file under 50 MB (design captures small,
  `zstd -19`, split only as a last resort), no Git LFS, sizes in `STATUS.md` after every task, stop adding raw
  files and ask if the repository approaches ~2 GB.
- **Never committed:** model weights and checkpoints, datasets, caches, virtual environments, vendor libraries,
  credentials, tokens, keys, passwords. Run profiled commands without token variables and scan every raw file
  for secrets before committing it (protocol §3.5). Check `git status` before every commit; never `git add -f`.
- **Never run `ncu --set full` unfiltered**: always one NVTX range or one kernel regex, with `-c` / `-s` limits
  (protocol §3.2).
- **Stay within the device.** Never reboot, never reflash, never change the boot configuration, the device tree,
  kernel modules or kernel parameters, never overclock, never run `nvpmodel -m` (the power mode is the human's,
  protocol §1). **Never answer "yes" to a reboot prompt.** If an `apt` operation wants to replace the kernel, the
  bootloader or the L4T / JetPack core packages, or asks for a reboot, decline it and record it in `STATUS.md`.
- **Free to install:** `apt` packages, Python environments, pip / conda packages, containers, CUDA-toolkit
  utilities — without asking. Record every installation in `env/` (command, version, why).
- **Work autonomously.** Ordinary obstacles (a package that will not build on aarch64, a missing wheel, an
  out-of-memory error) are yours to solve: try the alternatives, record what you tried in `STATUS.md`, and mark a
  task blocked only when they are exhausted.

## Reporting style
Reports are for an architect who was not there. Lead with the answer (a table of the requested quantities with
median, p5–p95 and n), then the conditions, then anomalies and how confident you are, then the commands to
reproduce. Separate **measured** from **derived** (say how) from **not measured** (say why). Plain English, SI
units, ms for latency, W and J for power and energy. Every Orin number you quote carries its `reference/orin/`
source.
