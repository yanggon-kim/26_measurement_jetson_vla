# ONBOARDING — jetson, branch `thor`

*Research-side stable facts for the `thor` branch. Owner: `jetson-owner`. The on-device agent never edits this file.*

## 1. Identity
Orphan branch `thor` of `git@github.com:yanggon-kim/26_measurement_jetson_vla.git`: the channel between the research
server (no GPU) and an on-device agent on a **Jetson AGX Thor** that repeats the Orin study (branch `main`) for six
VLA models — π0, π0.5, SmolVLA, GR00T N1.7 / N1.6 / N1.5 — in `eager_fp32` and `best_bf16`. No history is shared
with `main`; never merge the two.

## 2. Layout and write partition
Research side: `README.md`, `AGENTS.md` (`CLAUDE.md` = `@AGENTS.md`), `docs/`, `tasks/` bodies, `analysis/`,
`reference/` (read-only Orin snapshot from `main` @ `4b22260`), `ONBOARDING.md`, `HANDOFF.md`. Device: `env/`,
`bench/`, `results/`, `reports/`, `STATUS.md`, `PROGRESS.md`, status blocks appended to tasks, and one `.gitignore`
change (the nsys lines, only after T001 step 6 is clean).

## 3. Working on it from the research server
Use a separate worktree or clone of `thor`; never switch `main`'s checkout. Fetch, rebase, re-read before every edit;
push right after committing (pushing is how instructions reach the device). A conflict with the device is reported.

## 4. Traps
- Thor facts are hints until T001 / T003 record them; never write one into a task as a requirement.
- The ceilings come only from T003's `hw_constants.json`; `best_bf16` only from T005's `best_bf16.json`.
- Power mode: one mode (the maximum-performance one), never switched by the device; capped-mode work is dropped.
- Orin numbers on this branch live only in `reference/orin/` and `docs/00_context.md`, always labelled Orin.
