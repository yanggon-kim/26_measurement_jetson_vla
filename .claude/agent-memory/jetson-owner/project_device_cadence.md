---
name: device-agent-cadence-and-liveness
description: How to tell whether the on-device agent is live, blocked, or waiting on a human reboot — and where it keeps that state
metadata:
  type: project
---

The on-device agent runs long unattended campaigns and pushes a `PROGRESS: HH:MM tick` commit
roughly every 20 minutes (`bench/progress_monitor.py`), plus a `PROGRESS.md` it keeps current with
`todo`/`running`/`done`/`blocked` per item.

**Why:** "315 commits behind" is usually ticks, not 315 results, and `STATUS.md` alone cannot tell
you whether the device is working or stopped.

**How to apply:** check `git log -1 --format='%ci'` on the last device commit against now — a tick
within ~30 minutes means it is live. Read `PROGRESS.md`'s tail and `grep -n blocked PROGRESS.md`
before concluding anything is blocked. The capped power modes `MODE_30W` / `MODE_15W` sit as
`PENDING — ask the owner` in the standing item at the top of `STATUS.md` and each needs a human
reboot; the device schedules them **last**, so "PENDING" does not mean "waiting now" — check its
stated execution order. Task claims live as `status: claimed <date>` appended to `tasks/T0NN_*.md`.
