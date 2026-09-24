---
name: pull-and-report-discipline
description: On a pull-and-report task, commit only what was named, never queue a device task, and give numbers before narrative
metadata:
  type: feedback
---

On a "pull and report" task: commit only the files the brief names, write nothing into `docs/` or
`tasks/`, do not start/queue/request a measurement, and do not push without explicit approval.
Report **numbers first**, narrative second.

**Why:** pushing this repo is how instructions reach the device, so any push or task file is an
action on a running machine, not a note. The user asked for the numbers because a design decision
(GPU vision/VLM of frame t concurrent with the expert of frame t−1) turns on them.

**How to apply:** collect the answers, update `HANDOFF.md` with the detail, and list what is *owed*
to the device as "next steps" rather than writing it. If a `docs/` fix is obviously needed (e.g. the
device asking to extend a checker's `MODELS` set), report it as an owed item.
