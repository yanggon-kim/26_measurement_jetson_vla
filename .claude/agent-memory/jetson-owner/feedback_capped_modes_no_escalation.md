---
name: capped-power-modes-are-the-devices-business
description: Never escalate the MODE_30W / MODE_15W reboots to main or the user — carry the standing item in HANDOFF and wait for the device's queue
metadata:
  type: feedback
---

Do not report or escalate the two capped-power-mode reboots (`MODE_30W` id 2, `MODE_15W` id 1) as
something the user should schedule. Carry the standing item in my own `HANDOFF.md`, note it on each
pull, and let the on-device agent's queue reach it.

**Why:** user's decision 2026-09-24 — the reboots are the on-device agent's business, not theirs to
be reminded about, and they asked to just wait. They appear as `PENDING — ask the owner` at the top
of the device's `STATUS.md`, which reads like a request but is the device's own bookkeeping.

**How to apply:** on every pull, confirm the row is still pending and where it sits in the device's
stated order (they come last, after E5b and MODE_50W). Mention it only if the device marks the item
`blocked` or asks directly. See [[device-agent-cadence-and-liveness]].
