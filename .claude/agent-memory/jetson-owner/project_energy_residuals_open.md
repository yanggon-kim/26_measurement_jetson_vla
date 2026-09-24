---
name: e7-energy-residuals-block-energy-claims
description: Two unreconciled residuals in the E7 stage loops mean GPU-vs-PIM energy cannot be quoted yet — latency is unaffected
metadata:
  type: project
---

π0 N = 10 `bf16_both` in `reports/E7_power_energy.md` has two residuals found by microarch-owner on
2026-09-24: the stage loops sum to 418.64 ms against 424.68 ms measured (6.04 ms unplaced), and the
stage above-idle energies sum to 9509 mJ against 13 666 mJ (a 30 % gap).

**Why:** a 1.4 % time gap cannot produce a 30 % energy gap, so the two are not one discrepancy. It
is either real cost outside both loops or a difference in how each loop's idle baseline is
subtracted. Until it closes, **any GPU-versus-PIM energy comparison is not valid**; latency figures
are unaffected.

**How to apply:** block energy claims sourced from the `prefix`/`ae` sub-loops until reconciled, and
keep using the `full` loop for whole-frame energy. The fix is a question to the device about the
loop definitions and baseline subtraction, not a re-measurement.
See [[pim-premise-contradicted-by-device]].
