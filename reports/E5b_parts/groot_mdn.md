## Measured / derived / not measured

**Measured:** the four in-model ncu captures, the eight isolated sweep captures, and sweep
timings (n = 200 per point).
**Derived:** the §7 verdicts (thresholds applied to the measured stall shares), × floor
(against E1's measured 187.47 GB/s / 33.85 TFLOP/s), and coverage.
**Not measured:**
- E5b for GR00T N1.5 and N1.6; their C1 captures exist and the scripts take `n15` and `n16`.
- A sweep of the cross-attention k/v projections (2048-wide input).
- The S sweep for π0 (T007). π0.5 and SmolVLA were swept on 2026-09-25 (T008 step 6):
  `E5b_pi05.md`, `E5b_smolvla.md`.
