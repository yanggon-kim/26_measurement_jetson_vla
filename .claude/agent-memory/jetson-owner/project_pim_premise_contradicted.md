---
name: pim-premise-contradicted-by-device
description: The device's six-model campaign contradicts the study's starting premise — the action expert dominates only the EAGER baseline, and is memory-bound only AFTER compilation
metadata:
  type: project
---

The Jetson campaign of 2026-09-22..24 (six models: π0, π0.5, SmolVLA, GR00T N1.5/N1.6/N1.7)
contradicts three things the research side built on. Live numbers are in
`reports/E4d_phase_balance.md`, `E4i_best_combinations.md`, `E4g_c1_c2_bottlenecks.md`,
`E5_timeline.md`, `E7_power_energy.md` — read those, not this, for figures.

1. The action-expert share **peaks at the eager baseline** in all six models. No optimised
   deployment has an expert that dominates its frame.
2. "The expert is the memory-bound phase" is **false eager** (issue-limited, `imc_miss`) and
   **true compiled**. The phase that is memory-latency-bound eager is the **VLM prefix**.
3. **Latency is not a proxy for energy**: frame energy-saving ÷ time-saving spans 1.05 to 0.45.
   π0's expert is the *cheap* phase — a low energy share against its time share.

**Why:** a PIM case argued from eager numbers is arguing from launch overhead, and a PIM case
argued from latency alone overstates the joules. The device says this plainly and repeatedly.

**How to apply:** never quote an eager phase split as "the" split; always say which N and which
rung. When the paper or site cites an expert share, check it against the *optimised* row at the
model's *configured* N. See [[pull-and-report-discipline]].
