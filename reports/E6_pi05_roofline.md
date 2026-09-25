# E6 — π0.5 on the Orin's own roofline

**Task T008 step 4. Binding: plan E6, protocol §8 (the roof is this device's), §9 (floor and
gap ratio), §2 (region times). Result: `results/20260925_E6_pi05/roofline.csv`, with the
analytic inputs in `floor.csv` and `floor_inputs.json` beside it.**

> E6 did not exist for any model until 2026-09-25. `bench/roofline.py` had been written and
> never run, and `floor.csv` — the analytic bytes and FLOPs a roofline stands on — existed only
> for π0, because `run_e3.py`'s `floor_rows()` hardcodes π0's architecture and module paths.
> `bench/floors_models.py` was written to derive them per model from the **loaded** model.

> **⚠ CORRECTION 2026-09-25:** any cross-model comparison with GR00T in this report set an EAGER LeRobot `ae_step` against a CUDA-graphed GR00T one. Matched rungs are in `reports/E6_groot_roofline.md`'s correction block: eager GR00T is 33.2 % of peak bandwidth at 3.0× floor, and compiling its head takes it to 1.20×. The numbers for THIS model below are unaffected; they are eager and labelled so.

## The answer

π0.5, N = 10, MAXN. Roof = **this Orin's measured peaks** (E1/T005: 187.47 GB/s,
33.85 TFLOP/s bf16, ridge **180.57 FLOP/B**); spec peaks are in the CSV alongside.

| region | measured ms | intensity | % peak BW | % peak compute | floor ms | **gap × floor** |
|---|---:|---:|---:|---:|---:|---:|
| `ae_step` | 62.69 | **28.8122 FLOP/B** | **14.761 %** | **2.355 %** | 9.2539 | **6.774×** |

Full table for `vision` and `vlm_prefix` is in the CSV. Protocol §9's reading of
`gap_ratio`: < 1.5× near-optimal, 1.5–3× moderate, 3–10× substantial software overhead,
**> 10× a structural problem (launch-bound, idle GPU, too little parallelism)**.

## The cross-model finding, which is the point

The action expert of **every LeRobot model measured** sits on the memory side of the ridge and
uses almost none of either roof:

| model | `ae_step` ms | intensity | side of ridge (180.57) | % peak BW | % peak compute | gap × floor |
|---|---:|---:|---|---:|---:|---:|
| π0 | 52.79 | 60.0 | memory | 6.45 % | 2.14 % | 15.51× |
| π0.5 | 62.69 | 28.8 | memory | 14.76 % | 2.36 % | 6.77× |
| SmolVLA | 60.80 | 52.5 | memory | 1.78 % | 0.52 % | 56.04× |

**The study's premise holds analytically and fails operationally, for all three.** Each expert
is memory-side, so a memory-side accelerator aims at the right half of the chart. But none is
anywhere near that roof: 1.8–14.8 % of peak bandwidth, 0.5–2.4 % of peak compute, 6.8–56× the
physical floor. Memory is the *nearer* of two distant roofs, not a limit any of them reaches.

For π0.5 specifically the gap is **6.774×**. SmolVLA is the extreme case at 56×, consistent
with it having the highest kernel count of any model measured (1183.7 kernels/step, 92.73 % GPU
idle in E5) — the more kernels per step, the more of the phase is launch overhead.

**Consequence for a PIM case:** on these numbers the win from moving the action expert off the
GPU comes from eliminating per-kernel launch and dispatch overhead, **not** from relieving a
bandwidth limit that none of these models currently reaches. That is a change of mechanism, not
a verdict against the design, and the achievable win is bounded by what launch overhead costs.
E5, E5b and E12 §5 reach the same conclusion by three other routes for π0.

## Measured / derived / not measured

**Measured.** Region times, protocol §2, from
`results/20260922_E4d_pi05_paper_config_N10*/paper_comparison.csv`; the peaks (E1/T005);
the token counts, built through the model's own `embed_prefix` — P = 968 = 3x256 + 200.

**Derived.** Bytes and FLOPs are analytic, from the loaded model's own parameters and shapes
(`bench/floors_models.py`; inputs and resolved module paths recorded in `floor_inputs.json`).
Architecture: 18 VLM layers, 18 expert layers, 8 heads, 1 KV head, head_dim 256 -- config and loaded model agree.

**`vision` FLOPs are **counted**: 660.53 GFLOP, intensity 398.1 FLOP/B, and the 3 encoded image slots are confirmed independently by the measured token count (P = 968 = 3x256 + 200, exactly as pi0's P = 816 = 3x256 + 48).**

**Two derivations that change numbers, both stated rather than buried.**

1. **`ae_step` time is `ae_total / 10`.** π0.5 records only `ae_total`, not per-step regions.
   This needs the explicit `--ae-total-steps` flag, because a divisor under the study's central
   region should not come from a filename. It is an **average step, not a steady-state step**:
   π0, which does record per-step times, shows step 0 about 6 % above the rest.
2. **Compute columns are empty, not 0, where FLOPs were not counted.** `floor.csv` records
   `flops = 0` to mean "not counted analytically", and 0 would be read downstream as "does no
   arithmetic" — which is how `vision` sat in earlier π0 tables with no operational intensity
   until `vision_flops.py` was written.

**Not measured.** A compiled-rung roofline (these are the paper-config numbers; compiling cuts
π0's expert kernels from 968 to 372 per step, so the gap should fall sharply and quantifying
that is the obvious next step); per-kernel roofline points, which need an nsys–ncu join on
kernel identity; and off-chip-byte `source = ncu` rows, which exist for π0 only —
`20260922_E4g_pi0_ncu_eager` is the one capture carrying the L2 sysmem-miss counters.
