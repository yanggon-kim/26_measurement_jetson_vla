# E1 — measured peaks of this Orin (T005)

MAXN, 16 SMs, GPU pinned 1300.5 MHz, EMC 3199 MHz, all 12 CPU cores pinned at 2.2 GHz (recorded
in `meta.json.gate`). GEMM rows: median of 100 timed calls after 20 warm-ups, each verified
against an fp32 reference before its time was used. Bandwidth rows: 3 repeats of (20 warm-ups +
100 calls); the reported GB/s is the **best** repeat's median, because this is a peak.

| quantity | spec | measured | ratio |
|---|---|---|---|
| memory bandwidth | 204.8 GB/s | **187.47 GB/s** (`axpy_inplace`) | 0.915 |
| bf16 dense GEMM | 42 TFLOP/s | **33.85 TFLOP/s** (2048³) | 0.806 |
| ridge point | ≈ 205–208 FLOP/B | **180.6 FLOP/B** | 0.87 |

Replaces the two `[ASSUMPTION]` fractions in `docs/00_context.md`:
`eff_flops_frac` 0.40 → **0.806**, `eff_bw_frac` 0.70 → **0.915**.

**Odd, and unresolved: bandwidth varies run to run by up to 1.19×** with the GPU clock, the EMC
clock (sampled *during* the timed loop) and the die temperature all constant.
`raw/bandwidth_modes.jsonl` holds 22 fresh processes spanning 159.9–189.7 GB/s at 47.2–50.5 °C.
Whole-run peaks across the session were 161.8 / 183.4 / 185.6 / 187.5 / 189.1 GB/s (see
`results/20260921_E1_peak/` and its `raw/earlier_passes/`). The constant above is this run's
best. A floor derived from it carries about ±8 %; the sensitivity runs the safe way for E2's
conclusion, since a lower peak raises the floor and lowers the measured multiple of it.

**Not throttling:** `raw/sustained_load.csv` holds 12 repeats of an 8192² bf16 GEMM; the clock
stayed at 1300.5 MHz while the die reached 58.4 °C and throughput did not decay.

**int8 is not a hardware peak.** `torch._int_mm` reaches 10.5 TOP/s, ~⅛ of this GPU's dense
INT8 capability, bit-exact — a PyTorch kernel-selection limit on `sm_87`, recorded as measured
and explicitly not offered as a hardware constant.
