# E1 — measured peaks of this Orin (T005)

MAXN, 16 SMs, GPU pinned 1300.5 MHz, EMC 3199 MHz. GEMM rows: median of 100 timed calls after
20 warm-ups, each verified against an fp32 reference before its time was used. Bandwidth rows:
3 repeats of (20 warm-ups + 100 calls); the reported GB/s is the **best** repeat's median,
because this is a peak.

| quantity | spec | measured | ratio |
|---|---|---|---|
| memory bandwidth | 204.8 GB/s | **185.59 GB/s** (`mul_scalar_fp32`) | 0.906 |
| bf16 dense GEMM | 42 TFLOP/s | **33.55 TFLOP/s** (2048³) | 0.799 |
| ridge point | ≈ 205 FLOP/B | **180.8 FLOP/B** | 0.88 |

Also replaces the two `[ASSUMPTION]` fractions in `docs/00_context.md`:
`eff_flops_frac` 0.40 → **0.799**, `eff_bw_frac` 0.70 → **0.906**.

**Odd, and unresolved: bandwidth varies run to run by up to 1.14×** with the GPU clock, the EMC
clock and the die temperature all constant. `raw/bandwidth_modes.jsonl` holds 22 fresh
processes spanning 165.1–188.6 GB/s; `raw/earlier_passes/` holds three earlier whole-run passes
whose peaks were 183.4, 161.8 and 189.1 GB/s. The constant above is this run's best; any
floor derived from it carries that spread, and a measured-vs-floor ratio should use a peak from
the same session.

**Not throttling:** `raw/sustained_load.csv` holds 12 repeats of an 8192² bf16 GEMM; the clock
stayed at 1300.5 MHz throughout while the die went 51 → 56 °C, and throughput did not decay.

**int8 is not a hardware peak.** `torch._int_mm` reaches 10.5 TOP/s, ~⅛ of this GPU's dense
INT8 capability, bit-exact — a PyTorch kernel-selection limit on `sm_87`, recorded as measured
and explicitly not offered as a hardware constant.
