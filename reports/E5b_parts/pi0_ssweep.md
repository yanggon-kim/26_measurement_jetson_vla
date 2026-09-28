## S sweep (T007 step 3, added 2026-09-28 — hand-written addendum, not part of the generated extract)

`results/20260928_E5b_pi0_ssweep/`, MAXN, jetson_clocks, run 2026-09-28 12:49–12:55. The 2026-09-28
completeness audit found that the S sweep T007 asks for had been run for π0.5, SmolVLA and GR00T but never for π0;
this is that run, with the same method (`bench/micro_proj_sweep.py` + `bench/sweep_ncu_table.py`). The two targets are
read from the loaded model's action-expert layer 0 (`shapes.json`): the largest projection `mlp.gate_proj`
(1024 → 4096) and `self_attn.q_proj` (1024 → 2048), both **bf16** as π0 ships. S ∈ {1, 16, 51 (π0's own S: 50
actions + 1 state token), 256}; `F.linear` alone; one ncu capture of 3 kernels per point. Time = ncu kernel duration
per call (no split-K reduce kernel was launched at any point). Floor = max(bytes / 187.47 GB/s, FLOPs / 33.85 TFLOP/s).

| target | projection (in → out, dtype) | S | kernel µs (ncu) | × floor | GB/s | % measured BW | kernel | CUDA events µs (50-call mean) |
|---|---|---:|---:|---:|---:|---:|---|---:|
| `largest` | `layers[0].mlp.gate_proj` (1024 → 4096, bf16) | 1 | **57.2** | 1.28 | 146.9 | 78.4 | `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn` | 56.6 |
| `largest` | same | 16 | **58.2** | 1.28 | 147.0 | 78.4 | `…_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 57.9 |
| `largest` | same | 51 | **57.8** | 1.22 | 154.2 | 82.3 | `…_128x64_ldg8_f2f_stages_64x3_tn` | 57.1 |
| `largest` | same | 256 | **208.2** | 3.28 | 52.9 | 28.2 | `…_128x64_ldg8_f2f_stages_64x3_tn` | 206.9 |
| `q` | `layers[0].self_attn.q_proj` (1024 → 2048, bf16) | 1 | **32.7** | 1.46 | 128.3 | 68.4 | `…_64x64_ldg8_f2f_stages_64x5_tn` | 50.9 |
| `q` | same | 16 | **33.0** | 1.44 | 130.0 | 69.3 | `…_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | 52.5 |
| `q` | same | 51 | **35.7** | 1.49 | 126.1 | 67.3 | `…_64x128_sliced1x2_ldg8_f2f_stages_64x3_tn` | 50.8 |
| `q` | same | 256 | **65.8** | 2.07 | 87.7 | 46.8 | `…_128x64_ldg8_f2f_stages_64x3_tn` | 65.4 |

* **Measured:** at π0's own S = 51 both projections are weight streams — the time is flat from S = 1 to 51
  (`gate_proj` 57.2–58.2 µs, `q_proj` 32.7–35.7 µs) at 1.22–1.49× the bandwidth floor. Only at S = 256 do they leave
  the flat region (3.28× and 2.07× floor): there the GEMM is no longer bound by the weight read.
* The numbers match π0.5's bf16 rows within 0.5 µs (`reports/E5b_pi05.md` S sweep: 57.6 / 58.4 / 57.7 / 209.9 and
  32.8 / 32.8 / 35.7 / 65.7 µs) — the same expert shapes, the same cuBLAS kernels.
* The CUDA-event column times 50 back-to-back calls per event pair. It agrees with ncu for `gate_proj` but is
  15–20 µs above ncu for `q_proj` at S ≤ 51 (`event_overhead_us` in `s_sweep_ncu.csv`): a 33 µs kernel is short
  enough for launch overhead to show. The ncu column is the primary time.
