## S sweep (T008 step 6, added 2026-09-25)

`results/20260925_E5b_pi05_ssweep/` — the action expert's layer-0 projections read from the loaded
model (`shapes.json`): largest `mlp.gate_proj` (1024 → 4096) and `self_attn.q_proj` (1024 → 2048).
π0.5 ships **fp32**, so each is swept in fp32 (TF32, the eager rung) and in bf16 (the dtype of the
best compiled rung `bf16_both_graphs`). S ∈ {1, 16, 50 (own S), 256}; `F.linear` alone, one ncu
capture of 3 kernels per point. Time = ncu kernel duration per call (+ cuBLASLt's split-K reduce
kernel where launched); the CUDA-event column is superseded (see `E5b_groot.md` correction).

| target | projection (in → out, dtype) | S | kernel µs (ncu) | × floor | GB/s | % measured BW | kernel | events said µs (superseded) |
|---|---|---:|---:|---:|---:|---:|---|---:|
| `largest` | `layers[0].mlp.gate_proj` (1024 → 4096, float32) | 1 | **106.3** | 1.19 | 158.1 | 84.3 | `enable_if<!T7, void>::type kernel<int, int, floa` | 150.1 |
| `largest` | `layers[0].mlp.gate_proj` (1024 → 4096, float32) | 16 | **118.3** | 1.30 | 144.5 | 77.1 | `cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_6` | 164.0 |
| `largest` | `layers[0].mlp.gate_proj` (1024 → 4096, float32) | 50 | **149.0** | 1.57 | 119.5 | 63.7 | `cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_6` | 186.9 |
| `largest` | `layers[0].mlp.gate_proj` (1024 → 4096, float32) | 256 | **246.8** | 2.10 | 89.2 | 47.6 | `Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16` | 265.9 |
| `largest_bf16` | `layers[0].mlp.gate_proj` (1024 → 4096, bfloat16) | 1 | **57.6** | 1.29 | 145.8 | 77.8 | `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stage` | 99.3 |
| `largest_bf16` | `layers[0].mlp.gate_proj` (1024 → 4096, bfloat16) | 16 | **58.4** | 1.28 | 146.3 | 78.1 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8` | 100.8 |
| `largest_bf16` | `layers[0].mlp.gate_proj` (1024 → 4096, bfloat16) | 50 | **57.7** | 1.22 | 154.3 | 82.3 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stag` | 101.2 |
| `largest_bf16` | `layers[0].mlp.gate_proj` (1024 → 4096, bfloat16) | 256 | **209.9** | 3.31 | 52.5 | 28.0 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stag` | 252.4 |
| `q` | `layers[0].self_attn.q_proj` (1024 → 2048, float32) | 1 | **57.2** | 1.28 | 146.9 | 78.4 | `enable_if<!T7, void>::type kernel<int, int, floa` | 97.9 |
| `q` | `layers[0].self_attn.q_proj` (1024 → 2048, float32) | 16 | **67.9** | 1.48 | 126.4 | 67.4 | `cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_6` | 108.8 |
| `q` | `layers[0].self_attn.q_proj` (1024 → 2048, float32) | 50 | **82.0** | 1.71 | 109.8 | 58.6 | `cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_6` | 119.4 |
| `q` | `layers[0].self_attn.q_proj` (1024 → 2048, float32) | 256 | **80.8** | 1.31 | 142.8 | 76.2 | `Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16` | 129.8 |
| `q_bf16` | `layers[0].self_attn.q_proj` (1024 → 2048, bfloat16) | 1 | **32.8** | 1.46 | 128.1 | 68.3 | `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stage` | 89.7 |
| `q_bf16` | `layers[0].self_attn.q_proj` (1024 → 2048, bfloat16) | 16 | **32.8** | 1.43 | 130.8 | 69.8 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8` | 91.9 |
| `q_bf16` | `layers[0].self_attn.q_proj` (1024 → 2048, bfloat16) | 50 | **35.7** | 1.49 | 126.0 | 67.2 | `ampere_bf16_s16816gemm_bf16_64x128_sliced1x2_ldg` | 89.4 |
| `q_bf16` | `layers[0].self_attn.q_proj` (1024 → 2048, bfloat16) | 256 | **65.7** | 2.07 | 87.7 | 46.8 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stag` | 110.7 |

* **bf16 halves the weight read and the time follows**: `gate_proj` 106 → 58 µs at S = 1, flat
  to S = 50 (57.6–58.4 µs, 1.22–1.29× floor): at the model's own S the bf16 projection is a weight
  stream. In fp32 the time **rises with S** (106 → 149 µs from S = 1 to 50) because cuBLAS picks
  split-K TF32 kernels whose reduce pass grows with S (14.6 → 39.9 µs of the 149).
* At S = 256 both leave the flat region (fp32 2.10×, bf16 3.31× of the bandwidth floor): the GEMM
  is no longer weight-bound there. Floor = max(bytes / 187.47 GB/s, FLOPs / 33.85 TFLOP/s bf16);
  the fp32 rows use the same compute peak, which understates their floor (TF32 peak is lower), so
  their × floor is an upper bound.
