## S sweep (T008 step 6, added 2026-09-25)

`results/20260925_E5b_smolvla_ssweep/` — SmolVLA's expert alternates self- and cross-attention
layers, so layer 0 (self) and layer 1 (cross) were both read from the loaded model: largest
`mlp.gate_proj` (720 → 2048), `self_attn.q_proj` (720 → 960) in each, and layer 1's
`self_attn.k_proj`, which projects the VLM's features (**320 → 320, loaded fp32** — recorded as
found; also swept in bf16). S ∈ {1, 16, 50 (own S), 256}; time = ncu kernel duration per call; the
CUDA-event column is superseded (see `E5b_groot.md` correction).

| target | projection (in → out, dtype) | S | kernel µs (ncu) | × floor | GB/s | % measured BW | kernel | events said µs (superseded) |
|---|---|---:|---:|---:|---:|---:|---|---:|
| `k_cross` | `layers[1].self_attn.k_proj` (320 → 320, float32) | 1 | **18.8** | 8.56 | 21.9 | 11.7 | `std::enable_if<!T7, void>::type internal::kernel` | 95.8 |
| `k_cross` | `layers[1].self_attn.k_proj` (320 → 320, float32) | 16 | **19.7** | 8.21 | 22.8 | 12.2 | `ampere_sgemm_32x32_sliced1x4_tn` | 88.9 |
| `k_cross` | `layers[1].self_attn.k_proj` (320 → 320, float32) | 50 | **23.2** | 8.11 | 23.1 | 12.3 | `cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_6` | 99.3 |
| `k_cross` | `layers[1].self_attn.k_proj` (320 → 320, float32) | 256 | **16.2** | 2.85 | 65.9 | 35.1 | `Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6` | 91.6 |
| `k_cross_bf16` | `layers[1].self_attn.k_proj` (320 → 320, bfloat16) | 1 | **9.6** | 8.72 | 21.5 | 11.5 | `enable_if<!T7, void>::type kernel<int, int, __nv` | 89.1 |
| `k_cross_bf16` | `layers[1].self_attn.k_proj` (320 → 320, bfloat16) | 16 | **9.7** | 8.10 | 23.1 | 12.3 | `Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu` | 91.8 |
| `k_cross_bf16` | `layers[1].self_attn.k_proj` (320 → 320, bfloat16) | 50 | **12.1** | 8.43 | 22.2 | 11.9 | `Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gem` | 92.1 |
| `k_cross_bf16` | `layers[1].self_attn.k_proj` (320 → 320, bfloat16) | 256 | **12.8** | 4.51 | 41.5 | 22.2 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8` | 91.3 |
| `largest` | `layers[0].mlp.gate_proj` (720 → 2048, bfloat16) | 1 | **27.8** | 1.76 | 106.4 | 56.7 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stag` | 96.9 |
| `largest` | `layers[0].mlp.gate_proj` (720 → 2048, bfloat16) | 16 | **25.6** | 1.58 | 118.9 | 63.4 | `Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gem` | 93.0 |
| `largest` | `layers[0].mlp.gate_proj` (720 → 2048, bfloat16) | 50 | **26.3** | 1.53 | 122.6 | 65.4 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stag` | 91.4 |
| `largest` | `layers[0].mlp.gate_proj` (720 → 2048, bfloat16) | 256 | **46.5** | 2.00 | 93.8 | 50.0 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stag` | 89.9 |
| `q` | `layers[0].self_attn.q_proj` (720 → 960, bfloat16) | 1 | **16.5** | 2.23 | 84.0 | 44.8 | `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stage` | 89.6 |
| `q` | `layers[0].self_attn.q_proj` (720 → 960, bfloat16) | 16 | **16.4** | 2.13 | 87.8 | 46.8 | `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stage` | 91.1 |
| `q` | `layers[0].self_attn.q_proj` (720 → 960, bfloat16) | 50 | **17.2** | 2.08 | 90.1 | 48.1 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8` | 89.4 |
| `q` | `layers[0].self_attn.q_proj` (720 → 960, bfloat16) | 256 | **26.4** | 2.21 | 85.0 | 45.3 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stag` | 91.3 |
| `q_cross` | `layers[1].self_attn.q_proj` (720 → 960, bfloat16) | 1 | **16.6** | 2.25 | 83.3 | 44.4 | `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stage` | 89.2 |
| `q_cross` | `layers[1].self_attn.q_proj` (720 → 960, bfloat16) | 16 | **16.5** | 2.15 | 87.0 | 46.4 | `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stage` | 90.4 |
| `q_cross` | `layers[1].self_attn.q_proj` (720 → 960, bfloat16) | 50 | **16.9** | 2.05 | 91.6 | 48.9 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8` | 89.3 |
| `q_cross` | `layers[1].self_attn.q_proj` (720 → 960, bfloat16) | 256 | **26.2** | 2.19 | 85.8 | 45.7 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stag` | 91.4 |

* SmolVLA's projections are **small**: the largest is 2.9 MB and runs at 1.53–1.76× floor up to
  S = 50 (25.6–27.8 µs); the q projections (1.4 MB) at 2.05–2.25×. Self- and cross-attention q are
  the same shape and cost the same (16.4–17.2 µs at S ≤ 50).
* The cross-attention k projection (0.4 MB fp32 / 0.2 MB bf16) sits at **8–9× floor** at S ≤ 50:
  9.6–23 µs of kernel for 0.2–0.4 MB is launch and tail, not bandwidth. This is the regime where a
  GEMM is too small for the GPU, which is why SmolVLA's compiled step (4.33 ms for 16 layers,
  `E3_E4_models.md`) is still 4× its floor.
