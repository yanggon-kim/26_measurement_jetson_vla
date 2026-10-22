# E1 peaks and counter validation (T003) — 2026-10-22, MAXN, jetson_clocks, gate passed

**Ceilings** (`hw_constants.json`): bandwidth = best repeat median; compute = best burst median (calls before any OC event).

| ceiling | value | at | kernel | spread / note |
|---|---|---|---|---|
| bandwidth, best (read-only `x.sum()`) | **258.23 GB/s** | 1 GiB | `reduce_kernel` (SM) | 2 processes: 258.23 vs 253.82 (×1.017); NVIDIA spec 273 → 94.6 % |
| bandwidth, copy (SM, `z = 1.0·x`) / copy engine `copy_` | 236.10 / 237.72 GB/s | 1 GiB | elementwise / memcpy DtoD | repeat spread ≤ ×1.03 |
| bandwidth, axpy / add / write-only fill | 233.69 / 234.00 / 201.63 GB/s | 1 GiB | elementwise | |
| fp32 (`highest`) | **6.43 TFLOP/s** | 16384³ | `cutlass_80_simt_sgemm` (SIMT) | 80 % of the derived 8.06 SIMT peak (2,560 cores × 2 × 1.575 GHz) |
| tf32 | **62.37 TFLOP/s** | 1024³ | `cutlass3x_sm100 … tf32gemm` | falls to 21.9–26.5 at 16384³ |
| bf16 | **129.79 TFLOP/s** | 2048³ | `nvjet_sm110_tst_128x128…` | 82 at 4096³, 58 at 16384³ (no OC events) |
| fp16 | **129.82 TFLOP/s** | 2048³ | `nvjet_sm110_hsh_128x128…` | as bf16 |
| int8 (`torch._int_mm`) | **258.40 TOP/s** | 8192³ | `cutlass3x_sm100 … i256x256x32gemm` | |
| fp8 e4m3 (`torch._scaled_mm`, record only) | 302.92 TFLOP/s | 4096³ | `nvjet_sm110_qqtst_128x256…` | NVIDIA's only published figure: 2,070 FP4 sparse |
| ridge (FLOP/B, ÷ 258.23 GB/s) | fp32 24.9 · tf32 241.5 · bf16 502.6 · fp16 502.7 · int8 1000.7 | | | |

**Counter validation** (`validation.csv`, ncu form D, cold caches):
- DRAM reads: `lts__d_sectors_fill_sysmem` measures 1.0001–1.0005 × analytic on copy, axpy, sum and GEMV.
- DRAM writes: `lts__t_sectors_aperture_sysmem_op_write` measures 1.0006–1.0007 ×.
- Tensor FLOPs: `sm__ops_path_tensor_src_*` measures 1.000 for tf32, int8 and fp8. bf16 / fp16 read 1.0078, which is the tile padding 4128/4096. The M = 51 linear reads 1.098 (M padded to 56).
- SIMT FP32 FLOPs: 1.0001.
- `mcc__dram_throughput_*` is 0 on every kernel, so it is unusable.

Odd, and important for the roofline:
- **Large tensor GEMMs lose throughput as n grows**, with no over-current events. The 4096³ bf16 GEMM reads 3.57× its A + B bytes from DRAM: its operands exceed the 32 MiB L2. The 130 TFLOP/s ceiling is reached only when the working set is L2-resident (2048³).
- With per-call synchronisation the GPU idles between calls, so OC events were rare here (only fp8 4096³ in p2). The sustained, current-limited rate for back-to-back load is T002's: bf16 93.97 TFLOP/s for a 4096³ `x @ x` loop.
