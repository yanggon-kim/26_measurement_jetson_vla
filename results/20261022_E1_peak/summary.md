# E1 peaks and counter validation (T003) — 2026-10-22, MAXN, jetson_clocks, gate passed

**Ceilings** (`hw_constants.json`):
- Bandwidth: the best repeat median. The median over all repeats is listed beside it.
- Compute: the highest rate measured in either mode, isolated calls (this run) or back to back
  (`results/20261022_E1_peak_backtoback/`). Both modes and the sustained rate are listed.

| ceiling | value | at | mode / kernel | also |
|---|---|---|---|---|
| bandwidth, best (`x.sum()`, read-only) | **258.23 GB/s** | 1 GiB | `reduce_kernel` (SM) | median over all 30 repeats: 250.32; 2 processes 258.23 / 253.82; spec 273 → 94.6 % |
| bandwidth, copy (SM) / copy engine / axpy / add / fill | 236.10 / 237.72 / 233.69 / 234.00 / 201.63 GB/s | 1 GiB | | axpy differs ×1.026 and fill ×1.104 between processes |
| fp32 (`highest`, SIMT) | 6.427 TFLOP/s | 16384³ | back to back (isolated 6.425) | 80 % of the derived 8.06 |
| tf32 | 62.37 TFLOP/s | 1024³ | isolated | back to back 61.8, no OC |
| bf16 / fp16 | **129.79 / 129.82** TFLOP/s | 2048³ | isolated | back to back 111.3 / 109.1 first slice, then 104.3 / 95.2 under OC |
| int8 | 330.6 TOP/s | 8192³ | back to back, fastest slice (isolated 258.4) | state-dependent: 258 for 20 s, then 313–331 |
| fp8 e4m3 (record only) | 302.9 TFLOP/s | 4096³ | isolated | back to back 197.0 → 192.7 |
| ridge (FLOP/B, ÷ 258.23) | fp32 24.9 · tf32 241.5 · bf16 502.6 · fp16 502.7 · int8 1280.3 | | | |

**Counter validation** (`validation.csv`, ncu form D, cold caches):
- DRAM reads: L2 sysmem fill 1.0001–1.0005 × analytic, on copy, axpy, sum and GEMV.
- DRAM writes: sysmem op_write 1.0006–1.0007 ×, on copy and axpy.
- Tensor FLOPs:
  - 1.000 for tf32, int8 and fp8;
  - bf16 / fp16 at 4096³: 1.0078 (= 4128/4096, tile padding of M);
  - M = 51: 1.098 (56/51);
  - M = 1 GEMV: 8.125 (the split-K GEMV kernel pads M to 8).
- SIMT FP32: 1.0001.
- `mcc__*`: 0.

Odd:
- bf16 / fp16 / tf32 / fp8 throughput falls for operands larger than the L2: bf16 82 at 4096³, 58 at 16384³. The
  4096³ GEMM re-reads its operands from DRAM (3.57× A + B), yet draws only 158 GB/s (61 % of 258). So it is **not**
  bandwidth-bound, and the cause is not established here.
- Tensor and memory-path throughput are state-dependent; see `results/20261022_E1_state_probe/`.
