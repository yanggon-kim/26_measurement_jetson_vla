# Pseudocode — `Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tensor-core GEMM (HMMA fed from shared-memory tiles staged per K-tile; §5 fingerprint)**.

Demangled: `void Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(Params)`

```c
// === Phase: setup + pre-loop (PC 0x8622fb00-0x86231180), 361 instructions: ALU 189, Branch 53, FMA 51, LSU 42, Other 17, Uniform 8, Shuffle 1
//     thread/block ids (S2R 42), constant-memory params (LDC/ULDC 2), bounds checks (ISETP 56), early EXIT 1, first loads (LDG 18, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x86231190-0x86232340), 284 instructions
    // global loads LDG 16, async tile copies LDGSTS 0, shared loads LDS 0 / LDSM 0, shared stores STS 16
    // math: HMMA 16, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 1, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x86232340; 2.00 iterations per warp per launch (derived)
// === Phase: epilogue (PC 0x86232350-0x86233570), 291 instructions: ALU 104, FMA 97, Branch 55, LSU 27, Uniform 6, Other 2
//     reduction SHFL 0, BAR 8; output STG 5, EXIT 3
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`UIADD3 -> UIADD3 -> IADD3 -> LD.E -> HMMA.16816.F32.BF16 -> HMMA.16816.F32.BF16 -> HMMA.16816.F32.BF16 -> HMMA.16816.F32.BF16 -> HMMA.16816.F32.BF16 -> HMMA.16816.F32.BF16 -> HMMA.16816.F32.BF16`

11 dependent instructions; cycles in `sass/Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
