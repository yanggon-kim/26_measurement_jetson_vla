# Pseudocode — `Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nn_align2` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nn_align2_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tensor-core GEMM (HMMA fed from shared-memory tiles staged per K-tile; §5 fingerprint)**.

Demangled: `void Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nn_align2>(Params)`

```c
// === Phase: setup + pre-loop (PC 0x86236500-0x86237ae0), 351 instructions: ALU 151, Uniform 75, FMA 63, Branch 26, LSU 18, Other 17, Shuffle 1
//     thread/block ids (S2R 17), constant-memory params (LDC/ULDC 9), bounds checks (ISETP 35), early EXIT 1, first loads (LDG 18, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x86237af0-0x862382c0), 126 instructions
    // global loads LDG 16, async tile copies LDGSTS 0, shared loads LDS 0 / LDSM 0, shared stores STS 4
    // math: HMMA 8, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 2, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x862382c0; 3.00 iterations per warp per launch (derived)
// === Phase: epilogue (PC 0x862382d0-0x86239770), 331 instructions: ALU 129, FMA 112, Branch 52, LSU 36, Other 2
//     reduction SHFL 0, BAR 8; output STG 9, EXIT 3
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`LD.E -> MOVM.16.MT88 -> HMMA.16816.F32.BF16 -> HMMA.16816.F32.BF16 -> HMMA.16816.F32.BF16 -> HMMA.16816.F32.BF16`

6 dependent instructions; cycles in `sass/Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nn_align2_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
