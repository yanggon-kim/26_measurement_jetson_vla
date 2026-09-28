# Pseudocode — `Kernel2__cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_nn_align1` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/Kernel2__cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_nn_align1_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tensor-core GEMM (HMMA fed from shared-memory tiles staged per K-tile; §5 fingerprint)**.

Demangled: `void Kernel2<cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_nn_align1>(Params)`

```c
// === Phase: setup + pre-loop (PC 0x2ef28500-0x2ef29650), 278 instructions: ALU 163, FMA 57, Branch 18, LSU 18, Other 17, Uniform 4, Shuffle 1
//     thread/block ids (S2R 9), constant-memory params (LDC/ULDC 3), bounds checks (ISETP 37), early EXIT 1, first loads (LDG 18, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x2ef29660-0x2ef29b10), 76 instructions
    // global loads LDG 16, async tile copies LDGSTS 0, shared loads LDS 0 / LDSM 0, shared stores STS 2
    // math: HMMA 4, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 2, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x2ef29b10; 28.00 iterations per warp per launch (derived)
// === Phase: epilogue (PC 0x2ef29b20-0x2ef2b3f0), 398 instructions: ALU 175, FMA 115, LSU 56, Branch 49, Other 2, Uniform 1
//     reduction SHFL 0, BAR 8; output STG 17, EXIT 3
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`LD.E -> MOVM.16.MT88 -> HMMA.16816.F32.BF16 -> HMMA.16816.F32.BF16`

4 dependent instructions; cycles in `sass/Kernel2__cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_nn_align1_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
