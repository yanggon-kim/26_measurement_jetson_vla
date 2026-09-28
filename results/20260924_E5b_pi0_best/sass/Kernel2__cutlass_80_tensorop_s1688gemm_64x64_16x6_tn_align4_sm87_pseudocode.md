# Pseudocode — `Kernel2__cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/Kernel2__cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tensor-core GEMM (HMMA fed from shared-memory tiles staged per K-tile; §5 fingerprint)**.

Demangled: `void Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4>(Params)`

```c
// === Phase: setup + pre-loop (PC 0x2ef02200-0x2ef03e60), 455 instructions: ALU 193, FMA 116, Uniform 65, Branch 32, LSU 31, Other 17, Shuffle 1
//     thread/block ids (S2R 20), constant-memory params (LDC/ULDC 14), bounds checks (ISETP 30), early EXIT 1, first loads (LDG 2, LDGSTS 20)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x2ef03e70-0x2ef04680), 130 instructions
    // global loads LDG 0, async tile copies LDGSTS 4, shared loads LDS 0 / LDSM 8, shared stores STS 0
    // math: HMMA 16, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 1, DEPBAR 1
    stage_next_tile_and_accumulate();
}   // back-edge at 0x2ef04680; 32.00 iterations per warp per launch (derived)
// === Phase: epilogue (PC 0x2ef04690-0x2ef06870), 543 instructions: FMA 230, ALU 151, LSU 86, Branch 67, Uniform 7, Other 2
//     reduction SHFL 0, BAR 21; output STG 17, EXIT 3
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`UIADD3 -> UISETP.NE.AND -> UIADD3 -> LDSM.16.M88.4 -> FSETP.GEU.AND -> IMAD.IADD.U32`

6 dependent instructions; cycles in `sass/Kernel2__cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
