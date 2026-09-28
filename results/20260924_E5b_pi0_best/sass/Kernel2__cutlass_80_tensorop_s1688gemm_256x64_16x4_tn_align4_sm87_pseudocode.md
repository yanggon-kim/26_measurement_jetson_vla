# Pseudocode — `Kernel2__cutlass_80_tensorop_s1688gemm_256x64_16x4_tn_align4` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/Kernel2__cutlass_80_tensorop_s1688gemm_256x64_16x4_tn_align4_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tensor-core GEMM (HMMA fed from shared-memory tiles staged per K-tile; §5 fingerprint)**.

Demangled: `void Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_tn_align4>(Params)`

```c
// === Phase: setup + pre-loop (PC 0x2eef6c00-0x2eef9130), 596 instructions: ALU 237, FMA 143, Branch 83, Uniform 72, LSU 43, Other 17, Shuffle 1
//     thread/block ids (S2R 68), constant-memory params (LDC/ULDC 11), bounds checks (ISETP 37), early EXIT 1, first loads (LDG 2, LDGSTS 30)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x2eef9140-0x2eefa1f0), 268 instructions
    // global loads LDG 0, async tile copies LDGSTS 10, shared loads LDS 0 / LDSM 16, shared stores STS 0
    // math: HMMA 64, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 1, DEPBAR 1
    stage_next_tile_and_accumulate();
}   // back-edge at 0x2eefa1f0; 5.82 iterations per warp per launch (derived)
// === Phase: epilogue (PC 0x2eefa200-0x2eeff9f0), 1408 instructions: FMA 690, LSU 303, ALU 293, Branch 88, Uniform 32, Other 2
//     reduction SHFL 0, BAR 37; output STG 65, EXIT 3
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`UIADD3 -> UISETP.NE.AND -> UIADD3 -> LDSM.16.M88.4 -> FSETP.GEU.AND -> IMAD.IADD.U32`

6 dependent instructions; cycles in `sass/Kernel2__cutlass_80_tensorop_s1688gemm_256x64_16x4_tn_align4_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
