# Pseudocode — `CatArrayBatchedCopy_alignedK_contig__8239fae5` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/CatArrayBatchedCopy_alignedK_contig__8239fae5_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **elementwise / copy kernel (global load -> arithmetic -> global store, no K-loop)**.

Demangled: `void unnamed>::CatArrayBatchedCopy_alignedK_contig<unnamed>::OpaqueType<4>, unsigned int, 4, 128, 1, 16>(T1 *, unnamed>::CatArrInputTensorMetadata<T1, T2, T4, T5>, unnamed>::TensorSizeStride<T2, 4>, int, T2)`

```c
// === Phase: setup + pre-loop (PC 0x22e7400-0x22e7700), 49 instructions: Other 17, ALU 12, Branch 7, FMA 7, LSU 4, Uniform 2
//     thread/block ids (S2R 3), constant-memory params (LDC/ULDC 5), bounds checks (ISETP 6), early EXIT 1, first loads (LDG 0, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x22e7710-0x22e8380), 200 instructions
    // global loads LDG 1, async tile copies LDGSTS 0, shared loads LDS 0 / LDSM 0, shared stores STS 0
    // math: HMMA 0, FFMA 0, HFMA2 1, MUFU 3; sync: BAR 0, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x22e8380; 0.53 iterations per warp per launch (derived)
// === Phase: epilogue (PC 0x22e8390-0x22e97f0), 327 instructions: ALU 152, FMA 134, Branch 25, LSU 10, XU 6
//     reduction SHFL 0, BAR 0; output STG 5, EXIT 3
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`I2F.U32.RP -> MUFU.RCP -> IADD3 -> F2I.FTZ.U32.TRUNC.NTZ -> IMAD -> IMAD.HI.U32 -> IMAD.HI.U32 -> IMAD.MOV -> IMAD -> ISETP.GE.U32.AND -> IMAD.IADD -> ISETP.GE.U32.AND -> IADD3 -> SEL -> IMAD.HI.U32 -> IMAD.MOV -> IMAD -> ISETP.GE.U32.AND -> IADD3 -> ISETP.GE.U32.AND -> IADD3 -> SEL -> IMAD.HI.U32 -> IMAD.MOV -> IMAD -> ISETP.GE.U32.AND -> IMAD.IADD -> ISETP.GE.U32.AND -> IADD3 -> SEL -> IMAD.MOV -> IMAD -> IMAD -> IMAD -> IMAD -> IMAD.WIDE.U32 -> STG.E`

37 dependent instructions; cycles in `sass/CatArrayBatchedCopy_alignedK_contig__8239fae5_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
