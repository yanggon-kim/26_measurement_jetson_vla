# Pseudocode — `CatArrayBatchedCopy__2eb9aed4` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/CatArrayBatchedCopy__2eb9aed4_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **elementwise / copy kernel (global load -> arithmetic -> global store, no K-loop)**.

Demangled: `void unnamed>::CatArrayBatchedCopy<unnamed>::OpaqueType<4>, unsigned int, 3, 64, 64>(T1 *, unnamed>::CatArrInputTensorMetadata<T1, T2, T4, T5>, unnamed>::TensorSizeStride<T2, 4>, int, T2)`

```c
// === Phase: setup + pre-loop (PC 0x2242000-0x22428f0), 144 instructions: ALU 56, FMA 44, Other 17, LSU 11, Branch 8, XU 6, Uniform 2
//     thread/block ids (S2R 3), constant-memory params (LDC/ULDC 10), bounds checks (ISETP 16), early EXIT 2, first loads (LDG 1, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x2242900-0x2242ca0), 59 instructions
    // global loads LDG 1, async tile copies LDGSTS 0, shared loads LDS 0 / LDSM 0, shared stores STS 0
    // math: HMMA 0, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 0, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x2242ca0; 0.39 iterations per warp per launch (derived)
// === Phase: epilogue (PC 0x2242cb0-0x2242d70), 13 instructions: Branch 13
//     reduction SHFL 0, BAR 0; output STG 0, EXIT 1
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`IMAD.HI.U32 -> IADD3 -> IMAD -> ISETP.GE.U32.AND -> IMAD.IADD -> ISETP.GE.U32.AND -> IADD3 -> SEL -> IMAD.HI.U32 -> IADD3 -> IMAD -> ISETP.GE.U32.AND -> IMAD.IADD -> ISETP.GE.U32.AND -> IADD3 -> SEL -> IMAD.MOV -> IMAD -> IMAD -> IMAD -> IMAD.WIDE.U32 -> LDG.E -> STG.E`

23 dependent instructions; cycles in `sass/CatArrayBatchedCopy__2eb9aed4_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
