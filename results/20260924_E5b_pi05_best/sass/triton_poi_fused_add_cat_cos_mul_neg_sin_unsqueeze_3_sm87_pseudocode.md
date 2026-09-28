# Pseudocode — `triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_3` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_3_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tile-based SIMT GEMM (long backward BRA over K-tiles, LDS A/B tiles, unrolled FFMA; §5 fingerprint)**.

Demangled: `triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_3`

```c
// === Phase: setup + pre-loop (PC 0x228b800-0x228c7d0), 254 instructions: ALU 97, FMA 80, Branch 37, LSU 20, Other 18, Uniform 2
//     thread/block ids (S2R 10), constant-memory params (LDC/ULDC 1), bounds checks (ISETP 14), early EXIT 0, first loads (LDG 10, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x228c7e0-0x228c8b0), 14 instructions
    // global loads LDG 1, async tile copies LDGSTS 0, shared loads LDS 0 / LDSM 0, shared stores STS 0
    // math: HMMA 0, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 0, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x228c8b0
// === Phase: epilogue (PC 0x228c8c0-0x2291c70), 1340 instructions: FMA 577, ALU 496, Branch 124, LSU 120, XU 16, Other 7
//     reduction SHFL 0, BAR 8; output STG 2, EXIT 2
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`IADD3 -> LDG.E.CONSTANT -> IMAD.WIDE.U32 -> STL`

4 dependent instructions; cycles in `sass/triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_3_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
