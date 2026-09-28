# Pseudocode — `triton_poi_fused__to_copy_cat_cos_linspace_mul_pow_reciprocal_sin_unsqueeze_3` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/triton_poi_fused__to_copy_cat_cos_linspace_mul_pow_reciprocal_sin_unsqueeze_3_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **elementwise / copy kernel (global load -> arithmetic -> global store, no K-loop)**.

Demangled: `triton_poi_fused__to_copy_cat_cos_linspace_mul_pow_reciprocal_sin_unsqueeze_3`

```c
// === Phase: setup + pre-loop (PC 0x2275e00-0x2279040), 805 instructions: XU 236, FMA 232, ALU 176, Branch 105, LSU 34, Other 17, Uniform 5
//     thread/block ids (S2R 4), constant-memory params (LDC/ULDC 1), bounds checks (ISETP 48), early EXIT 2, first loads (LDG 25, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x2279050-0x2279210), 29 instructions
    // global loads LDG 1, async tile copies LDGSTS 0, shared loads LDS 0 / LDSM 0, shared stores STS 0
    // math: HMMA 0, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 0, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x2279210
// === Phase: epilogue (PC 0x2279220-0x2279970), 118 instructions: ALU 66, FMA 25, Branch 20, LSU 5, Uniform 2
//     reduction SHFL 0, BAR 0; output STG 0, EXIT 0
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`IMAD.MOV.U32 -> LDG.E.64.CONSTANT -> IMAD.HI.U32 -> IMAD.X -> IADD3.X -> IADD3.X -> IMAD.MOV.U32`

7 dependent instructions; cycles in `sass/triton_poi_fused__to_copy_cat_cos_linspace_mul_pow_reciprocal_sin_unsqueeze_3_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
