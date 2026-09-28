# Pseudocode — `triton_poi_fused_cat_5` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/triton_poi_fused_cat_5_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **elementwise / copy kernel (global load -> arithmetic -> global store, no K-loop)**.

Demangled: `triton_poi_fused_cat_5`

```c
// === Phase: setup + pre-loop (PC 0x2293b00-0x2293b00), 0 instructions: 
//     thread/block ids (S2R 0), constant-memory params (LDC/ULDC 0), bounds checks (ISETP 0), early EXIT 0, first loads (LDG 0, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x2293b00-0x2293e70), 56 instructions
    // global loads LDG 2, async tile copies LDGSTS 0, shared loads LDS 0 / LDSM 0, shared stores STS 0
    // math: HMMA 0, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 0, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x2293e70
// === Phase: epilogue (PC 0x2293e70-0x2293e70), 0 instructions: 
//     reduction SHFL 0, BAR 0; output STG 0, EXIT 0
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`S2R -> IMAD.SHL.U32 -> LOP3.LUT -> IADD3 -> IMAD.WIDE -> LDG.E.64 -> SEL -> STG.E.64`

8 dependent instructions; cycles in `sass/triton_poi_fused_cat_5_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
