# Pseudocode — `triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_9` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_9_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **reduction kernel (SHFL / REDUX ladder)**.

Demangled: `triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_9`

```c
// === Phase: setup + pre-loop (PC 0x2297200-0x2297200), 0 instructions: 
//     thread/block ids (S2R 0), constant-memory params (LDC/ULDC 0), bounds checks (ISETP 0), early EXIT 0, first loads (LDG 0, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x2297200-0x2297af0), 144 instructions
    // global loads LDG 11, async tile copies LDGSTS 0, shared loads LDS 2 / LDSM 0, shared stores STS 2
    // math: HMMA 0, FFMA 16, HFMA2 2, MUFU 1; sync: BAR 2, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x2297af0
// === Phase: epilogue (PC 0x2297af0-0x2297af0), 0 instructions: 
//     reduction SHFL 0, BAR 0; output STG 0, EXIT 0
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`S2R -> SHF.L.U32 -> LOP3.LUT -> LEA -> IMAD.WIDE -> LDG.E.128 -> FFMA -> FFMA -> FMUL -> FFMA -> FFMA -> FFMA -> FADD -> FADD -> FADD -> FADD -> FADD -> STS`

18 dependent instructions; cycles in `sass/triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_9_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
