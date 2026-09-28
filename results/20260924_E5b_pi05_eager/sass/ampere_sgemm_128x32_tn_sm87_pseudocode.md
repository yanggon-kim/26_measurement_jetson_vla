# Pseudocode — `ampere_sgemm_128x32_tn` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/ampere_sgemm_128x32_tn_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tile-based SIMT GEMM (long backward BRA over K-tiles, LDS A/B tiles, unrolled FFMA; §5 fingerprint)**.

Demangled: `ampere_sgemm_128x32_tn`

```c
// === Phase: setup + pre-loop (PC 0x2275900-0x22763a0), 171 instructions: ALU 114, Other 17, Branch 16, LSU 13, FMA 10, Uniform 1
//     thread/block ids (S2R 4), constant-memory params (LDC/ULDC 0), bounds checks (ISETP 15), early EXIT 0, first loads (LDG 8, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x22763b0-0x2277110), 215 instructions
    // global loads LDG 10, async tile copies LDGSTS 0, shared loads LDS 18 / LDSM 0, shared stores STS 10
    // math: HMMA 0, FFMA 128, HFMA2 0, MUFU 0; sync: BAR 2, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x2277110
// === Phase: epilogue (PC 0x2277120-0x2278270), 278 instructions: ALU 165, Branch 49, FMA 37, LSU 23, Other 4
//     reduction SHFL 0, BAR 4; output STG 5, EXIT 1
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`LDS.128 -> FFMA -> FFMA -> FFMA -> FFMA -> FFMA -> FFMA -> FFMA -> FFMA`

9 dependent instructions; cycles in `sass/ampere_sgemm_128x32_tn_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
