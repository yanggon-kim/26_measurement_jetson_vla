# Pseudocode — `ampere_sgemm_32x32_sliced1x4_tn` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/ampere_sgemm_32x32_sliced1x4_tn_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tile-based SIMT GEMM (long backward BRA over K-tiles, LDS A/B tiles, unrolled FFMA; §5 fingerprint)**.

Demangled: `ampere_sgemm_32x32_sliced1x4_tn`

```c
// === Phase: setup + pre-loop (PC 0x4cb95700-0x4cb96610), 242 instructions: ALU 151, LSU 42, Branch 22, Other 17, FMA 9, Uniform 1
//     thread/block ids (S2R 12), constant-memory params (LDC/ULDC 0), bounds checks (ISETP 27), early EXIT 0, first loads (LDG 18, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x4cb96620-0x4cb97ce0), 365 instructions
    // global loads LDG 16, async tile copies LDGSTS 0, shared loads LDS 24 / LDSM 0, shared stores STS 16
    // math: HMMA 0, FFMA 256, HFMA2 0, MUFU 0; sync: BAR 1, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x4cb97ce0; 2.00 iterations per warp per launch (derived)
// === Phase: epilogue (PC 0x4cb97cf0-0x4cb98ef0), 289 instructions: ALU 163, FMA 53, Branch 51, LSU 14, Other 8
//     reduction SHFL 0, BAR 4; output STG 2, EXIT 1
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LDG.E.LTC128B.CONSTANT -> STS`

17 dependent instructions; cycles in `sass/ampere_sgemm_32x32_sliced1x4_tn_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
