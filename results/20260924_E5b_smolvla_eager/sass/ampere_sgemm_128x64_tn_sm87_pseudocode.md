# Pseudocode — `ampere_sgemm_128x64_tn` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/ampere_sgemm_128x64_tn_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tile-based SIMT GEMM (long backward BRA over K-tiles, LDS A/B tiles, unrolled FFMA; §5 fingerprint)**.

Demangled: `ampere_sgemm_128x64_tn`

```c
// === Phase: setup + pre-loop (PC 0x4cb3de00-0x4cb3ebc0), 221 instructions: ALU 145, LSU 32, Other 17, Branch 16, FMA 10, Uniform 1
//     thread/block ids (S2R 4), constant-memory params (LDC/ULDC 0), bounds checks (ISETP 22), early EXIT 0, first loads (LDG 15, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x4cb3ebd0-0x4cb41670), 683 instructions
    // global loads LDG 24, async tile copies LDGSTS 0, shared loads LDS 36 / LDSM 0, shared stores STS 24
    // math: HMMA 0, FFMA 512, HFMA2 0, MUFU 0; sync: BAR 2, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x4cb41670
// === Phase: epilogue (PC 0x4cb41680-0x4cb42f70), 400 instructions: ALU 212, FMA 89, Branch 51, LSU 40, Other 8
//     reduction SHFL 0, BAR 4; output STG 9, EXIT 1
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`ISETP.GE.AND -> IADD3 -> IADD3 -> IADD3 -> IADD3 -> IADD3 -> IADD3 -> IADD3 -> IADD3 -> IADD3 -> IADD3 -> IADD3 -> IADD3 -> IADD3 -> IADD3 -> IADD3 -> LDG.E.LTC128B.CONSTANT`

17 dependent instructions; cycles in `sass/ampere_sgemm_128x64_tn_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
