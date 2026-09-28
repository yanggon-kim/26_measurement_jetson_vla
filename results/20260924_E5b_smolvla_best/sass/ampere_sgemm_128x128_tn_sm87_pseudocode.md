# Pseudocode — `ampere_sgemm_128x128_tn` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/ampere_sgemm_128x128_tn_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tile-based SIMT GEMM (long backward BRA over K-tiles, LDS A/B tiles, unrolled FFMA; §5 fingerprint)**.

Demangled: `ampere_sgemm_128x128_tn`

```c
// === Phase: setup + pre-loop (PC 0x4f684f00-0x4f685ad0), 190 instructions: ALU 125, LSU 27, Other 17, Branch 13, FMA 7, Uniform 1
//     thread/block ids (S2R 4), constant-memory params (LDC/ULDC 0), bounds checks (ISETP 18), early EXIT 0, first loads (LDG 10, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x4f685ae0-0x4f6882f0), 642 instructions
    // global loads LDG 16, async tile copies LDGSTS 0, shared loads LDS 36 / LDSM 0, shared stores STS 16
    // math: HMMA 0, FFMA 512, HFMA2 0, MUFU 0; sync: BAR 2, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x4f6882f0
// === Phase: epilogue (PC 0x4f688300-0x4f689bf0), 400 instructions: ALU 212, FMA 89, Branch 51, LSU 40, Other 8
//     reduction SHFL 0, BAR 4; output STG 9, EXIT 1
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`ISETP.GE.AND -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> IADD3 -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LEA -> LDG.E.LTC128B.CONSTANT`

17 dependent instructions; cycles in `sass/ampere_sgemm_128x128_tn_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
