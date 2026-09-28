# Pseudocode — `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tensor-core GEMM (HMMA fed from shared-memory tiles staged per K-tile; §5 fingerprint)**.

Demangled: `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn`

```c
// === Phase: setup + pre-loop (PC 0x4f144500-0x4f145e40), 405 instructions: ALU 249, Branch 53, LSU 51, FMA 19, Other 17, Uniform 16
//     thread/block ids (S2R 36), constant-memory params (LDC/ULDC 2), bounds checks (ISETP 45), early EXIT 0, first loads (LDG 3, LDGSTS 27)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x4f145e50-0x4f146720), 142 instructions
    // global loads LDG 0, async tile copies LDGSTS 12, shared loads LDS 0 / LDSM 24, shared stores STS 0
    // math: HMMA 64, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 1, DEPBAR 1
    stage_next_tile_and_accumulate();
}   // back-edge at 0x4f146720; 12.00 iterations per warp per launch (derived)
// === Phase: epilogue (PC 0x4f146730-0x4f147f70), 389 instructions: ALU 196, FMA 95, Branch 54, LSU 38, Other 4, Uniform 2
//     reduction SHFL 0, BAR 4; output STG 5, EXIT 1
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`LDSM.16.M88.4 -> HMMA.16816.F32.BF16 -> HMMA.16816.F32.BF16 -> HMMA.16816.F32.BF16`

4 dependent instructions; cycles in `sass/ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
