# Pseudocode — `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tensor-core GEMM (HMMA fed from shared-memory tiles staged per K-tile; §5 fingerprint)**.

Demangled: `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn`

```c
// === Phase: setup + pre-loop (PC 0x85b08000-0x85b098e0), 399 instructions: ALU 220, Branch 85, LSU 42, FMA 19, Other 17, Uniform 16
//     thread/block ids (S2R 68), constant-memory params (LDC/ULDC 2), bounds checks (ISETP 37), early EXIT 0, first loads (LDG 3, LDGSTS 20)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x85b098f0-0x85b0a070), 121 instructions
    // global loads LDG 0, async tile copies LDGSTS 8, shared loads LDS 0 / LDSM 16, shared stores STS 0
    // math: HMMA 64, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 1, DEPBAR 1
    stage_next_tile_and_accumulate();
}   // back-edge at 0x85b0a070; 64.00 iterations per warp per launch (derived)
// === Phase: epilogue (PC 0x85b0a080-0x85b0c270), 544 instructions: ALU 235, FMA 207, Branch 53, LSU 39, Other 8, Uniform 2
//     reduction SHFL 0, BAR 4; output STG 5, EXIT 1
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`IADD3 -> ISETP.GT.AND -> MOV -> R2P`

4 dependent instructions; cycles in `sass/ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
