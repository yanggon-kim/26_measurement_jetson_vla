# Pseudocode — `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tensor-core GEMM (HMMA fed from shared-memory tiles staged per K-tile; §5 fingerprint)**.

Demangled: `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn`

```c
// === Phase: setup + pre-loop (PC 0x2ee0fd00-0x2ee11330), 356 instructions: ALU 211, Branch 53, LSU 40, FMA 19, Other 17, Uniform 16
//     thread/block ids (S2R 36), constant-memory params (LDC/ULDC 2), bounds checks (ISETP 37), early EXIT 0, first loads (LDG 3, LDGSTS 20)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x2ee11340-0x2ee11880), 85 instructions
    // global loads LDG 0, async tile copies LDGSTS 8, shared loads LDS 0 / LDSM 12, shared stores STS 0
    // math: HMMA 32, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 1, DEPBAR 1
    stage_next_tile_and_accumulate();
}   // back-edge at 0x2ee11880; 32.00 iterations per warp per launch (derived)
// === Phase: epilogue (PC 0x2ee11890-0x2ee12e70), 351 instructions: ALU 166, FMA 95, Branch 52, LSU 32, Other 4, Uniform 2
//     reduction SHFL 0, BAR 4; output STG 3, EXIT 1
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`IADD3 -> ISETP.GT.AND -> MOV -> R2P`

4 dependent instructions; cycles in `sass/ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
