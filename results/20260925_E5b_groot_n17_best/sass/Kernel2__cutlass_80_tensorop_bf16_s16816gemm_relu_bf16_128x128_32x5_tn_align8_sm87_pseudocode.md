# Pseudocode — `Kernel2__cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/Kernel2__cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tensor-core GEMM (HMMA fed from shared-memory tiles staged per K-tile; §5 fingerprint)**.

Demangled: `void Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8>(Params)`

```c
// === Phase: setup + pre-loop (PC 0x86215400-0x86217230), 484 instructions: ALU 229, Branch 79, FMA 69, LSU 46, Uniform 43, Other 17, Shuffle 1
//     thread/block ids (S2R 68), constant-memory params (LDC/ULDC 8), bounds checks (ISETP 41), early EXIT 1, first loads (LDG 2, LDGSTS 32)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x86217240-0x86217a80), 133 instructions
    // global loads LDG 0, async tile copies LDGSTS 8, shared loads LDS 0 / LDSM 16, shared stores STS 0
    // math: HMMA 64, FFMA 0, HFMA2 0, MUFU 0; sync: BAR 1, DEPBAR 1
    stage_next_tile_and_accumulate();
}   // back-edge at 0x86217a80; 64.00 iterations per warp per launch (derived)
// === Phase: epilogue (PC 0x86217a90-0x8622a2f0), 4743 instructions: FMA 1885, ALU 1751, LSU 563, Other 258, Branch 242, Uniform 44
//     reduction SHFL 0, BAR 69; output STG 97, EXIT 10
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`UIADD3 -> UISETP.NE.AND -> UIADD3 -> LDSM.16.M88.4`

4 dependent instructions; cycles in `sass/Kernel2__cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
