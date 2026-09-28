# Pseudocode — `Kernel2__cutlass_80_tensorop_bf16_s16816gemm_bf16_64x64_32x6_nn_align2` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/Kernel2__cutlass_80_tensorop_bf16_s16816gemm_bf16_64x64_32x6_nn_align2_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **tensor-core GEMM (HMMA fed from shared-memory tiles staged per K-tile; §5 fingerprint)**.

Demangled: `void Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16_64x64_32x6_nn_align2>(Params)`

```c
// === Phase: setup + pre-loop (PC 0x86245d00-0x86245d00), 0 instructions: 
//     thread/block ids (S2R 0), constant-memory params (LDC/ULDC 0), bounds checks (ISETP 0), early EXIT 0, first loads (LDG 0, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x86245d00-0x8624b570), 1416 instructions
    // global loads LDG 24, async tile copies LDGSTS 96, shared loads LDS 36 / LDSM 12, shared stores STS 32
    // math: HMMA 16, FFMA 32, HFMA2 0, MUFU 0; sync: BAR 24, DEPBAR 3
    stage_next_tile_and_accumulate();
}   // back-edge at 0x8624b570
// === Phase: epilogue (PC 0x8624b570-0x8624b570), 0 instructions: 
//     reduction SHFL 0, BAR 0; output STG 0, EXIT 0
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`S2R -> SHF.R.S32.HI -> LEA.HI -> SHF.R.S32.HI -> SHF.R.S32.HI -> LEA.HI -> LOP3.LUT -> IMAD.IADD -> LEA.HI -> SHF.R.S32.HI -> IMAD -> IMAD -> IMAD -> IMAD.SHL.U32 -> IMAD.WIDE.U32 -> IADD3 -> SEL -> IMAD.WIDE.U32 -> IADD3 -> IADD3 -> IADD3 -> LDG.E.LTC128B -> LOP3.LUT -> FMUL -> FFMA -> FMNMX -> FMNMX -> F2FP.BF16.PACK_AB -> STG.E`

29 dependent instructions; cycles in `sass/Kernel2__cutlass_80_tensorop_bf16_s16816gemm_bf16_64x64_32x6_nn_align2_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
