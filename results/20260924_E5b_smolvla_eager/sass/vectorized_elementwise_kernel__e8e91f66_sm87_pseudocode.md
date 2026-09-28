# Pseudocode — `vectorized_elementwise_kernel__e8e91f66` (sm87)

**Mechanical decompilation of the committed SASS** (`raw/vectorized_elementwise_kernel__e8e91f66_sm87.sass`, bench/e5b_static.py): phases are PC ranges cut at the inner loop found in §3, contents are the opcode families counted in each range. It is a reading aid; it does not compile and names no source variable the SASS does not show.

Algorithm (§5 fingerprints): **elementwise / copy kernel (global load -> arithmetic -> global store, no K-loop)**.

Demangled: `void vectorized_elementwise_kernel<4, unnamed>::silu_kernel(TensorIteratorBase &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 2)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, 2>>(int, T2, T3)`

```c
// === Phase: setup + pre-loop (PC 0x4cb92300-0x4cb92300), 0 instructions: 
//     thread/block ids (S2R 0), constant-memory params (LDC/ULDC 0), bounds checks (ISETP 0), early EXIT 0, first loads (LDG 0, LDGSTS 0)
setup_indices_from_blockIdx_threadIdx();
for (iter = 0; iter < n_iter; ++iter) {   // === Phase: inner loop (PC 0x4cb92300-0x4cb94e70), 696 instructions
    // global loads LDG 10, async tile copies LDGSTS 0, shared loads LDS 0 / LDSM 0, shared stores STS 0
    // math: HMMA 0, FFMA 184, HFMA2 22, MUFU 34; sync: BAR 0, DEPBAR 0
    stage_next_tile_and_accumulate();
}   // back-edge at 0x4cb94e70
// === Phase: epilogue (PC 0x4cb94e70-0x4cb94e70), 0 instructions: 
//     reduction SHFL 0, BAR 0; output STG 0, EXIT 0
write_output_tile();
```

## Longest dependency chain in one iteration (static register def-use, deepest)

`S2R -> SHF.L.U32 -> IMAD.WIDE.U32 -> IMAD.WIDE -> ISETP.GE.AND -> IADD3 -> ISETP.GE.AND -> IADD3 -> ISETP.GE.AND -> IADD3 -> ISETP.GE.AND -> IADD3 -> ISETP.GE.AND -> IADD3 -> ISETP.GE.AND -> IADD3 -> ISETP.GE.AND -> IADD3 -> ISETP.GE.AND -> IADD3 -> IMAD.WIDE.U32 -> LDG.E -> FFMA.SAT -> FFMA.RM -> FADD -> FFMA -> FFMA -> MUFU.EX2 -> FFMA -> MOV -> SHF.R.U32.HI -> LOP3.LUT -> IADD3 -> ISETP.GE.AND -> FFMA -> IADD3 -> MUFU.RCP -> FFMA -> FFMA -> FFMA -> FFMA -> FFMA -> FFMA -> FFMA -> SHF.R.U32.HI -> LOP3.LUT -> IADD3 -> IADD3 -> SHF.L.U32 -> ISETP.NE.AND -> PLOP3.LUT -> SEL -> LOP3.LUT -> LOP3.`

63 dependent instructions; cycles in `sass/vectorized_elementwise_kernel__e8e91f66_sm87_latency_budget.csv` (which also gives the chain with the most §9-documented cycles). Loads not on the chain can be in flight together (memory-level parallelism).

```
one iteration:  [loads issue] ---> [wait: scoreboard] ---> [math on the loaded tile] ---> [back-edge]
```
