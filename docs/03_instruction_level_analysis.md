# 03 — Instruction-level analysis of the action-expert kernels (experiment E5b)

**The question E5b answers.** The research claims that the action expert's kernels are *memory-bound*: that the
GPU's streaming multiprocessors (SMs) spend the time inside a skinny GEMM / GEMV waiting for weight loads, not
computing and not (only) waiting for the CPU to launch the next kernel. E5 shows this with counters; E5b backs it
with **compiled, instruction-level evidence**: what the kernel's machine code (SASS) does in its inner loop, which
instructions the warps stall on, and how that changes (a) with the token count S and (b) under `torch.compile`.
If the evidence says the kernels are compute-bound, launch-bound or something else, that is the result.

This document is self-contained: you need no other guide. It applies to **every model in the plan** (π0, π0.5,
SmolVLA, GR00T N1.7); tasks say which model and when. `docs/02_protocol.md` stays binding (timing, result schema,
raw files committed under `results/<run>/raw/` with the size rules of §6, "never run `ncu --set full` unfiltered").

Vocabulary: *SM* streaming multiprocessor; *warp* the group of threads an SM schedules together; *SASS* the
GPU's machine instructions; *PTX* the virtual ISA above it; *pipe* an execution pipe inside the SM — LSU
(load/store), FMA (multiply-add, also integer multiply-add), ALU, XU (transcendentals, FP64, shuffle), ADU
(address generation), CBU (control), FP16.

## 0. Inputs and outputs
Inputs: the E5 results of the same model and mode — nsys top kernels per region, `ncu/kernels.csv`
(`docs/02_protocol.md` §3.3) — and the measured Orin constants (`hw_constants.json`, protocol §8).

Output directory: `results/<date>_E5b_<model>_<mode>_<short>/` with

| file | content |
|---|---|
| `meta.json` | protocol §5, plus `gpu_arch`, `cuobjdump_version`, library paths + sha256 of every `.so` / `.cubin` dumped |
| `sass/targets.csv` | `kernel_short, mangled_symbol, demangled, library_or_cubin, arch, region, role, S, share_of_ae_step_kernel_time, launches_per_step, grid, block` |
| `sass/<kernel>_<arch>_total_inventory.csv` and `.md` | whole-kernel instruction inventory (§4) |
| `sass/<kernel>_<arch>_inner_inventory.csv` and `.md` | inner-loop inventory (§4), with the PC range in the file header |
| `sass/<kernel>_<arch>_backward_bra.csv` | `source_pc, target_pc, span_bytes, role` (§3) |
| `sass/<kernel>_<arch>_hotloop.sass.md` | the **annotated hot-loop excerpt** (§5) — an excerpt; the full dump is in `raw/` |
| `sass/<kernel>_<arch>_pseudocode.md` | decompiled C-like pseudocode with phases and PC ranges (§5) |
| `ncu/parsed.json` | per-kernel pipes, LSU split, stalls (§6; produced by `bench/parse_ncu.py`) |
| `sass/<kernel>_<arch>_crosscheck.md` | static-vs-dynamic cross-check (§8) |
| `sass/<kernel>_<arch>_latency_budget.csv` | inner-loop latency budget (§9) |
| `raw/<kernel>_<arch>.sass.zst`, `raw/<kernel>_<arch>.ptx.zst`, `raw/<kernel>_<arch>.res-usage.txt` | the **full** SASS and PTX dumps and resource usage of every target kernel (compressed per protocol §6) |
| `raw/cubin/*.cubin` | the extracted / JIT-compiled kernel binaries that were dumped (not the vendor `.so`: record its path, version and sha256) |
| `raw/*.ncu-rep.zst`, `ncu/*_details.csv`, `ncu/*_raw.csv` | every ncu report of the run and both full page exports |
| `raw/*.stdout.log` | stdout / stderr of the cuobjdump and ncu commands |
| `summary.md` | 5–15 lines |

Raw files are **committed** (protocol §3.5): the excerpts and inventories are in addition to the full dumps,
never instead of them. Every raw file is in `meta.json.raw_files` with sha256 and bytes; the size rules of
protocol §6 apply (a full SASS dump of one kernel is text and compresses well; never dump a whole library —
dump the target functions). Prose goes to `reports/E5b_<model>.md` (§10).

## 1. Choose the target kernels
From E5 (nsys `cuda_gpu_kern_sum` restricted to `ae_step`, and `ncu/kernels.csv`), take for each model and mode:
1. the GEMM / GEMV kernels that serve the expert's projections at the model's **own S** — for π0 / π0.5 the
   shapes of `docs/01_measurement_plan.md` E2 (q 1024→2048, kv 1024→512, o 2048→1024, gate/up 1024→4096,
   down 4096→1024; S = 51 / 50); for SmolVLA and GR00T read the shapes from the loaded model and list them in
   `targets.csv`. Several projections often share one kernel symbol; distinct symbols are distinct targets;
2. the attention kernel, **if** it ranks in the top-5 kernels of `ae_step` by time;
3. anything else above 10 % of `ae_step` kernel time (an elementwise or norm kernel may surprise us).
Usually this gives 2–5 targets per model and mode. Record each target's share of `ae_step` kernel time; the
targets together should cover ≥ 60 % of it — if not, say what the remainder is made of (many tiny kernels is
itself a finding).

Do **not** run the deep analysis (§5–§9) on a kernel whose `Compute (SM) Throughput` is below 5 % *and* whose
duration is comparable to the launch gap: it is launch-bound, and §2–§4 (dump + inventories) plus the E5 gap
data are the evidence. Say so in the report.

## 2. Find the symbol and dump the SASS
1. **Architecture string.** We expect the Orin to be `sm_87` (compute capability 8.7). Confirm it: compute
   capability from `torch.cuda.get_device_capability()`, and the list of architectures actually present in the
   library from step 3. Record it as `gpu_arch`.
2. **Kernel name.** Take the full kernel name exactly as ncu's `details` page prints it (`Kernel Name`), do not
   paraphrase or guess; demangle for reading with `c++filt`.
3. **Locate the binary.** Eager-mode GEMM kernels live in the cuBLAS / cuBLASLt libraries that PyTorch loads
   (find the exact file: `python -c "import torch, os; print(os.path.dirname(torch.__file__))"`, then
   `ldd`/`find` for `libcublas*.so*`, `libcublasLt*.so*`; the JetPack wheel may use the system copies under
   `/usr/local/cuda/lib64` or `/usr/lib/aarch64-linux-gnu`). Attention kernels may live in `libtorch_cuda.so` or
   in the `flash_attn` extension. **Compiled-mode (Inductor / Triton) kernels are generated at run time**: set
   `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to a directory under `scratch/<run>/` (ignored by git)
   before the run, look there for the compiled kernel binaries (`.cubin`), and copy the ones you analyse —
   with the Triton / Inductor source of the same kernel if present — to `results/<run>/raw/cubin/`; `max-autotune` may still pick cuBLAS for GEMM — the ncu
   kernel name tells you which.
4. **List and dump.** `cuobjdump` ships with the CUDA toolkit; use the one from the toolkit that built the
   library. Two option spellings are documented for different cuobjdump versions; run `cuobjdump --help`, save
   it to `env/tool_help/cuobjdump.txt`, use whichever your version accepts, and record the exact command:
   ```
   # spelling 1 (newer)
   cuobjdump --list-text <library.so>                       # one line per (kernel, sm_arch) pair
   cuobjdump --dump-sass -arch=sm_<XX> --function <MANGLED_SYMBOL> <library.so> > results/<run>/raw/<kernel>_sm<XX>.sass
   # spelling 2 (older)
   cuobjdump -symbols <library.so> | c++filt | grep -i "gemm\|gemv"
   cuobjdump -sass -fun "<kernel_name>" <library.so>        > results/<run>/raw/<kernel>_sm<XX>.sass
   ```
   Notes: `-arch=sm_87` takes `=` (a space is an error); `--list-text` prints lines such as
   `SASS text section <N> : x-<mangled>...sm_<XX>.elf.bin` — pick the entry for this GPU's architecture (vendor
   libraries carry several); a benign `Function not found` warning may precede the listing; for a `.cubin`
   point cuobjdump at that file instead of a `.so`. If the library has no SASS for the device architecture,
   or the kernel is JIT-compiled from PTX at load time, record exactly what you found and go as far as the
   available binary allows.
5. Also useful, same tool: `cuobjdump -res-usage <binary>` (registers per thread, shared memory, stack frame,
   spills per kernel), `cuobjdump -ptx <binary>` (virtual ISA), `cuobjdump -xelf all <binary>` then
   `nvdisasm -g -sf <kernel>.sm_<XX>.cubin` (control-flow graph; `-bbnum` numbers basic blocks, `-plr` prints
   live registers). Put the resource usage and the launch configuration in `targets.csv` / the report:

   | kernel | grid | block | regs/thread | shared mem (B) | theoretical occupancy | achieved occupancy |
   |---|---|---|---|---|---|---|

The dump has **two physical lines per instruction**: the mnemonic line `/*PC*/ [@predicate] OPCODE.MOD args ;`
and an encoding line (whitespace, `/*`, hex). `wc -l` ≈ 2 × instructions.

## 3. Identify the inner loop
Every loop ends with a `BRA` whose target PC is lower than its own PC. Enumerate them
(`bench/sass_backward_bra.py`, §11) and classify by span:

| span | likely role |
|---|---|
| 32–128 bytes | small spin: bisection, retry, warp-vote loop |
| 128–512 bytes | medium body: index walk, partition discovery, reduction shuffle ladder |
| > 1 KB | the real inner loop or an outer iteration loop |
| > 10 KB | outer loop with a long (unrolled) body |

A looped inner block has PC range `[BRA_target, BRA_source]`. If there is no long-span backward `BRA`, the
per-element work is fully unrolled and the inner range is the whole unrolled sequence. If your ncu offers a
per-instruction sampling set (documented for some GPUs as `--set sass-source`; check `ncu --help` and the list
from T001), use it — with the same filters as always — to confirm which PCs are hot; otherwise rely on the
`BRA` enumeration plus the algorithm fingerprints of §5. Sanity check: inner-loop counts × trip count must be
able to explain the dynamic LSU activity; if they are far too small, the range is wrong.

## 4. Two instruction inventories
Classify every opcode (base mnemonic before the first `.`) into a pipe family and count — once for the **whole
kernel**, once for the **inner-loop PC range**:

| family | opcodes |
|---|---|
| LSU – Load | `LDG` `LD` `LDS` `LDC` `LDCU` `LDL` `LDSM` `LDGSTS` `TLD4` `TLDS` |
| LSU – Store | `STG` `ST` `STS` `STL` `STSM` |
| LSU – Atomic | `ATOM` `ATOMG` `ATOMS` `RED` `REDG` |
| FMA | `FFMA` `FMUL` `FADD` `FSEL` `FSETP` `FCMP` `FMNMX` `IMAD` (`IMAD.WIDE`) `IMUL` `HFMA2` `HMUL2` `HADD2` |
| ALU | `IADD3` `IADD` `ISETP` `LOP3` `SHF` `MOV` `SEL` `LEA` `PRMT` `FLO` `POPC` `I2I` `I2F` `F2I` `F2F` `IMNMX` `VIMNMX` `BMSK` `PLOP3` |
| Shuffle | `SHFL.*` |
| XU | `MUFU` and FP64 `DMUL` `DADD` `DFMA` `DSETP` |
| Reduce | `REDUX.*` (newer architectures only) |
| Uniform | U-prefixed datapath: `UMOV` `ULEA` `UISETP` `ULOP3` `USHF` `UIADD3` `ULDC` `ULDCU` … |
| Branch / control | `BRA` `BRX` `BSSY` `BSYNC` `EXIT` `NOP` `BAR` (`BAR.SYNC`) `DEPBAR` `WARPSYNC` `S2R` `S2UR` `VOTE` `CALL` `RET` `JMP` … |

Tensor-core GEMM opcodes (documented for Ampere as `HMMA.16816.F16` / `.F32`; newer parts `WGMMA…`) are the
heart of a half-precision GEMM: count them as their **own row "Tensor core"**, do not let them fall into
"Other". Any opcode you do not recognise is listed by name in the report — never silently binned. Reading
hints: `HFMA2 R, -RZ, RZ, 0, 0` is a register-zeroing idiom, not FP16 work; many `ISETP` = bounds/partition
logic; `@P` prefixes = predicated (branch-free conditional) execution.

Each inventory file has three blocks: per-opcode (`opcode, count, category, lsu_subclass`), per-category
(`category, count, pct_of_total`, with `TOTAL`), and the LSU split (`lsu_subclass, count, pct_of_lsu,
pct_of_total`). From the inner-loop inventory also compute the **compute-to-memory ratio** = (FP + tensor-core
+ non-address integer arithmetic) ÷ (global loads + global stores), and the static load:store ratio.

## 5. Annotate the hot loop and decompile
Walk the SASS in PC order and partition it into phases with PC ranges: **setup** (`S2R` thread / block ids,
`IMAD` index computation, bounds `ISETP` + `@P EXIT`, launch parameters read from constant memory) →
**pre-loop** (pointers, strides, tile indices) → **inner loop** → **reduction** (`SHFL`, `BAR.SYNC`, if any) →
**output write / epilogue** (`STG`, `EXIT`). Annotate the inner loop line by line in this format, and commit
only this excerpt (≤ ~200 instructions; longer unrolled bodies: one representative repetition + a note "× k"):
```
// === Phase: inner loop (PC 0x0150–0x0260), one iteration = one K-tile ===
/*0150*/  LDG.E R6, [R10] ;            // w = W[row, k]            <- global load: weight, DRAM
/*0160*/  LDS   R7, [R11] ;            // x = x_tile[k]            <- shared-memory load: activation
/*0170*/  FFMA  R8, R6, R7, R8 ;       // acc += w * x
...
/*0260*/  @P1 BRA 0x150 ;              // back-edge
```
Then write C-like pseudocode of the whole kernel with the phases and PC ranges in comments (it need not
compile; it must be readable), and **name the algorithm** from its fingerprints:
- *tile-based GEMM*: long-span backward `BRA` over K-tiles; per tile shared-memory loads of an A-tile and a
  B-tile (`LDS.*`); hundreds of unrolled `FFMA`; predicated `STG` of the C-tile at the end;
- *tensor-core GEMM*: tensor-core opcodes (`HMMA…`), cooperative per-tile load with `BAR.SYNC`;
- *GEMV-like / skinny* kernels: one long dot-product loop per output row, global loads of the weight row
  dominating, little or no tiling;
- *FlashAttention-style*: two nested tile loops (Q-block × K-block), GEMM → online softmax (`MUFU` exponent /
  reciprocal, running max / sum) → GEMM, butterfly row-reduce.
State which loads touch **weights** (global memory, W is 1–8 MB per projection, re-read every step),
which touch activations (S × hidden: a few hundred KB) and which touch shared / constant memory.

**Dependency chain.** In the inner loop, trace register def-use (`dest ← sources`) and write down the longest
chain in which each instruction waits for the previous one (e.g. `LDG weight → FFMA`; for indirect access
`LDG index → IMAD → LDG value → FFMA`), plus which loads are independent and can be in flight together
(memory-level parallelism). Add a small ASCII pipeline diagram of one iteration.

## 6. Dynamic side: pipes, LSU split, stalls (from the E5 ncu exports)
Write `bench/parse_ncu.py` (§11). For each target kernel (match by `Kernel Name`, join the two pages on `ID`,
average over the captured launches, keep n):
- per-pipe utilisation: the `sm__inst_executed_pipe_<p>.avg.pct_of_peak_sustained_active` (and `_elapsed`)
  columns of protocol §3.3;
- **LSU split** — apportion `lsu_active_pct` by request counts from the raw page:
  ```
  loads   = global_op_ld + local_op_ld + shared_op_ld
  stores  = global_op_st + local_op_st + shared_op_st
  atomics = global_op_red + global_op_atom + shared_op_atom
  total   = loads + stores + atomics
  load_pct = lsu_active_pct × loads ÷ total      store_pct, atomic_pct likewise
  write_pct = store_pct + atomic_pct             load_to_write_ratio = loads ÷ max(1, stores + atomics)
  ```
  Writes are stores **plus** reductions **plus** atomics; counting only `op_st` under-reports writes. Also keep
  the global / local / shared breakdown of loads: for our claim the question is how much of the LSU activity
  is *global* loads (weights) versus *shared* loads (tiles already staged);
- warp-stall reasons as exported (protocol §3.3), `Executed Ipc Active`, `Achieved Occupancy`,
  `Achieved Active Warps Per SM`, `Memory Throughput`, `DRAM Throughput`, `L2 Hit Rate`,
  `gpc__cycles_active.avg.per_warp`.
If a named counter is absent on the Orin, say so and use the nearest exported one (record its name); if targeted
collection is needed, `ncu --metrics <comma-separated metric names>` with the usual filter collects named
metrics only.

## 7. Map stall reasons to instruction classes
| stall reason | caused by |
|---|---|
| long scoreboard | long-latency dependency: a **global-memory load** (`LDG` / `LD.E`) issued, the dependent instruction cannot fire until data returns |
| short scoreboard | short-latency dependency: shared-memory load (`LDS`), constant-cache miss, `SHFL`, `MUFU` |
| memory / MIO throttle | LSU issue queue full: too many loads or stores in flight |
| LG throttle | load/store address-calculation or shared-memory bank conflict |
| math pipe throttle | FMA / tensor-core pipe full: back-pressure from compute |
| barrier | `BAR.SYNC`, `WARPSYNC`: waiting for the other warps (load imbalance, or too few warps to hide it) |
| wait | fixed-latency operation in flight (transcendental, integer divide, FP64), `DEPBAR` |
| not selected | the warp was ready but another was issued — **a sign of ample warp-level parallelism, not a problem** |
| dispatch stall / drain / misc | pipe imbalance, pipeline drain after barrier or branch — normally small |

For each target, a table `inner-loop instruction(s) → stall reason → measured share → evidence`, covering every
stall reason above 5 %. Interpretation: long scoreboard ≥ 60 % is the signature of a **memory-latency-bound**
kernel; `DRAM Throughput` ≥ 90 % of peak is **memory-bandwidth-bound**; the two can coexist. Math-pipe throttle
> 10 % indicates a compute-bound region. High barrier + low occupancy at small S indicates **too little
parallel work for the SMs** — a third possibility that matters for skinny kernels; check the launch grid
(`grid × block` threads ÷ SM count) and `Achieved Active Warps Per SM`.

## 8. Static-vs-dynamic cross-check
One table per target: static counts and shares (LSU, of which Load / Store / Atomic; FMA; tensor core; ALU;
XU + Shuffle; other) and static ratios LSU/FMA, LSU/ALU, load:store — next to the dynamic pipe percentages,
LD %, WR %, LD:WR and dynamic LSU/FMA for each profiled configuration (each S, eager and compiled). Then a
paragraph: do static and dynamic agree in **direction** (LSU-heavy vs FMA-heavy)? They never agree in magnitude,
for four reasons to discuss concretely for this kernel: predicated instructions that rarely fire; trip counts
(a loop body is counted once statically, many times dynamically); different peak issue rates per pipe (100 % LSU
and 100 % FMA have different denominators); per-instruction issue cost (a vector load is one instruction but
several LSU cycles; NOPs and uniform ops cost nothing). If the directions disagree, find out whether the
offending instructions sit on a rarely taken predicated path.

## 9. Inner-loop latency budget
Per-instruction latencies **documented for other architectures (Ada / Blackwell, GDDR memory)** — on the Orin
(Ampere, LPDDR5) they are `[ASSUMPTION]`, order-of-magnitude only; a global load to LPDDR5 is likely *slower*:

| instruction | cycles | | instruction | cycles |
|---|---|---|---|---|
| `LDG` / `LD.E`, DRAM | ~400–500 | | `FFMA` `FMUL` `FADD` | ~4 |
| `LDG`, L2 hit | ~150–200 | | `IMAD` | ~4–6 |
| `LDG`, L1 hit | ~30–40 | | `IADD3` `LOP3` `SHF` `ISETP` | ~2 |
| `LDS` shared | ~30 | | `SHFL.*` | ~5 |
| `LDC` constant | ~1–4 | | `BAR.SYNC` | ~10–20 + wait for other warps |
| `STG` | fire-and-forget (no stall) | | `BRA` predicated | ~2–4 |
| `RED` / `ATOM` | contention-dependent | | `MUFU` | ~16 |

Procedure: (1) list the inner-loop body in order; (2) sum the latencies along the longest dependency chain of
§5 → *serial cycles per iteration* (give the DRAM-hit and the L2-hit variant; a 4–8 MB weight matrix may or may
not stay in L2 between launches — the `L2 Hit Rate` tells you); (3) measured cycles per iteration =
`gpc__cycles_active.avg.per_warp` ÷ inner-loop iterations per warp; (4) the ratio (2) ÷ (3) is how many
dependent chains the SM overlaps through warp-level parallelism; compare it with `Achieved Active Warps Per SM`.
Output `…_latency_budget.csv`: `instruction, role, on_chain, cycles_assumed, source_of_cycles` plus the four
summary numbers. If `gpc__cycles_active` is not exported, give (2) only and say (3) was not measurable.

## 10. Sweeps and the report
- **S sweep (isolated kernels).** Using the E2 micro-benchmark (`bench/micro_ae_kernels.py`) as the ncu target
  with `-k "regex:<target kernel>" -c 5`: S ∈ {1, 16, 51 (the model's own S), 256} for the model's largest
  projection (π0: `mlp_up` 1024→4096) and its `q_proj`. cuBLAS may select a **different kernel** at different
  S: record the kernel name per S; a new symbol is a new target for §2–§4 (deep analysis §5–§9 only for the
  model's own S and for S = 256).
- **Eager vs compiled.** Repeat target selection under the best compiled mode of E4. If the GEMM kernels are the
  same symbols, say so and report only the dynamic metrics again; if Inductor / Triton generated its own, they
  are new targets.
- **Report** `reports/E5b_<model>.md`: (1) the answer in one paragraph — *what do the dominant action-expert
  kernels wait for?* — with the numbers; (2) target table; (3) per target: algorithm, the two inventory
  tables, annotated hot loop, dependency chain, pipe and LSU load/store/atomic table, stall → instruction table,
  cross-check, latency budget; (4) how (3) changes with S and under `torch.compile`; (5) the root-cause chain
  and gap table of protocol §10, connecting floor → E5 idle/launch gaps → in-kernel stalls; (6) not measured /
  not available on this device; (7) self-review (protocol §11). Verification list to satisfy before "done":
  every stall reason > 5 % traced to instructions; chain length consistent with the budget; instruction mix
  consistent with the roofline class of E6; differences across S explained; gap shares sum to 100 % ± 10.

## 11. Scripts to write under `bench/` (specifications)
Reference implementations of the first, third and fourth are in `docs/reference_scripts/` (read-only, written
for desktop GPUs, not validated on Orin): copy them into `bench/`, adapt, and commit your versions.

1. **`bench/ncu_target.py`** — the program ncu launches. Args: `--model`, `--mode {eager,<compile mode>}`,
   `--num-steps N`, `--warmup` (default 20), `--region` (informational). Loads model and the fixture
   observation, runs the warm-up, then `cudaProfilerStart` → exactly one inference with the protocol §2 NVTX
   ranges → `synchronize` → `cudaProfilerStop`. Must run standalone.
2. **`bench/parse_ncu.py`** — input: one or more `.ncu-rep` (or the two exported CSVs). Runs
   `ncu --import <rep> --csv --page details` and `--page raw`; parses with `csv.DictReader`; details page →
   `{ID_kernelname: {Metric Name: {value, unit, section}}}`; raw page → per-`ID` pipe columns and LSU op counts;
   numeric parsing strips `,` and `%`. Output: `ncu/kernels.csv` and `ncu/region_utilisation.csv` (protocol
   §3.3) and `ncu/parsed.json` = `{run_id: {kernel_short: {duration, n_launches, pipe_utilization: {lsu_active_pct,
   fma_active_pct, alu_active_pct, xu_active_pct, adu_active_pct, cbu_active_pct, fp16_active_pct, *_elapsed_pct},
   pipe_lsu_split: {load_pct, write_pct, store_pct, atomic_pct, load_global_pct, load_local_pct,
   load_shared_pct, store_global_pct, store_shared_pct, atomic_global_pct, atomic_shared_pct,
   load_to_write_ratio, op_request_counts}, stalls: {...}, occupancy: {...}, classification}}}`.
3. **`bench/build_inst_inventory.py`** — input: a SASS dump, optional `--inner-loop-range 0xLO-0xHI`, `--csv`,
   `--label`. Match mnemonic lines with
   `^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?P[0-9TF]\s+)?(?:@U?P[0-9TF]\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)`
   (group 1 = PC, group 2 = opcode with modifiers; encoding lines do not match); keep PCs inside the range if
   given; count per full opcode; map the base mnemonic to the families of §4 (add the "Tensor core" family);
   sub-classify LSU into Load / Store / Atomic; emit the three blocks of §4 as CSV or markdown; exit non-zero
   if nothing matched.
4. **`bench/crosscheck_sass.py`** — input: `--inventory <total inventory csv>`, `--ncu-json ncu/parsed.json`,
   `--target-kernel <kernel_short>`, `--single-run`, `--gpu-arch`. Reads the category block of the inventory,
   prints static counts, shares and ratios, then one row per run with LSU %, FMA %, ALU %, LD %, WR %, LD:WR,
   dynamic LSU/FMA and LSU/ALU, the direction-agreement sentence, and the four reasons of §8.
5. **`bench/sass_backward_bra.py`** — input: a SASS dump. For each line matching
   `^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?P[0-9TF]\s+)?(BRA)\b(.*?)(?:;|$)`, take the first `0x…` operand as target;
   if target < PC emit `source_pc, target_pc, span_bytes` and the role class of §3.
