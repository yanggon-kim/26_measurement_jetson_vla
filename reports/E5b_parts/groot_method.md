## Method, and what the window covers

* **Targets** were chosen by `bench/pick_e5b_targets.py --only-groot` from C1's committed
  capture (`results/20260923_E4g_c1_groot_n17_{eager,best}`), applying §1 (skip if SM < 5 %
  and per-launch duration < the 62 µs launch gap) and §5 (≥ 60 % coverage). Coverage came out
  at 86.8 % eager and 62.2 % best. **It is coverage of C1's captured window**, the first 40
  kernels of `ae_total`, not of the whole region. GR00T's denoising loop has no per-step wrap
  point, so its window is `ae_total` where the LeRobot models used `ae_step_0`.
* **Captures** used `bench/run_ncu_e5b_groot.sh`. It carries `run_ncu_e5b.sh`'s full metric set
  (per-pipe issue, LSU requests, 14 stall ratios, launch and occupancy limits, InstructionStats)
  with `run_ncu_c1_groot.sh`'s GR00T configuration copied verbatim, so the kernels captured are
  the ones C1 ranked. Each capture was capped at 20 launches, and pipe metrics were present in
  all four.
* **SASS** was exported from the reports (`ncu_sass_dump.py`), i.e. what actually executed.
  These cuBLAS/CUTLASS kernels live in `libcublasLt` and have no `.cubin` on disk. Inventories
  and static-vs-dynamic cross-checks were in each directory's `inventory/` (2026-09-28: moved to
  the docs/03 §0 paths `sass/<kernel>_sm87_*`, one per SYMBOL, by `bench/e5b_static.py`; see that
  directory's meta.json `e5b_static`). Each cross-check is
  paired with the details CSV of the **same** kernel. The older queue took the first CSV in the
  directory, which in a multi-target directory would pair a kernel with another's metrics.
* **Relation to T007's kernels.** GR00T's GEMMs are the `_relu_` epilogue variants, which are
  different symbols from π0's `…128x64_ldg8_f2f_stages_64x3_tn`, so the static part was done
  afresh rather than referenced. `Kernel2` shares a short name with a π0 target, but CUTLASS
  names many instantiations `Kernel2`, and identity is not claimed.
