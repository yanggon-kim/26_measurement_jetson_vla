# T007 — π0 instruction-level analysis of the action-expert kernels (E5b)  · priority P1 · depends on: T004 (step 2), T005 (E2 script)

Model: π0 (openpi PyTorch), S = 51, eager and the best compiled mode of T003. Follow
`docs/03_instruction_level_analysis.md` end to end; it is self-contained.
1. Targets (§1): the GEMM / GEMV kernels behind q, kv, o, gate/up, down at S = 51, the attention kernel if it is
   in the top-5 of `ae_step`, anything else > 10 % of `ae_step` kernel time → `sass/targets.csv`.
2. For each target: confirm the architecture string (expected `sm_87`), dump SASS with `cuobjdump` (§2), backward-`BRA`
   table and inner-loop range (§3), whole-kernel and inner-loop inventories (§4), annotated hot loop, pseudocode,
   algorithm and dependency chain (§5), pipes + LSU load / store / atomic split + stalls (§6), stall →
   instruction mapping (§7), static-vs-dynamic cross-check (§8), inner-loop latency budget (§9).
3. S sweep on isolated kernels with `bench/micro_ae_kernels.py` as ncu target, S ∈ {1, 16, 51, 256}, for `mlp_up`
   (1024→4096) and `q_proj` (1024→2048), filter `-k "regex:<kernel>" -c 5`; record the kernel chosen at each S (§10).
4. Eager vs compiled (§10).
Scripts to write: `bench/build_inst_inventory.py`, `bench/crosscheck_sass.py`, `bench/sass_backward_bra.py`
(specs in §11; reference copies in `docs/reference_scripts/`).
Result directories: `results/<date>_E5b_pi0_<mode>_S51/` and `results/<date>_E5b_pi0_eager_Ssweep/` with the files of §0.
Acceptance: `reports/E5b_pi0.md` with the structure of §10, answering *what do the dominant π0 action-expert
kernels wait for — memory loads, compute, or launch — and how does it change with S and under `torch.compile`?*;
every stall reason > 5 % traced to instructions; items that the device's tools could not provide are listed as
"not available" with the exact failing command. Full SASS dumps and `.ncu-rep` stay in `raw/`. Self-review (§11).
