# T007 — π0 instruction-level analysis of the action-expert kernels (E5b)  · priority P1 · depends on: T004 (step 2), T005 (E2 script)

Model: π0 — LeRobot's π0 as in T002–T005 (`lerobot @ f0d2b37beb`, `lerobot/pi0_libero_finetuned_v044`), S = 51, eager and
`reduce-overhead` (T003's best stock mode). Off-chip bytes per protocol §3.3's two-counter rule (2026-09-22). Follow
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
"not available" with the exact failing command. The **full** SASS / PTX dumps, the analysed `.cubin` files and
every `.ncu-rep` are committed under `results/<run>/raw/` (compressed, protocol §3.5 / §6) and listed in
`meta.json.raw_files`; repository-size lines in `STATUS.md` (§6f). Self-review (§11).

status: claimed 2026-09-24

status: done 2026-09-25

**Results.** Six configurations captured — π0, π0.5 and SmolVLA × {eager, best} — in
`results/20260924_E5b_<model>_<mode>/`, with `stall_attribution.csv` in each and
`reports/E5b_summary.md` generated from them. 24 ncu captures, 0 non-zero exits.
bench commit `b450d7945d4c`.

**Headline: the expert is NOT memory-latency bound by §7's own threshold.**

Time-weighted `long_scoreboard` share of waiting, launch-bound kernels excluded per §1:

| config | time-weighted | max across kernels | §7 ≥ 60 %? |
|---|---:|---:|---|
| π0.5 best | 47.6 % | 74.9 % | no |
| π0 best | 46.3 % | 77.0 % | no |
| SmolVLA best | 30.8 % | 57.5 % | no |
| π0 eager | 32.4 % | 44.0 % | no |
| SmolVLA eager | 30.0 % | 41.5 % | no |
| π0.5 eager | 19.5 % | 36.6 % | no |

**None of the six meets it.** I first reported 52–73 % from the MAXIMUM across kernels,
which is exactly what a threshold on the weighted figure exists to prevent; corrected.

By instruction class, the picture is not uniform:

* **π0 compiled is the one memory-dominated case**, 67.2 % LSU — the best case in the set
  for a memory-side accelerator.
* **π0 eager is ALU/XU dominated**, 44.9 %, driven by `wait` at 45 % (fixed-latency ops),
  not memory. Concluding "memory-bound" from the eager configuration of the model this
  study started from would be wrong.
* Compiling moves most models toward LSU (π0 34.5 → 67.2, π0.5 27.2 → 48.4) but **not**
  SmolVLA (41.1 → 35.0), so "optimise and the memory wall appears" is a tendency, not a rule.

**§5 coverage.** Five configurations meet the ≥ 60 % bar. **π0.5 eager reaches only 43.4 %**
and the shortfall is reported rather than the bar lowered: eight of its kernels are
launch-bound (SM 0.1–2.5 %, a few µs per launch against a 62 µs gap) carrying ~57 % of its
`ae_step` time. That is itself the finding — a memory-side accelerator cannot help a phase
waiting on launches.

**Deviations and defects.**

1. `bench/pick_e5b_targets.py`'s §1 gate compared a kernel's TOTAL duration against a
   single launch gap. Fixed to per-launch; it changed a real verdict (π0.5 eager's claimed
   coverage fell from 65.7 % to 43.4 %) and would otherwise have spent captures on eight
   launch-bound kernels.
2. **π0.5/best's first capture profiled nothing and exited 0.** Its target, `internal::kernel`
   at 96.4 % of that configuration's `ae_step` time, was taken from the C1 capture; a fresh
   run of the same `--compile-mode reduce-overhead` path launched only Triton kernels.
   Kernel selection is not reproducible across runs, so **any target list derived from an
   earlier capture can name a kernel the current run never launches.** Re-captured with a
   `triton_` family regex (17 kernels). `run_ncu_e5b.sh` now exits 3 on a zero-kernel
   capture instead of 0.
3. E5's per-region kernel lists must **not** be used to pick E5b targets: E5b's "best" is
   `--compile-mode reduce-overhead` and E5's is the ladder rung. Different configurations,
   different kernels.
4. Captures contain more kernels than their filename says, because ncu matches a regex on
   short names that are substrings of one another. Attribution is unaffected (rows merge by
   `_short_name`), but 19 of the report's 49 sections are launch-bound and are now marked
   as such.

**Open question for the research side.** `imc_miss` is π0.5 eager's largest single wait at
27 % and §7 does not document it. C1 independently found the eager expert `imc_miss`-limited
in all three LeRobot models, so it is a reproduced observation, not noise. What does it mean
on this device?
