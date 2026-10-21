# T003 — π0 under torch.compile (E4)  · priority P0 · depends on: T002

Same harness as T002 with `pytorch_compile_mode` ∈ {`default`, `reduce-overhead`, `max-autotune`,
`max-autotune-no-cudagraphs`}, with the **full sweep N ∈ {1, 2, 4, 6, 8, 10}** for the best mode (needed for the slope
definition of per-step time, protocol §2) and N ∈ {4, 10} for the others; plus the variant that compiles only the
single-step function. Record compile time, graph breaks, recompiles, whether CUDA graphs
were used, steady-state latency per region, peak memory.
`max-autotune` on this GPU: Inductor logs "Not enough SMs to use max_autotune_gemm mode" (`min_sms = 68`, the Orin
has 16; T001 at MAXN), i.e. the mode runs without Triton GEMM templates. Measure and report it **as it is**, named
as you proposed — "max-autotune (= default + CUDA graphs + ATen autotuning on this device)" — that is the stack a
user gets. Additionally, at N ∈ {4, 10} only, one labelled variant
`compile_mode = max-autotune+gemm_templates_forced`, in which that size check is overridden before compilation
(candidate route: replace the checking function in `torch/_inductor/utils.py` — the one that logs the message —
with one that returns `True`, in the benchmark process only, never by editing the installed package; use whichever
route works on the installed version and record exactly what you did). Report whether Triton GEMM templates were
then chosen for the action-expert projections (Inductor's autotune log / `output_code`), and the latency. If the
override does not work, record the error; it does not block the task. The ladder's "best compiled" row uses
unmodified modes only; the forced variant is reported beside it.
Acceptance: `reports/E4_pi0_compile.md` with the speed-up per region vs T002 and the completed ladder
(floor → eager → best compiled, ms per denoising step). If compilation fails on this stack, the report documents
the failure precisely and the best configuration that does work.
Raw files (protocol §3.5 / §6): per-iteration samples, run stdout, and the compile logs (`TORCH_LOGS="graph_breaks,recompiles"`
output, Inductor's `output_code` for the action-expert graph) committed under each result directory's `raw/`.

status: claimed 2026-09-21

status: done 2026-09-22

- Result directories: `results/2026092{1,2}_E4_pi0_*_N*/` (20 runs), plus
  `results/regions_E4_pi0_reduce-overhead+step_only_N{4,10}/` (per-region under compilation)
  and `results/20260921_E4_pi0_summary.csv`
- Report: `reports/E4_pi0_compile.md`
- Harness: `bench/run_e4.py`, `analyze_e4.py`, `parse_autotune.py`, `check_equivalence.py`,
  `regen_floor.py`; per-file hashes in each `meta.json.bench_files_sha256`

**The ladder** (ms per denoising step): floor 3.115 → eager 52.56 (16.9×) → **compiled best
stock `reduce-overhead` 16.30 (5.2×)**. The research side's ~90 ms estimate is 5.5× too
pessimistic. Measured compile gain is 3.22× per step; the transplanted 1.51× was an
end-to-end figure and is not transferable across N.

Mechanism: compilation 2.32×, CUDA graphs a further 1.39× (confirmed by log, with a negative
control), autotuning nothing measurable.

**Triton GEMM templates win no action-expert shape.** With the `min_sms = 68` gate defeated
in-process: 0 of 126 expert blocks, 54 of 697 vision/prefix blocks; cuBLAS is ~2× faster on
four of five expert shapes. The committed `output_code` shows the expert graph is 127
`extern_kernels.mm` with Triton only for fused RoPE/GeGLU/RMSNorm — so there were no Triton
GEMMs to gain from.

**The one line that dominates.** `copy.deepcopy(past_key_values)` (`modeling_pi0.py:919`) is
every graph break (33/33 events across all logs). It **cannot be deleted** — the negative
control fails because the expert appends its suffix K/V despite `use_cache=False` — but
`DynamicCache.crop(P)` is an exact substitute: bitwise-identical actions, 0 graph breaks,
**10.37 ms/step (3.3× floor)**, i.e. 1.67× the "optimistic bound" `docs/00_context.md` calls a
stand-in for a perfect software stack.

Deviations:
1. **A result directory was overwritten** (`reduce-overhead_N{4,10}`), against protocol §5.
   The superseded numbers survive at commit `6474fa9` and differ by 0.5 %, which usefully
   bounds this harness's run-to-run reproducibility.
2. **`throttled: false` has no tegrastats log** for any E4 run; it rests on the GPU clock
   being pinned `min == max`. `emc_hz` reads 0 in all E4 runs (debugfs read returned empty).
3. **Per-region latency is absent from the whole-model compiled runs** by necessity (the
   instrumentation graph-breaks them); measured instead via `step_only`, §4.1.
4. **`docs/00_context.md` edited** (ladder rows, the compile-gain row, the corrected
   deepcopy paragraph) under the owner's standing instruction; disclosed in the report and
   the commits.

Errors caught and fixed: two by me before reporting (instrumentation graph-breaks a compiled
model; the `forced` variant reused cached kernels until both arms were re-run cold), three by
the independent review (empty `raw/` and `raw_files: []` in all 20 runs — the same defect
T002's review found, reintroduced in new code; CUDA-graph use inferred rather than measured;
autotune block counts double-counted, plus a quoted log line that existed in no file).

Review: self-review in the report §9 plus an independent sub-agent pass; its findings and the
fixes are recorded there.
