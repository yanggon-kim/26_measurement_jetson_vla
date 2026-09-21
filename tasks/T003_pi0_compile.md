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
