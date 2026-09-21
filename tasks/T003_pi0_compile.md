# T003 — π0 under torch.compile (E4)  · priority P0 · depends on: T002

Same harness as T002 with `pytorch_compile_mode` ∈ {`default`, `reduce-overhead`, `max-autotune`,
`max-autotune-no-cudagraphs`}, with the **full sweep N ∈ {1, 2, 4, 6, 8, 10}** for the best mode (needed for the slope
definition of per-step time, protocol §2) and N ∈ {4, 10} for the others; plus the variant that compiles only the
single-step function. Record compile time, graph breaks, recompiles, whether CUDA graphs
were used, steady-state latency per region, peak memory.
Acceptance: `reports/E4_pi0_compile.md` with the speed-up per region vs T002 and the completed ladder
(floor → eager → best compiled, ms per denoising step). If compilation fails on this stack, the report documents
the failure precisely and the best configuration that does work.
Raw files (protocol §3.5 / §6): per-iteration samples, run stdout, and the compile logs (`TORCH_LOGS="graph_breaks,recompiles"`
output, Inductor's `output_code` for the action-expert graph) committed under each result directory's `raw/`.
