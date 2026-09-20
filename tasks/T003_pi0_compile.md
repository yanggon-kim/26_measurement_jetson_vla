# T003 — π0 under torch.compile (E4)  · priority P0 · depends on: T002

Same harness as T002 with `pytorch_compile_mode` ∈ {`default`, `reduce-overhead`, `max-autotune`,
`max-autotune-no-cudagraphs`}, N ∈ {4, 10}. Record compile time, graph breaks, recompiles, whether CUDA graphs
were used, steady-state latency per region, peak memory.
Acceptance: `reports/E4_pi0_compile.md` with the speed-up per region vs T002 and the completed ladder
(floor → eager → best compiled, ms per denoising step). If compilation fails on this stack, the report documents
the failure precisely and the best configuration that does work.
