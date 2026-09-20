# T002 — π0 end-to-end, eager (E3)  · priority P0 · depends on: T001

Goal: measured π0 latency per region and **per denoising step** on this device.
1. Clone openpi at the pinned commit (docs/00_context.md); use the **PyTorch** model. Get a π0 checkpoint that has
   a LIBERO fine-tune (`pi0_libero`, converted to PyTorch with openpi's conversion script if only JAX is published);
   record checkpoint name and source. Set `pytorch_compile_mode=None` for this task.
2. Save one real LIBERO observation (images + state + prompt) to `bench/fixtures/` and drive `sample_actions` with it.
3. Instrument the regions of protocol §2 with CUDA events + NVTX (wrap, do not rewrite the model). Print P and S.
4. Run the sweep N ∈ {1, 2, 4, 6, 8, 10} per protocol §2.
Acceptance: one result directory per N; `reports/E3_pi0_eager.md` with the region table, ms per step vs N, the
comparison with 920 ms / 110-258-543 ms, and the first two rows of the ladder (floor 3.15 ms vs measured eager).
If π0 does not fit or run, fall back in this order: lerobot's π0 implementation, then SmolVLA, and say so.
