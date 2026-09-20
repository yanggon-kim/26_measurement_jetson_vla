# T002 — π0 end-to-end, eager (E3)  · priority P0 · depends on: T001

Goal: measured π0 latency per region and **per denoising step** on this device.
1. Clone openpi at the pinned commit (docs/00_context.md); use the **PyTorch** model in its own environment
   (relax pins as described in docs/00_context.md). Follow the PyTorch section of openpi's README exactly, including
   its required patch of the installed `transformers` package (copy `src/openpi/models_pytorch/transformers_replace/*`
   over the matching files of the transformers version the README names). Checkpoint: the LIBERO fine-tune
   `pi0_libero` from openpi's public checkpoint bucket (`gs://openpi-assets/checkpoints/pi0_libero`), converted with
   `examples/convert_jax_model_to_pytorch.py`. The conversion needs JAX on CPU; if JAX will not install on aarch64,
   record that under "Questions for the research side" (the conversion can be done off-device and the result
   copied over by the human) and meanwhile proceed with lerobot's PyTorch π0 checkpoint from Hugging Face.
   Record checkpoint name, source and sha256. Set `pytorch_compile_mode=None` for this task.
2. Save one real LIBERO observation per protocol §2 to `bench/fixtures/` (PNG + JSON) and drive `sample_actions` with it.
3. Instrument the regions of protocol §2 with CUDA events + NVTX (wrap, do not rewrite the model). Print P and S.
4. Run the sweep N ∈ {1, 2, 4, 6, 8, 10} per protocol §2.
Acceptance: one result directory per N; `reports/E3_pi0_eager.md` with the region table, ms per step vs N, the
comparison with 920 ms / 110-258-543 ms, and the first two rows of the ladder (floor 3.15 ms vs measured eager).
If π0 does not fit or run, fall back in this order: lerobot's π0 implementation, then SmolVLA, and say so.
