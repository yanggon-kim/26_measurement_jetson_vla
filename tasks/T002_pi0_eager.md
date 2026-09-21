# T002 — π0 end-to-end, eager (E3)  · priority P0 · depends on: T001

**Power-mode gate (protocol §1).** Every timed run only at MAXN + `jetson_clocks` applied in this session, with
16 SMs and the GPU `cur_freq` recorded; steps 1–3 do not depend on it. Environment: start from `env/setup.sh`'s three
Jetson CUDA wheels (torch 2.8.0 / torchvision 0.23.0 / triton 3.4.0, by full URL, `--no-deps`) in a **new** venv
for openpi; install openpi and its dependencies so that pip cannot replace torch (protocol §12: the CPU-only-wheel
trap; re-check `torch.version.cuda` and `is_available()` after each step). If openpi's PyTorch path needs a torch
version other than 2.8.0, record what it asks for, try 2.8.0 first, and report what had to change.

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
Acceptance: one result directory per N, each with the full per-iteration `latency_samples.csv` and the run's
stdout under `raw/` (protocol §3.5); `reports/E3_pi0_eager.md` with the region table, ms per step vs N, the
comparison with 920 ms / 110-258-543 ms, and the first two rows of the ladder (floor 3.15 ms vs measured eager).
If π0 does not fit or run, fall back in this order: lerobot's π0 implementation, then SmolVLA, and say so.

status: claimed 2026-09-21

status: done 2026-09-21

- Result directories: `results/20260921_E3_pi0_eager_N{1,2,4,6,8,10}/`
- Report: `reports/E3_pi0_eager.md`
- Environment: `env/setup_pi0.sh`, `env/requirements_pi0.txt`, `env/README.md` §10
- Fixture: `bench/fixtures/` — `HuggingFaceVLA/libero`, episode 0, frame 0, task_index 0
- Harness commit: see each run's `meta.json.bench_commit` + `bench_files_sha256`

Headline: **52.94 ms per denoising step** (slope over N ∈ {1,2,4,6,8,10}, r² = 0.99999),
against the ladder's assumed 136 ms and a 3.115 ms bandwidth floor — 17.0× the floor.
End-to-end 694.1 ms at N = 4, 1011.6 ms at N = 10. P = 816, S = 51, both measured.

**Deviation 1 — LeRobot's π0 instead of openpi's PyTorch port.** Step 1 of this task
prescribed openpi. Researched in full and rejected on setup cost, none of which measures
anything: a 12 GB orbax/OCDBT checkpoint to convert on-device; a mandatory JAX install
(`openpi.models.gemma` imports `jax`/`flax.linen` at module level even on the PyTorch path);
a `transformers.__version__ == "4.53.2"` string-equality check enforced at model construction
with five files to copy over site-packages; and a `requires-python >=3.11` gate on a
Python-3.10-only device. LeRobot's implementation is a direct port of the same architecture
(the model prints so on load) with a ready-made PyTorch checkpoint. **The weights are a
different fine-tune**, which is stated wherever the paper comparison appears. The openpi
route stays available and is the natural cross-check on the phase-split disagreement.
`docs/00_context.md` and this file were updated to match (owner decision).

**Deviation 2 — the JAX conversion and its fallback were not needed**, since no conversion was
done. For the record: `jaxlib==0.5.3` *does* have a cp310 aarch64 wheel, so the openpi route
is feasible on this device if it is ever wanted.

**Deviation 3 — `compile_model` and `gradient_checkpointing` overridden before construction.**
The checkpoint ships `compile_model: true, compile_mode: "max-autotune"`;
`PI0Pytorch.__init__` compiles `sample_actions` the moment it sees that, so overriding after
`from_pretrained` is too late and T002 would have measured a compiled model with its NVTX
ranges swallowed.

**Deviation 4 — protocol §2's "known property" does not apply.** §2 says openpi's denoising
loop tests a GPU tensor each iteration (`while time >= -dt/2`), forcing a CPU–GPU sync per
step, and that this is to be reported and not fixed. LeRobot's loop is a plain
`for step in range(num_steps)` with Python floats and has no such sync. The 52.94 ms/step is
therefore not inflated by per-step syncs.

**A trap worth recording for every future model environment.** PyPI `lerobot==0.4.4` matches
the checkpoint's `_v044` name and would have produced a **randomly initialised model**: it
needs a patched transformers fork, it lacks the `lm_head → embed_tokens` alias fix this
checkpoint requires, and `PI0Policy.from_pretrained` catches load failures and returns the
model anyway with only a printed warning. A random π0 runs and yields plausible latencies.
`bench/pi0_common.py:assert_real_weights()` now refuses to measure unless 400 sampled tensors
match the checkpoint file.

Findings for T003/T004: the 17× gap to the floor (protocol §9 calls that structural);
denoising step 0 costs +3.13 ms over every later step; 80.2 ms per inference (33.3 % of
`vision`) encodes an all-−1 dummy camera whose attention mask is zero.

Review: self-review in the report §9, plus an independent sub-agent pass (protocol §11).
