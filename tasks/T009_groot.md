# T009 — GR00T N1.7: full set (E3, E4, E5, E5b, E6, E7)  · priority P3 · depends on: T008

Model: GR00T N1.7, github.com/NVIDIA/Isaac-GR00T at the pinned commit (`docs/00_context.md`); action head = 16 DiT
blocks, hidden 1536, S = 41, N = 4. Use its Orin path (`scripts/activate_orin.sh`, `scripts/deployment/orin/`), its
published checkpoint (record name, source, sha256) and a fixture observation in its own input format (record the
dataset, episode, frame). If the model cannot be installed or does not fit, document exactly what failed and mark
the task blocked — do not substitute another model.
1. E3 / E4 as T002 / T003: regions per protocol §2 (`vision`, `vlm_prefix` = the backbone, `ae_step[i]` = one DiT
   denoising step, `ae_layer[j]` = one DiT block); N ∈ {1, 2, 4, 6, 8, 10}; `torch.compile` modes as T003. Also run
   the repository's own `scripts/deployment/benchmark_inference.py` (eager / compile; TensorRT modes belong to
   E10) and compare its per-component numbers with yours.
2. `floor.csv`, E5 nsys + ncu, E6, E7 (MAXN) exactly as T008 steps 1–5.
3. E5b as T008 step 6 with S sweep S ∈ {1, 16, 41, 256}.
Acceptance: `reports/E3_E4_groot.md`, `reports/E5_groot_timeline.md`, `reports/E5b_groot.md`,
`reports/E6_groot_roofline.md`, `reports/E7_groot_power.md`; a GR00T row added to `reports/E5_E5b_models.md`.
Self-review (§11).
