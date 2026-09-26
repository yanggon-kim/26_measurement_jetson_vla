# T009 — GR00T N1.7: full set (E3, E4, E5, E5b, E6, E7)  · priority P3 · depends on: T008

Model: GR00T N1.7, github.com/NVIDIA/Isaac-GR00T at the pinned commit (`docs/00_context.md`); action head = 16 DiT
blocks, hidden 1536, S = 41, N = 4. Use its Orin path (`scripts/activate_orin.sh`, `scripts/deployment/orin/`), its
published checkpoint (record name, source, sha256) and a fixture observation in its own input format (record the
dataset, episode, frame). If the model cannot be installed or does not fit, document exactly what failed and mark
the task blocked — do not substitute another model. E11 for this model is a separate task, T012.
1. E3 / E4 as T002 / T003: regions per protocol §2 (`vision`, `vlm_prefix` = the backbone, `ae_step[i]` = one DiT
   denoising step, `ae_layer[j]` = one DiT block); N ∈ {1, 2, 4, 6, 8, 10}; `torch.compile` modes as T003. Also run
   the repository's own `scripts/deployment/benchmark_inference.py` (eager / compile; TensorRT modes belong to
   E10) and compare its per-component numbers with yours.
2. `floor.csv`, E5 nsys + ncu, E6, E7 (MAXN) exactly as T008 steps 1–5.
3. E5b as T008 step 6 with S sweep S ∈ {1, 16, 41, 256}.
Acceptance: `reports/E3_E4_groot.md`, `reports/E5_groot_timeline.md`, `reports/E5b_groot.md`,
`reports/E6_groot_roofline.md`, `reports/E7_groot_power.md`; a GR00T row added to `reports/E5_E5b_models.md`.
Raw files committed under each result directory's `raw/` as in T004 / T007 (protocol §3.5 / §6); repository-size
lines in `STATUS.md` (§6f). Self-review (§11).

status: claimed 2026-09-24

status: done 2026-09-26

**Deliverables:** `reports/E3_E4_groot.md`, `reports/E5_groot_timeline.md`, `reports/E5b_groot.md`
(+ `E5b_groot_analysis.md`), `reports/E6_groot_roofline.md`, `reports/E7_groot_power.md`; GR00T rows in
`reports/E5_E5b_models.md`; repository-size lines in `STATUS.md`.

**Results:** E3/E4 N-sweep `results/20260925_E3E4_groot_n17_N{1,2,4,6,8,10}_{eager_tf32,vlm_compile}` —
per-step 51.75 → 18.13 ms, 1.05× the weight-streaming floor compiled. NVIDIA's own
`scripts/deployment/benchmark_inference.py`: `results/20260925_E3E4_groot_n17_benchmark_inference` (failed:
torchcodec missing, kept) and `…_torchcodec` (after installing torchcodec `--no-deps` from the repo's own Orin
wheel, `env/INSTALL_LOG.md`) — eager action head 204.42 ms vs ours 205.72 ms at N = 4 (0.6 %); the other
components differ by definition, analysed from the script in `E3_E4_groot.md` §4. E5
`results/20260924_E5_groot_n1{5,6,7}_{eager,best}`; E5b `results/20260925_E5b_groot_n17_{eager,best,ssweep}`
(S ∈ {1, 16, 41, 256}); E6 `results/20260925_E6_groot_n1{5,6,7}` and `results/20260925_E6_rungs/groot_*`; E7
`results/20260924_E7_groot_n17_*`.

**Deviations, stated:** `ae_step[i]` / `ae_layer[j]` events are not emitted inside the DiT loop
(`groot_phases.py` times `ae_total`), so per-step is the §2 slope; GR00T's byte floor is a lower bound
(cross-attended context not counted) and it has no FLOP floor (DiT not in the π0 formula); TensorRT modes
belong to E10. The GR00T E6 block in STATUS.md first compared mismatched rungs; corrected (6d27541) and the
matched-rung E6 is in `results/20260925_E6_rungs`.

bench commit `b2f58f79b27b`.
