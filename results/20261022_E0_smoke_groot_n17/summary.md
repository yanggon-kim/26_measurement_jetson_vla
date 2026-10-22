# E0 smoke model GR00T N1.7 (T001 step 8, after H2) — 2026-10-22, MAXN — smoke test, **not a measurement**

| item | result |
|---|---|
| load | `nvidia/GR00T-N1.7-3B` @ 2fc962b9 with backbone `nvidia/Cosmos-Reason2-2B` @ 9ce19a19 (`main` on 2026-10-22; not pinned by the docs) |
| weights | 400/400 sampled tensors equal to the checkpoint; bf16; 4 denoising steps |
| input | `oxe_droid_relative_eef_relative_joint`, 2 cameras × 2 frames, 256×256 (Orin method, seed 1234) |
| tokens | P = 273 (LLM input = DiT `encoder_hidden_states`), S = 41: the Orin's DROID input (`docs/00_context.md`) |
| eager | actions `eef_9d` [1, 40, 9], `gripper_position` [1, 40, 1], `joint_position` [1, 40, 7], finite |
| compiled DiT | finite, relative difference vs eager 3.2e-3 at the same seed |

Wall-clock times are in `smoke_groot_n17.json` for completeness only (GPU shared); they are not latency results.
