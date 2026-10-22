# E0 smoke models (T001 step 8, A16) — 2026-10-22, MAXN — smoke tests, **not measurements**

| model | env | weights | P (valid) / S | actions | eager | compiled per-step callable |
|---|---|---|---|---|---|---|
| π0 | `~/venvs/lerobot` | 400/400 equal | 816 (533) / 51 | [1, 50, 7], finite | pass | pass, rel. diff 4.9e-3 |
| π0.5 | `~/venvs/lerobot` | 400/400 | 968 (574) / 50 | [1, 50, 7], finite | pass | pass, rel. diff 7.6e-8 (fp32 checkpoint) |
| SmolVLA | `~/venvs/lerobot` | 400/400 | 177 (151) / 50 | [1, 50, 6], finite | pass | pass, rel. diff 2.4e-3 |
| GR00T N1.7 | `~/venvs/groot17` | — | — | — | **blocked** | **blocked** |
| GR00T N1.6 | `~/venvs/groot16` | 400/400 | not read | 5 keys × [1, 16, ·], finite | pass | pass, rel. diff 3.1e-2 |
| GR00T N1.5 | `~/venvs/groot15` | 400/400 | not read | 3 keys × [16, ·], finite | pass | pass, rel. diff 1.2e-2 |
| LIBERO | `~/venvs/libero` | — | — | `libero_spatial` task 0 reset rendered (EGL, GPU), 256×256, mean 119.8 | pass | — |

Wall-clock times of the 5 calls are in `smoke.csv` and the per-model JSON for completeness only. The GPU was shared
with other users' sessions; none of them is a latency result.

- **N1.7 blocked:** its pinned backbone `nvidia/Cosmos-Reason2-2B` is gated (`gated: auto`, HTTP 401 without a
  token, `raw/groot_n17_backbone_gating_check.txt`). The environment itself works: with a public Qwen3-VL-2B stand-in
  config (not the pinned configuration), all 1031 GR00T parameters came from the checkpoint and actions were finite.
- **π0 attempt 1 loaded random weights.** lerobot `from_pretrained` drops the `revision` argument
  (`kwargs.get("revision")`, modeling_pi0.py:1023), looks up `main`, and offline falls back to random initialisation
  with only a printed warning. The weight check refused it. The harness now loads from the pinned snapshot directory.
- GR00T compiled-vs-eager differences of 1–3 % are not explained yet. T005 checks equivalence for every compiled
  configuration.
