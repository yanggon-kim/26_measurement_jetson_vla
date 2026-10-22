# E0 addenda (T001, after the independent review) — 2026-10-22, MAXN

Files for values that `env/README.md` cited without a saved source, plus two step items that were missing:

| item | result | file |
|---|---|---|
| GR00T N1.6 tokens (`robocasa_panda_omron`, 3 cameras × 1 frame) | P = 284 into the LLM and the DiT's `encoder_hidden_states`; S = 51 (DiT `hidden_states`) | `tokens_groot_n16.json` |
| GR00T N1.5 tokens (`oxe_droid`, 3 cameras × 1 frame) | P = 814; S = 49; vision input `[3, 3, 224, 224]` | `tokens_groot_n15.json` |
| checkpoint files | 116 files, 49,260,960,405 B, sha256 each, at the pinned revisions (+ SmolVLM2 @ 7b375e1b) | `checkpoint_files.csv` |
| GPU properties | L2 33,554,432 B; smem/SM 233,472 B; regs/SM 65,536; max threads/SM 1,536; 20 SMs; CC 11.0 | `raw/gpu_properties.json` |
| GPU frequency table | `gpu-gpc-0` available 54–1575 MHz in 9 MHz steps (nvpmodel minimum 314 MHz) | `raw/devfreq.txt` |
| JetPack | `nvidia-jetpack` not installed (candidate 7.0-b128) | `raw/apt_jetpack.txt` |
| `cuobjdump --list-text` on cuBLASLt | 5,668 `sm_110` + 2 `sm_110a` entries | `raw/cublaslt_list_text*` |

Token counts come from forward pre-hooks during one real `get_action`; nothing here is timed. N1.7: blocked (H2).
