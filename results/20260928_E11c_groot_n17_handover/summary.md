# E11c - GR00T N1.7 hand-over object (protocol §13.4)

Run directory: `20260928_E11c_groot_n17_handover`  
Written by `bench/groot_phases.py` at 2026-09-28T20:28:03Z.

## Answer

**Answer: the hand-over object is the backbone's hidden states (`backbone_features`, 273 tokens x
2048, bfloat16, 1,118,208 B), not a KV cache. It is passed by reference to every
denoising step, but the 16 cross-attention blocks project it through their own `to_k` / `to_v` at EVERY step:
26,836,992 B of K/V are re-computed per step (107,347,968 B per inference at N = 4),
costing 3.1775 ms per step (CUDA events, n = 200).**

## Hand-over object

| stage | tensor | shape | dtype | bytes | valid_tokens | padded_tokens | notes |
|---|---|---|---|---|---|---|---|
| backbone_output | backbone_features | (1, 273, 2048) | bfloat16 | 1118208 | 273 | 0 | returned by model.backbone (the hand-over object) |
| backbone_output | backbone_attention_mask | (1, 273) | bool | 273 | 273 | 0 | mask carried with the features |
| backbone_output | image_mask | (1, 273) | bool | 273 | 273 | 0 | mask carried with the features |
| after_encode_features | backbone_features | (1, 273, 2048) | bfloat16 | 1118208 | 273 | 0 | vlln + vl_self_attention applied once per inference (gr00t_n1d7.py:175-179, :307); this is what every DiT step reads as encoder_hidden_states |
| after_encode_features | state_features | (1, 1, 1536) | bfloat16 | 3072 |  |  | state_encoder once per inference (gr00t_n1d7.py:321); concatenated with the action tokens every step (:412) |

## Boundary tensors

| tensor | direction | shape | dtype | bytes | per | notes |
|---|---|---|---|---|---|---|
| backbone_features | gpu_to_action_head | (1, 273, 2048) | bfloat16 | 1118208 | inference | the hand-over object (backbone output) |
| noisy_actions | into_action_head | (1, 40, 132) | bfloat16 | 10560 | step | action_encoder input each step (gr00t_n1d7.py:405) |
| timestep | into_action_head | (1,) | int64 | 8 | step | discretised timestep tensor (gr00t_n1d7.py:401-404) |
| state_features | inside_action_head | (1, 1, 1536) | bfloat16 | 3072 | inference | state encoded ONCE per inference (gr00t_n1d7.py:321) and concatenated each step (:412) |
| dit_hidden_states | into_dit | (1, 41, 1536) | bfloat16 | 125952 | step | sa_embs = cat(state_features, action_features) |
| velocity | out_of_action_head | (1, 41, 132) | bfloat16 | 10824 | step | action_decoder output; the last action_horizon rows are the Euler update's velocity (gr00t_n1d7.py:430-435) |
| total | both |  |  | 21392 | step | noisy_actions + timestep + velocity (dit_hidden_states is inside the head) |

## Cross-attention K/V per step

| block | tokens | kv_bytes_written | median_ms | p5_ms | p95_ms | n |
|---|---|---|---|---|---|---|
| 0 | 273 | 1677312 | 0.2448 | 0.236712 | 0.261912 | 200 |
| 2 | 273 | 1677312 | 0.238576 | 0.234197 | 0.253614 | 200 |
| 4 | 273 | 1677312 | 0.237488 | 0.232275 | 0.248046 | 200 |
| 6 | 273 | 1677312 | 0.23648 | 0.232734 | 0.247928 | 200 |
| 8 | 273 | 1677312 | 0.236944 | 0.232822 | 0.253861 | 200 |
| 10 | 273 | 1677312 | 0.236192 | 0.230966 | 0.24965 | 200 |
| 12 | 273 | 1677312 | 0.236496 | 0.230942 | 0.250254 | 200 |
| 14 | 273 | 1677312 | 0.237136 | 0.23235 | 0.248381 | 200 |
| 16 | 273 | 1677312 | 0.244976 | 0.237306 | 0.267952 | 200 |
| 18 | 273 | 1677312 | 0.23808 | 0.233355 | 0.251715 | 200 |
| 20 | 273 | 1677312 | 0.240224 | 0.234901 | 0.264171 | 200 |
| 22 | 273 | 1677312 | 0.23936 | 0.233429 | 0.254149 | 200 |
| 24 | 273 | 1677312 | 0.243072 | 0.235072 | 0.268915 | 200 |
| 26 | 273 | 1677312 | 0.23864 | 0.232926 | 0.252478 | 200 |
| 28 | 273 | 1677312 | 0.239856 | 0.235584 | 0.25615 | 200 |
| 30 | 273 | 1677312 | 0.23856 | 0.233405 | 0.253314 | 200 |

## Conditions

GR00T N1.7 eager (bf16), MAXN + jetson_clocks, N = 4, P = 273 backbone tokens, S = 41. Transfers at these sizes: xfer_sizes.json -> bench/xfer_exact.py.

## Reproduce

```
python bench/groot_phases.py --version n17 --checkpoint nvidia/GR00T-N1.7-3B --rung eager --tf32 on --e11 handover --outdir results/20260928_E11c_groot_n17_handover
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
