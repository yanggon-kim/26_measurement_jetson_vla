# How the denoising steps use the backbone output (GR00T N1.7)

**Answer: the hand-over object is the backbone's hidden states (`backbone_features`, 273 tokens x
2048, bfloat16, 1,118,208 B), not a KV cache. It is passed by reference to every
denoising step, but the 16 cross-attention blocks project it through their own `to_k` / `to_v` at EVERY step:
26,836,992 B of K/V are re-computed per step (107,347,968 B per inference at N = 4),
costing 3.1775 ms per step (CUDA events, n = 200).**

## Evidence from the source
* `Gr00tN1d7.get_action` (gr00t_n1d7.py:603-612): backbone -> action_head.get_action(backbone_outputs, ...).
* `get_action_with_features` (gr00t_n1d7.py:397-435): the denoising loop passes the same `vl_embeds` as
  `encoder_hidden_states` to `self.model(...)` at every step.
* `AlternateVLDiT.forward` (dit.py:349-415): even blocks cross-attend (idx % 4 == 0 onto the non-image tokens, the
  other even blocks onto the image tokens), odd blocks self-attend.
* The attention processor projects the features inside every call: `/home/yanggon/venvs/groot/lib/python3.10/site-packages/diffusers/models/attention_processor.py:2747 (AttnProcessor2_0.__call__)`.

## Evidence from the running process (one real inference, N = 4, `raw/hook_log.json`)
| quantity | value |
|---|---|
| DiT forward calls | 4 |
| `to_k` calls per cross block | {"0": 4, "2": 4, "4": 4, "6": 4, "8": 4, "10": 4, "12": 4, "14": 4, "16": 4, "18": 4, "20": 4, "22": 4, "24": 4, "26": 4, "28": 4, "30": 4} |
| distinct `encoder_hidden_states` data_ptr across steps | 1 (the features are referenced, not copied) |
| K/V bytes written per step (all cross blocks) | 26,836,992 B |
| to_k/to_v time per step, all cross blocks | 3.1775 ms (gpu_event), 3.2846 ms (wall) |
| valid backbone tokens | 273 of 273 (256 image, 17 text) |

T016's `kvonce` variant (bench/groot_savings.py) computes these K/V once per frame instead and was measured in
`reports/E14_prefix_savings.md`.
