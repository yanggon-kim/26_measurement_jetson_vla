# E11c — the VLM→expert hand-over object (protocol §13.4)

Run directory: `20260922b_E11c_pi0_handover`  
Written by `bench/handover.py` at 2026-09-22T13:24:49Z.

## Answer — the hand-over object

The object the VLM prefix hands to the action expert is the prefix KV cache.

| tensor | shape | dtype | bytes | valid | padded | notes |
|---|---|---|---|---|---|---|
| key+value | (18, 2, 816, 1, 256) | bfloat16 | 15040512 | 533 | 283 | 18 layers x 2 tensors; 835584 B per layer; cross-check 18*2*816*1*256*2 = 15040512 |

## Per-layer detail

| layer | tensor | shape | dtype | stride | contig | bytes | valid | padded |
|---|---|---|---|---|---|---|---|---|
| 0 | key | (1, 1, 816, 256) | bfloat16 | (208896, 208896, 256, 1) | True | 417792 | 533 | 283 |
| 0 | value | (1, 1, 816, 256) | bfloat16 | (208896, 208896, 256, 1) | True | 417792 | 533 | 283 |
| 1 | key | (1, 1, 816, 256) | bfloat16 | (208896, 208896, 256, 1) | True | 417792 | 533 | 283 |
| 1 | value | (1, 1, 816, 256) | bfloat16 | (208896, 208896, 256, 1) | True | 417792 | 533 | 283 |
| 2 | key | (1, 1, 816, 256) | bfloat16 | (208896, 208896, 256, 1) | True | 417792 | 533 | 283 |
| 2 | value | (1, 1, 816, 256) | bfloat16 | (208896, 208896, 256, 1) | True | 417792 | 533 | 283 |
| 3 | key | (1, 1, 816, 256) | bfloat16 | (208896, 208896, 256, 1) | True | 417792 | 533 | 283 |
| 3 | value | (1, 1, 816, 256) | bfloat16 | (208896, 208896, 256, 1) | True | 417792 | 533 | 283 |

(first rows; the full table is `handover.csv`)

## Boundary tensors

| tensor | dir | shape | dtype | bytes | per | notes |
|---|---|---|---|---|---|---|
| prefix_kv_cache | gpu_to_expert | (18, 2, 816, 1, 256) | bfloat16 | 15040512 | inference | the hand-over object itself |
| x_t_noisy_actions | into_expert | (1, 50, 32) | float32 | 6400 | step | current noisy action chunk |
| state | into_expert | (1, 32) | float32 | 128 | step | re-embedded every step by embed_suffix |
| timestep | into_expert | (1,) | float32 | 4 | step | scalar per step |
| v_t_velocity | out_of_expert | (1, 50, 32) | float32 | 6400 | step | the step's output, consumed by the Euler update |
| total | both |  |  | 12932 | step | sum of the per-step tensors; excludes the KV cache, which the model assumes is loaded once |

## Is it copied? — measured, not read off the source

`copy.deepcopy(past_key_values)` at `modeling_pi0.py:918` runs once per denoising step. The evidence and its cost are in `kv_reuse.md`; the timing is:

| what | median ms | p5 | p95 | n | bytes | GB/s |
|---|---|---|---|---|---|---|
| copy.deepcopy(past_key_values) | 5.5824 | 5.5445 | 5.6274 | 1000 | 15040512 |  |

## Conditions

π0, bf16, MAXN + jetson_clocks, P = 816 processed prefix positions of which 533 carry a valid mask, S = 51, N = 4. Shapes and strides are read from the live `past_key_values` of a real inference, not from the config.

## Reproduce

```
python bench/handover.py --outdir results/20260922b_E11c_pi0_handover
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
