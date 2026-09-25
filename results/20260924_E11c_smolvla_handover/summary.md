# E11c — the VLM→expert hand-over object (protocol §13.4)

Run directory: `20260924_E11c_smolvla_handover`  
Written by `bench/handover.py` at 2026-09-24T23:18:28Z.

## Answer — the hand-over object

The object the VLM prefix hands to the action expert is the prefix KV cache.

| tensor | shape | dtype | bytes | valid | padded | notes |
|---|---|---|---|---|---|---|
| key+value | (16, 2, 177, 177, 64) | bfloat16 | 3624960 | 151 | 26 | 16 layers x 2 tensors; 226560 B per layer; cross-check 16*2*177*177*64*2 = 128323584 |

## Per-layer detail

| layer | tensor | shape | dtype | stride | contig | bytes | valid | padded |
|---|---|---|---|---|---|---|---|---|
| 0 | key | (1, 177, 5, 64) | bfloat16 | (56640, 320, 64, 1) | True | 113280 | 151 | 26 |
| 0 | value | (1, 177, 5, 64) | bfloat16 | (56640, 320, 64, 1) | True | 113280 | 151 | 26 |
| 1 | key | (1, 177, 5, 64) | bfloat16 | (56640, 320, 64, 1) | True | 113280 | 151 | 26 |
| 1 | value | (1, 177, 5, 64) | bfloat16 | (56640, 320, 64, 1) | True | 113280 | 151 | 26 |
| 2 | key | (1, 177, 5, 64) | bfloat16 | (56640, 320, 64, 1) | True | 113280 | 151 | 26 |
| 2 | value | (1, 177, 5, 64) | bfloat16 | (56640, 320, 64, 1) | True | 113280 | 151 | 26 |
| 3 | key | (1, 177, 5, 64) | bfloat16 | (56640, 320, 64, 1) | True | 113280 | 151 | 26 |
| 3 | value | (1, 177, 5, 64) | bfloat16 | (56640, 320, 64, 1) | True | 113280 | 151 | 26 |

(first rows; the full table is `handover.csv`)

## Boundary tensors

| tensor | dir | shape | dtype | bytes | per | notes |
|---|---|---|---|---|---|---|
| prefix_kv_cache | gpu_to_expert | (16, 2, 177, 177, 64) | bfloat16 | 3624960 | inference | the hand-over object itself |
| x_t_noisy_actions | into_expert | (1, 50, 32) | float32 | 6400 | step | current noisy action chunk |
| state | into_expert | (1, 32) | float32 | 128 | step | re-embedded every step by embed_suffix |
| timestep | into_expert | (1,) | float32 | 4 | step | scalar per step |
| v_t_velocity | out_of_expert | (1, 50, 32) | float32 | 6400 | step | the step's output, consumed by the Euler update |
| total | both |  |  | 12932 | step | sum of the per-step tensors; excludes the KV cache, which the model assumes is loaded once |

## Is it copied? — measured, not read off the source

`copy.deepcopy(past_key_values)` at `modeling_pi0.py:918` runs once per denoising step. The evidence and its cost are in `kv_reuse.md`; the timing is:

| what | median ms | p5 | p95 | n | bytes | GB/s |
|---|---|---|---|---|---|---|
| copy.deepcopy(past_key_values) | 4.3689 | 4.3405 | 4.4011 | 1000 | 3624960 |  |

## Conditions

π0, bf16, MAXN + jetson_clocks, P = 816 processed prefix positions of which 533 carry a valid mask, S = 51, N = 4. Shapes and strides are read from the live `past_key_values` of a real inference, not from the config.

## Reproduce

```
python bench/handover.py --outdir results/20260924_E11c_smolvla_handover
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
