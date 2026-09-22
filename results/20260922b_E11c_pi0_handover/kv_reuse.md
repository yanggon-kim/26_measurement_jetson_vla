# How the denoising steps use the prefix KV cache (pi0)

**Answer: every step deep-copies the whole cache before using it, so the prefix KV is
re-materialised 4 times per inference — once
per denoising step — not loaded once.**

## Evidence from the source

`PI0Pytorch.denoise_step`, `lerobot/policies/pi0/modeling_pi0.py:918`:

```python
past_key_values = copy.deepcopy(past_key_values)
```

It sits immediately before the expert forward, inside the per-step function, so it runs once
per denoising step. The prefix itself is computed once per inference
(`modeling_pi0.py:847-853`, `use_cache=True`).

## Evidence from the running process

`raw/kv_pointers.csv` records `data_ptr()` of layer 0's key tensor at two points in every
`denoise_step` of one real inference at N = 4: on **entry**, and at the expert
forward, i.e. **after** the deepcopy.

| step | on entry | after the deepcopy |
|---|---|---|
| 0 | 18093846528 | 8667475968 |
| 1 | 18093846528 | 8667475968 |
| 2 | 18093846528 | 8667475968 |
| 3 | 18093846528 | 8667475968 |

The **entry** address is stable across all 4
steps: `sample_actions` holds one cache object and passes the same one in every step. Read
alone, that would say the cache is referenced rather than copied — and it would be wrong.
The address **after** the deepcopy is a *different buffer from the original*, which is the
proof that a copy is made: the expert reads the copy, never the cache `sample_actions` holds.
That copy address is itself stable across the 4 steps, and that is allocator
reuse rather than reuse of the data — each step's copy is freed when the step ends, so PyTorch's
caching allocator hands the same 15.04 MB block back for the next one. The bytes
are re-written every step regardless, which is what 5.5824 ms and
`kv_copied_bytes_per_step = 15,040,512` measure.

## What it costs

| quantity | value |
|---|---|
| KV bytes per layer | 835,584 B |
| KV bytes total | 15,040,512 B (15.04 MB) |
| valid prefix tokens | 533 of 816 (283 padded but still stored) |
| `copy.deepcopy` per step | **5.5824 ms** |
| `kv_copied_bytes_per_step` | **15,040,512 B** (the whole cache) |

The analytic model assumes `kv_copied_bytes_per_step = 0` and `t_kv_materialise = 0`. Both are
measurably not zero: the copy moves the entire 15.04 MB object every step and
costs 5.5824 ms, which at N = 10 is 55.82 ms per inference of pure copying.

**283 of 816 prefix tokens are padding** (35 %) and are
stored and copied like any other. A mapping that moved only the valid tokens would carry
9,824,256 B instead.
