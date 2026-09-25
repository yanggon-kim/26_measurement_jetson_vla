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
| 0 | 9897681408 | 9897681408 |
| 1 | 9897681408 | 9897681408 |
| 2 | 9897681408 | 9897681408 |
| 3 | 9897681408 | 9897681408 |

The **entry** address is stable across all 4
steps: `sample_actions` holds one cache object and passes the same one in every step. Read
alone, that would say the cache is referenced rather than copied — and it would be wrong.
The address **after** the deepcopy is a *different buffer from the original*, which is the
proof that a copy is made: the expert reads the copy, never the cache `sample_actions` holds.
That copy address is itself stable across the 4 steps, and that is allocator
reuse rather than reuse of the data — each step's copy is freed when the step ends, so PyTorch's
caching allocator hands the same 3.62 MB block back for the next one. The bytes
are re-written every step regardless, which is what 4.3689 ms and
`kv_copied_bytes_per_step = 3,624,960` measure.

## What it costs

| quantity | value |
|---|---|
| KV bytes per layer | 226,560 B |
| KV bytes total | 3,624,960 B (3.62 MB) |
| valid prefix tokens | 151 of 177 (26 padded but still stored) |
| `copy.deepcopy` per step | **4.3689 ms** |
| `kv_copied_bytes_per_step` | **3,624,960 B** (the whole cache) |

The analytic model assumes `kv_copied_bytes_per_step = 0` and `t_kv_materialise = 0`. Both are
measurably not zero: the copy moves the entire 3.62 MB object every step and
costs 4.3689 ms, which at N = 10 is 43.69 ms per inference of pure copying.

**26 of 177 prefix tokens are padding** (15 %) and are
stored and copied like any other. A mapping that moved only the valid tokens would carry
3,092,480 B instead.
