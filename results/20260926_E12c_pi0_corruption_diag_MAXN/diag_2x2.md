# E12c — T014's compiled co-run corruption, 2 × 2 (T016 step 5, EV-13)

`bench/e12c_diagnose.py`, one process per cell, 100 frames each (5 E12 fixtures × a seeded ±2/255 image
perturbation per frame, per-frame noise seeds 1000+k), whole GPU, MAXN, bf16_both, N = 10. Pipeline actions
compared with the SAME cell's serial run (bitwise); device-side finite check on every action; stale control =
the expert handed another frame's cache (must change the actions; 66 of 100 frames have a different source frame).

| pool \ KV | double-buffered (dbuf_kv) | single buffer, no wait |
|---|---|---|
| **own** (owngraph_expert, own CUDAGraph + pool) | 0 / 100 failures · max\|Δ\| 0 · 0 non-finite · stale 66/66 | **98 / 100 failures · max\|Δ\| 2.01** · 0 non-finite · stale 64/66 |
| **shared** (rung's CUDAGraphTrees denoise_step, pool shared with embed_image's tree) | 0 / 100 failures · max\|Δ\| 0 · 0 non-finite · stale 66/66 | **HANG** (twice): GPU at 99.9 %, host blocked in `CUDAGraph.replay()` inside the graph-tree `denoise_step`, in the concurrent pipeline pass (serial pass completed); stopped by the 30-min guard, stack in `raw/` |

**Reading.** The operative cause of wrong actions is the **single KV buffer** — the next prefix overwrites the
cache the expert is still reading: with its own pool, the single buffer corrupts 98 of 100 frames (finite but
wrong, max |Δ| 2.0), while the shared graph-tree pool with double buffering is bitwise clean. The shared pool is
not harmless in combination: with a single buffer and concurrent streams it **hangs** the GPU (reproduced in the
6-frame smoke and the 100-frame cell) rather than returning NaN. T014 saw NaN / wrong actions instead of a hang;
its pipeline passed caches by reference (fresh tensors) instead of copying into one buffer, which is the
difference from this harness's single-buffer cell — so this 2 × 2 isolates the mechanisms E12b removed, it is not
a replay of T014's exact code. E12b's fix (dbuf_kv) is necessary and sufficient for correctness here; its
own-pool change is not needed for correctness but avoids the hang if a single buffer were ever reintroduced.

Harness control: a 6-frame own/double smoke passed before the cells (bitwise, stale 4/4). First attempt of the
shared/single cell (6-frame smoke, 16:35) hung the same way; its stack is `shared_pool_single_kv/raw/pyspy_first_attempt_smoke_6frames.txt`.
