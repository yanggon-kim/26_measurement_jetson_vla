# E6 — GR00T N1.7 on the Orin's own roofline

**Task T009 step 2. Binding: plan E6, protocol §8, §9, §2. Result:
`results/20260925_E6_groot_n17/roofline.csv`, with `floor.csv` and `floor_inputs.json` beside
it.**

> **⚠ CORRECTION 2026-09-25 — the first version of this report compared rungs that do not
> match.** GR00T's region times came from `20260923_E4d_groot_n17_ladder/phase_split.csv`,
> which is the `vlm_graphs` run with a **CUDA-graphed action head** (`--head-mode
> reduce-overhead`). The π0, π0.5 and SmolVLA rows it was set against are **eager**
> (`compile_mode: None`). "86 % of peak bandwidth, 1.16× floor" is true of the compiled head
> only, and was presented as a contrast with eager LeRobot models. Matched rungs, GR00T N1.7,
> from its own ladder rows (the weights and therefore the bytes are identical across these
> rungs; they change tf32 and compilation only):
>
> | GR00T N1.7 rung | `ae_step` ms | GB/s | % peak BW | gap × floor |
> |---|---:|---:|---:|---:|
> | `eager_tf32` | 52.13 | 62.2 | **33.2 %** | **3.02×** |
> | `eager_fp32` | 52.78 | 61.4 | 32.8 % | 3.05× |
> | `ae_compile` (head compiled only) | 20.82 | 155.7 | **83.0 %** | **1.20×** |
> | `ae_graphs` | 19.87 | 163.1 | 87.0 % | 1.15× |
> | `vlm_graphs` (the row first reported) | 20.06 | 161.5 | 86.2 % | 1.16× |
>
> **What survives, and it is a cleaner result than the original:** eager-to-eager, GR00T's
> action step is 33.2 % of peak bandwidth at 3.0× floor against π0's 6.45 % at 15.5× — still
> about five times closer to its roof, because it moves 5× the bytes per step through fewer,
> larger kernels. And **compiling the action head alone takes GR00T from 3.0× to 1.20× its
> floor**: once launch overhead is removed its DiT is bandwidth-bound. Whether π0's expert does
> the same when compiled is measured in the matched-rung E6 (`results/*_E6_*_rungs`), which
> replaces the mixed table.

## The answer as first written (see the correction above)

GR00T N1.7, N = 4, MAXN. Roof = this Orin's measured peaks (E1/T005: 187.47 GB/s,
33.85 TFLOP/s; spec alongside in the CSV).

| region | measured ms | bytes read | achieved GB/s | **% peak BW** | floor ms | **gap × floor** |
|---|---:|---:|---:|---:|---:|---:|
| `vlm_backbone` | 161.09 | 3 047 000 064 | 18.9 | 10.09 % | 16.25 | 9.91× |
| **`ae_step`** | **20.06** | **3 241 031 936** | **161.5** | **86.17 %** | **17.29** | **1.161×** |

**GR00T N1.7's action step is genuinely bandwidth-bound and within 16 % of the physical floor.**
It streams 3.24 GB of action-head weights per denoising step and sustains 161.5 GB/s doing it.
Protocol §9 reads < 1.5× as **near-optimal**.

### Cross-model, this changes which model the PIM argument fits

| model | `ae_step` ms | bytes/step | % peak BW | gap × floor | reading |
|---|---:|---:|---:|---:|---|
| π0 | 52.79 | 638 MB | 6.45 % | 15.51× | launch-bound |
| π0.5 | 62.69 | 1 735 MB | 14.76 % | 6.77× | launch-bound |
| SmolVLA | 60.80 | 203 MB | 1.78 % | 56.04× | launch-bound, extreme |
| **GR00T N1.7** | **20.06** | **3 241 MB** | **86.17 %** | **1.161×** | **bandwidth-bound** |

The three LeRobot models sit on the memory side of the ridge while using 1.8–14.8 % of the
bandwidth they already have — memory is their nearer roof, not their limit; they are limited by
kernel launch overhead (E5's 84.2 % GPU idle, E12 §5's finding that π0's expert runs the same on
4 SMs as on 16). **GR00T N1.7 is the case where the bandwidth story is literally true**: a
1.09-billion-parameter DiT whose weights must cross the memory bus every denoising step, already
running at 86 % of what the bus can deliver.

**This is the strongest measured support in the campaign for a memory-side accelerator — and it
points at a different model than the study started from.** For GR00T N1.7 there is no software
overhead left to remove: at 1.161× the floor, the only way to make the action step faster is to
move less data or move it faster, which is exactly what processing-in-memory offers. For π0 and
SmolVLA the same accelerator would target a ceiling they use a fraction of.

### Why the byte count can be trusted, which matters because the whole result rests on it

`ae_step` bytes are the action head's full parameter bytes, i.e. the assumption that every
denoising step reads all of them. That assumption is **validated by the gap ratio itself**: a
`gap_ratio` below 1.0 is physically impossible, because no execution can beat the floor computed
from the bytes it moves. Had the bytes been over-counted by even 20 %, the floor would have
exceeded the measured 20.06 ms and the ratio would have dropped below 1. At **1.161×** the count
is consistent and cannot be a large overestimate. (Had it come out at, say, 0.6, that would have
been a self-check failing, not a discovery.)

## Measured / derived / not measured

**Measured.** Region times from `results/20260923_E4d_groot_n17_ladder/phase_split.csv`
(protocol §2); the peaks (E1/T005).

**Derived.** Parameter bytes per region, summed from the **loaded** model
(`bench/floors_groot.py`; resolved module paths in `floor_inputs.json`):
`model.action_head` 3241.0 MB, `model.backbone` 3047.0 MB. `ae_step` time is
`ae_total / 4` — GR00T records only the total — which requires the explicit
`--ae-total-steps` flag and is an **average step, not a steady-state step**.

**FLOPs are NOT COUNTED, deliberately, and the compute columns are empty rather than 0.**
GR00T's action head is a DiT whose `transformer_blocks` **alternate** self-attention and
cross-attention (`bench/segments_groot.py`), so a step's arithmetic is self-attention over the
action tokens, cross-attention of those queries against the backbone's keys, and the block
MLPs — a different expression with a different token pairing in each term from π0's
linear-plus-self-attention formula. Applying π0's formula would have produced an
authoritative-looking wrong number. A 0 in a compute column reads downstream as "does no
arithmetic", which is the error that kept π0's `vision` out of the roofline until
`bench/vision_flops.py` was written, so those cells are blank instead.

**Also not measured.**

* **`vision` as a separate region.** `floors_groot.py` walks the backbone for a submodule named
  `vision_model`/`vision_tower` and found none — Cosmos-Reason2-2B names it otherwise — so
  vision has no byte count here rather than a wrong one. GR00T's own phase split does time it
  (71.39 ms), and `vlm_backbone` includes its weights.
* **The DiT's cross-attention context bytes**, which depend on the backbone's prefix length.
  `ae_step` bytes are therefore a **lower bound**; the true achieved bandwidth is at or above
  161.5 GB/s, which given the 187.47 GB/s roof leaves little headroom and is itself further
  evidence the step is bus-limited.
* **N1.5 and N1.6.** Only N1.7 is loadable in this environment; the other two live on separate
  `Isaac-GR00T` branches with their own venvs (`model_registry.py` records `installed: True`
  for N1.7 alone), so measuring them means switching branches. Outstanding for T009.
* **A compiled-rung roofline**, and `source = ncu` off-chip-byte rows, which exist only for π0
  (`20260922_E4g_pi0_ncu_eager` is the single capture carrying the L2 sysmem-miss counters).

## GR00T N1.5 and N1.6 (added 2026-09-25, `results/20260925_E6_groot_n1{5,6}`, `results/20260925_E6_rungs/groot_n1{5,6}_*`)

Same method (bytes from the loaded model via `bench/floors_groot.py`, through each version's own
worktree and the `groot16` venv; unprofiled ladder times; FLOPs deliberately not counted). The
compiled rung is each version's best, `all_compile_autotune`.

| `ae_step` | bytes / step | eager: ms · % peak BW · gap | compiled: ms · % peak BW · gap |
|---|---:|---|---|
| GR00T N1.5 | 2 137.6 MB | 24.02 · 47.5 % · 2.11× | 15.14 · **75.3 %** · **1.33×** |
| GR00T N1.6 | 2 837.2 MB | 50.65 · 29.9 % · 3.35× | 18.21 · **83.1 %** · **1.20×** |
| GR00T N1.7 | 3 241.0 MB | 52.13 · 33.2 % · 3.02× | 19.89–20.82 · **83–87 %** · **1.15–1.20×** |

**The finding holds across all three GR00T versions:** once compiled, the action step runs at 75–87 %
of the measured peak bandwidth within 1.15–1.33× of its memory floor, while π0, π0.5 and SmolVLA stay
3–5× off theirs compiled (`reports/E5_E5b_models.md`).
