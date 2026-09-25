# E6 — GR00T N1.7 on the Orin's own roofline

**Task T009 step 2. Binding: plan E6, protocol §8, §9, §2. Result:
`results/20260925_E6_groot_n17/roofline.csv`, with `floor.csv` and `floor_inputs.json` beside
it.**

## The answer, and it is the opposite of every other model measured

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
