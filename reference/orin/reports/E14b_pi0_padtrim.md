# E14b — π0 prefix padding trim, decomposed: `pad` / `nocam` / `trimprompt` / `nopad` (T017)

π0 `lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc`, LeRobot @ `f0d2b37b`, batch 1, N = 10, Jetson AGX Orin 64 GB at
MAXN + `jetson_clocks` (GPU 1300.5 MHz, EMC 3199 MHz, 16 SMs), 2026-09-27. All four rows are labelled **variants**;
none is a baseline (whether a trim enters the GPU reference is the user's OD-1). ρ = variant / `pad`, same session.

## Pre-registered rows, answered

| # | quantity | measured | band |
|---|---|---|---|
| 1 | ρ_prefix(`nopad`) | **0.734** (180.1 / 245.4 ms) | **0.64–0.74: inside the ~164–183 ms bracket** (near its top) |
| 2 | ρ_vision(`nocam`); `trimprompt` vision | **0.665** (36.45 / 54.81 ms); `trimprompt` 0.999 | **0.62–0.72: equal per-image cost in compiled mode too** |
| 3 | ρ_prefix(`trimprompt`) | **1.028** (252.3 ms) — **slower** than `pad` | ≥ 0.99 (the "buys ~nothing" band) — and it costs time: +6.9 ms |
| 4 | ρ_prefix(`nocam`) − ρ_prefix(`nopad`) | **−0.010** (0.724 − 0.734): with the camera gone, trimming the prompt *adds* 2.3 ms. Interaction savings(nopad) − savings(nocam) − savings(trimprompt) = 0.266 − 0.276 + 0.028 = **+0.018** | ≤ 0.01 band (no gain), with the sign reversed |
| 5 | bytes ratio, prefix, `nopad` / `pad` | **0.792** reads + writes (17.10 / 21.60 GB); reads alone 0.826 (14.64 / 17.73) | **0.72–0.80: as case 003's padding row assumes (0.742)** — at its upper end |
| 6 | ρ_expert(`nopad`) | **0.653** (66.3 / 101.6 ms) | **< 0.97: attention over the prefix keys costs far more than its bytes** |
| 7 | serial frame, `nopad` | **248.5 ms** (pad 347.6 ms, ρ 0.715) | below the `[BRAINSTORM]` 260–282 ms (the expert saves more than row 6's band assumed) |
| 8 | K/V handover | **exact for all four**: 15 040 512 / 10 321 920 / 14 542 848 / 9 824 256 B (36 tensors [1, 1, T, 256], T = 816 / 560 / 789 / 533) | exactly the Step 2c bytes |
| 9 | action verdicts | fp32 gate **passes** for all three (worst REL 1.5 × 10⁻⁶); compiled bf16 **not bitwise** for any (0 / 64) and **PASS** for all three; real-camera embeddings under `nocam`: **bitwise equal to `pad`, 10 / 10** (5 frames × 2 cameras, one `embed_image` call per camera) | as the prior (fp32 pass, not bitwise, PASS) |

The research side's prior (rows 1 and 5 in their middle bands, row 3 in its first): all three hold; row 3's sign is new — the prompt trim alone is a small slowdown in the prefix and a large speed-up in the expert.

## 1. Time per stage (`results/20260927_E14b_pi0_padtrim_stages_MAXN`)

E12b `graphs_both` harness exactly as `results/20260925_E12_pi0_compiled_stages_N10_MAXN` (`bf16_both+hoist+
owngraph_expert+dbuf_kv+owngraph_prefix+memo_681+prefix_first`, whole GPU), one harness per variant built on that
variant's trimmed inputs; `split_vision` = two further graphs per variant — vision (every camera's `embed_image`,
projector included, one call per camera) and `vlm_prefix` (LeRobot's `embed_prefix` + Gemma prefill, with
`embed_image` answered from the vision graph's outputs). Fixture length only (5 E12 frames, 21 valid prompt tokens).
Bracket pad → nocam → trimprompt → nopad → pad → nopad → trimprompt → nocam → pad, 60 frames per stage per block.

| variant (tokens) | vision | vlm_prefix | vision + vlm (split) | **prefix** (one graph) | **expert** | **serial** |
|---|---|---|---|---|---|---|
| `pad` (816), n = 180 | 54.81 [54.77–55.01] | 190.22 [189.83–192.94] | 245.05 | **245.41** [245.00–246.85] | **101.60** [101.35–102.16] | **347.64** [347.01–348.91] |
| `nocam` (560), n = 120 | 36.45 [36.43–36.52] | 140.84 [140.60–141.05] | 177.30 | **177.77** [177.49–178.99] | **88.35** [88.29–88.62] | **269.37** [269.13–269.82] |
| `trimprompt` (789), n = 120 | 54.77 [54.74–55.23] | 196.94 [196.51–197.45] | 251.71 | **252.28** [251.80–254.44] | **69.81** [69.78–70.07] | **323.41** [322.55–325.07] |
| `nopad` (533), n = 120 | 36.62 [36.59–36.74] | 143.32 [143.02–144.90] | 179.93 | **180.12** [179.87–182.19] | **66.32** [66.27–66.52] | **248.48** [248.24–248.70] |

Median [p5–p95] ms. ρ: prefix 0.724 / 1.028 / 0.734, vision 0.665 / 0.999 / 0.668, vlm_prefix 0.740 / 1.035 / 0.753,
expert 0.870 / 0.687 / 0.653, serial 0.775 / 0.930 / 0.715 (nocam / trimprompt / nopad).

**Session:** `pad` prefix blocks 245.46 / 245.38 / 245.41 ms — drift **0.03 %** (no extra bracket); the reference's
246.79 ms is 0.6 % away, its expert 101.48 and serial 348.73 ms within 0.3 %. GPU 1300.5 MHz and EMC 3199 MHz at every
block boundary, never throttled, peak 61.6 °C, **0 non-finite**. The split sum is within 0.2–0.6 ms of the one-graph
prefix for every variant. Build (4 harnesses + split graphs) 65.9 s; 13 unique Dynamo graphs.

**What the two surprises look like (measured, not explained).** (a) Removing the 27 padded prompt tokens makes the
Gemma prefill *slower* (190.2 → 196.9 ms at 816 → 789 tokens; 140.8 → 143.3 ms at 560 → 533): 816 and 560 are
multiples of 16, 789 and 533 are odd, and the prefill's GEMMs run over M = sequence length — kernel choice at an
unaligned M is the obvious candidate, not tested. (b) The *expert* gains far more from the 27 prompt tokens than from
the 256 camera tokens: 101.6 → 69.8 ms (`trimprompt`) vs → 88.3 ms (`nocam`), 66.3 ms for both. The expert attends
to the prefix K/V at every one of its 10 steps; which property of the K/V (length alignment, the position of the
masked block — trailing prompt padding vs a masked block in the middle) selects the faster path is not isolated.
The bytes (§2) and a per-kernel look would separate them; neither was in T017's scope.

## 2. DRAM bytes per frame (`results/20260927_E14b_pi0_padtrim_bytes_MAXN`)

E12b graph-level method (`ncu --graph-profiling graph --cache-control none`, E12b `lts__` metrics), same harness and
graphs as §1, 5 frames per stage, run right after the stages session. Reads = L2 fills from memory; writes ≤ L2 write
sectors (upper bound); 32 B per sector.

| variant | prefix reads / writes ≤ (total) GB | vision (split) | vlm_prefix (split) | expert, 10 steps | prefix total ρ |
|---|---|---|---|---|---:|
| `pad` | 17.73 / 3.86 (**21.60**) | 5.74 / 0.84 | 12.05 / 3.03 | 7.03 / 1.26 | 1 |
| `nocam` | 14.78 / 2.52 (**17.30**) | 3.83 / 0.56 | 10.93 / 1.96 | 6.92 / 1.05 | 0.801 |
| `trimprompt` | 17.76 / 3.81 (**21.58**) | 5.75 / 0.84 | 12.12 / 2.98 | 7.01 / 1.12 | 0.999 |
| `nopad` | 14.64 / 2.46 (**17.10**) | 3.83 / 0.56 | 10.81 / 1.90 | 6.91 / 0.99 | **0.792** |

The `pad` prefix total reproduces E12b's 21.64 GB within 0.2 % (reads 17.73 vs 17.80, writes 3.86 vs 3.83), and
T016 step 3's `pad` / `nopad` prefix (17.65 / 14.64 GB reads) within 0.5 %. Vision bytes scale with the camera count
(0.667); the Gemma prefill's do not (`nocam` 0.91 of `pad` reads for 0.69 of the tokens) — its weights are streamed
once per frame whatever the length, the token-independent part row 5's upper band describes. The prompt trim moves
no prefix bytes (0.999) and ~2 % of the expert's — so the expert's 31 % time saving under `trimprompt` (§1) is not a
bytes effect. Profiled kernel time (ncu, serialised): prefix 251.7 / 183.1 / 259.3 / 182.6 ms, expert 104.8 / 91.6 /
72.9 / 69.5 ms — the same ordering as the event timing.

## 3. Action difference (`results/20260927_E14b_pi0_padtrim_equiv`)

**Input set, fixed before any run** (`bench/e14b_inputs.py`, committed with the 64 observations before the first
T017 run): HuggingFaceVLA/libero through the datasets-server rows API (same pipeline as `bench/e12_fixtures.py`),
8 tasks × 8 frames — task_index 0, 16, 10, 13, 14, 21, 3, 36, i.e. **8 distinct valid-prompt lengths (21, 6, 8, 9,
11, 12, 14, 17)** from all four suites; episodes spread over each task, the frame moving through the trajectory;
input 0 = row offset 0 = the committed fixture (pixels bit-identical). Noise: a `torch.Generator` per input, seed
17 000 + i, the same tensor for every variant. Action chunk `predict`-shape [1, 50, 32]; primary metrics on the 7 dims
the environment uses (all 32 also in `summary.json`).

**Step 1 — fp32 gate: PASS.** Shipped model cast to fp32 (`policy.float()`), eager, TF32 off, matmul "highest":

| variant | worst REL (64 inputs) | worst MAXABS |
|---|---:|---:|
| `nocam` | 1.5 × 10⁻⁶ | 6.0 × 10⁻⁶ |
| `trimprompt` | 2.3 × 10⁻⁷ | 1.0 × 10⁻⁶ |
| `nopad` | 1.1 × 10⁻⁶ | 5.5 × 10⁻⁶ |

**Failed attempt, reported:** the first gate ran the shipped model *as loaded*, taking "checkpoint dtype" for fp32.
This checkpoint stores **3.61 B of its parameters in bf16** (0.42 B fp32), so that pass measured bf16 arithmetic:
REL 1.8–2.5 × 10⁻³ on the 4-input smoke → FAIL (stopped before anything was timed; `scratch/failed_runs/`). The
gate was re-run in fp32 as the step's heading requires; the checkpoint-dtype pass is kept beside it on the full set
(`checkpoint_dtype_attempt` rows): worst REL `nocam` 2.9 × 10⁻³ (22 / 64 bitwise), `trimprompt` 4.8 × 10⁻²
(MAXABS 1.04 on one input), `nopad` 8.9 × 10⁻³ — bf16 sensitivity, not a trim defect (fp32 is ~10⁻⁶).

**Step 3 — compiled `bf16_both` rung (as E7), static shapes** (`automatic_dynamic_shapes` off; 18 first-call compiles —
one per variant and prompt length: ~11–18 s at the fixture length, ~87–94 s at each new length; 55 unique graphs):

| comparison | bitwise | MAXABS | REL (max) | median REL |
|---|---:|---:|---:|---:|
| control (i): `pad` vs `pad`, second pass, same process | **64 / 64** | 0 | 0 | 0 |
| floor F: shipped (eager, padded) vs `bf16_both` `pad` | 0 / 64 | **0.2961** | **0.02446** | 0.0034 |
| `nocam` vs `pad` | 0 / 64 | 0.0733 | 0.00562 | 0.0013 |
| `trimprompt` vs `pad` | 0 / 64 | 0.1250 | 0.00593 | 0.0012 |
| `nopad` vs `pad` | 0 / 64 | 0.0418 | 0.00340 | 0.0013 |

**Verdicts (pre-registered): `nocam` PASS, `trimprompt` PASS, `nopad` PASS** — none bitwise, each moves the actions
less than the rung itself does (MAXABS ≤ F_maxabs and REL ≤ F_rel). No LIBERO run was triggered (no FAIL). 0 non-finite.

## 4. Comparison with T016 step 3's `nopad` row

Same implementation (`bench/pi0_padtrim.py`, harness-level, LeRobot untouched). T016 timed it on the rung path
(`prefix_once` / `sample_actions`): prefix 253.0 → 189.6 ms (0.749), frame 420.5 → 319.5 ms (0.760). T017, in the
E12b `graphs_both` harness: prefix 245.4 → 180.1 ms (**0.734**), serial 347.6 → 248.5 ms (**0.715**). Same direction;
the graph harness removes launch overhead and saves a little more. T016's LIBERO pair (empty camera masked):
70 vs 66 / 100, p = 0.29 — consistent with T017's PASS.

## 5. Conditions, anomalies, not measured

- The `nocam` dummy: LeRobot's `_preprocess_images` still builds its −1 tensor for the missing camera; the variant
  drops it from the list before the image encoder (no SigLIP / projector call, no tokens, no mask entry). It is built
  outside every timed and profiled region.
- `kernel_bytes.csv` per variant (T016 1d's per-kernel tooling, optional in 2b) was not made: a prefix frame is
  ~1 150 kernels, i.e. 3 chunked ncu processes per frame per variant; the graph-level totals answer row 5.
- Not measured: why the prompt trim slows the prefill and speeds the expert (§1).

## Reproduce

```
python bench/e14b_inputs.py --meta <episodes/tasks parquet dir> --out bench/fixtures/e14b      # already committed
~/venvs/pi0/bin/python bench/e14b_padtrim.py --mode gate    --outdir results/<date>_E14b_pi0_padtrim_equiv
~/venvs/pi0/bin/python bench/e14b_padtrim.py --mode equiv   --outdir results/<date>_E14b_pi0_padtrim_equiv
~/venvs/pi0/bin/python bench/e14b_padtrim.py --mode verdict --outdir results/<date>_E14b_pi0_padtrim_equiv
~/venvs/pi0/bin/python bench/e14b_padtrim.py --mode time --frames 60 --outdir results/<date>_E14b_pi0_padtrim_stages_MAXN
~/venvs/pi0/bin/python bench/e14b_padtrim.py --mode ncu --frames 2          # plain warm-up first
bash bench/run_ncu_e14b.sh results/<date>_E14b_pi0_padtrim_bytes_MAXN 5
~/venvs/pi0/bin/python bench/e14b_padtrim.py --mode visioncheck --outdir results/<date>_E14b_pi0_padtrim_stages_MAXN
```
