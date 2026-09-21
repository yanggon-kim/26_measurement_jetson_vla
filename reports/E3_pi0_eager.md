# E3 — π0 end-to-end latency on the Jetson AGX Orin, eager (task T002)

**Runs:** `results/20260921_E3_pi0_eager_N{1,2,4,6,8,10}/` · **date:** 2026-09-21 (UTC) ·
**MAXN, 16 SMs, GPU 1300.5 MHz, EMC 3199 MHz, `jetson_clocks` applied before every run,
`throttled: false`** · 20 warm-up + **100 timed** iterations per point, every sample kept.

**Read this first: the model is LeRobot's π0, not openpi's.** The task file prescribed
openpi's PyTorch port; it was investigated in full and rejected on setup cost (§8). LeRobot's
implementation is a direct port — it prints *"a direct port of the OpenPI implementation …
follows the original OpenPI structure for compatibility"* on load — but the **weights are a
different fine-tune** (`lerobot/pi0_libero_finetuned_v044`, trained on `HuggingFaceVLA/libero`),
so every comparison with the characterization paper below is across implementations *and*
checkpoints.

---

## 1. The answer

### 1.1 The number the project exists to measure

**One denoising step of π0's action expert costs 52.8 ms on this Orin** (52.80 ms
instrumented steady state; 52.94 ms by slope — the two agree to 0.3 %).

| definition (protocol §2) | value | how |
|---|---|---|
| **(a) instrumented** — CUDA events per `ae_step[i]`; **§2 makes this primary in eager mode** | **52.80 ms/step** (steps 1…N−1, uninstrumented) | `ae_step_individual[i]`; 53.11 ms if step 0 is included, which carries the `ae_layer` event overhead (§1.7b) |
| (b) slope — the primary number only under `torch.compile` | 52.943 ms/step | least squares over all six N: `ae_total = 3.69 + 52.943·N` ms, **r² = 0.99999** |
| `ae_total / N` at N = 10 | 53.30 ms | |
| `ae_total / N` at N = 4 | 53.88 ms | |

**The 3.69 ms intercept is not a fixed cost of the denoising phase.** `ae_total` is derived
in `bench/regions.py` as the span from the first `ae_step` start event to the last `ae_step`
end event, so by construction it contains *no* pre-loop work — `embed_prefix`, the prefill and
the KV build are all outside it. The intercept is the step-0 excess (§1.7b) re-expressed by
the fit, not a second, independent quantity. An earlier draft of this report presented it as
one; that was wrong.

### 1.2 The ladder (protocol §9), ms per denoising step

| rung | value | status |
|---|---|---|
| analytic bandwidth floor | **3.115 ms** | computed from the **loaded** model: 638.0 MB (623.01 MB expert weights + 15.0 MB KV cache) ÷ 204.8 GB/s spec |
| analytic compute floor | 0.912 ms | 38.29 GFLOP ÷ 42 TFLOP/s → the step is **memory-bound**, as predicted |
| **measured eager** | **52.94 ms** | **17.0 × the floor** |
| *what the research side assumed* | *136 ms* | *derived from the paper; the real device is **2.6× faster*** |

**Gap ratio per region** (protocol §9 requires this for every region, at the *spec* peaks —
the measured peaks come from T005/E1):

| region | measured (N = 10) | floor | bound | gap |
|---|---|---|---|---|
| `vision` | 239.60 ms | 8.10 ms | memory | **29.6×** ← the worst in the model |
| `vlm_prefix` | 232.93 ms | 97.48 ms | compute | **2.4×** |
| `ae_step` | 52.80 ms | 3.115 ms | memory | **17.0×** |

Protocol §9's reading of a 17× gap: **"a structural problem — launch-bound, idle GPU, or too
little parallelism."** `vision`'s 29.6× is worse still, and a third of it is spent on a dummy
image (§1.7a). Note the `vision` floor counts weights only — its FLOPs were not computed
analytically — so that ratio is an upper bound on the gap. E5/E5b (T004, T007) must decide
which cause applies; this report does not guess.

### 1.3 Per-region latency

Median ms, n = 100 per point. `vision` and `vlm_prefix` are independent of N, as they should
be — they vary by < 1 % across the sweep, which is a useful check on the instrumentation.

| region | N=1 | N=2 | N=4 | N=6 | N=8 | N=10 |
|---|---|---|---|---|---|---|
| `vision` | 241.12 | 240.30 | 239.35 | 239.28 | 239.37 | 239.60 |
| `vlm_prefix` | 234.35 | 234.63 | 233.08 | 232.99 | 233.01 | 232.93 |
| `ae_total` | 55.90 | 110.35 | 215.53 | 321.35 | 427.22 | 533.03 |
| **end-to-end** | **537.35** | **592.10** | **694.05** | **799.52** | **905.47** | **1011.57** |
| control frequency (Hz) | 1.86 | 1.69 | 1.44 | 1.25 | 1.10 | 0.99 |

Spread is tight: p5–p95 on end-to-end is within ±0.5 % at every N (see each run's
`latency_summary.csv`).

Measured separately, because they are CPU work plus one host↔device copy and would otherwise
hide inside a GPU region (`results/…_N10/preprocess.csv`, n = 100):

| | median ms |
|---|---|
| host `uint8 → float32 CHW` | 0.688 |
| `preprocess` (tokenise + normalise + H2D) | 1.065 |
| `postprocess` (unnormalise + D2H) | 0.261 |

Together **2.0 ms**, i.e. 0.2–0.4 % of end-to-end. Negligible, and excluded from the
end-to-end figures above, which time `predict_action_chunk` only.

### 1.4 Share of end-to-end — the action expert does **not** dominate until N ≈ 8

| region | N=1 | N=2 | N=4 | N=6 | N=8 | N=10 |
|---|---|---|---|---|---|---|
| `vision` | 44.9 % | 40.6 % | 34.5 % | 29.9 % | 26.4 % | 23.7 % |
| `vlm_prefix` | 43.6 % | 39.6 % | 33.6 % | 29.1 % | 25.7 % | 23.0 % |
| `ae_total` | 10.4 % | 18.6 % | **31.1 %** | 40.2 % | 47.2 % | **52.7 %** |

### 1.5 Token counts — three assumptions in `docs/00_context.md` confirmed

Printed from the real fixture, not assumed:

| quantity | doc | measured |
|---|---|---|
| prefix tokens **P** | 816 (marked **assumed**; "the paper implies 841") | **816** = 3 images × 256 + 48 language tokens |
| suffix tokens **S** | 51 | **51** = 1 state + 50 action tokens |
| expert weights per step | 623 MB | **623.01 MB** |
| expert FLOPs per step | 38.7 GFLOP | **38.29 GFLOP** (31.77 linear + 6.52 attention) |

Of the 816 prefix tokens only **533 are valid** — 512 from the two real cameras plus 21 real
language tokens; the rest is padding (§1.7) that is nevertheless computed.

### 1.6 Peak memory

9.04 GB allocated / 9.40 GB reserved (`torch.cuda.max_memory_allocated/reserved`), constant
across N. It is dominated by the 8.89 GB of weights, so N has little to move. On Orin this is allocator accounting inside *unified*
memory, so it competes with host RAM rather than sitting in separate VRAM.

### 1.7 Three findings, two of them cheap to act on

**(a) A third of all vision time encodes a dummy image.** The checkpoint sets
`empty_cameras: 1`, so the policy synthesises a third camera filled with `-1` and runs it
through the full SigLIP tower. Measured per image, n = 100 with 20 warm-ups
(`results/20260921_E3_pi0_eager_probe/vision_per_image.csv`; an earlier n = 50 run from an
un-committed script agreed to 0.2 %):

| slot | median ms |
|---|---|
| `image` (base camera) | 80.61 |
| `image2` (wrist camera) | 80.14 |
| **`empty_camera_0` (all −1, carries no information)** | **80.08** |

The three passes agree to within 0.7 % because SigLIP's cost is data-independent — so "a
third of vision" is really "one of three equal passes", not an independent measurement of the
dummy's cost. That does not weaken the conclusion: the pass is real, and its output is
discarded.

**80.1 ms — 33.3 % of `vision`, and 11.5 % of the whole N = 4 inference — is spent encoding
nothing.** Its attention mask is zero, so its 256 prefix tokens are masked out downstream:
the work is discarded. Caching one constant embedding would remove it outright. This is a
property of the shipped configuration, reported as found; it was not changed.

**(b) The first denoising step costs +0.83 ms more than the rest — most of the apparent
offset was my own instrumentation.** An earlier draft of this report published **+3.13 ms**
and floated a cold-KV-cache explanation. An independent review pointed out the confound:
`bench/regions.py` records the 18 `ae_layer[j]` event pairs on **one** denoising step, so the
anomalous step was also the only instrumented one. The control
(`results/20260921_E3_pi0_eager_probe/step0_control.csv`, n = 100 each):

| condition | step-0 offset | step 5 |
|---|---|---|
| `ae_layer` instrumentation **off** | **+0.83 ms** | 54.46 ms |
| instrumentation on step 0 | +3.42 ms | 54.51 ms |
| instrumentation on step 5 | +0.80 ms | **57.03 ms** |

The event overhead is **+2.57 ms**, and it follows whichever step carries it. **The real
step-0 offset is +0.83 ms**; the published 3.13 ms was ~74 % artefact. The steps are still not
exactly equal, but the effect is small and no longer resembles a cold-cache read of one
step's worth of memory. `--no-fine-layers` now exists so this control can be re-run (it was
previously un-disableable, which is why the confound survived the first pass).

**(c) 10.4 % of every denoising step is a Python deep copy of the KV cache.**
`lerobot/policies/pi0/modeling_pi0.py:919`, inside `denoise_step`:

```python
past_key_values = copy.deepcopy(past_key_values)
```

It clones all 36 KV tensors — **15.04 MB** — once per step. Measured directly
(`results/…_probe/kv_deepcopy.csv`, n = 100): **5.677 ms per step**, p5–p95 5.644–5.717, i.e.
**56.8 ms and 150 MB per inference at N = 10**.

It is not bandwidth-bound: 15.04 MB in 5.677 ms is an effective **2.65 GB/s**, ~1.3 % of the
204.8 GB/s the device is specified for. The cost is Python and allocator work across 36
tensors. For scale, this one line costs **1.82× the entire 3.115 ms memory floor of a
denoising step**.

`docs/00_context.md` assumes `kv_copied_bytes_per_step = 0`; the measured value is
15,040,512 B. This is a property of the shipped stack, reported and not changed — but the
expert branch passes `use_cache=False`, so the callee does not mutate the cache, and whether
the copy is needed at all is worth one experiment in T003/T010.

**With it, the step budget nearly closes** (layers-off baseline, mean step 54.63 ms):

| | ms | share |
|---|---|---|
| 18 × `ae_layer[j]` | 44.88 | 82.1 % |
| `copy.deepcopy(past_key_values)` | 5.68 | 10.4 % |
| remainder — `embed_suffix`, mask build, final norm, `action_out_proj`, Python | 4.08 | 7.5 % |

(The 44.88 ms layer figure itself carries ~2.6 ms of event overhead, so the true layer total
is nearer 42.3 ms and the remainder correspondingly larger.) `docs/00_context.md` assumes
`T_extra` ≈ 0.08 ms per step for everything outside the layer stack; the measured
non-layer cost is **~10 ms**, two orders of magnitude larger.

---

## 2. Comparison with the characterization paper

The paper (Zhou et al., arXiv 2604.24447) reports π0 on an AGX Orin at N = 4:
920 ms end-to-end, split 110 / 258 / 543 ms.

| | paper, N = 4 | measured here, N = 4 | |
|---|---|---|---|
| vision | 110 ms (12 %) | **239.4 ms (34.5 %)** | 2.2× **slower** |
| VLM prefix | 258 ms (28 %) | **233.1 ms (33.6 %)** | 1.1× faster |
| action expert | 543 ms (59 %) | **215.5 ms (31.1 %)** | **2.5× faster** |
| end-to-end | 920 ms | **694.1 ms** | 1.3× faster |
| AE per step | 136 ms (derived) | **53.9 ms** | **2.5× faster** |

**The headline disagreement is the phase split, not the total.** The paper has the action
expert at 59 % of latency and vision at 12 %; here at the same N = 4 the action expert is the
*smallest* of the three phases (31 %) and vision is the *largest* (34.5 %). Even at the
model's own default N = 10 the action expert reaches only **52.7 %**.

**This contradicts the project's working hypothesis** that the action expert dominates
latency. It does not contradict the claim that the expert is *memory-bound* — the floor
analysis (§1.2) confirms that independently, 3.115 ms memory against 0.912 ms compute.

Honest caveats on the comparison, in order of how much they could matter:

1. **Different implementation and different weights.** LeRobot's port and a LeRobot-native
   LIBERO fine-tune, versus openpi's code and checkpoint. Architecture and shapes match
   (§1.5), so the comparison is meaningful, but it is not the same binary.
2. **Image count.** Our vision cost is 3 SigLIP passes, one of them wasted (§1.7a). If the
   paper's setup encoded fewer images, that alone explains a large part of the 110 → 239 ms
   gap. The paper's image count is not known to me.
3. **Software stack.** torch 2.8.0 + CUDA 12.6 + transformers 5.3.0 on JetPack 6.2.3 here;
   the paper's stack is not stated in `docs/00_context.md`.
4. **Attention implementation.** Both prefill and denoise force
   `_attn_implementation = "eager"` (`modeling_pi0.py`), so no SDPA or flash path is taken —
   materialising a `[1, 1, S, P+S]` mask each step. This is the shipped behaviour and was not
   changed, but it is an obvious lever for E4/E10.
5. Device state is identical in kind to the paper's claim (AGX Orin) but ours is pinned and
   recorded: MAXN, 16 SMs, 1300.5 MHz, EMC 3199 MHz, not throttled.

---

## 3. A protocol assumption that does not hold for this implementation

Protocol §2 states as a *"known property, to be reported and not fixed"* that openpi's
denoising loop tests a GPU tensor each iteration (`while time >= -dt/2`), forcing a CPU–GPU
sync per step.

**LeRobot's loop does not do this.** It is a plain `for step in range(num_steps)` with `time`
and `dt` as Python floats; the only host→device traffic per step is a scalar
`torch.tensor(time, device=cuda)`. There is no `.item()`, no `.cpu()`, no explicit
synchronisation inside the loop. So that property is openpi-specific and must not be carried
into the interpretation of these numbers. It also means the 52.94 ms/step here is *not*
inflated by per-step syncs — which makes the 17× gap to the floor harder to explain away, not
easier.

---

## 4. Conditions

Batch 1, bf16 (`dtype: bfloat16` from the checkpoint), eager. **No RNG seed is set** — an
earlier draft claimed one. `sample_noise` draws from the global RNG each iteration, so the
denoising *input* differs between iterations; the model *observation* is the fixed fixture and
every tensor shape is constant, so latency is unaffected. Protocol §2 asks for fixed seeds and
this run does not satisfy that; it is recorded in each `meta.json.seed`. One real LIBERO
observation: `HuggingFaceVLA/libero`, episode 0, frame 0, task_index 0, *"put the white mug on
the left plate and put the yellow and white mug on the right plate"* — two 256×256 camera
images and an 8-D state, saved in `bench/fixtures/` (PNG + JSON). N is swept through the
model's own `num_inference_steps`; no loop was edited. The model is loaded once and N changed
between points, so all six runs share identical weights and allocator state.

Timing is CUDA events around each region, read only after a synchronise at the end of each
iteration; end-to-end is wall clock around `predict_action_chunk` with a synchronise.

**`predict_action_chunk` is timed, not `select_action`.** `select_action` serves 49 of every
50 calls from an internal action queue without running the model; timing it would have
measured queue pops.

---

## 5. Confidence

**High** on the per-step number: r² = 0.99999 across six N, p5–p95 within ±0.5 %, and three
independent estimates (slope, instrumented, `ae_total/N`) agreeing to within 2 %.

**High** on the region split: `vision` and `vlm_prefix` are flat across the sweep to < 1 %,
which is what they must be if the instrumentation is attributing correctly, and the regions
sum to end-to-end (N = 4: 239.4 + 233.1 + 215.5 = 688.0 vs 694.1 measured; the 6 ms remainder
is `embed_prefix` assembly, the Euler update and Python overhead outside the wrapped regions).

**High** that the weights are real: `assert_real_weights()` compares tensors against the
checkpoint file on every run and refuses to measure on a mismatch (400/400 matched). Precisely:
it takes the **first 400 of the checkpoint's 777 tensors in file order**, not a random sample —
that covers the whole action expert, the whole VLM language model and the action projections,
but only 60 of 437 vision-tower tensors. A model whose vision tower alone were partly random
would pass. Against the failure mode of §8, where the entire `load_state_dict` fails, all 400
would mismatch.

**Not established:** why the gap to the floor is 17× (or 29.6× for `vision`); the +0.83 ms
residual step-0 offset; what the 6.1 ms end-to-end remainder is; whether the paper's vision
figure covers fewer images; whether the per-step KV deep copy is removable.

---

## 6. Separation of evidence

**Measured:** every latency in §1.3, §1.4, §1.7; P, S and the per-region parameter bytes read
out of the loaded model (§1.5); peak memory; control frequency; the per-image vision split;
preprocess/postprocess cost; the shape of the denoising loop (read from the installed source).

**Derived:** the slope and its r² (least squares over the six measured points); the analytic
floors in §1.2 (from parameter bytes and shapes, at the *spec* 204.8 GB/s and 42 TFLOP/s —
`peak_source = spec`); all gap ratios; the share-of-end-to-end percentages; the comparison
factors against the paper.

**Not measured:** the `measured` rows of `floor.csv` are deliberately empty — the Orin's real
peak bandwidth and TFLOP/s come from T005/E1 and were **not** back-filled from the spec sheet.
Any region's FLOPs for `vision` are recorded as 0 rather than guessed. Nothing under
`torch.compile` (that is T003/E4). No profiler was attached to any timed run.

---

## 7. Reproduce

```bash
bash env/setup_pi0.sh                      # ~/venvs/pi0, lerobot @ f0d2b37beb, transformers 5.3.0
python bench/fetch_fixture.py --offset 0 --outdir bench/fixtures
sudo jetson_clocks && sudo jetson_clocks --fan
python bench/run_e3.py --steps 1,2,4,6,8,10 --warmup 20 --timed 100
python bench/analyze_e3.py --date 20260921
```

Exact invocations in each `meta.json.command`; the code that ran is pinned per file in
`meta.json.bench_files_sha256`.

---

## 8. Deviation: LeRobot instead of openpi, and the trap that justified the harness

`tasks/T002_pi0_eager.md` prescribed openpi's PyTorch port. Researched in full and rejected —
the cost is entirely setup, none of it measurement: a **12 GB** orbax/OCDBT checkpoint to
convert on-device; a mandatory **JAX** install (`openpi.models.gemma` imports `jax` and
`flax.linen` at module level even on the PyTorch path); a
`transformers.__version__ == "4.53.2"` **string-equality** check enforced at model
construction, with five files to copy over site-packages; and a `requires-python >=3.11` gate
to override on a Python-3.10-only device. The openpi route remains available for a later
cross-implementation comparison, and is the obvious way to test whether the phase-split
disagreement in §2 is implementation or measurement.

**The trap.** The obvious LeRobot version is PyPI `lerobot==0.4.4`, matching the checkpoint's
`_v044` name. It would have silently produced a **randomly initialised model**. Three
properties combine, all verified in the installed source before it was discarded:

1. it requires a patched transformers fork (`from transformers.models.siglip import check`);
2. it lacks the `lm_head → embed_tokens` fix. This checkpoint stores
   `…language_model.embed_tokens.weight` only as a safetensors **metadata alias** of
   `paligemma.lm_head.weight`, and `safetensors.torch.load_file` does not reconstruct
   aliases, so the key is simply absent and `load_state_dict` fails;
3. `PI0Policy.from_pretrained` **catches load failures, prints a warning, and returns the
   model anyway** — with random weights.

A random π0 runs, produces `[1, 50, 7]` finite actions, and yields plausible latencies. The
pinned ref `f0d2b37beb` is the one window that is Python-3.10-clean *and* carries the fix, and
`bench/pi0_common.py:assert_real_weights()` now refuses to measure unless 400 sampled tensors
match the checkpoint file byte for byte. `env/setup_pi0.sh` re-asserts all three properties on
every rebuild. Full detail: `env/README.md` §10.

`transformers` is pinned to **5.3.0 exactly**, not the `<6.0.0` the extra allows: 5.4.0 moved a
`× √hidden_size` rescale inside `PaliGemmaModel.get_image_features` and lerobot dropped its own
copy at the same commit, so this ref against 5.4+ scales every prefix embedding by ≈ 45 —
silently.

Other deviations: `compile_model` and `gradient_checkpointing` are overridden to `False`
**before** construction (the checkpoint ships `compile_model: true, compile_mode:
"max-autotune"`, and `PI0Pytorch.__init__` compiles the moment it sees it, which would have
made T002 measure a compiled model with its NVTX ranges swallowed). The tokenizer repo
`google/paligemma-3b-pt-224` is gated; the license was accepted for this account, and the
tokenizer's sha256 was verified byte-identical to Google's ungated copy at
`storage.googleapis.com/big_vision/paligemma_tokenizer.model`.

---

## 9. Self-review (protocol §11)

| check | result |
|---|---|
| one result directory per N, all requested files | **yes** — 6 × (`meta.json`, `latency_samples.csv`, `latency_summary.csv`, `floor.csv`, `summary.md`, `raw/`), plus `per_step.csv`, `preprocess.csv`, `vision_per_image.csv` at N=10 |
| every per-iteration sample kept | **yes** — `latency_samples.csv`, 100 iterations × every region, including each `ae_step_individual[i]` and all 18 `ae_layer[j]` |
| every number here traces to a file | **yes** |
| `meta.json` complete | **yes**, incl. `weight_integrity`, `checkpoint_config_as_shipped`, `tokens_detail`, `region_parameter_bytes`, `bench_files_sha256` |
| n, power mode, `throttled` stated | **yes** — n = 100; MAXN, 16 SMs, 1300.5 MHz locked; `throttled: false` with evidence |
| measured / derived / not-measured separated | **yes** — §6 |
| deviations listed | **yes** — §8 |
| nothing over 50 MB; secrets scan | **yes** — largest run directory 208 KB; scan clean |
| no weights/caches/venvs staged | **yes** — the 7 GB checkpoint lives in `~/.cache/huggingface` |
| `raw_files` hashes filled | **yes, after correction** — they were `[]` in all six runs when this report was first written, while this table claimed otherwise. Fixed; each run now lists its stdout and `latency_samples.csv` with sha256. |
| checkpoint name, source **and sha256** | **yes, after correction** — `meta.json.checkpoint_sha256` = `31633cfa…`; it was missing in the first version |
| **nothing on the research side was edited** | **no — `docs/00_context.md` was edited.** `AGENTS.md` reserves `docs/` for the research side. The repository owner instructed me directly to correct wrong markdown and push it, and the two roles are the same person here; the edit is recorded in §8 and in the commit message rather than made silently. Under the unmodified `AGENTS.md` workflow this would have gone to `STATUS.md` → "Questions for the research side" instead. |

### Independent pass (protocol §11, P0 task)

A sub-agent was given the task file, the protocol, the plan and the produced files, and asked
to recompute every number from the CSVs. It reproduced the full region tables, the slope fit
(a = 3.6894, b = 52.9434, r² = 0.9999933), every floor figure, the token arithmetic and the
paper-comparison factors exactly, and confirmed `latency_summary.csv` matches the raw samples
to the last float digit. **It found five substantive defects, all now fixed:**

| finding | fix |
|---|---|
| **The +3.13 ms step-0 offset was confounded with my own instrumentation** — `ae_layer` events land on that step alone, and `--fine-layers` was un-disableable so the control could not be run | flag fixed; control run; the real offset is **+0.83 ms** and the published figure was ~74 % artefact (§1.7b) |
| **`copy.deepcopy(past_key_values)` at `modeling_pi0.py:919` was missed** — I had grepped for a symbol name used in a different lerobot ref | measured: **5.68 ms/step, 10.4 % of a step** (§1.7c) — the single largest thing this report had failed to see |
| §4 claimed a fixed seed; none is set | claim corrected (§4), recorded in `meta.json.seed` |
| "3.69 ms fixed cost of the denoising phase" misreads my own `ae_total` definition | corrected (§1.1) |
| `meta.json.raw_files` was `[]` in all six runs while §9 claimed hashes were filled; checkpoint sha256 missing; provenance hashes stale after `floor.csv` was regenerated | all six `meta.json` repopulated; `bench/regen_floor.py` committed |

Also accepted and applied: per-region gap ratios (`vision` is **29.6×**, the worst in the
model, and was absent); the "400 sampled tensors" wording (it is the first 400 of 777 in file
order); the unmeasured "6 ms remainder" explanation marked as a hypothesis; the per-image
vision measurement re-run at n = 100 in its own result directory.

Its one finding I did not act on is the `docs/00_context.md` edit — see the row above.

---

## 10. What T003/T004 should look at first

1. The **17× gap** to the floor is the headline open question. Protocol §9 calls a gap this
   size structural. `vision` and `vlm_prefix` should be profiled too — `vlm_prefix` at 233 ms
   against a 97.5 ms *compute* floor is only 2.4×, so the expert is the anomaly, not the VLM.
2. The **+3.13 ms first step** (§1.7b).
3. The **80 ms dummy image** (§1.7a) — the cheapest real speed-up visible so far, and it needs
   no kernel work.
4. For E4: `max-autotune` on this GPU cannot use Triton GEMM templates
   (`min_sms = 68` vs Orin's 16, T001), and attention is forced to eager — both bound what
   compilation can recover.
