# E4 — π0 under `torch.compile` on the Jetson AGX Orin (task T003)

**Runs:** `results/2026092{1,2}_E4_pi0_*_N*/` · **MAXN, 16 SMs, GPU 1300.5 MHz, EMC 3199 MHz,
`jetson_clocks` before every run, `throttled: false`** · 20 warm-up + **100 timed** iterations
per point · LeRobot π0, `lerobot/pi0_libero_finetuned_v044`, bf16, batch 1, the T002 fixture.

---

## 1. The answer

### 1.1 The ladder is complete

| rung | ms per denoising step | × floor | status |
|---|---|---|---|
| analytic bandwidth floor | **3.115** | 1.0× | analytic, from the loaded model (T002) |
| research side's "optimistic bound" (2 × floor) | 6.230 | 2.0× | *assumed out of reach* |
| **measured eager** (T002) | **52.56** | 16.9× | measured |
| **measured compiled, best stock mode** | **16.30** | **5.2×** | measured, `reduce-overhead` |
| *research side's estimate for the compiled stack* | *~90* | *29×* | **5.5× too pessimistic** |

The ~90 ms figure came from applying Jetson **Thor's** 1.51× `torch.compile` gain to a derived
136 ms eager step. Both inputs were wrong for this device: eager is 52.56 ms, not 136, and the
compile gain is **3.22×**, not 1.51×.

### 1.2 Every mode, end-to-end, uninstrumented

Per-step is the slope of end-to-end over N (protocol §2's definition (b), which is *primary*
under compilation). `fixed` is the fit intercept = vision + prefix + per-inference overhead.

| configuration | e2e N=4 | e2e N=10 | ms/step | fixed | vs eager | × floor | compile |
|---|---|---|---|---|---|---|---|
| eager (T002) | 694.0 | 1011.6 | 52.56 | 485.1 | 1.00× | 16.9× | — |
| `default` | 528.0 | 663.6 | 22.61 | 437.6 | 2.32× | 7.3× | 40 s |
| `max-autotune-no-cudagraphs` | 530.3 | 667.4 | 22.86 | 438.8 | 2.30× | 7.3× | 213 s |
| `max-autotune` | 505.2 | 607.2 | 16.99 | 437.2 | 3.09× | 5.5× | 185 s |
| **`reduce-overhead`** | **497.8** | **597.3** | **16.30** | 433.6 | **3.22×** | **5.2×** | 124 s |

`reduce-overhead` was swept over all six N ∈ {1,2,4,6,8,10}; the rest at N ∈ {4,10}.

### 1.3 What each mechanism actually contributes

The four stock modes separate cleanly into two pairs:

| | without CUDA graphs | with CUDA graphs |
|---|---|---|
| `default` / `reduce-overhead` | 22.61 | **16.30** |
| `max-autotune-*` | 22.86 | 16.99 |

Columns differ by **1–4 %**; rows by **36 %**. So:

* **compilation itself**: 52.56 → 22.6 ms/step (**2.33×**)
* **CUDA graphs**: 22.6 → 16.3 ms/step (**1.39×**)
* **autotuning**: **nothing measurable** — and it costs 5× the compile time.

**`max-autotune` is a trap on this GPU, and here is the direct evidence.** A cold-cache run
logs **494 autotune blocks and 0 `triton_mm` candidates**: Inductor's `is_big_gpu` check
(`min_sms = 68` for CUDA, against the Orin's 16 — read from the installed source in T001)
excludes Triton GEMM templates from the search entirely, so `max-autotune` degenerates to
`default` plus CUDA graphs. Use **`reduce-overhead`**: same latency, 124 s instead of 185 s.

---

## 2. What the SM-count gate costs (labelled override variant)

T003 asks for one variant with that check defeated. Patched **in the benchmark process only**
(`bench/run_e4.py:force_gemm_templates`, which replaces Inductor's `is_big_gpu` in memory); the
installed package is never edited. Both arms re-run with **cold, separate Inductor caches** —
see §5.1 for why that matters.

| | AUTOTUNE blocks | `triton_mm` candidates | ms/step | fixed | e2e N=4 |
|---|---|---|---|---|---|
| `max-autotune` stock | 494 | **0** | 16.72 | 440.0 | 506.9 |
| `max-autotune` + gate defeated | 1646 | **6514** | 16.23 | 382.4 | 447.3 |
| gain | | | **2.9 %** | **13.1 %** | 11.8 % |

Triton GEMM templates *were* chosen once allowed. But the benefit lands almost entirely
**outside** the action expert: 13.1 % off vision + VLM prefix, only **2.9 %** off the denoising
step. That is consistent with the shapes — the expert's projections are skinny (S = 51) and
memory-bound, where tiling choice cannot beat cuBLAS, while vision and prefix push 816 tokens
through a 2 B-parameter model where it can.

**So the restriction costs ~13 % of the phases this study is *not* moving to memory, and ~3 %
of the phase it is.** It is not the reason the expert sits 5.2× above its floor.

---

## 3. The single line that dominates what remains

### 3.1 It is every graph break

With region instrumentation off, the compiled model still breaks its graph 4 times, and
**every break is the same line** (`raw/compile.log`):

```
Graph break in user code at .../lerobot/policies/pi0/modeling_pi0.py:919
Graph Break Reason: copy.deepcopy()
  Explanation: Dynamo does not support copy.deepcopy()
  Developer debug context: copy.deepcopy(UserDefinedObjectVariable(DynamicCache))
```

T002 measured that line at **5.68 ms/step (10.4 % of an eager step)**, cloning the whole
15.04 MB prefix KV cache. It is also the sole thing fragmenting the compiled denoising step.

### 3.2 It cannot simply be deleted — negative control

T002's report, and a paragraph I added to `docs/00_context.md`, said the expert passes
`use_cache=False` so the callee does not mutate the cache, and that *"whether the copy is
needed at all is worth one experiment"*. **I ran the experiment and I was wrong.** Deleting
the copy (`--variant nodeepcopy`) fails on the second denoising step:

```
RuntimeError: Attempting to broadcast a dimension of length 867 at -1!
  ... had torch.Size([1, 1, 51, 867]); but expected broadcastable to [1, 8, 51, 918]
```

867 = P + S = 816 + 51; 918 = 867 + 51. The expert **does** append its suffix K/V to the shared
cache regardless of `use_cache=False`, so without a private copy the cache grows by S every
step and the attention mask stops broadcasting. **The deep copy is load-bearing.** Both
documents are corrected.

### 3.3 But the *clone* is not needed — only the isolation (`cropcache`)

`transformers.DynamicCache` has `crop(max_length)`. Letting the expert append and then
cropping back to P is semantically identical and is a slice, not a 15 MB clone:

| | ms/step | graph breaks | × floor | vs eager | e2e N=4 |
|---|---|---|---|---|---|
| `reduce-overhead` (stock) | 16.30 | **4** | 5.2× | 3.22× | 497.8 |
| **`reduce-overhead` + crop-not-clone** | **10.37** | **0** | **3.3×** | **5.07×** | 468.7 |

One line removes **all four graph breaks** and a further **36 %** of the step. It is reported
**beside** the ladder, not on it: T003 requires the ladder's compiled rung to come from
unmodified modes.

**It computes exactly the same actions.** Checked before reporting the numbers, because a
faster model that computes something different is not a result: eager, `num_steps = 10`, fixed
noise (seed 1234), stock versus cropped —

| check | result |
|---|---|
| **bitwise identical** | **True** |
| max absolute difference | **0.000e+00** |
| control: reference vs itself | identical (harness is deterministic) |

Evidence: `results/20260922_E4_pi0_reduce-overhead+cropcache_N10/equivalence.md`.

**This is the result with the most consequence for the study.** `docs/00_context.md` presents
2 × floor = 6.23 ms as *"a stand-in for a perfect software stack"*, assumed unreachable. A
one-line change to stock LeRobot reaches **10.37 ms — 1.67× that bound**, on real hardware,
today. The headroom a PIM design must beat is much smaller than the original ladder implied.

---

## 4. `step_only` — where the gain comes from

Protocol §2 asks for a variant that compiles only the single-step function and loops in Python.

| | fixed (vision + prefix) | ms/step |
|---|---|---|
| eager | 485.1 | 52.56 |
| `step_only` | 484.3 (unchanged) | **16.34** |
| whole `reduce-overhead` | 433.6 (1.12×) | 16.30 |

Compiling the denoising step alone recovers the **entire** per-step gain (16.34 vs 16.30).
Compiling vision and prefix as well buys ~51 ms of fixed cost and nothing per step. Useful for
the hybrid model: the part being moved into memory is exactly the part compilation helps.

---

## 5. Two measurement errors I made and caught

Recorded because both would have produced plausible, wrong numbers.

### 5.1 The `forced` variant first reused cached kernels
All runs initially shared one `TORCHINDUCTOR_CACHE_DIR`. `forced` uses the same *mode* as stock
`max-autotune`, hence the **same cache key**: it loaded the stock kernels and the `is_big_gpu`
patch never took effect. The symptoms were visible — compile fell from 185 s to 79 s, and the
autotune log was empty. Both arms were re-run with cold, separate caches; the stock arm
reproduced its warm-cache numbers to 0.3 %, confirming only the `forced` cell was affected.
The contaminated directories are in `scratch/T003/invalid/`, not in `results/`.

### 5.2 My own instrumentation graph-breaks the compiled model
`bench/regions.py` wraps calls in Python context managers, and dynamo cannot inline
`contextlib.__enter__` (`SKIPPED INLINING <code object __enter__>`, at `regions.py:112`). Every
region boundary became a graph break, so an instrumented compiled run measures a graph that
compilation never produces. E4's primary measurement therefore runs with **instrumentation
off** and takes ms/step from the slope, exactly as protocol §2 prescribes. The consequence is
that E4 has no per-region breakdown — `ae_total`, `vision` and `vlm_prefix` are **not
measured** under compilation. T004's nsys capture is the right instrument for that.

---

## 6. Comparison with the characterization paper

The paper reports a **1.51× `torch.compile` gain on Jetson Thor**, which the research side
transplanted to Orin to estimate ~90 ms/step.

| | paper (Thor) | measured here (Orin) |
|---|---|---|
| compile gain, end-to-end at N = 4 | 1.51× | **1.39×** (694.0 → 497.8) |
| compile gain, end-to-end at N = 10 | — | **1.69×** (1011.6 → 597.3) |
| compile gain, **per denoising step** | — | **3.22×** |

End-to-end the transplanted 1.51× is about right — but only by coincidence, since it sits
between our N = 4 and N = 10 values and depends entirely on N. **Per step, the quantity the
ladder actually needs, it understates the gain by more than 2×.** A single end-to-end speed-up
factor is not transferable across N; the slope is.

---

## 7. Separation of evidence

**Measured:** every latency in §1.2, §2, §3.3, §4; the graph-break counts and their source
line; the autotune-candidate counts (494/0 stock vs 1646/6514 forced); compile times; the
`nodeepcopy` failure and its exact shape error; peak memory per run.

**Derived:** all slopes and intercepts (least squares over the N points); every speed-up ratio
and × floor figure; the mechanism decomposition in §1.3 (differences between measured modes);
the §2 split of the Triton-template gain into step vs fixed (from slope and intercept).

**Not measured:** any per-region breakdown under compilation (§5.2) — `vision`, `vlm_prefix`
and `ae_total` are absent from every E4 run by design; whether CUDA graphs were *actually*
captured rather than silently skipped (inferred from the 36 % gap between the pairs in §1.3,
not from a capture log); the combination of `forced` + `cropcache`, which is the obvious next
configuration and belongs to E10; the measured peaks (T005) — all × floor figures use the
**spec** 204.8 GB/s.

---

## 8. Reproduce

```bash
sudo jetson_clocks && sudo jetson_clocks --fan
# stock modes
TORCH_LOGS=graph_breaks,recompiles python bench/run_e4.py --mode reduce-overhead \
    --steps 1,2,4,6,8,10 --warmup 20 --timed 100
# variants
python bench/run_e4.py --mode reduce-overhead --variant step_only  --steps 4,10
python bench/run_e4.py --mode reduce-overhead --variant cropcache  --steps 4,10
python bench/run_e4.py --mode reduce-overhead --variant nodeepcopy --steps 4,10   # expected to fail
# the override — COLD, SEPARATE Inductor cache per arm, or it silently reuses kernels
TORCHINDUCTOR_CACHE_DIR=<fresh> TORCH_LOGS="graph_breaks,recompiles,+torch._inductor.select_algorithm" \
  python bench/run_e4.py --mode max-autotune --variant forced --steps 4,10
python bench/analyze_e4.py --date 20260921
```

---

## 9. Review (protocol §11)

### Self-review

| check | result |
|---|---|
| all four modes, full sweep for the best | **yes** — `reduce-overhead` over six N; others at N ∈ {4,10} |
| `step_only` variant | **yes** (§4) |
| labelled `gemm_templates_forced` variant, override never editing the package | **yes** (§2), patched in-process, both arms cold-cache |
| whether Triton GEMM templates were then chosen | **yes** — 0 → 6514 candidates |
| compile time, graph breaks, recompiles, CUDA graphs, peak memory | compile time and graph breaks **yes**; recompiles logged; **CUDA-graph capture inferred, not confirmed** (§7) |
| ladder uses unmodified modes only | **yes** — `cropcache` and `forced` sit beside it |
| ms/step by protocol §2's definition for compiled runs | **yes** — slope, with the reason stated (§5.2) |
| n, power mode, `throttled` | n = 100 per point; MAXN, 16 SMs, 1300.5 MHz locked; `throttled: false` |
| measured / derived / not-measured separated | **yes** — §7 |
| deviations and errors disclosed | **yes** — §5 |
| nothing over 50 MB; no weights/caches/venvs staged | **yes** |
| research-side files edited | **yes** — `docs/00_context.md`, under the owner's standing instruction; disclosed here and in the commit |

### Independent pass

Recorded below once complete.
