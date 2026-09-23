# Parts A, B and C — what they are, why they exist, and where everything is

**For the research side.** These three parts are not in `docs/01_measurement_plan.md` and
they are not part of the T001–T013 task numbering. They were added by the repository owner
on 2026-09-22 and 2026-09-23, mid-campaign, and they took priority over the remaining
planned tasks. If you pulled this repository and found six models' worth of results that no
task file asked for, this document is the explanation.

Nothing in the original plan was deleted. T004's remainder, T006–T013 and E7's capped power
modes are still open and still listed in `STATUS.md`.

---

## 1. Why these parts exist

The campaign's premise, from `docs/00_context.md`, is that **the action expert is the
memory-bound phase that dominates VLA latency**, which is what makes it the candidate for
processing-in-memory. T004's optimisation work produced a result that put that premise in
doubt, on π0 at N = 4:

| region | eager fp32 | best measured | speedup | share of the optimised frame |
|---|---:|---:|---:|---:|
| vision | 241.54 | 99.56 | 2.43× | 27 % |
| **`vlm_prefix`** | 234.50 | **193.32** | **1.21×** | **52 %** |
| action expert | 233.88 | 70.00 | 3.34× | 19 % |

The action head had become the *smallest* phase, and the VLM prefix — which had gained
least — dominated. The owner's instruction (2026-09-22, verbatim):

> *"I don't want to optimize (make it fast) for action head only. I want to equally optimize
> all three parts. This is the priority work for me."*

and then, on scope:

> *"skip the diffusion policy, ACT. You can do all GROOT N1.5/1.6/1.7 from GROOT family +
> pi 0.5, 0, SmolVLA as well."*

That is Parts A and B. Part C was added on 2026-09-23:

> *"1) using ncu, i want you to profile the eager and best for all models on three parts …
> 2) analyze how best optimize the eager for all models on each part … 3) check the
> dimension of expert like 51 skinny gemm of pi0 for all models"*

with a clarification on (3):

> *"my intention was the dimension of operands (matrix dimensions). There must be some
> 51 x something which is something like long shallow matrix form"*

The ordering was also set by the owner: **Part A and B for every model, error-free, before
any Part C work.** That ordering is why Part C started only after all six ladders were
clean.

---

## 2. The six models, and the one rule that makes them comparable

| model | checkpoint | N (its own default) | N field name |
|---|---|---:|---|
| π0 | `lerobot/pi0_libero_finetuned_v044` | 10 | `num_inference_steps` |
| π0.5 | `lerobot/pi05_libero_base` | 10 | `num_inference_steps` |
| SmolVLA | `lerobot/smolvla_base` | 10 | **`num_steps`** |
| GR00T N1.7 | `nvidia/GR00T-N1.7-3B` | 4 | **`num_inference_timesteps`** |
| GR00T N1.6 | `nvidia/GR00T-N1.6-3B` | 4 | `num_inference_timesteps` |
| GR00T N1.5 | `nvidia/GR00T-N1.5-3B` | 4 | `num_inference_timesteps` |

**Each model runs at its own configured N**, not a common N. That was the owner's
instruction ("you need to run the max N for each model's configuration") and it is the
deployment setting the authors' own leaderboard numbers use. The field name differs three
ways, which is why `bench/model_registry.py` exists.

π0 additionally has a complete **N = 4 ladder**, kept as a within-model control on N. It is
reported separately and never mixed into the N = 10 table — see §6 for why that warning is
there.

**Device, identical for every number:** Jetson AGX Orin 64 GB, MAXN, 16 SMs, GPU pinned
1300.5 MHz, JetPack 6.2.3 / L4T R36.5.0 / CUDA 12.6, `sm_87`. Measured peaks (T005/E1):
187.47 GB/s, 33.85 TFLOP/s bf16, **ridge 180.57 FLOP/B** — measured, not from a spec sheet.

---

## 3. Part A — reproduce the published characterisation, per phase

**Question.** Where does the time actually go, per phase, and does
[arXiv:2604.24447](https://arxiv.org/abs/2604.24447) (Zhou et al., *Characterizing VLA
Models across XPUs*) reproduce on this device?

**Method.** `bench/paper_repro.py` runs each model in the paper's configuration (eager,
TF32, the model's own N) with a matched fp32 control, and `bench/regions.py` splits every
inference into three regions by wrapping three call sites:

    <container>.embed_image   -> vision
    <container>.forward       -> vlm_prefix
    model.denoise_step        -> ae_step[i], summed to ae_total

**The one method detail that matters:** regions are installed **after** compilation, never
before. A region boundary inside a compiled graph disappears. This "compile first,
instrument second" order is what makes the compiled per-phase split possible at all.

**Result.** The paper reproduces to within **0.45–0.6 %** end-to-end on π0 (916.45 ms
measured against their 920.58), with every phase within 10 %.

| where | files |
|---|---|
| paper configuration + fp32 control | `results/20260922_E4d_{pi0,pi05,smolvla}_paper_config_N10{,_fp32}/` |
| per-phase SM utilisation | `results/20260922_E4d_pi0_phase_util_N10/` |
| report | **`reports/E4d_phase_balance.md`** |

---

## 4. Part B — the optimisation ladder, one factor at a time

**Question.** Which lever moves which phase, for each model?

**Method.** `bench/opt_ladder.py`. Each rung changes **exactly one thing** from the rung
below, so a speedup can be attributed. Rungs, with what each isolates:

| rung | what it adds |
|---|---|
| `eager_fp32` | baseline, `matmul_precision=highest` |
| `eager_tf32` | TF32 only — this was a hidden confound, see below |
| `bf16_base` | an fp32 checkpoint cast to bf16 through the model's **own** supported path |
| `ae_compile` / `ae_graphs` | compile the denoising step, then add CUDA graphs |
| `vis_compile` / `vis_graphs` | the same two steps for `embed_image` |
| `vlm_compile` / `vlm_graphs` | the same two steps for the prefix |
| `whole_graphs` | LeRobot's own path: compile `sample_actions` as one unit |
| `bf16_vision` / `bf16_norms` / `bf16_both` | remove LeRobot's fp32 islands |

`reduce-overhead` and `max-autotune-no-cudagraphs` are used deliberately as the pair that
**separates compilation from CUDA graphs** — the same compile, with and without graphs.

**Three things this part found that are worth your attention:**

1. **TF32 was a hidden confound.** `modeling_pi0.py:591-595` sets
   `torch.set_float32_matmul_precision("high")` *inside* `if config.compile_model:`. Any
   comparison of compiled against eager in LeRobot therefore also compares TF32 against
   fp32 unless TF32 is made explicit. The ladder makes it its own rung.

2. **`params_to_keep_float32` is a training constraint, not an accuracy one.** Its own
   comment says the fp32 list exists to avoid an optimizer "same dtype" error. There is no
   optimizer at inference. Removing it is rungs B2/B3.

3. **The checkpoints do not ship the same dtype.** π0 ships `bfloat16`, π0.5 ships
   `float32`, SmolVLA loads its VLM bf16 with no keep-list at all. So "remove the fp32
   islands" means something different for each, which is why `bf16_base` exists and why
   SmolVLA's B2/B3 are recorded as **not applicable with a reason** rather than as failures.

**Result — the headline table**, every pair at the same N:

| model | N | AE share eager | AE share best | e2e eager (ms) | e2e best (ms) | speedup |
|---|---:|---:|---:|---:|---:|---:|
| SmolVLA | 10 | 85.3 % | 48.6 % | 727.46 | 80.97 | 8.98× |
| GR00T N1.6 | 4 | 61.5 % | 39.5 % | 327.22 | 186.12 | 1.76× |
| GR00T N1.7 | 4 | 56.7 % | 33.1 % | 372.48 | 240.49 | 1.55× |
| π0 | 10 | 54.9 % | 39.7 % | 1069.04 | 424.46 | 2.52× |
| GR00T N1.5 | 4 | 38.6 % | 31.0 % | 255.22 | 200.85 | 1.27× |
| π0.5 | 10 | 20.4 % | 38.2 % | 3028.56 | 489.15 | 6.19× |

**A prediction recorded in advance, and refuted.** The plan predicted action-head dominance
would track N, putting the GR00T family lowest at N = 4. It does not: N1.6 and N1.7 at
N = 4 sit *above* π0 at N = 10. Part C §3 gives the mechanism.

**For π0.5, precision beats compilation on the phase compilation cannot touch:** compiling
moved its VLM prefix 534 → 516 ms (1.04×); casting the fp32 checkpoint to bf16 moved it
516 → 239 ms (**2.16×**), and lifted all three phases at once.

| where | files |
|---|---|
| ladders, all six models | `results/20260922_E4c_{pi0,pi05,smolvla}_opt_ladder/`, `results/20260923_E4d_groot_{n17,n16,n15}_ladder/` |
| per-rung raw rows | `<dir>/raw/rows/<rung>_N<N>.json` — one file per rung, written as it completes |
| report | **`reports/E4d_phase_balance.md`** (generated by `bench/make_phase_report.py`) |

---

## 5. Part C — what each phase is *bound by*, and what the fix actually fixed

### C1 — Nsight Compute, per phase, eager and best, all six models

36 captures: 6 models × {eager, best} × 3 phases. Four sections plus the 14
`smsp__average_warps_issue_stalled_*` metrics requested **by name** — the sections alone do
not carry the stall breakdown, and "Long Scoreboard" is the specific signal this study
needs.

**How a phase is judged:** every per-kernel number is averaged **weighted by kernel
duration**, and the bound thresholds are ncu's own `SOLBottleneck` rule (80 % of SM or
memory throughput), not values invented here. Below 60 % on both is latency bound, and the
dominant stall names which latency.

### C2 — eager → best, per phase

Reported as **counter changes only**, deliberately. `-c 40` captures the first 40 kernels of
a range and the kernel mix differs between modes, so summed capture duration is not a
speedup and is not presented as one. Phase wall times live in Part B.

### C3 — the operand dimensions of every expert GEMM

`bench/expert_dims.py`, from each checkpoint's own tensor shapes. **S is measured** on a
real inference, never taken from `chunk_size`/`action_horizon`.

### The results

**The action expert, all six models:**

| model | eager | best |
|---|---|---|
| π0 | `imc_miss` — SM 22.7, Mem 21.6 | **`long_scoreboard`** — Mem 56.5 |
| π0.5 | `imc_miss` — SM 4.9, Mem 12.2 | **`long_scoreboard`** — Mem 60.3 |
| SmolVLA | `imc_miss` — SM 15.6, Mem 18.8 | **`long_scoreboard`** — Mem 35.5 |
| GR00T N1.7 | mixed — SM 52.4, Mem 70.4 | mixed — Mem 75.0 |
| GR00T N1.6 | `long_scoreboard` — SM 36.7, Mem 24.8 | `long_scoreboard` — Mem 53.8 |
| GR00T N1.5 | mixed — SM 54.2, Mem 71.7 | mixed — Mem 76.0 |

**This is the result that bears directly on the PIM premise, and it is conditional:**

* **LeRobot flow-matching experts** (π0, π0.5, SmolVLA) are **not** memory-bound when
  eager. They are *issue*-starved — constant-cache misses 32–35 %, instruction-fetch
  starvation ~18 %, Long Scoreboard only third at ~16 %, memory throughput 12–22 %. They
  become memory-latency bound **only after compilation**.
* **GR00T DiT experts** are memory-led **from the eager baseline**.

So "the action expert is the memory-bound phase" is false for the eager LeRobot baseline
and true for the compiled one, and unconditionally truer for GR00T. A PIM design argued
from *eager* profiles of a π0-class model is sized against launch and issue overhead, not
bandwidth. arXiv:2604.24447's characterisation is eager.

**C3 predicts the C1 split**, which is the strongest internal cross-check in the study:

| model | S (measured) | expert layers | GEMMs | skinniest GEMM | K/M |
|---|---:|---:|---:|---|---:|
| π0 | **51** | 18 | 126 | `[51×4096] × [4096×1024]` | 80.3 |
| π0.5 | 50 | 18 | 162 | `[50×4096] × [4096×1024]` | 81.9 |
| SmolVLA | 50 | 16 | 112 | `[50×2048] × [2048×720]` | 41.0 |
| GR00T N1.7 | 41 | 32 | 224 | `[41×6144] × [6144×1536]` | 149.9 |
| GR00T N1.6 | 51 | 32 | 224 | `[51×6144] × [6144×1536]` | 120.5 |
| GR00T N1.5 | 49 | **16** | 112 | `[49×6144] × [6144×1536]` | 125.4 |

The owner's guess was right: π0's skinniest expert GEMM is literally **51 × 4096**. Every
model's I ≈ S sits far below the measured ridge of 180.57 FLOP/B, so *analytically* every
expert GEMM is memory-bound by construction — which is exactly what the measured counters
contradict for the eager LeRobot models. **Arithmetic intensity bounds what a phase could
achieve; it does not say what limits it now.** A roofline argument alone gets this wrong in
both directions.

**Measuring S rather than trusting config mattered:**

| model | measured S | config says | |
|---|---:|---:|---|
| GR00T N1.7 | 41 | 40 | +1, a state token is prepended |
| GR00T N1.6 | 51 | 50 | +1 |
| **GR00T N1.5** | **49** | **16** | **3× off** |

**And the DiT block count explains the Part B ranking:** N1.6 and N1.7 carry **32**
transformer blocks, N1.5 carries **16**, all width 1536. At identical N = 4 in one family,
with only the head differing, N1.5's action head is 38.6 % of the eager frame against
N1.6's 61.5 %. Architecture beats step count.

| where | files |
|---|---|
| C1 captures | `results/20260923_E4g_c1_<model>_{eager,best}/ncu/` — `*_details.csv` and `*.ncu-rep.zst` |
| C1/C2 summary | `results/20260923_E4g_c1_summary/{c1_bounds.csv,c2_eager_vs_best.csv,c1_detail.json}` |
| C3 dimensions | `results/20260922_E4f_expert_dims/expert_dims.json` |
| C3 measured S (GR00T) | `results/20260923_E4f_groot_S_{n17,n16,n15}/suffix_tokens.json` |
| report | **`reports/E4g_c1_c2_bottlenecks.md`** (generated by `bench/analyze_c1.py`) |

---

## 6. How to trust these numbers — the gates, and what they caught

Both reports are **generated from the raw files** (`bench/make_phase_report.py`,
`bench/analyze_c1.py`). Nothing is typed in, so prose cannot drift from data when a rung is
re-run. Beyond that, three gates run over Part C, and each exists because it caught a real
defect:

| gate | what it checks | what it caught |
|---|---|---|
| `bench/c1_capture_ok.py` | CSV has all 14 stall metrics and > 300 rows | an orphaned profiler wrote a 130-row CSV with **0** stall metrics, and a file-existence test called it done |
| `bench/c1_validity_check.py` | eager vs best, and vision vs prefix, have **different kernels** | (a) SmolVLA's compiled captures silently measured **eager** — dynamo hit the recompile limit and fell back; (b) GR00T's `vlm_prefix` range sat on `backbone.forward`, which *contains* vision, so it profiled vision under the prefix's name |
| regions-sum check | regions sum to end-to-end within a few percent | regions summed over the whole loop instead of per iteration — they came to 3.4× end-to-end |

**Corrections already issued, so you do not rediscover them as discrepancies:**

* **π0's optimised action-head share is 39.7 % at N = 10, not 18.9 %.** An earlier
  cross-model table paired the N = 10 *eager* share with the N = 4 *optimised* share,
  because π0's directory holds both ladders and the generator keyed rows by rung alone.
  18.9 % is π0's N = 4 figure and is correct only there.
* **SmolVLA's first `best` C1 captures were withdrawn**, not merely superseded. They
  profiled an uncompiled model. Its `best` now compiles the three callables the Part B
  ladder compiles, validated outside ncu against the `vlm_graphs` rung.

**Known non-uniformity, accepted by the owner:** C1's `best` is LeRobot's own
`compile_model=True` path for π0 and π0.5, and the ladder's three callables for SmolVLA,
because LeRobot's whole-`sample_actions` path does not converge for SmolVLA. Every capture
records which under `compile_style` in its JSON. Do not compare the two as if identical.

**Still open at the time of writing:** the closed-loop accuracy check for the bf16 rungs
(`bench/libero_eval.py`). π0's and π0.5's best rungs are `bf16_both`, which changes
numerics, and until that check reports, those are **speed results without an accuracy
statement**. `STATUS.md` carries the current state.

---

## 7. Raw data that is not in this repository

Four `.ncu-rep` files from the superseded first C1 attempt are 88–203 MB each. GitHub
rejects anything over 100 MB and protocol §6 caps a committed file at 50 MB, so they are
published instead at

> **`https://huggingface.co/datasets/yang-gon/vla-orin-ncu-raw`**
> private dataset on the owner's Hugging Face account (`yang-gon`), 707.6 MB.
> Ask the owner for access.

`results/20260922_E4g_pi0_ncu_eager/RAW_FILES.md` carries each file's **sha256 and byte
count**, and the uploaded sizes match it exactly, so the published copy is verifiably the
same data.

They were scanned before upload for credential patterns and for environment-variable-shaped
strings: **zero hits in all four**, which is positive evidence that `.ncu-rep` does not
embed the process environment. That check is not a formality — `.nsys-rep` on this device
**was** verified to carry system-wide process metadata including a session token, which is
why no `.nsys-rep` is ever published and why `.gitignore` blocks them.

The current C1 captures are small enough to commit: four targeted sections instead of
`--set full -c 300` brings each report to about **900 KB** after `zstd -19`, against the
89–203 MB the first attempt produced, answering the same question.

---

## 8. Reproducing any of it

```bash
# Part A — paper configuration and its fp32 control
~/venvs/pi0/bin/python bench/paper_repro.py --model pi0 --tf32 on

# Part B — the full ladder for one model, at its own N
~/venvs/pi0/bin/python bench/opt_ladder.py --model pi05 \
    --outdir results/<date>_E4c_pi05_opt_ladder --steps 10 --warmup 20 --timed 10

# Part B — GR00T (its own checkout and environment; see env/setup_groot_n1d5_n1d6.md)
bash bench/queues/queue_n1d6.sh

# Part C1 — one capture
bash bench/run_ncu_c1.sh pi0 eager vision results/<date>_E4g_c1_pi0_eager

# Part C — regenerate both reports from the raw files
~/venvs/pi0/bin/python bench/make_phase_report.py --out reports/E4d_phase_balance.md
~/venvs/pi0/bin/python bench/analyze_c1.py --out results/20260923_E4g_c1_summary \
    --md reports/E4g_c1_c2_bottlenecks.md

# Part C — the gates
~/venvs/pi0/bin/python bench/c1_validity_check.py
```

Every run's `meta.json` carries the exact command, the code commit, the checkpoint, the
software versions and the device state, per protocol §5.

---

## 9. Three questions for the research side

These are also in `STATUS.md`.

1. **Does the conditional result change the PIM target?** If a π0-class expert is only
   memory-bound *after* compilation, a study quoting eager numbers is sizing against launch
   overhead. GR00T-class DiT heads are memory-led either way — should they be the target
   instead?
2. **Should the paper's Finding #4 be re-examined?** It calls the VLM compute-bound from
   operational intensity (840 FLOP/B analytic; ours 816). Our counters show the *eager*
   prefix stalling **74 %** on Long Scoreboard at 40 % SM and 49 % memory throughput. High
   arithmetic intensity does not help when latency-bound at low occupancy.
3. **Is fp32 the intended upstream deployment for π0.5?** It ships `dtype: float32` while
   π0 ships bf16, and casting it to bf16 through LeRobot's own path gives **2.16×** on the
   VLM prefix where compilation gave 1.04×.
