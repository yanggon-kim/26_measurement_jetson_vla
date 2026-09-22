# 01 — Measurement plan

Priorities: **P0** = E0, E3, E4 for π0 · **P1** = **E11 for π0** (inputs for the hybrid GPU + PIM model — right
after E3 / E4, because it reuses their harness and fixture), then E5 (nsys + ncu), E5b, E6, E7 for π0, and E2, E1 ·
**P2** = π0.5 and SmolVLA through E3, E4, **E11**, **E5, E5b, E6, E7**, and E8 · **P3** = GR00T N1.7 through the
same set including **E11**, E9, E10. The profiling experiments (E5, E5b, E6, E7) **and E11** are **required for
every model that runs**, not only for π0;
π0 goes first. First deliverable (end of P0): the ladder table for π0 —
*bandwidth floor 3.15 ms → measured eager → measured compiled*, in ms per denoising step — with conditions.
Second deliverable (first item of P1): `model_inputs.csv` for π0 (E11, protocol §13.8).
Before profiling any model, compute its **physical floor** per region (protocol §9).
Each experiment below lists what to run, the sweep, and the files we expect. Follow `02_protocol.md` throughout.

**Task order.** Work through `tasks/` in this order — by priority, not by task number — taking the first open
task whose dependencies are done:

| order | task | experiments | priority | depends on |
|---|---|---|---|---|
| 1 | T001 | E0 bring-up | P0 | — |
| — | T013 | withdrawn (the MAXN record was completed inside T001); never claim | — | — |
| 2 | T002 | E3 π0 eager | P0 | T001 |
| 3 | T003 | E4 π0 compiled | P0 | T002 |
| 4 | **T010** | **E11 π0** | P1 | T003 |
| 5 | T004 | E5, E6, E7 π0 | P1 | T003 |
| 6 | T005 | E1, E2 | P1 | T001 |
| 7 | T007 | E5b π0 | P1 | T004, T005 |
| 8 | T006 | E3, E4 π0.5 + SmolVLA; E8 | P2 | T003 |
| 9 | **T011** | **E11 π0.5 + SmolVLA** | P2 | T006 (step 1), T010 |
| 10 | T008 | E5, E5b, E6, E7 π0.5 + SmolVLA | P2 | T006, T004, T007 |
| 11 | T009 | GR00T N1.7: E3, E4, E5, E5b, E6, E7 | P3 | T008 |
| 12 | **T012** | **E11 GR00T N1.7** | P3 | T009 (step 1), T010 |

## E0 — Platform record
Everything in protocol §1, plus `nvidia-smi` (if present), `free -h`, storage, `tegrastats` idle snapshot, the list
of `nvpmodel` modes with their clock caps. Output: `results/<date>_E0_platform/` and `env/README.md` describing
how the software stack was installed (exact wheels / containers, e.g. NVIDIA's PyTorch for JetPack, and the
torch / triton versions that make `torch.compile` work on aarch64).

## E1 — Peak microbenchmarks (what the silicon can do)
1. Memory bandwidth as the GPU sees it: large device-to-device `memcpy` / `tensor.clone()` (≥ 1 GiB, several sizes),
   and a bandwidth-bound elementwise kernel (`y = a*x + y` on 1 GiB bf16). Report GB/s; compare with 204.8.
2. Compute: square GEMM `n × n` for n ∈ {1024, 2048, 4096, 8192} in bf16, fp16, fp32 and int8 (`torch._int_mm` if
   available): achieved TFLOP/s; compare with 42 TFLOP/s bf16.
3. Write `hw_constants.json` (protocol §8): SM count, compute capability / `sm_` architecture, measured peak
   GB/s and TFLOP/s per dtype, ridge point = peak FLOP/s ÷ peak bytes/s (expected ≈ 208 FLOP/B; verify).
Output: `latency_summary.csv` with columns `kernel, dtype, shape, bytes, flops, ms, GBps, TFLOPps`.

## E2 — Action-expert kernel sweep (where the floor-multiple comes from)
For every linear layer of the π0 expert, time `y = x @ W.T` with x of shape `[S, K]`, W `[N, K]`, bf16, cuBLAS via
`torch.nn.functional.linear`, for **S ∈ {1, 2, 4, 8, 16, 32, 51, 64, 128, 256}**:

| layer | K (in) | N (out) | weight bytes (bf16) |
|---|---|---|---|
| q_proj | 1024 | 2048 | 4.19 MB |
| k_proj | 1024 | 256 | 0.52 MB |
| v_proj | 1024 | 256 | 0.52 MB |
| o_proj | 2048 | 1024 | 4.19 MB |
| mlp_gate, mlp_up | 1024 | 4096 | 8.39 MB each |
| mlp_down | 4096 | 1024 | 8.39 MB |

> **[CORRECTED ON DEVICE — T005, 2026-09-21]** This table previously carried a single row
> `kv_proj (k and v) | 1024 | 512 | 1.05 MB`. The expert is **multi-query**
> (`num_key_value_heads = 1`, `head_dim = 256`), so k and v are two *separate* `1024 -> 256`
> projections, not one fused `1024 -> 512`. The combined weight bytes are unchanged, but it is
> two kernel launches rather than one — and at S = 51, where E2 finds these layers are
> launch- and weight-streaming-bound rather than FLOP-bound, the launch count is what matters.
> Confirmed twice: from the checkpoint's own module shapes (`k_proj.weight [256, 1024]`, T002)
> and from the Inductor autotune log, which shows `36 x mm(51x1024, 1024x256)` = 18 layers x 2
> separate projections (T003).

Also attention for S = 51 queries (8 heads × 256) against P + S = 867 keys with one KV head (SDPA), RMSNorm, RoPE,
GeLU, softmax and an elementwise add at these sizes. For each: kernel time (CUDA events over 1,000 calls), the
**time per call when calls are issued back to back from Python** (this exposes launch overhead), achieved GB/s
(weight bytes ÷ time) and FLOP/s, and the **ratio to the bandwidth floor** (weight bytes ÷ the **measured** peak from E1, not the
204.8 GB/s spec — protocol §8; see the bandwidth-variability note in
`reports/E1_E2_microbench.md`).
Questions to answer in the report: at which S does a layer stop being launch/latency-bound; what is the
CPU-side cost of launching one small kernel on this machine (µs); how does CUDA-graph replay change it.

## E3 — End-to-end latency, eager mode
Models: π0 first; then π0.5, SmolVLA; GR00T N1.7 optional. One real LIBERO observation as input.
Sweep **N (denoising steps) ∈ {1, 2, 4, 6, 8, 10}**; π0's default is 10 in openpi, the characterization paper used 4 —
run both and say which is which. Report per region (protocol §2): median ms, share of total; **ms per denoising
step and whether steps are equal**; the actual prefix token count P and suffix count S; peak GPU memory;
wall-clock control frequency (Hz). Compare with the paper's 920 ms and 110 / 258 / 543 ms and explain differences
(software versions, N, image count, precision).

## E4 — Same, under `torch.compile`
Modes: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`; compile the action expert
alone, and the whole policy if it works. Record: compile time, number of graph breaks (`TORCH_LOGS=graph_breaks`),
recompilations, steady-state latency per region, peak memory, and whether CUDA graphs were actually used.
Report the speed-up per region against E3, and **ms per denoising step** — this replaces our 90 ms estimate.
T001 showed that every mode compiles on this stack (torch 2.8.0 + triton 3.4.0). Known limit to report, not to
fix: Inductor refuses Triton GEMM templates under `max-autotune` when the GPU has fewer SMs than a fixed minimum
(protocol §12; `min_sms = 68` against the Orin's 16, confirmed at MAXN by T001) — so `max-autotune` here means
"default + CUDA graphs + ATen autotuning" and is presented as such; T003 adds one labelled override variant.
If Triton / Inductor does not work on the device's PyTorch build, document exactly what fails, try NVIDIA's
JetPack PyTorch container and the newest available wheel, and report the best working configuration.

## E5 — Kernel timeline (nsys) and SM utilisation (ncu) on E3 and E4
For each model, eager and the best compiled mode, at the N the characterization paper used (π0: 4) and at the
model's default N (π0, SmolVLA: 10; GR00T: 4):
1. **nsys**, per protocol §3.1: kernels per denoising step, summed kernel time vs region time, GPU idle fraction
   inside `ae_step`, gap histogram, top kernels (are they GEMM, attention, elementwise, or memcpy?), the CPU
   thread that launches them and whether it is the bottleneck.
2. **ncu**, per protocol §3.2–3.3, one filtered run per region — `vision/`, `vlm_prefix/`, `ae_step[0]/` (and
   `ae_layer[0]/` first, as the small case): **SM occupancy and activity** per kernel and per region
   (`kernel_weighted` and `region_weighted`), **per-pipe utilisation** (`pipe_lsu`, `pipe_fma`, `pipe_alu`,
   `pipe_xu`, `pipe_adu`, `pipe_cbu`; on this Orin there is no `pipe_fp16` column — keep the exported
   `sm__pipe_tensor_op_hmma_cycles_active.*` columns instead), **warp-stall reasons** (raw page on this Orin,
   protocol §3.3), SM and memory (= L2, protocol §3.3) throughput, off-chip bytes from the L2 sysmem-miss sectors,
   the memory-bound / compute-bound / balanced split of kernel time. Never unfiltered; one step, not N steps.
   Outputs: `ncu/*_details.csv`, `ncu/*_raw.csv` (full exports), `ncu/columns.txt`, `ncu/kernels.csv`,
   `ncu/region_utilisation.csv`, the raw `.ncu-rep` under `raw/` and the full nsys CSV exports under `nsys/` (protocol §3.5, §6; the nsys binaries are not committed on this device — §3.5 exception).
Questions to answer: **of the time one denoising step takes, how much is kernels executing and how much is the
GPU waiting for the CPU? While kernels execute, how occupied are the SMs, which pipes are busy, and what do the
warps stall on?** The region-weighted SM occupancy of the action-expert phase is the number to set against the
characterization paper's "SM utilisation 22 %"; tegrastats `GR3D_FREQ` is reported next to it as a coarse
cross-check only. End with the root-cause chain of protocol §10.

## E5b — Instruction-level analysis of the dominant action-expert kernels
Full procedure: `docs/03_instruction_level_analysis.md`. For each model, for the skinny GEMM / GEMV kernels
that serve the expert's projections at the model's own S, plus the attention kernel if it ranks in the top-5 of
`ae_step`: SASS dump with `cuobjdump` for this GPU's architecture (expected `sm_87`; confirm), whole-kernel and
inner-loop instruction inventories, annotated hot loop and pseudocode with phases, load / store / atomic split
of the LSU pipe, mapping of ncu stall reasons to instruction classes, static-vs-dynamic cross-check, inner-loop
latency budget; an S sweep S ∈ {1, 16, own S, 256} on isolated kernels, and eager vs compiled.
Question: **is there compiled, instruction-level evidence that the action-expert kernels stall on memory loads
(memory-bound) rather than on compute or on launch, and how does that change with S and under `torch.compile`?**
Outputs: `results/<date>_E5b_<model>_<mode>_<short>/` and `reports/E5b_<model>.md`.

## E6 — Roofline points
For `vision`, `vlm_prefix` and `ae_step` of each model, and for the E5b target kernels: FLOPs and bytes moved —
analytic from shapes (always; the primary rows), **and** from ncu where exported — on this Orin that means FLOPs
from the SM throughput counters and **off-chip bytes from the L2 sysmem-miss sectors** (protocol §3.3; there are no
DRAM counters, and ncu's `Memory Throughput` is L2 throughput and is *not* a roofline x- or y-value) — ÷ measured
time (nsys kernel time or §2 region time, never ncu `Duration`) → achieved
FLOP/s and GB/s, arithmetic intensity (FLOPs per byte). Place them against the **Orin's own** roof (measured
peaks from E1 and the spec values, protocol §8 — not any other GPU's). Output a CSV we can plot:
`model, region_or_kernel, flops, bytes, ms, intensity, achieved_TFLOPps, achieved_GBps, pct_peak_compute, pct_peak_bw, source`
with `source` ∈ {`analytic`, `ncu`}, plus the floor and gap ratio of protocol §9.

## E7 — Power and energy
Per protocol §4 for each model (π0 first; eager and best compiled), at the paper's N and the model's default N: mean power per rail for idle, prefix-only loop,
action-expert-only loop, full inference loop; energy per inference and per denoising step, total and above idle.
Power is sampled from sysfs for all four rails (protocol §4; tegrastats omits the memory rail on this board).
MAXN is required for every model. Capped modes (answer to question 3 of 2026-09-21; rules in protocol §1): for π0
try `MODE_50W` without a reboot as the last item of the last MAXN session; `MODE_30W` / `MODE_15W` need one human
reboot each and are measured only if a later task announces such a session — they are **optional** and block
nothing. For the other models the capped modes are optional.

## E8 — LIBERO closed loop
Using openpi's `examples/libero` (policy server + client) and lerobot's LIBERO environment: success rate on
`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`. Start with 10 episodes per task to validate the
pipeline, then the standard 50 if time allows; always report episodes and seeds. Factors: eager vs compiled;
N ∈ {10, 4, 2}; action-chunk replanning interval. Separate simulator + rendering time from policy time, and report
the achieved control frequency. Stretch: success rate when the last k denoising steps reuse a **stale** prefix KV
cache from the previous observation (k ∈ {0, 1, 3, 5}), reproducing the paper's Table 7 — this bounds an
overlap scheme we rely on. MuJoCo rendering on the Jetson needs EGL (`MUJOCO_GL=egl`); if LIBERO cannot be
installed natively, use the project's Dockerfile and record the image.

## E9 — Precision (optional)
fp16 vs bf16 for E3; weight-only INT8 / INT4 and W8A8 through `torchao` if it installs: latency per region and,
if E8 works, success rate. Orin's Ampere GPU has INT8 tensor cores and no FP8.

## E10 — Best engineered stack (optional)
GR00T's `scripts/deployment/benchmark_inference.py` in all its modes (eager, compile, TensorRT action head, TensorRT
full pipeline); if feasible a TensorRT or CUDA-graph version of the π0 action expert. Question: how close to the
6.30 ms (2 × bandwidth floor) per step does the best stack get on real hardware?

## E11 — Inputs for the hybrid GPU + PIM model (all models)
**Why.** The system we will model keeps the vision encoder and the VLM prefix on the Jetson GPU and executes the
action expert's large matrix products (the projections, and the attention products against the cached prefix) in
LPDDR5 memory (a Ramulator-based PIM simulator on the research side). Between those products the GPU still runs
small **host segments** in every expert layer — norms, RoPE, softmax, residual adds, the gated activation — and,
per denoising step, the suffix / time embedding, the output projection and the Euler update. The CPU synchronises
with the memory side between segments, the prefix KV cache is handed over once per inference, and in the
pipelined scheme the GPU already computes the **next** frame's prefix while the memory denoises. Today all of
this is an analytic guess (`docs/00_context.md`, "Inputs of the hybrid model"): segment time =
max(FLOPs ÷ 16.8 TFLOP/s, bytes ÷ 143 GB/s) + 20 µs per kernel, transfers = bytes ÷ 143 GB/s with **no fixed
cost**, synchronisation free, prefix unaffected by memory-side activity. E11 measures each of these terms.
Methods, file formats and the `model_inputs.csv` schema are in **protocol §13**; E1–E7 are not repeated here —
E11 reuses their harness, regions, fixtures and loops (E3 / E4 regions and N sweep, E2 launch cost, E1 peaks,
E7 phase loops) and adds only what they do not give. Models: π0, π0.5, SmolVLA, GR00T N1.7 (when it runs).
MAXN + `jetson_clocks`, batch 1, bf16, the fixture observation, eager and the best compiled mode of E4.

**Part 1 — host segments as the mapping defines them** (protocol §13.1–13.2). Derive the segment list per model
from the model code → `segments.csv`. For **π0** (S = 51 = 1 state + 50 action tokens, hidden D = 1024, 8 query
heads × 256, 1 KV head, FFN 4096, T = P + S keys with P printed by E3, expected ≈ 816) the research side's
mapping uses these names — use them as `key`, check every shape against the loaded model, and report differences:

| segment | operations | tensors (π0) | analytic FLOPs / bytes used today |
|---|---|---|---|
| `rmsnorm1` | pre-attention RMSNorm | x `[1, S, 1024]` | 8·S·D / 3·(S·D·2 B) |
| `rope` | rotary embedding of q and k (+ the query scale if the code applies it there) | q `[1, S, 8, 256]`, k `[1, S, 1, 256]`, positions P … P+S−1 | 6·S·(8·256 + 256) / 2·S·(8·256 + 256)·2 B |
| `softmax` | mask add + softmax over the attention scores, in the dtypes the code uses; **also** the mapping's variant: fp32 scores in → softmax → fp16 out (`variant = dtype=fp32_in_fp16_out`) | scores `[1, 8, S, T]` (≈ 1.41 MB as fp32, ≈ 0.71 MB as fp16: ≈ 2.1 MB per layer cross the boundary) | 5·8·S·T / 8·S·T·(4 + 2) B |
| `residual_rmsnorm2` | x + attention output, then pre-FFN RMSNorm | `[1, S, 1024]` | 9·S·D / 4·(S·D·2 B) |
| `gelu_gate_mul_up` | GeLU(gate) ⊙ up (the GeLU variant of the code) | gate, up `[1, S, 4096]` | 9·S·4096 / 3·S·4096·2 B |
| `partials_residual` | `mapping_only`: sum of 4 partial results of the down projection + residual: `partials.sum(0) + x` | partials `[4, S, 1024]`, x `[S, 1024]` | 5·S·D / 3·(S·D·2 B) |
| `residual` | the plain residual add alone (the variant without partial sums) | `[1, S, 1024]` | S·D / 3·(S·D·2 B) |
| `attention_gpu` | optional design variant: the whole attention product on the GPU — the model's attention call for S queries against T keys, 1 KV head (as in E2) | q `[1, 8, S, 256]`, K, V `[1, 1, T, 256]` | — |
| `cast_scores_fp32_fp16` | the dtype cast of the score matrix alone | `[1, 8, S, T]` | — |

Per-step extras for π0: `suffix_embed` (the model's own suffix embedding: state and action input projections,
sinusoidal time embedding, the two-layer time MLP with its activation), `final_norm_out_proj` (final RMSNorm +
`action_out_proj` 1024 → 32 on 50 tokens), `euler_update` (x + dt·v on `[1, 50, 32]`).
**π0.5**: the same six layer segments at S = 50, plus whatever its adaptive RMSNorm adds (the dense layer on the
time-conditioning vector and the scale / shift / gate path) — put those operations into the norm and residual
segments where the code executes them and list them in `segments.csv`; its suffix embedding differs from π0's
(no state token; time MLP feeds the norms) — derive it. **SmolVLA** (16 expert layers, self- and cross-attention
layers, hidden 720) and **GR00T N1.7** (16 DiT blocks, hidden 1536, S = 41; adaptive LayerNorm, cross-attention
to the backbone features, its own MLP activation, state / action encoders and decoder): the research side has
**not** established their segment lists — derive them from the model code with the §13.2 definition ("everything
that is not a large linear layer or an attention matrix product"), one list per layer kind, with shapes read
from the loaded model, and name them by analogy with the π0 names.
Time every segment and extra in the three styles and timings of §13.1 (`eager`, `compiled`, `cudagraph`;
`wall_isolated` with `gap_ms` ∈ {0, 2}, `gpu_event`, `wall_launch`), at layer 0 and a middle layer. Report per
segment: median / p5 / p95 ms, kernels per call, analytic bytes read + written and FLOPs (yours from the real
shapes, next to the formulas above where they differ), achieved GB/s and TFLOP/s; per layer **`t_host_layer`**
per style as `sum_of_segments`, `sequence` and `whole_layer_graph`; per step `t_step_extra`; and
`t_launch_per_kernel`. Result directory `results/<date>_E11a_<model>_hostseg/`.

**Part 2 — small-transfer and synchronisation latency** (protocol §13.3; model-independent, measured once, in
the π0 task; later models add only their exact boundary sizes). Sizes {4 KB, 16 KB, 64 KB, 256 KB, 1 MB, 2 MB,
4 MB, 16 MB} + the exact sizes of part 3 and of the π0 score matrix (fp32 and fp16), methods `pageable`,
`pinned`, `zero_copy`, both directions; fit t = a + b·bytes per method and direction; the five synchronisation
costs; whether the waiting CPU spins. Result directory `results/<date>_E11b_xfer/` with `xfer_summary.csv`
(`method, direction, size_bytes, median_ms, p5_ms, p95_ms, n`) and `xfer_fit.csv` (`method, direction, fit_a_ms,
fit_b_ms_per_MB, fit_r2, GBps_from_b, n_points`), `sync_summary.csv`.

**Part 3 — the hand-over object** (protocol §13.4). `handover.csv` (per layer shape, dtype, stride, bytes,
valid vs padded tokens, total), `t_kv_materialise` for the `stacked` and `v_transposed` layouts, `kv_reuse.md`
(computed once or per step; referenced or re-copied per step; bytes copied per step — with code lines and
run-time evidence), `boundary.csv` (tensors crossing the prefix → expert boundary per inference and per step).
Cross-check for π0: 18 layers × 2 × P × 1 × 256 × 2 B ≈ 15.0 MB at P = 816. For GR00T the object is what its
backbone hands to the DiT (features rather than a KV cache, if that is what the code does): describe it as it
is, and say whether the cross-attention K / V projections of it are recomputed at every step.
Result directory `results/<date>_E11c_<model>_handover/`.

**Part 4 — prefix latency as a function of the input.** Sweep the number of camera images {1, 2, 3} (as many
as the model's input format accepts; say how a missing camera is represented — dropped, or a masked slot that is
still encoded and still occupies prefix tokens) × prompt length {short ≈ 3–5 words, typical = the fixture's
instruction, long ≈ the model's maximum prompt length}; record the prompt's token count, the image token count,
P as processed and P valid (models that pad the prompt to a fixed length will show a constant processed P —
report that, it is a result). Per point, a full inference at the model's default N, eager and the best compiled
mode (warm up each shape until no recompilation; record compile time per shape): `vision`, `vlm_prefix`,
`ae_step` (instrumented in eager, slope not required), `t_e2e`, P, peak memory. Fit `vision` vs number of images
and `vlm_prefix` and `ae_step` vs processed P (linear; give the residuals, and say if a line is a poor model).
Result directories `results/<date>_E11d_<model>_<mode>_prefixsweep/` with `prefix_sweep.csv`
(`n_images, prompt_label, prompt_tokens, image_tokens, P_processed, P_valid, region, median_ms, p5_ms, p95_ms, n`)
and `prefix_fit.csv`.

**Part 5 — the GPU under memory pressure, and the CPU** (protocol §13.5–13.6). Prefix-only loop (the E7 loop; if
E7 has not run yet, write it here and E7 reuses it): latency per region and throughput alone, then concurrently
with `gpu_hog` and `cpu_hog` at duty cycles 25 / 50 / 75 / 100 % and with the two controls at 100 %; achieved
background GB/s alone and during the run; the slowdown curve `prefix_slowdown` vs achieved background GB/s.
Optional: the same for the action-expert-only loop and for the per-layer host sequence of part 1
(`host_layer_slowdown`). **This is a proxy for PIM activity on the same DRAM; repeat the limits of protocol §13.5
in the report.** CPU utilisation per phase (§13.6): launching thread, process, and per core, for `vision`,
`vlm_prefix`, `ae_total`. tegrastats is logged in every loop (protocol §4), which also gives `p_mean_W` per phase
and rail. Result directory `results/<date>_E11e_<model>_<mode>_mempressure/` with `mempressure.csv`
(`load_label, duty_pct, bg_GBps_alone, bg_GBps_during, region, median_ms, p5_ms, p95_ms, n, throughput_inf_per_s,
slowdown`), `cpu_util.csv` (`phase, scope, pct, n`), `power.csv`.

**Part 6 — pipelining numbers** (protocol §13.7). (a) `prefix_throughput`, `ae_throughput`, `e2e_throughput`
from back-to-back loops, eager and best compiled. (b) The GPU idle time inside `ae_total` of the ordinary run —
`gpu_idle_ae_total` = region time − union of kernel intervals from the E5 nsys trace of the same configuration
(restated per §13.8 if E5 exists; otherwise take the 1–3-inference nsys capture of protocol §3.1 here and let E5
reuse it) — stated as "GPU time available for the next frame's prefix". (c) The pipelined case itself: the
prefix-only loop running **while** the per-layer host sequence of part 1 is issued the way the target system
would issue it (segments separated by a 2 ms pause, 18 × N layers per inference for π0; the model's own layer
count and default N otherwise), as two processes and as two CUDA streams in one process: latency distribution of
the host segments and `t_host_layer` alone vs under the prefix load (`host_layer_slowdown`, key `prefix_loop`),
and the prefix slowdown (`prefix_slowdown`, key `host_segments`). Result directory
`results/<date>_E11f_<model>_pipelining/`.

**Part 7 — `model_inputs.csv`.** Every E11 result directory carries a `model_inputs.csv` with the rows it can
support from its own files (protocol §13.8), checked with `docs/reference_scripts/check_model_inputs.py`. One more
directory per model, `results/<date>_E11_<model>_inputs/`, holds the **consolidated** file: all rows of parts
1–6, plus the region latencies of E3 / E4 (`t_preprocess` … `t_ae_total`, `t_e2e`, for N = 4 and the default N,
eager and best compiled, instrumented and slope) restated from their result directories by the copy rule of
§13.8. Report: `reports/E11_<model>_hybrid_inputs.md` — the table of quantities, what differs from the analytic
assumptions listed in `docs/00_context.md`, the limits of part 5, and "not measured" items with the failing
command.
