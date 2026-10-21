# 01 — Measurement plan (Thor)

The experiments keep the Orin study's E-ids, so every Thor result has an Orin partner with the same name
(`reference/orin/`). Follow `02_protocol.md` throughout. All six models go through every core experiment:
π0, π0.5, SmolVLA, GR00T N1.7, N1.6, N1.5 (pins in `00_context.md`), in this order inside each task.

**Task order.** Take the first open task in this table whose dependencies are done. Within the core (T001–T009),
do not skip ahead past a blocked task's dependants; work on any other open task instead. A new task is added as
a new row.

| order | task | experiments | depends on | priority |
|---|---|---|---|---|
| 1 | T001 setup and availability | stack, tools, access, checkpoints, one inference per model, nsys secrets re-test | — | P0 |
| 2 | T002 platform record | E0 | T001 | P0 |
| 3 | T003 peaks and expert kernel sweep | E1 → `hw_constants.json`, E2 | T002 | P0 |
| 4 | T004 baseline latency, all models | E3 (eager, N sweep), E4 stock compile modes (π0), `floor.csv` | T003 | P0 |
| 5 | T005 optimisation ladder and best BF16 | E4 Part A, Part B ladder, E4i search, E4d phase balance — **deliverable 1** | T004 | P0 |
| 6 | T006 roofline | E5 nsys captures, E6 per part and per kernel, C3 expert dimensions — **deliverable 2** | T005 | P0 |
| 7 | T007 timelines and phase bounds | E5 timeline analysis, E4g C1/C2 ncu per part — **deliverable 3a** | T006 | P1 |
| 8 | T008 per-kernel bottlenecks | E5b SASS (expert, prefix, vision), E4h, E5c — **deliverable 3b** | T007 | P1 |
| 9 | T009 core synthesis | `reports/SYNTHESIS_thor.md`, Thor vs Orin | T005–T008 | P1 |
| 10 | T010 power and energy | E7 + E7v at the current mode | T005 | P2 |
| 11 | T011 LIBERO closed loop | E8, E8s | T005 | P2 |
| 12 | T012 hybrid-model inputs | E11 a–f + `model_inputs.csv`, all models | T005, T007 | P2 |
| 13 | T013 co-run | E12, E12b (π0) | T005 | P2 |
| 14 | T014 prefix bandwidth and savings | E13, E14, E14b | T013 | P2 |
| 15 | T015 full synthesis and repository check | all | T009–T014 | P2 |

**Dropped from the Orin plan, and why.** Every item below needs a power-mode switch, and on Jetson a switch that
changes the GPU configuration needs a reboot, which this campaign never asks for: Orin's T013 (E0 again after a
mode change), E7's capped modes (`MODE_50W`, `MODE_30W`, `MODE_15W` and their Thor equivalents), Orin's T004
step 4b (latency in capped modes) and T014 Step 4 (the co-run at a reduced SM count in a capped mode). Thor runs
every experiment in one power mode, the maximum-performance one (protocol §1). Orin's E9 (precision) and E10
(TensorRT) were optional and are not scheduled; FP8 appears only as a recorded peak in E1.

## E0 — Platform record
Everything in protocol §1 at the **current** power mode (never switched), plus `free -h`, storage, an idle
power-monitor snapshot, the list of `nvpmodel` modes with their caps (read from `/etc/nvpmodel.conf` and
`nvpmodel -q --verbose`, never by switching), CPU topology, GPU properties (SM count, L2 size, shared memory per
SM, max warps per SM, clock range), and the static half of `hw_constants.json` (protocol §8).

## E1 — Peak microbenchmarks (the roofline ceilings)
1. Bandwidth as the GPU sees it: device-to-device copy (`tensor.clone()` / `copy_`) of ≥ 1 GiB at several sizes; a
   bandwidth-bound elementwise kernel (`y = a*x + y`, 1 GiB bf16); a read-only reduction (`x.sum()`, 1 GiB) —
   GB/s counted as bytes read + written. Repeat each ≥ 5 times across ≥ 2 processes and give the spread (the Orin
   saw up to 1.19× between processes, `reference/orin/reports/E1_E2_microbench.md`).
2. Compute: square GEMM n × n, n ∈ {1024, 2048, 4096, 8192, 16384 if memory allows}, in fp32 (`highest`), TF32,
   bf16 and fp16 (`torch.matmul`), int8 (`torch._int_mm`) if available, and FP8 (`torch._scaled_mm`) **as a
   record only** (no model uses it). Achieved TFLOP/s; the best n per dtype is the measured peak.
3. `hw_constants.json` (protocol §8): the static fields of E0 plus the measured peaks and the ridge per dtype.
   Output also `latency_summary.csv` (`kernel, dtype, shape, bytes, flops, ms, GBps, TFLOPps`).

## E2 — Action-expert kernel sweep
For every linear layer of the π0 expert, time `y = x @ W.T` (x `[S, K]`, W `[N, K]`, bf16,
`torch.nn.functional.linear`) for **S ∈ {1, 2, 4, 8, 16, 32, 51, 64, 128, 256}**: q 1024→2048, k 1024→256,
v 1024→256, o 2048→1024, gate / up 1024→4096, down 4096→1024. Also attention for S = 51 queries (8 heads × 256)
against P + S = 867 keys with one KV head (SDPA), RMSNorm, RoPE, GeLU, softmax and an elementwise add at these
sizes. Per op: kernel time (CUDA events over 1,000 calls), time per call issued back to back from Python (launch
overhead), the same under CUDA-graph replay, achieved GB/s and FLOP/s, the kernel name cuBLAS picked, and the ratio
to the bandwidth floor at the **measured** peak of E1. Questions: at which S does a layer stop being
launch / latency-bound; what does launching one small kernel cost on this CPU (µs); how much does graph replay save.

## E3 — End-to-end latency, eager
Each model, the fixture input, eager. Sweep **N ∈ {1, 2, 4, 6, 8, 10}** with the model's own N parameter (GR00T:
also its default 4 is in the list). Per region (protocol §2): median ms and share of total; ms per denoising step
both ways (instrumented and slope) and whether steps are equal; P (processed and valid) and S; peak GPU memory;
control frequency (Hz).

## E4 — `torch.compile`, and the optimisation ladder
- **Stock modes (π0):** `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`; the action
  expert alone and the whole policy. Compile time, graph breaks (`TORCH_LOGS=graph_breaks`), recompilations,
  steady-state latency, ms per step by slope, peak memory, whether CUDA graphs were used. Record whether Inductor
  logs "Not enough SMs to use max_autotune_gemm mode" (its `is_big_gpu` check compares the SM count with a fixed
  minimum; read the threshold from the installed torch) and present `max-autotune` for what it really did.
- **Part A:** the published characterisation's configuration (eager + TF32, the model's own N) with a matched fp32
  control (`reference/orin/bench/paper_repro.py`).
- **Part B, the ladder** (`reference/orin/bench/opt_ladder.py`): each rung changes exactly one thing from the rung
  below — `eager_fp32`, `eager_tf32`, `bf16_base`, `ae_compile`, `ae_graphs`, `vis_compile`, `vis_graphs`,
  `vlm_compile`, `vlm_graphs`, `whole_graphs`, `bf16_vision`, `bf16_norms`, `bf16_both`; GR00T's ladder as on the
  Orin (`reference/orin/bench/groot_phases.py`), **cumulative** (vision and LLM compiled together, the defect the
  Orin's first GR00T ladder had). A rung that does not apply to a model is recorded `not applicable` with the reason.
- **E4i, the combination search:** combine the winning factors per model — bf16 everywhere it is legal, compile
  and CUDA graphs on each part, forced GEMM autotuning (`all_compile_autotune`) — and keep the fastest
  configuration **whose weights and activations are BF16**: that is `best_bf16`. Also report the configuration that
  maximises the expert's share. Every combination is checked for finite outputs and against eager bf16 actions.
- **E4d:** the per-model phase-balance table (vision / prefix / expert / other, ms and %) for `eager_fp32` and
  `best_bf16`, and every rung, at the model's own N.

## E5 — Kernel timeline (nsys) and per-part counters
For each model, `eager_fp32` and `best_bf16`, at the model's default N: nsys per protocol §3.1 (1–3 steady-state
inferences, `--cuda-graph-trace=node` for graphed modes): kernels per region and per step, summed kernel time vs
region time, GPU idle = region − union of kernel intervals, gap histogram, top-20 kernels per region with grids,
memcpy time, the launching CPU thread. **Every** kernel is kept in the per-kernel tables (no top-N cap).

## E4g — C1 / C2: what each part is bound by (ncu)
C1: per model × {`eager_fp32`, `best_bf16`} × {vision, prefix, one denoising step}: ncu with the sections and
metrics of protocol §3.3 on **every kernel of the part** (chunked with `-s` / `-c` when ncu's per-process limit
bites), duration-weighted per part: SM and memory throughput, occupancy, tensor-pipe utilisation, the full stall
breakdown, the bound verdict. C2: the eager → best change per part, counters only. Captured at the **exact**
`best_bf16` configuration, through the same compiled callables that are timed (the Orin's C1 "best" was a
different rung from E4i's best; never again).

## E6 — Roofline points (per part and per kernel)
Per model × mode × part, and per kernel: FLOPs and DRAM bytes, analytic (from shapes) **and** measured (ncu
counters, protocol §3.3), time from nsys (kernels) or §2 regions (parts), never from ncu `Duration`. Output the
plottable CSV of protocol §5 (`roofline.csv`) against the **measured** ceilings of `hw_constants.json`, plus the
floor and gap ratio of protocol §9.

## E5b — Instruction-level analysis
Full procedure: `03_instruction_level_analysis.md`. Targets: the dominant kernels of **each part** (expert, prefix,
vision) in both modes, covering ≥ 60 % of the part's kernel time. SASS for this device's own architecture
(T001). Question: what do the dominant kernels of each part wait for, and how does that change with S and under
compilation?

## E4h — prefix kernels, and E5c — the expert's component split
E4h: the VLM prefix's kernels by class (GEMM by shape, attention, norm, elementwise, copy) with time, FLOP/s and
bytes per class, eager and best, and the same for vision. E5c: one denoising step split into projections,
attention, host segments and per-step extras (`reference/orin/bench/e5_expert_components.py`).

## E7 — Power and energy
Per protocol §4, current mode only: mean power per rail for idle, vision-only (E7v), prefix-only, expert-only and
full loops (≥ 60 s each, timed window only), `eager_fp32` and `best_bf16`, every model; energy per inference, per
part and per denoising step, total and above idle, and a reconciliation (parts vs full loop, residual < 5 % or the
reason).

## E8 — LIBERO closed loop, and E8s — the seeded N sweep
LIBERO at the pinned commit, `MUJOCO_GL=egl`. E8: success on `libero_spatial`, `libero_object`, `libero_goal`,
`libero_10` (10 tasks × 10 episodes), `best_bf16` vs `eager_fp32` (π0 first), policy vs simulator time, control
Hz. E8s: noise from a `torch.Generator` seeded per (suite, task, episode), N ∈ {1, 2, 4, 10} paired, with the
repeat control (Orin: `reference/orin/tasks/T016_prefix_bandwidth_and_followups.md` step 2).

## E11 — Inputs for the hybrid GPU + PIM model
As on the Orin (`reference/orin/specs/orin_02_protocol.md` §13 for the history; the binding text is protocol §13
here): host segments (E11a), transfers and synchronisation (E11b), the hand-over object (E11c), prefix latency vs
input (E11d), the GPU under memory pressure and CPU utilisation (E11e), pipelining numbers (E11f), and the
consolidated `model_inputs.csv` per model. All six models, in bf16: eager (the rung `bf16_both`'s precision
without compilation, or the model's shipped bf16 path; say which) and `best_bf16`.

## E12 / E12b — co-running the prefix and the action expert (π0)
Specifications `reference/orin/specs/E12_corun/README.md` and `reference/orin/specs/E12b_compiled_corun/README.md`
with the Thor adaptations of T013 and protocol §14–§15: the pipelined two-stream co-run, then both stages together
inside green-context SM partitions, with splits derived from **this device's** SM count.

## E13, E14, E14b — the prefix under stolen bandwidth, prefix savings, the padding trim
Specifications `reference/orin/tasks/T016_prefix_bandwidth_and_followups.md` (E13 a/b/c/d, E14) and
`reference/orin/tasks/T017_pi0_padding_trim.md` (E14b), with the Thor adaptations of T014 and protocol §16–§17.
