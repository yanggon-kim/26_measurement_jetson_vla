# 01 — Measurement plan

Priorities: **P0** = E0, E3, E4 for π0 · **P1** = E5 (nsys + ncu), E5b, E6, E7 for π0, and E2, E1 ·
**P2** = π0.5 and SmolVLA through E3, E4, **E5, E5b, E6, E7**, and E8 · **P3** = GR00T N1.7 through the same set,
E9, E10. The profiling experiments (E5, E5b, E6, E7) are **required for every model that runs**, not only for π0;
π0 goes first. First deliverable (end of P0): the ladder table for π0 —
*bandwidth floor 3.15 ms → measured eager → measured compiled*, in ms per denoising step — with conditions.
Before profiling any model, compute its **physical floor** per region (protocol §9).
Each experiment below lists what to run, the sweep, and the files we expect. Follow `02_protocol.md` throughout.

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
| kv_proj (k and v) | 1024 | 512 | 1.05 MB |
| o_proj | 2048 | 1024 | 4.19 MB |
| mlp_gate, mlp_up | 1024 | 4096 | 8.39 MB each |
| mlp_down | 4096 | 1024 | 8.39 MB |

Also attention for S = 51 queries (8 heads × 256) against P + S = 867 keys with one KV head (SDPA), RMSNorm, RoPE,
GeLU, softmax and an elementwise add at these sizes. For each: kernel time (CUDA events over 1,000 calls), the
**time per call when calls are issued back to back from Python** (this exposes launch overhead), achieved GB/s
(weight bytes ÷ time) and FLOP/s, and the **ratio to the bandwidth floor** (weight bytes ÷ 204.8 GB/s).
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
   `pipe_xu`, `pipe_adu`, `pipe_cbu`, `pipe_fp16`), **warp-stall reasons**, SM and memory throughput,
   the memory-bound / compute-bound / balanced split of kernel time. Never unfiltered; one step, not N steps.
   Outputs: `ncu/*_details.csv`, `ncu/*_raw.csv` (full exports), `ncu/columns.txt`, `ncu/kernels.csv`,
   `ncu/region_utilisation.csv`, and the raw `.nsys-rep` / `.ncu-rep` under `raw/` (protocol §3.5, §6).
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
analytic from shapes, **and** from ncu's SM / memory throughput where exported — ÷ measured time → achieved
FLOP/s and GB/s, arithmetic intensity (FLOPs per byte). Place them against the **Orin's own** roof (measured
peaks from E1 and the spec values, protocol §8 — not any other GPU's). Output a CSV we can plot:
`model, region_or_kernel, flops, bytes, ms, intensity, achieved_TFLOPps, achieved_GBps, pct_peak_compute, pct_peak_bw, source`
with `source` ∈ {`analytic`, `ncu`}, plus the floor and gap ratio of protocol §9.

## E7 — Power and energy
Per protocol §4 for each model (π0 first; eager and best compiled), at the paper's N and the model's default N: mean power per rail for idle, prefix-only loop,
action-expert-only loop, full inference loop; energy per inference and per denoising step, total and above idle.
Repeat the full-inference point at MAXN, 50 W, 30 W, 15 W with the latency at each mode. The power-mode sweep is required for π0;
for the other models MAXN is required and the capped modes are optional.

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
