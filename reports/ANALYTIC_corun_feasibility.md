# Can the prefix and the action stage share one Orin? — an analytic, resource-budget answer (owner request, 2026-09-28)

**Derived only; no new measurement.** Every demand and supply is read from a committed file by
`bench/analytic_corun.py`; the numbers are in `results/20260929_ANALYTIC_corun_feasibility/` (`demand.csv`,
`bounds.csv`, `rate_feasibility.csv`, `pi0_floors.csv`). All floors are **lower bounds** on the frame period.

## The answer

1. **Capacity is never the constraint in one process.** Parameters are 0.8–6.9 GB and the measured torch peak
   0.96–16.6 GB, 1.5–25 % of Orin's 65.9 GB. It only binds when a deployment duplicates the model: T014's two full π0
   processes were OOM-killed.
2. **Resources allow little overlap gain even with a perfect scheduler.** Resources add when two stages share the
   GPU, so the best co-run period is `max(ΣF / P, ΣM / BW)`, against `t_A + t_B` serial (each stage at its own
   roofline). The headroom this leaves is only **1.15–1.22× for π0 / π0.5**: their VLM prefill alone nearly fills
   the compute budget, and the expert's compute is small. It is **1.00–1.02× for SmolVLA and GR00T N1.7**, whose two
   stages are both bandwidth-bound and so compete for the same resource.
3. **The large measured serial-vs-overlap gaps come from inefficiency, not resources.** Measured, the stages run
   1.8–12× above their floors, so the GPU has idle resources a co-runner could use. π0's co-run experiments show what
   takes that room away (next section).
4. **At 10 Hz the resource budget excludes π0, π0.5 and GR00T N1.7 under any schedule.**
   - π0 needs 1.52× Orin's bf16 compute per second (VLM prefill over 816 tokens).
   - π0.5 needs 1.81×.
   - GR00T N1.7 needs 1.32× its bandwidth with measured bytes: its DiT re-reads 3.24 GB of weights per step.
   - **Only SmolVLA fits** (5 % compute, 14 % bandwidth). Its measured 81–89 ms frame is 6× above its 14 ms floor,
     so for SmolVLA the limit is efficiency, not resources.
   - At 2 Hz, the rate of π0's own chunk schedule, every model fits.

## Supply (Orin AGX 64 GB, MAXN, measured — `results/20260922_E1_peak/hw_constants.json`)

| resource | value |
|---|---:|
| bf16 tensor compute `P` | 33.85 TFLOP/s |
| DRAM bandwidth `BW` (best stream) | 187.47 GB/s (a hog reached 178.5 GB/s, E12b M4) |
| memory (shared CPU + GPU) | 65.9 GB |
| SMs | 16 |

## Demand per frame (best configuration of each model; N = 10 LeRobot, 4 GR00T)

| model | F_A prefix | F_B action | M_A prefix (analytic / measured) | M_B action (analytic / measured) | capacity params / peak alloc | measured A / B (E7v) |
|---|---:|---:|---:|---:|---:|---:|
| π0 `bf16_both` | 4.75 TFLOP | 0.38 TFLOP | 5.85 / **21.64** GB | 6.38 / 8.21 GB | 6.5 / 8.9 GB | 252.6 / 170.1 ms |
| π0.5 `bf16_both_graphs` | 5.52 TFLOP | 0.61 TFLOP | 5.85 / — GB | 10.94 / — GB | 6.9 / 16.6 GB | 293.1 / 183.7 ms |
| SmolVLA `vlm_graphs` | 0.07 TFLOP ¹ | 0.11 TFLOP | 0.61 / — GB | 2.03 / — GB | 0.8 / 0.96 GB | 38.8 / 42.1 ms |
| GR00T N1.7 `vlm_compile` | 0.61 TFLOP ¹ ² | 0.75 TFLOP ² | 3.05 / **12.96** GB | 12.96 / 11.77 GB | 6.3 / 6.5 GB | 155.9 / 81.6 ms |

Sources and derivations:
- **FLOPs** are E6's analytic count (2 × tokens × parameters, matrix work only); vision is from `vision_flops.json`.
  ¹ Vision FLOPs are not counted, so the compute floor is loose. ² GR00T's FLOPs are derived here with E6's convention,
  because E6 did not count them: language prefill over P = 273 tokens; per step, the DiT over S = 41 tokens plus the 16
  cross-attention blocks' to_k / to_v on the 273 backbone tokens (`results/20260928_E11c_groot_n17_handover`).
- **Analytic bytes** read the weights once per frame, and the expert's weights and KV cache once per denoising step
  (they do not fit in the 4 MB L2).
- **Measured bytes** are ncu graph-level DRAM bytes from `results/20260925_E12b_pi0_dram_bytes_MAXN` (π0) and
  `results/20260926_E14_groot_n17_savings_MAXN/bytes` (GR00T, the `ref` variant). π0's prefix moves 3.7× its analytic
  bytes (E12b / E13).

## Floors on the frame period

| model | bytes | serial floor `t_A + t_B` | co-run floor `max(ΣF/P, ΣM/BW)` | limited by | overlap headroom | measured serial `A + B` | measured ideal overlap `max(A, B)` |
|---|---|---:|---:|---|---:|---:|---:|
| π0 | analytic | 174.5 ms | 151.8 ms | compute | 1.15× | 422.7 ms | 252.6 ms |
| π0 | measured | 184.3 | 159.2 | bandwidth | 1.16× | 〃 | 〃 |
| π0.5 | analytic | 221.3 | 181.1 | compute | 1.22× | 476.8 | 293.1 |
| SmolVLA | analytic | 14.1 | 14.1 | bandwidth | 1.00× | 80.9 | 42.1 |
| GR00T N1.7 | analytic | 87.2 | 85.4 | bandwidth | 1.02× | 237.4 | 155.9 |
| GR00T N1.7 | measured | 131.9 | 131.9 | bandwidth | 1.00× | 〃 | 〃 |

**Reading.** The measured ideal-overlap column is what co-running would give if each stage kept its isolated speed.
It is always far above the co-run floor. So resources never forbid that overlap: the stages leave most of Orin idle.
Whether a real co-run gets there depends on how the idle resources are shaped, which the π0 section measures.

## Rate feasibility (`rate_feasibility.csv`) — necessary conditions

| model | 2 Hz (compute / BW needed) | 10 Hz | 20 Hz | capacity (peak alloc) |
|---|---|---|---|---:|
| π0 | 0.30 / 0.32 ✔ | **1.52** / **1.59** ✘ | 3.04 / 3.18 ✘ | 13.5 % |
| π0.5 | 0.36 / 0.18 ✔ | **1.81** / 0.90 ✘ | 3.62 / 1.79 ✘ | 25.3 % |
| SmolVLA | 0.01 / 0.03 ✔ | 0.05 / 0.14 ✔ | 0.11 / 0.28 ✔ | 1.5 % |
| GR00T N1.7 | 0.08 / 0.26 ✔ | 0.40 / **1.32** ✘ ³ | 0.80 / 2.64 ✘ | 9.9 % |

Values are fractions of Orin's supply; > 1 means no schedule can reach that rate. The worse of analytic or measured
bytes is shown. ³ With analytic bytes GR00T needs 0.85 of bandwidth at 10 Hz, so it passes on paper. The measured
bytes (the DiT's 4 × 3.24 GB of weight reads plus activation traffic) exceed the supply.

## π0 in detail: from the resource floor to the measured co-run (`pi0_floors.csv`)

These floors use the stage times the co-run experiments actually measured, not E7v's.

**Compiled** (`graphs_both`, fair expert, E12b: A_iso 246.3, B_iso 101.4 ms):

| floor | T | O = T / max(A, B) | what it assumes |
|---|---:|---:|---|
| fluid resources, measured bytes, 178.5 GB/s | 246.3 ms | 1.00 | compute and bandwidth are perfectly shareable: ΣM / BW = 167 ms < A |
| SM space, best measured split (12/4) | 267.7 ms | 1.087 | each stage confined to its SMs (measured per-partition times), no memory coupling |
| **measured** best co-run (green 12/4) | **307.1 ms** | **1.247** | reality: SM space + memory-system interference (E12b M4) |

The analytic resource budget would allow perfect overlap. SM space costs 1.09×, because the prefix needs 12 SMs
and the expert slows 2.5× on the remaining 4. Memory-system coupling costs a further 1.15×. This is the E12b result,
now decomposed against an analytic floor.

**Eager** (E12: A_iso 474.1, B_iso 523.8 ms):

| floor | T | O | what it assumes |
|---|---:|---:|---|
| one issuing thread in program order | 997.8 ms | 1.905 | the host-bound expert (E5: 84 % GPU idle, 969 kernels/step) occupies the single thread for ≈ B |
| **measured** in-process co-run | **994.9 ms** | **1.900** | — |
| two independent issuers, GPU work serialised | 607.8 ms | 1.160 | GPU busy per frame from E5 nsys: prefix 472.4 + expert 135.4 ms; stages no longer wait on one thread |

The eager measurement sits on the one-thread floor. If each stage had its own issuer, the floor would fall to about
608 ms: a 1.64× faster frame than serial, bounded by the prefix's GPU time plus the expert's kernel time. **That is
a prediction, and E12d is measuring it now.** E12d runs the expert as one CUDA graph of its eager kernels, and runs
two slim processes that hand the KV cache over through shared memory.

## Caveats

- These are necessary conditions. Passing a floor does not make a schedule achievable: π0's compiled co-run shows
  SM space and memory-system coupling adding 25 % above a floor of 1.00.
- Uncounted FLOPs (SmolVLA and GR00T vision) only loosen the compute floors. No verdict above changes: those two
  models are bandwidth-limited, not compute-limited.
- Analytic bytes assume weights are read once per frame (per step for the expert). Measured bytes are 1.3–3.7×
  higher, because GEMM tiles re-read weights and activations spill to DRAM (E13 / E14).

Reproduce: `python3 bench/analytic_corun.py --out results/20260929_ANALYTIC_corun_feasibility`.
