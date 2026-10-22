# E1 / E2: this Thor's ceilings, the counters that place kernels against them, and the π0 expert's kernels (T003)

Jetson AGX Thor, MAXN (id 0), GPU locked at 1575 MHz, EMC 4266 MHz, 20 SMs, `jetson_clocks` + fan. The protocol §1 gate
(including GPU idle from the system-wide `nvidia-smi` utilisation) passed before every run. 2026-10-22.

Results:
- `results/20261022_E1_peak/`: ceilings, `hw_constants.json`, `validation.csv`;
- `results/20261022_E1_peak_backtoback/`;
- `results/20261022_E1_state_probe/`;
- `results/20261022_E2_ae_kernels_v3/` (authoritative; v1 superseded, v2 = fast state);
- `results/20261022_E1_bw_gemm_stream/`;
- `results/20261022_E1_load_latency_v2/` (v1 superseded).

## The answer first: the ceilings (`results/20261022_E1_peak/hw_constants.json`)

| ceiling | Thor measured | at, mode | NVIDIA spec (Thor) | Orin measured | Thor ÷ Orin |
|---|---|---|---|---|---|
| DRAM bandwidth, best (read-only `x.sum()`) | **258.23 GB/s** | 1 GiB | 273 GB/s (94.6 %) | 187.47 GB/s (axpy) | 1.38 |
| bandwidth, axpy / copy (SM) | 233.69 / 236.10 GB/s | 1 GiB | | 187.47 / 164.47 | 1.25 / 1.44 |
| bf16 dense GEMM | **129.79 TFLOP/s** | 2048³, isolated calls | not published | 33.85 | 3.83 |
| fp16 | 129.82 TFLOP/s | 2048³, isolated | not published | 32.15 | 4.04 |
| tf32 | 62.37 TFLOP/s | 1024³, isolated | not published | 15.96 | 3.91 |
| fp32 (`highest`, SIMT) | 6.427 TFLOP/s | 16384³, back to back | 8.06 derived (2,560 cores × 2 × 1.575 GHz), 80 % | 3.48 | 1.85 |
| int8 (`torch._int_mm`) | 330.6 TOP/s | 8192³, back to back, fastest 0.5 s slice | not published | 10.50 (not a hardware peak there) | — |
| fp8 e4m3 (record only) | 302.9 TFLOP/s | 4096³, isolated | only "2070 TFLOPS (FP4—Sparse)" | — | — |
| **ridge, bf16** | **502.6 FLOP/B** | | | 180.6 | 2.78 |
| ridge fp32 / tf32 / fp16 / int8 | 24.9 / 241.5 / 502.7 / 1280.3 | | | | |

Sources:
- Spec: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/ (accessed 2026-10-22). The core
  count used in the derived fp32 figure is from the same page.
- Orin: `reference/orin/reports/E1_E2_microbench.md` ("The answer first", E1 tables; Orin, MAXN).

**Definitions and their biases:**
- **Bandwidth:**
  - Value: (bytes read + written) ÷ the median per-call time of one repeat of 100 calls.
  - `best`: the highest of 30 repeat medians (3 sizes × 5 repeats × 2 processes). That is an optimistic estimator by
    construction; the median over all repeats of the same kernel is **250.32 GB/s**, about 3 % lower.
  - The two processes agree within ×1.017 for the read kernel. axpy differs ×1.026 and fill ×1.104.
  - Mixed read / write kernels reach 234–236 GB/s, so a read-only ceiling slightly flatters them.
- **Compute:** the highest rate measured in either of two modes.
  - **Isolated calls:** E1 synchronises after every call and reads the OC counters, so the GPU idles between calls.
    The rate is the median of the calls before any over-current event.
  - **Back to back:** `results/20261022_E1_peak_backtoback/`, distinct A and B, 30 s in 0.5 s slices.
  - **Sustained:** the back-to-back median from the first OC event onward, listed per dtype.

| dtype, shape | isolated calls | back to back, first slice | back to back, fastest slice | sustained |
|---|---|---|---|---|
| bf16 2048³ | **129.79** | 111.3 | 111.3 | 104.3 |
| fp16 2048³ | **129.82** | 109.1 | 109.1 | 95.2 |
| tf32 1024³ | **62.37** | 61.8 | 61.9 | 61.8 (no OC) |
| fp32 16384³ | 6.425 | 6.43 | **6.427** | 6.43 (no OC) |
| int8 8192³ | 258.4 | 258.0 | **330.6** | 313.1 |
| fp8 4096³ | **302.9** | 197.0 | 197.0 | 192.7 |

## Three device properties that a single peak hides
1. **The ≈ 100 W rail limit** (T002).
   - VDD_GPU is current-limited at 5,060 mA. Back-to-back heavy GEMMs hit it and soctherm `oc3` throttles them; devfreq
     doesn't show it.
   - bf16 2048³ drops from 111.3 to 104.3 TFLOP/s.
   - The T002 probe (4096³, `x @ x`) drops from 99.5 to 93.97 TFLOP/s and keeps falling as the die warms.
2. **Tensor and memory-path throughput are state-dependent** (`results/20261022_E1_state_probe/`).
   - After about 55 s of sustained heavy load (GPU ≈ 75 °C), a cold weight stream (the π0 expert's dominant pattern)
     speeds up from ≈ 50 to ≈ 34 µs per 8 MiB weight (166 → 245 GB/s). The state lasts a few seconds after the load
     ends.
   - The int8 GEMM jumps the same way, 258 → 313–331 TOP/s, at the same 1575 MHz.
   - No clock in the kernel clock tree, BPMP `emc` / `mcf` or devfreq changes. The mechanism is not identified (ncu
     exposes a "Tensor Core boost state" option; that is a hypothesis only).
   - Isolated-call and back-to-back rates differ (bf16 129.8 vs 111.3; fp8 303 vs 197) for reasons this run does not
     resolve.
   - Every later timing run classifies its state with `bench/gate.py: memory_state_probe()` (question in `STATUS.md`).
3. **GEMMs larger than the L2 slow down, and not because of bandwidth.**
   - bf16 falls from 130 TFLOP/s at 2048³ to 82 at 4096³ and 58 at 16384³; tf32 from 62 to 22–27; fp8 from 303 to
     99–101.
   - The 4096³ bf16 GEMM re-reads its operands from DRAM (3.57× A + B, `validation.csv`) but draws only 158 GB/s,
     61 % of 258, at an intensity (503 FLOP/B) on the ridge.
   - It is therefore **not** bandwidth-bound, and the cause is not established. An earlier version of this report
     said "bandwidth-limited"; that is retracted.
   - Only tf32 4096³ (209 GB/s at 230 FLOP/B) plausibly sits on the memory roof. No DRAM traffic was measured at
     16384³.

## Counter validation (`results/20261022_E1_peak/validation.csv`; ncu 2025.3.0, form D, cold caches)

| counter | kernels (known traffic / work) | measured ÷ analytic | used by the roofline |
|---|---|---|---|
| `lts__d_sectors_fill_sysmem.sum` × 32 B | SM copy, axpy, sum, GEMV 8192² | **1.0001–1.0005** | DRAM read bytes |
| `lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum` × 32 B | same | 1.0002–1.0004 | cross-check |
| `lts__t_sectors_aperture_sysmem_op_write.sum` × 32 B | SM copy, axpy (2 kernels) | **1.0006–1.0007** | DRAM write bytes |
| SIMT FP: fadd + fmul + 2·ffma (`smsp__sass_thread_inst_executed_op_*_pred_on.sum`) | fp32 SIMT GEMM 4096³ | **1.0001** | SIMT FLOPs |
| tensor: Σ `sm__ops_path_tensor_src_*.sum` | tf32, int8, fp8 GEMM 4096³ | **1.000** | tensor FLOPs |
| | bf16 / fp16 GEMM 4096³ | 1.0078 = 4128 / 4096 | executed, tile-padded work |
| | x [51, 1024] · W [4096, 1024]ᵀ | 1.098 = 56 / 51 | |
| | GEMV x [1, 8192] (split-K kernel) | 8.125 (M = 1 padded to 8) | |
| `mcc__dram_throughput_*` (memory controller) | all | 0 on every kernel | not usable |

Notes:
- There are no `dram__*` metrics on GB10B.
- The tensor counter counts executed MMA work, padding included. Which tile dimension is padded is inferred from the
  kernel names and the exact ratios, not shown in SASS. T006 reports measured and useful FLOPs side by side; for M = 1
  rows the measured tensor FLOPs are 8× the useful ones.
- The write counter is validated on 2 kernels only (copy, axpy).
- The GEMV's split-K partials write 16.8× its output bytes. That is real workspace traffic.

## E2: the π0 action expert, kernel by kernel (`results/20261022_E2_ae_kernels_v3/`, normal memory-path state)

Real layer weights from the checkpoint (18 layers, 33.0 MiB per layer):
- **kernel time:** CUDA-graph replay of whole passes over ≥ 3× L2 of weight copies (includes the gap between graph
  nodes);
- **issue time:** back to back from Python;
- **floors:** weight bytes ÷ 258.23 GB/s (`kernels_floors.csv`); activations stay in L2 when calls run back to back.

| at S = 51 | q | k | v | o | gate | up | down | per layer | per step (×18) |
|---|---|---|---|---|---|---|---|---|---|
| kernel, cold (µs) | 20.1 | 4.1 | 4.1 | 18.5 | 49.6 | 50.0 | 36.4 | 182.9 | **3.29 ms** |
| kernel, L2-resident (µs) | 5.5 | 3.0 | 3.0 | 6.0 | 9.2 | 9.1 | 9.9 | 45.7 | 0.82 ms |
| issued from Python (µs) | 20.9 | 12.4 | 12.4 | 19.1 | 49.9 | 50.2 | 37.1 | 202.0 | 3.64 ms |
| ÷ weight floor | 1.24 | 2.03 | 2.03 | 1.14 | 1.53 | 1.54 | 1.12 | | 1.37 (floor 2.41 ms) |
| weight GB/s | 208 | 127 | 127 | 227 | 169 | 168 | 230 | | |
| same, **fast state** (v2) (µs) | 18.0 | 4.1 | 4.1 | 17.7 | 34.1 | 33.9 | 33.1 | 145.1 | 2.61 ms |

Orin, same experiment (`reference/orin/reports/E1_E2_microbench.md` items 3 and 5): 4.41 ms per step as one CUDA graph,
8.66 ms issued from Python, 1.33× the memory floor. Thor: 3.29 ms kernel time and 3.64 ms issued in the normal state;
2.61 ms in the fast state.

**The plan's three E2 questions:**
1. **At which S does a layer stop being launch- or latency-bound?**
   - q, o, gate, up and down are weight-streaming-bound at every S from 1 to 256. Kernel time is flat to S = 64 (gate
     48.3–49.6 µs) and issue ≈ kernel.
   - gate / up get faster at S = 128–256 (45.2, 37.7 µs). cuBLAS's tile N grows with S: 256×8 at S ≤ 8, then
     256×16 / 32 / 56 / 64, 448×64 at 128, 144×128 at 256.
   - None becomes compute-bound by S = 256: gate's bf16 compute floor at S = 256 is 16.5 µs.
   - k and v (0.5 MiB) are **launch-bound at every S**: 4.1 µs of kernel vs 12.4 µs to issue. Why they reach only
     49 % of bandwidth even in a graph is not established: graph node gaps (3.0 µs hot) may dominate.
   - Attention against 816 + S keys is flat to S ≈ 32 (19.8–21.9 µs): 23.9 µs at 51, 45.7 at 128, 65.8 at 256.
2. **What does launching one small kernel cost on this CPU?**
   - 6.3 µs per empty kernel issued back to back; 16.4 µs between one event pair.
   - Smallest GEMM: 12.1 µs. q_proj chain fit at S = 51: 12.5 µs per call.
   - Unfused single ops are issue-bound at S = 51: RMSNorm 65.1 µs issue vs 16.4 µs kernel; RoPE 45.1 vs 17.8.
3. **How much does graph replay save?**
   - Projections: −0.04 to 1.41 µs per call (per layer at S = 51, 202.0 issued vs 182.9 kernel, −9 %).
   - k / v: 8.0–8.4 µs each (−67 %).
   - RMSNorm / RoPE: 27–49 µs each.
   - The expert step's dispatch overhead is in its non-GEMM operators.

A layer's weights (33 MiB) just exceed the 32 MiB L2. Resident, the projections would take 45.7 µs per layer instead of
182.9.

## Other measurements
- **Load latency** (`results/20261022_E1_load_latency_v2/`, pointer chase, cycles at 1.575 GHz):
  - L1 ≈ 40 cycles (25 ns);
  - L2 ≈ 255 cycles (162 ns) from 1 to 24 MiB;
  - **DRAM ≈ 817–832 cycles (≈ 520 ns)**: L2 flushed, cold random walk, repeats within 2 %.
  - v1's 387–549 cycles were partly L2 hits (every repeat walked the same 25.6 MB) and are superseded.
- **Skinny-GEMM weight streaming** (`results/20261022_E1_bw_gemm_stream/`): over 1 GiB of distinct weights, weight
  bytes ÷ time is 172–219 GB/s (67–85 % of 258), consistent with E2 in the normal state. Counting the L2-resident
  activations as DRAM bytes would give up to 293 GB/s, above the 273 GB/s spec. That accounting error was mine, and
  is corrected in E2's floors.
- **E2 ncu check** (`results/20261022_E2_ae_kernels/ncu/`): with a cold L2 each projection reads its weight bytes
  (1.003×). Isolated, gate S = 256 takes 52.96 µs with real weights and 47.52 µs with random weights of the same shape
  (both quoted).

## Anomalies and confidence
- High confidence: bandwidth ceiling, counter validation (0.1 %), E2 in the normal state (v1 and v3 agree within 9 %),
  and DRAM / L2 / L1 latency.
- Lower confidence: what a "peak" means for tensor GEMMs on this device. Rates depend on the call pattern and on a
  hidden state. `hw_constants.json` lists every mode, and T006 should quote the mode beside any tensor ceiling.
- Not established: the cause of the large-GEMM slowdown, the cause of the state switch, and why small projections
  stream at 49 %.
- Not measured: power during E1 (T010).

## Reproduce
```bash
source env/thor_env.sh
for p in p1 p2; do for part in bw gemm; do ~/venvs/base/bin/python bench/micro_peak.py --part $part --proc-tag $p --out results/<run>; done; done
for s in bf16:2048 fp16:2048 tf32:1024 fp32:16384 int8:8192 fp8_e4m3:4096; do sleep 30; ~/venvs/base/bin/python bench/oc_throttle_probe.py \
  --operands distinct --dtype ${s%%:*} --n ${s##*:} --seconds 30 --out results/<b2b>/backtoback_${s%%:*}_${s##*:}.csv; done
for c in copy_sm axpy read_sum gemv skinny_bf16 gemm_{fp32,tf32,bf16,fp16,int8,fp8}; do
  env/rootdo.sh ncu --clock-control none --metrics "$(paste -sd, results/<run>/ncu/validation_metrics.txt)" \
    --nvtx --nvtx-include "val/" --profile-from-start off -o results/<run>/raw/val_$c -f ~/venvs/base/bin/python bench/ncu_validation.py --case $c; done
~/venvs/base/bin/python bench/parse_ncu_validation.py --dir results/<run>
~/venvs/base/bin/python bench/hw_constants_build.py --e1 results/<run> --t002 results/20261022_E0_platform_MAXN/hw_constants.json --b2b results/<b2b>
HF_HUB_OFFLINE=1 ~/venvs/lerobot/bin/python bench/micro_ae_kernels.py --hw results/<run>/hw_constants.json --out results/<E2>
~/venvs/base/bin/python bench/e2_floors.py --e2 results/<E2> --hw results/<run>/hw_constants.json
~/venvs/base/bin/python bench/state_probe.py --idle-s 20 --load-s 90 --recover-s 300 --period-s 2 --out results/<state>/state_probe_load90.csv
```

## Deviations
- **Compute ceiling.** It is the highest rate in either mode, with both listed (STATUS.md question). The plan's E1 is
  isolated calls; back-to-back probes were added after the independent review.
- **E2 kernel time.** CUDA-graph replay, not "CUDA events over 1,000 calls". That chain measures max(kernel, CPU issue),
  reported as `chain_ms` / `issue_ms`. "Cold" = ≥ 3× L2 of weight copies, replayed in whole passes.
- **E2 floors.** Weight bytes only (`kernels_floors.csv`).
- **Result-file names.** `latency_summary_{bw,gemm}_{p1,p2}.csv` per part and process instead of one
  `latency_summary.csv`; per-call samples in `raw/`. The "4-dtype run with matmul flags printed": the flags are in
  `raw/run_gemm_p*.json` (`matmul_flags`).
- **Superseded directories.** `results/20261022_E2_ae_kernels/` and `results/20261022_E1_load_latency/` (each has a
  `SUPERSEDED.md`).
- **Added beyond the task:** back-to-back ceilings, the state probe, the GEMM-stream addendum, and the E2 ncu check.

## Self-review (protocol §11)
| check | result |
|---|---|
| `hw_constants.json` complete: bandwidth with spread and repeat median, TFLOP/s fp32 / tf32 / bf16 / fp16 (+ int8, fp8) in both modes + sustained, ridge, counter validation, each with its CSV row | yes |
| `validation.csv`: ≥ 4 byte kernels (reads; writes on 2) and every dtype's GEMM for FLOPs | yes, with the write caveat |
| E2: every projection at every S, kernel names, launch cost | yes (v3) |
| ceilings line of `PROGRESS.md` ticked | yes |
| meta.json complete, raw files hashed, secrets scan clean, no file > 50 MB | yes (9 directories) |
| independent review (P0) | done; every finding fixed or disclosed above (ceiling modes, retracted bandwidth claim, latency v2, E2 v3, unit fix, wrong statements, selection bias, sources) |
