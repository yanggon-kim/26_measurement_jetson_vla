# E2 π0 action-expert kernels (T003 step 6) — 2026-10-22, MAXN, jetson_clocks, gate passed, 0 OC events

π0 expert (real layer weights from the checkpoint): hidden 1024, q 1024→2048, k / v 1024→256, o 2048→1024,
gate / up 1024→4096, down 4096→1024, 18 layers. Kernel time = CUDA-graph replay; "cold" cycles through weight copies
≥ 3× the 32 MiB L2; floors from `kernels_floors.csv` (weight bytes ÷ 258.23 GB/s; activations stay in L2).

| at S = 51 | q | k | v | o | gate | up | down | layer | ×18 = per step |
|---|---|---|---|---|---|---|---|---|---|
| kernel, cold (µs) | 19.7 | 4.1 | 4.1 | 18.4 | 48.5 | 48.2 | 38.0 | 181.0 | **3.26 ms** |
| kernel, hot / L2-resident (µs) | 5.5 | 3.0 | 3.0 | 6.0 | 9.1 | 9.1 | 9.9 | 45.6 | 0.82 ms |
| issued back to back from Python (µs) | 20.9 | 12.5 | 12.6 | 19.4 | 49.0 | 48.7 | 38.7 | 201.8 | 3.63 ms |
| kernel ÷ weight floor | 1.21 | 2.03 | 2.03 | 1.13 | 1.49 | 1.48 | 1.17 | | floor 2.41 ms (594 MiB) |

- The expert's projections are **weight-streaming-bound**, at 67–89 % of the measured bandwidth (k / v ≈ 49 %:
  0.5 MiB kernels are too small to keep enough loads in flight).
- L2 residency would cut a layer's projections from 181 to 46 µs. That would require a layer's 33 MiB of weights
  to fit in the 32 MiB L2; they do not. Only k / v are launch-bound (12.5 µs to issue vs 4.1 µs of kernel).
- gate / up at S = 1 and 51 stream at ≈ 175 GB/s, but at 219 GB/s at S = 256: cuBLAS picks a 256×56 tile at
  S ≤ 51 and 144×128 at S = 256.
- Launch cost on this CPU: an empty kernel issues in 6.4 µs back to back (16.5 µs with an event pair around it); a
  q_proj chain fit gives 12.9 µs per call at S = 51.
- Attention at S = 51 vs 867 keys (SDPA → flash split-KV) takes 23.9 µs of kernel time. RMSNorm (unfused: 16.5 µs
  kernel, 65.7 µs to issue) and RoPE (17.8 / 45.5 µs) are CPU-issue-bound as single PyTorch ops.
- `kernels.csv`'s own `floor_ms` counted activations as DRAM bytes. Several S = 256 rows fell "below the floor";
  `kernels_floors.csv` corrects that.
- ncu, with a cold L2 flush, confirms each projection reads its weight bytes from DRAM (1.003×) and runs 47.5 µs
  isolated (gate, S = 256) vs 38.4 µs back to back (`ncu/ncu_check_*`).
