# E2 π0 action-expert kernels, v3 — AUTHORITATIVE (T003 step 6) — 2026-10-22, MAXN, jetson_clocks, gate passed, 0 OC events, memory-path state normal

Supersedes `results/20261022_E2_ae_kernels/` (v1: the cold replay for q / o reused 4 weight copies within 16 MiB, and
its `floor_ms` counted activations). `results/20261022_E2_ae_kernels_v2/` is the same code measured in the *fast*
memory-path state (`results/20261022_E1_state_probe/`). Kernel time = CUDA-graph replay of whole passes over ≥ 3× L2 of
weight copies; floors = weight bytes ÷ 258.23 GB/s (`kernels_floors.csv`).

| at S = 51 | q | k | v | o | gate | up | down | per layer | ×18 = per step |
|---|---|---|---|---|---|---|---|---|---|
| kernel, cold (µs) | 20.1 | 4.1 | 4.1 | 18.5 | 49.6 | 50.0 | 36.4 | 182.9 | **3.29 ms** |
| kernel, L2-resident (µs) | 5.5 | 3.0 | 3.0 | 6.0 | 9.2 | 9.1 | 9.9 | 45.7 | 0.82 ms |
| issued back to back from Python (µs) | 20.9 | 12.4 | 12.4 | 19.1 | 49.9 | 50.2 | 37.1 | 202.0 | 3.64 ms |
| kernel ÷ weight floor | 1.24 | 2.03 | 2.03 | 1.14 | 1.53 | 1.54 | 1.12 | | floor 2.41 ms (594 MiB) |
| weight GB/s | 208 | 127 | 127 | 227 | 169 | 168 | 230 | | |

- Projection kernel time is flat in S up to 64 (gate 48.3–49.6 µs). gate / up get faster at S = 128–256 (45.2,
  37.7 µs). cuBLAS's tile N grows with S: 256×8 at S ≤ 8, 256×16 / 32 / 56 / 64, then 448×64 at 128 and 144×128 at 256.
- Replaying projections as a graph instead of issuing them from Python saves −0.04 to 1.41 µs per call. k / v save
  8.0–8.4 µs per call: they are launch-bound at every S (4.1 µs kernel vs 12.4 µs issue).
- At S = 51: attention against 816 + S keys 23.9 µs (19.8 at S = 1, 65.8 at 256); RMSNorm 16.4 µs kernel vs
  65.1 µs issue; RoPE 17.8 vs 45.1; softmax 14.1; GeLU 2.8; add 1.5.
- Launch cost: empty kernel 6.3 µs issued back to back (16.4 µs between one event pair); smallest GEMM 12.1 µs; q_proj
  chain fit 12.5 µs per call.
- Graph-replay kernel time includes the gap between graph nodes. k / v's 3.0 µs hot vs 4.1 µs cold point to a fixed
  per-node cost, so why the small projections reach only 49 % of bandwidth is not established.
