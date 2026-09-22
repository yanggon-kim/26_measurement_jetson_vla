# E11e — memory pressure and CPU (protocol §13.5, §13.6)

Run directory: `20260922b_E11e_pi0_reduce-overhead_mempressure`  
Written by `bench/mempressure.py` at 2026-09-22T14:46:46Z.

## Answer — what a memory-bandwidth competitor does to the prefix

| load | duty % | bg GB/s alone | bg GB/s during | % of 187.47 | median ms | p5 | p95 | n | inf/s | slowdown |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 |  |  |  | 339.530 | 339.250 | 340.666 | 177 | 2.943 | 1.0 |
| gpu_hog_25 | 25 | 53.525 | 26.762 | 14.3 | 421.173 | 420.418 | 422.385 | 143 | 2.374 | 1.2405 |
| gpu_hog_50 | 50 | 107.051 | 53.526 | 28.6 | 549.837 | 547.820 | 551.864 | 110 | 1.819 | 1.6194 |
| gpu_hog_75 | 75 | 133.817 | 60.634 | 32.3 | 608.176 | 563.090 | 630.387 | 100 | 1.661 | 1.7912 |
| gpu_hog_100 | 100 | 173.157 | 68.155 | 36.4 | 658.577 | 654.751 | 665.286 | 92 | 1.517 | 1.9397 |
| cpu_hog_25 | 25 | 54.291 | 47.493 | 25.3 | 384.582 | 383.126 | 386.626 | 156 | 2.599 | 1.1327 |
| cpu_hog_50 | 50 | 27.9 | 47.471 | 25.3 | 383.694 | 382.030 | 385.278 | 157 | 2.606 | 1.1301 |
| cpu_hog_75 | 75 | 27.758 | 47.479 | 25.3 | 384.335 | 382.812 | 385.909 | 157 | 2.602 | 1.132 |
| cpu_hog_100 | 100 | 31.868 | 47.468 | 25.3 | 384.281 | 382.168 | 386.102 | 157 | 2.603 | 1.1318 |
| gpu_spin_control | 100 | 22.296 | 0.707 |  | 393.536 | 392.805 | 394.877 | 153 | 2.540 | 1.1591 |
| cpu_spin_control | 100 | 190.06 | 182.548 |  | 352.316 | 352.219 | 353.088 | 171 | 2.837 | 1.0377 |

## CPU utilisation per phase

| phase | scope | pct | median | p5 | p95 | n | cores busy |
|---|---|---|---|---|---|---|---|
| prefix | thread | 81.51 | 81.86 | 81.55 | 82.59 | 9 |  |
| prefix | process | 81.61 | 81.95 | 81.65 | 82.71 | 9 |  |
| prefix | system_all_cores | 8.08 | 8.08 | 6.33 | 57.71 | 60 | 0.97 |
| prefix | busiest_core | 88.0 | 88.0 | 75.46 | 100.0 | 60 |  |
| ae | thread | 99.73 | 99.74 | 99.72 | 99.74 | 9 |  |
| ae | process | 99.8 | 99.8 | 99.8 | 99.81 | 9 |  |
| ae | system_all_cores | 8.5 | 8.5 | 8.33 | 9.14 | 59 | 1.02 |
| ae | busiest_core | 100.0 | 100.0 | 100.0 | 100.0 | 59 |  |
| full | thread | 11.03 | 11.03 | 11.0 | 11.11 | 9 |  |
| full | process | 11.11 | 11.1 | 11.07 | 11.19 | 9 |  |
| full | system_all_cores | 1.16 | 1.16 | 0.67 | 1.68 | 60 | 0.14 |
| full | busiest_core | 10.0 | 10.0 | 7.91 | 16.4 | 60 |  |

Three scopes, because they answer different questions: `thread` is what the Python loop itself burns, `process` adds the CUDA driver's worker threads, and `system_all_cores` / `busiest_core` come from the kernel's own `/proc/stat` accounting, which is the only one that sees the whole machine. Per-core samples are in `raw/cpu_util_samples.csv`.

## Power per phase (four INA3221 rails)

| phase | rail | mean mW | median | p5 | p95 | n |
|---|---|---|---|---|---|---|
| prefix | VDD_GPU_SOC | 36151.6 | 36225.0 | 36204.4 | 36225.0 | 60 |
| prefix | VDD_CPU_CV | 2381.6 | 2388.0 | 2388.0 | 2389.0 | 60 |
| prefix | VIN_SYS_5V0 | 9565.8 | 9591.0 | 9490.0 | 9591.0 | 60 |
| prefix | VDDQ_VDD2_1V8AO | 3894.0 | 3931.0 | 3830.0 | 3931.0 | 60 |
| prefix | module | 51992.9 | 52135.0 | 51912.4 | 52136.0 | 60 |
| ae | VDD_GPU_SOC | 11156.3 | 10757 | 10757.0 | 12191.6 | 85 |
| ae | VDD_CPU_CV | 2784.4 | 2789 | 2789.0 | 2789.0 | 85 |
| ae | VIN_SYS_5V0 | 5650.8 | 5553 | 5544.0 | 5826.0 | 85 |
| ae | VDDQ_VDD2_1V8AO | 1251.4 | 1210 | 1210.0 | 1310.0 | 85 |
| ae | module | 20842.8 | 20400 | 20300.0 | 22116.6 | 85 |
| full | VDD_GPU_SOC | 32953.6 | 33466.0 | 31672.8 | 33466.0 | 56 |
| full | VDD_CPU_CV | 1601.1 | 1594.0 | 1594.0 | 1594.0 | 56 |
| full | VIN_SYS_5V0 | 8857.0 | 8884.0 | 8733.5 | 8985.0 | 56 |
| full | VDDQ_VDD2_1V8AO | 3387.4 | 3427.0 | 3301.0 | 3427.0 | 56 |
| full | module | 46799.2 | 47371.0 | 45301.2 | 47472.0 | 56 |

`VDDQ_VDD2_1V8AO` is the memory **regulator input**, not DRAM power; it is never rescaled to 1.8 V. The four rails do not double-count (T004/E7).

## Conditions

π0, MAXN + jetson_clocks. The competitor is a separate process on cores 6-11 (the foreground is pinned to 0-5), 20 ms period, duty ∈ {25, 50, 75, 100} %. Its achieved rate is measured inside its own duty window and compared with the **measured** peak of 187.47 GB/s (T005/E1), not the 204.8 GB/s spec. Each foreground point runs ≥ 60 s with tegrastats attached.

## Limits — what this is and is not (protocol §13.5, repeated verbatim as it asks)

This is a **proxy** for memory-side activity on the same DRAM. A real PIM occupies DRAM channels through the command bus and blocks ordinary accesses to them; a software hog instead competes at the memory controller and in the caches, and additionally takes GPU time slices (`gpu_hog`) or CPU cores (`cpu_hog`). The controls bound those side effects; **the result is the sensitivity of the GPU work to lost memory bandwidth, not a measurement of PIM interference.**

The two controls at d = 100 % are `gpu_spin_control` (a compute-bound GPU load with a small memory footprint — small GEMMs) and `cpu_spin_control` (the same worker processes on the same cores doing arithmetic that stays in cache). They separate "the GPU or CPU was taken" from "the memory bandwidth was taken"; both are rows of the table above.

## Anomaly to keep in mind

T005 measured up to 1.19× run-to-run variation in achieved bandwidth with clocks and temperature held constant and no competing process; the cause is not established. Every bandwidth-derived figure here carries that ±8 %.

## Reproduce

```
python bench/mempressure.py --outdir results/20260922b_E11e_pi0_reduce-overhead_mempressure --seconds 60
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
