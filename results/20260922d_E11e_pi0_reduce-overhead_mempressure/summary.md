# E11e — memory pressure and CPU (protocol §13.5, §13.6)

Run directory: `20260922d_E11e_pi0_reduce-overhead_mempressure`  
Written by `bench/mempressure.py` at 2026-09-22T17:12:21Z.

## Answer — what a memory-bandwidth competitor does to the prefix

| load | duty % asked | duty % got, alone | duty % got, during | bg GB/s alone | bg GB/s during | % of 187.47 | median ms | p5 | p95 | n | inf/s | slowdown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 |  |  |  |  |  | 339.435 | 338.839 | 340.691 | 177 | 2.945 | 1.0 |
| gpu_hog_25 | 25 | 26.65 | 30.69 | 36.799 | 8.697 | 4.6 | 374.553 | 371.830 | 376.736 | 161 | 2.671 | 1.1035 |
| gpu_hog_50 | 50 | 51.86 | 59.74 | 73.531 | 16.391 | 8.7 | 409.969 | 406.904 | 413.594 | 147 | 2.438 | 1.2078 |
| gpu_hog_75 | 75 | 76.41 | 82.19 | 109.456 | 21.612 | 11.5 | 438.827 | 436.183 | 442.207 | 137 | 2.278 | 1.2928 |
| gpu_hog_100 | 100 | 99.99 | 99.99 | 146.955 | 24.666 | 13.2 | 456.408 | 455.257 | 458.752 | 132 | 2.190 | 1.3446 |
| cpu_hog_25 | 25 | 30.37 | 30.58 | 27.082 | 24.839 | 13.2 | 356.551 | 354.167 | 359.418 | 169 | 2.804 | 1.0504 |
| cpu_hog_50 | 50 | 64.24 | 57.15 | 33.213 | 39.845 | 21.3 | 373.917 | 370.176 | 377.047 | 161 | 2.675 | 1.1016 |
| cpu_hog_75 | 75 | 87.41 | 82.89 | 31.412 | 43.727 | 23.3 | 379.538 | 378.417 | 380.355 | 159 | 2.635 | 1.1181 |
| cpu_hog_100 | 100 | 99.99 | 99.98 | 29.541 | 47.175 | 25.2 | 382.120 | 381.212 | 390.450 | 157 | 2.601 | 1.1258 |
| gpu_spin_control | 100 | 99.99 | 99.99 | 21.529 | 0.704 |  | 393.282 | 392.264 | 394.763 | 153 | 2.542 | 1.1586 |
| cpu_spin_control | 100 | 99.99 | 99.99 | 190.794 | 182.07 |  | 352.311 | 352.163 | 352.436 | 171 | 2.838 | 1.0379 |

**Read the achieved duty, not the requested one.** The unit of work cannot be made arbitrarily small — a 32 MiB GPU chunk is ~0.4 ms against a 5 ms window at duty 25 — so a hog asked for 25 % delivers more. The achieved figure is measured from the time the hog actually spent working in each 20 ms period.

## CPU utilisation per phase

| phase | scope | pct | median | p5 | p95 | n | cores busy |
|---|---|---|---|---|---|---|---|
| prefix | thread | 81.47 | 81.83 | 81.51 | 82.53 | 9 |  |
| prefix | process | 81.57 | 81.93 | 81.61 | 82.64 | 9 |  |
| prefix | system_all_cores | 7.81 | 7.81 | 6.49 | 57.71 | 60 | 0.94 |
| prefix | busiest_core | 88.0 | 88.0 | 75.51 | 100.0 | 60 |  |
| ae | thread | 99.72 | 99.72 | 99.71 | 99.72 | 9 |  |
| ae | process | 99.8 | 99.8 | 99.79 | 99.81 | 9 |  |
| ae | system_all_cores | 8.49 | 8.49 | 8.32 | 8.86 | 60 | 1.02 |
| ae | busiest_core | 100.0 | 100.0 | 100.0 | 100.0 | 60 |  |
| full | thread | 11.01 | 11.01 | 10.98 | 11.04 | 9 |  |
| full | process | 11.09 | 11.09 | 11.06 | 11.13 | 9 |  |
| full | system_all_cores | 1.16 | 1.16 | 0.67 | 1.68 | 60 | 0.14 |
| full | busiest_core | 10.77 | 10.77 | 6.12 | 16.0 | 60 |  |

Three scopes, because they answer different questions: `thread` is what the Python loop itself burns, `process` adds the CUDA driver's worker threads, and `system_all_cores` / `busiest_core` come from the kernel's own `/proc/stat` accounting, which is the only one that sees the whole machine. Per-core samples are in `raw/cpu_util_samples.csv`.

## Power per phase (four INA3221 rails)

| phase | rail | mean mW | median | p5 | p95 | n |
|---|---|---|---|---|---|---|
| prefix | VDD_GPU_SOC | 36172.8 | 36225 | 36225.0 | 36225.0 | 61 |
| prefix | VDD_CPU_CV | 2381.8 | 2388 | 2388.0 | 2389.0 | 61 |
| prefix | VIN_SYS_5V0 | 9589.3 | 9591 | 9591.0 | 9692.0 | 61 |
| prefix | VDDQ_VDD2_1V8AO | 3906.2 | 3931 | 3830.0 | 3931.0 | 61 |
| prefix | module | 52050.2 | 52135 | 52034.0 | 52237.0 | 61 |
| ae | VDD_GPU_SOC | 11151.6 | 10757.0 | 10757.0 | 12151.0 | 86 |
| ae | VDD_CPU_CV | 2784.4 | 2789.0 | 2789.0 | 2789.0 | 86 |
| ae | VIN_SYS_5V0 | 5706.3 | 5645.0 | 5645.0 | 5830.8 | 86 |
| ae | VDDQ_VDD2_1V8AO | 1289.3 | 1310.0 | 1210.0 | 1310.0 | 86 |
| ae | module | 20931.6 | 20501.0 | 20401.0 | 22080.8 | 86 |
| full | VDD_GPU_SOC | 33044.8 | 33466.0 | 31672.8 | 33466.0 | 56 |
| full | VDD_CPU_CV | 1601.1 | 1594.0 | 1594.0 | 1594.0 | 56 |
| full | VIN_SYS_5V0 | 8842.6 | 8884.0 | 8758.8 | 8884.0 | 56 |
| full | VDDQ_VDD2_1V8AO | 3383.8 | 3427.0 | 3301.0 | 3427.0 | 56 |
| full | module | 46872.2 | 47371.0 | 45326.5 | 47371.0 | 56 |

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
python bench/mempressure.py --outdir results/20260922d_E11e_pi0_reduce-overhead_mempressure --seconds 60
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
