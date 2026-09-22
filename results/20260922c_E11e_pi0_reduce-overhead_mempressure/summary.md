# E11e — memory pressure and CPU (protocol §13.5, §13.6)

Run directory: `20260922c_E11e_pi0_reduce-overhead_mempressure`  
Written by `bench/mempressure.py` at 2026-09-22T16:20:47Z.

## Answer — what a memory-bandwidth competitor does to the prefix

| load | duty % asked | duty % got, alone | duty % got, during | bg GB/s alone | bg GB/s during | % of 187.47 | median ms | p5 | p95 | n | inf/s | slowdown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 |  |  |  |  |  | 339.173 | 339.017 | 339.420 | 177 | 2.948 | 1.0 |
| gpu_hog_25 | 25 | 26.6 | 30.38 | 36.799 | 8.564 | 4.6 | 373.731 | 371.511 | 375.435 | 161 | 2.676 | 1.1019 |
| gpu_hog_50 | 50 | 51.58 | 59.9 | 73.534 | 16.391 | 8.7 | 410.736 | 407.872 | 412.686 | 147 | 2.436 | 1.211 |
| gpu_hog_75 | 75 | 76.66 | 81.92 | 110.192 | 21.543 | 11.5 | 438.862 | 435.217 | 441.465 | 137 | 2.279 | 1.2939 |
| gpu_hog_100 | 100 | 99.99 | 99.99 | 145.627 | 24.66 | 13.2 | 456.422 | 455.728 | 458.999 | 132 | 2.189 | 1.3457 |
| cpu_hog_25 | 25 | 30.43 | 31.16 | 30.088 | 26.242 | 14.0 | 357.984 | 355.639 | 361.240 | 168 | 2.792 | 1.0555 |
| cpu_hog_50 | 50 | 63.78 | 57.46 | 32.046 | 40.612 | 21.7 | 375.249 | 372.176 | 377.209 | 161 | 2.667 | 1.1064 |
| cpu_hog_75 | 75 | 88.79 | 83.21 | 31.756 | 43.916 | 23.4 | 380.328 | 379.544 | 381.068 | 158 | 2.629 | 1.1213 |
| cpu_hog_100 | 100 | 99.99 | 99.99 | 32.135 | 47.181 | 25.2 | 382.921 | 382.027 | 385.215 | 157 | 2.609 | 1.129 |
| gpu_spin_control | 100 | 99.99 | 99.99 | 21.814 | 0.698 |  | 392.940 | 392.221 | 393.585 | 153 | 2.545 | 1.1585 |
| cpu_spin_control | 100 |  |  |  |  |  | 337.441 | 336.999 | 339.188 | 178 | 2.963 | 0.9949 |

**Read the achieved duty, not the requested one.** The unit of work cannot be made arbitrarily small — a 32 MiB GPU chunk is ~0.4 ms against a 5 ms window at duty 25 — so a hog asked for 25 % delivers more. The achieved figure is measured from the time the hog actually spent working in each 20 ms period.

## CPU utilisation per phase

| phase | scope | pct | median | p5 | p95 | n | cores busy |
|---|---|---|---|---|---|---|---|
| prefix | thread | 81.1 | 81.51 | 81.48 | 81.78 | 9 |  |
| prefix | process | 81.18 | 81.58 | 81.56 | 81.86 | 9 |  |
| prefix | system_all_cores | 7.15 | 7.15 | 6.34 | 7.8 | 60 | 0.86 |
| prefix | busiest_core | 82.17 | 82.17 | 74.51 | 88.0 | 60 |  |
| ae | thread | 99.67 | 99.68 | 99.65 | 99.68 | 9 |  |
| ae | process | 99.77 | 99.77 | 99.75 | 99.78 | 9 |  |
| ae | system_all_cores | 8.49 | 8.49 | 8.33 | 8.96 | 60 | 1.02 |
| ae | busiest_core | 100.0 | 100.0 | 100.0 | 100.0 | 60 |  |
| full | thread | 10.82 | 10.8 | 10.79 | 10.86 | 9 |  |
| full | process | 10.9 | 10.89 | 10.88 | 10.95 | 9 |  |
| full | system_all_cores | 1.0 | 1.0 | 0.67 | 1.52 | 59 | 0.12 |
| full | busiest_core | 9.8 | 9.8 | 6.11 | 16.0 | 59 |  |

Three scopes, because they answer different questions: `thread` is what the Python loop itself burns, `process` adds the CUDA driver's worker threads, and `system_all_cores` / `busiest_core` come from the kernel's own `/proc/stat` accounting, which is the only one that sees the whole machine. Per-core samples are in `raw/cpu_util_samples.csv`.

## Power per phase (four INA3221 rails)

| phase | rail | mean mW | median | p5 | p95 | n |
|---|---|---|---|---|---|---|
| prefix | VDD_GPU_SOC | 36126.3 | 36211 | 35813.0 | 36225.0 | 61 |
| prefix | VDD_CPU_CV | 2381.5 | 2388 | 2388.0 | 2389.0 | 61 |
| prefix | VIN_SYS_5V0 | 9569.5 | 9591 | 9490.0 | 9591.0 | 61 |
| prefix | VDDQ_VDD2_1V8AO | 3823.4 | 3830 | 3830.0 | 3830.0 | 61 |
| prefix | module | 51900.7 | 52020 | 51521.0 | 52035.0 | 61 |
| ae | VDD_GPU_SOC | 11152.1 | 10757 | 10757.0 | 12115.9 | 87 |
| ae | VDD_CPU_CV | 2779.9 | 2789 | 2789.0 | 2789.0 | 87 |
| ae | VIN_SYS_5V0 | 5668.7 | 5645 | 5544.0 | 5816.0 | 87 |
| ae | VDDQ_VDD2_1V8AO | 1257.3 | 1210 | 1210.0 | 1310.0 | 87 |
| ae | module | 20858.0 | 20401 | 20300.0 | 22031.9 | 87 |
| full | VDD_GPU_SOC | 32833.0 | 33067.0 | 31663.0 | 33466.0 | 56 |
| full | VDD_CPU_CV | 1601.0 | 1594.0 | 1593.0 | 1594.0 | 56 |
| full | VIN_SYS_5V0 | 8842.6 | 8884.0 | 8733.5 | 8884.0 | 56 |
| full | VDDQ_VDD2_1V8AO | 3327.9 | 3326.0 | 3301.0 | 3427.0 | 56 |
| full | module | 46604.5 | 46871.0 | 45291.5 | 47357.0 | 56 |

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
python bench/mempressure.py --outdir results/20260922c_E11e_pi0_reduce-overhead_mempressure --seconds 60
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
