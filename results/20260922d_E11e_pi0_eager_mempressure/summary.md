# E11e — memory pressure and CPU (protocol §13.5, §13.6)

Run directory: `20260922d_E11e_pi0_eager_mempressure`  
Written by `bench/mempressure.py` at 2026-09-22T16:48:11Z.

## Answer — what a memory-bandwidth competitor does to the prefix

| load | duty % asked | duty % got, alone | duty % got, during | bg GB/s alone | bg GB/s during | % of 187.47 | median ms | p5 | p95 | n | inf/s | slowdown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 |  |  |  |  |  | 479.122 | 478.276 | 480.737 | 126 | 2.086 | 1.0 |
| gpu_hog_25 | 25 | 26.56 | 30.56 | 36.799 | 8.629 | 4.6 | 528.444 | 524.432 | 531.180 | 114 | 1.893 | 1.1029 |
| gpu_hog_50 | 50 | 51.51 | 59.89 | 72.796 | 16.322 | 8.7 | 578.866 | 575.145 | 582.974 | 104 | 1.727 | 1.2082 |
| gpu_hog_75 | 75 | 75.2 | 82.22 | 107.05 | 21.609 | 11.5 | 619.272 | 616.218 | 622.413 | 97 | 1.615 | 1.2925 |
| gpu_hog_100 | 100 | 99.99 | 99.99 | 145.725 | 24.656 | 13.2 | 647.647 | 645.473 | 651.548 | 93 | 1.543 | 1.3517 |
| cpu_hog_25 | 25 | 29.67 | 30.93 | 26.741 | 27.709 | 14.8 | 495.278 | 492.802 | 501.094 | 122 | 2.017 | 1.0337 |
| cpu_hog_50 | 50 | 63.06 | 57.91 | 32.882 | 39.547 | 21.1 | 507.192 | 502.857 | 509.758 | 119 | 1.973 | 1.0586 |
| cpu_hog_75 | 75 | 89.11 | 84.7 | 32.711 | 46.029 | 24.6 | 512.369 | 510.830 | 516.720 | 117 | 1.949 | 1.0694 |
| cpu_hog_100 | 100 | 99.99 | 99.98 | 32.911 | 48.933 | 26.1 | 516.522 | 512.934 | 521.361 | 116 | 1.933 | 1.0781 |
| gpu_spin_control | 100 | 99.99 | 99.99 | 21.906 | 0.705 |  | 558.805 | 556.220 | 581.100 | 107 | 1.781 | 1.1663 |
| cpu_spin_control | 100 | 99.99 | 99.99 | 186.006 | 183.042 |  | 491.436 | 490.620 | 493.119 | 123 | 2.034 | 1.0257 |

**Read the achieved duty, not the requested one.** The unit of work cannot be made arbitrarily small — a 32 MiB GPU chunk is ~0.4 ms against a 5 ms window at duty 25 — so a hog asked for 25 % delivers more. The achieved figure is measured from the time the hog actually spent working in each 20 ms period.

## CPU utilisation per phase

| phase | scope | pct | median | p5 | p95 | n | cores busy |
|---|---|---|---|---|---|---|---|
| prefix | thread | 69.1 | 70.25 | 66.58 | 70.32 | 9 |  |
| prefix | process | 69.22 | 70.36 | 66.7 | 70.43 | 9 |  |
| prefix | system_all_cores | 21.0 | 21.0 | 18.33 | 71.45 | 61 | 2.52 |
| prefix | busiest_core | 74.51 | 74.51 | 72.0 | 100.0 | 61 |  |
| ae | thread | 99.12 | 99.21 | 98.46 | 99.69 | 9 |  |
| ae | process | 99.23 | 99.32 | 98.56 | 99.81 | 9 |  |
| ae | system_all_cores | 10.99 | 10.99 | 9.0 | 21.05 | 60 | 1.32 |
| ae | busiest_core | 100.0 | 100.0 | 98.04 | 100.0 | 60 |  |
| full | thread | 56.53 | 56.53 | 56.45 | 56.65 | 8 |  |
| full | process | 56.65 | 56.64 | 56.57 | 56.76 | 8 |  |
| full | system_all_cores | 6.34 | 6.34 | 4.17 | 17.03 | 61 | 0.76 |
| full | busiest_core | 66.0 | 66.0 | 38.0 | 100.0 | 61 |  |

Three scopes, because they answer different questions: `thread` is what the Python loop itself burns, `process` adds the CUDA driver's worker threads, and `system_all_cores` / `busiest_core` come from the kernel's own `/proc/stat` accounting, which is the only one that sees the whole machine. Per-core samples are in `raw/cpu_util_samples.csv`.

## Power per phase (four INA3221 rails)

| phase | rail | mean mW | median | p5 | p95 | n |
|---|---|---|---|---|---|---|
| prefix | VDD_GPU_SOC | 39130.1 | 39410 | 38652.2 | 39410.0 | 43 |
| prefix | VDD_CPU_CV | 2665.9 | 2388 | 2388.0 | 3981.0 | 43 |
| prefix | VIN_SYS_5V0 | 8782.4 | 8797 | 8684.3 | 8899.0 | 43 |
| prefix | VDDQ_VDD2_1V8AO | 3321.3 | 3326 | 3326.0 | 3416.9 | 43 |
| prefix | module | 53899.8 | 53921 | 53050.5 | 55218.0 | 43 |
| ae | VDD_GPU_SOC | 11197.8 | 10757 | 10757.0 | 12270.4 | 85 |
| ae | VDD_CPU_CV | 3412.2 | 3187 | 2789.0 | 4781.0 | 85 |
| ae | VIN_SYS_5V0 | 5830.2 | 5755 | 5646.8 | 6138.8 | 85 |
| ae | VDDQ_VDD2_1V8AO | 1382.4 | 1310 | 1310.0 | 1512.0 | 85 |
| ae | module | 21822.5 | 21009 | 20510.0 | 24702.2 | 85 |
| full | VDD_GPU_SOC | 23828.9 | 23914.0 | 23654.7 | 23919.8 | 28 |
| full | VDD_CPU_CV | 2476.6 | 2391.0 | 2391.0 | 2790.7 | 28 |
| full | VIN_SYS_5V0 | 7261.8 | 7269.0 | 7203.4 | 7334.6 | 28 |
| full | VDDQ_VDD2_1V8AO | 2246.5 | 2218.0 | 2218.0 | 2318.0 | 28 |
| full | module | 35813.8 | 35792.5 | 35467.7 | 36363.2 | 28 |

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
python bench/mempressure.py --outdir results/20260922d_E11e_pi0_eager_mempressure --seconds 60
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
