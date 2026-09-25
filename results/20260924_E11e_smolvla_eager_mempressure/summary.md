# E11e — memory pressure and CPU (protocol §13.5, §13.6)

Run directory: `20260924_E11e_smolvla_eager_mempressure`  
Written by `bench/mempressure.py` at 2026-09-24T23:43:39Z.

## Answer — what a memory-bandwidth competitor does to the prefix

| load | duty % asked | duty % got, alone | duty % got, during | bg GB/s alone | bg GB/s during | % of 187.47 | median ms | p5 | p95 | n | inf/s | slowdown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 |  |  |  |  |  | 97.082 | 96.586 | 99.407 | 616 | 10.266 | 1.0 |
| gpu_hog_25 | 25 | 26.54 | 27.81 | 36.665 | 20.803 | 11.1 | 100.472 | 99.717 | 103.554 | 595 | 9.902 | 1.0349 |
| gpu_hog_50 | 50 | 51.59 | 53.44 | 73.391 | 34.585 | 18.4 | 104.587 | 102.793 | 108.329 | 573 | 9.535 | 1.0773 |
| gpu_hog_75 | 75 | 76.5 | 78.08 | 110.13 | 49.709 | 26.5 | 108.409 | 106.318 | 112.398 | 553 | 9.208 | 1.1167 |
| gpu_hog_100 | 100 | 99.99 | 99.99 | 145.668 | 62.182 | 33.2 | 110.528 | 109.739 | 114.361 | 541 | 9.008 | 1.1385 |
| cpu_hog_25 | 25 | 30.31 | 31.03 | 30.089 | 25.773 | 13.7 | 100.185 | 99.527 | 101.831 | 598 | 9.963 | 1.032 |
| cpu_hog_50 | 50 | 63.67 | 57.85 | 32.046 | 43.724 | 23.3 | 102.904 | 102.353 | 104.561 | 582 | 9.692 | 1.06 |
| cpu_hog_75 | 75 | 88.17 | 86.04 | 32.225 | 48.197 | 25.7 | 105.897 | 103.772 | 113.863 | 562 | 9.366 | 1.0908 |
| cpu_hog_100 | 100 | 99.99 | 99.98 | 33.29 | 51.165 | 27.3 | 105.189 | 103.823 | 108.112 | 569 | 9.479 | 1.0835 |
| gpu_spin_control | 100 | 99.99 | 99.99 | 21.892 | 3.392 |  | 102.885 | 102.488 | 104.384 | 583 | 9.706 | 1.0598 |
| cpu_spin_control | 100 | 99.99 | 99.99 | 191.779 | 190.16 |  | 96.729 | 96.386 | 97.211 | 620 | 10.332 | 0.9964 |

**Read the achieved duty, not the requested one.** The unit of work cannot be made arbitrarily small — a 32 MiB GPU chunk is ~0.4 ms against a 5 ms window at duty 25 — so a hog asked for 25 % delivers more. The achieved figure is measured from the time the hog actually spent working in each 20 ms period.

## CPU utilisation per phase

| phase | scope | pct | median | p5 | p95 | n | cores busy |
|---|---|---|---|---|---|---|---|
| prefix | thread | 96.95 | 96.98 | 96.83 | 97.04 | 9 |  |
| prefix | process | 97.05 | 97.08 | 96.94 | 97.14 | 9 |  |
| prefix | system_all_cores | 58.17 | 58.17 | 7.86 | 58.71 | 59 | 6.98 |
| prefix | busiest_core | 100.0 | 100.0 | 95.99 | 100.0 | 59 |  |
| ae | thread | 99.68 | 99.68 | 99.67 | 99.69 | 9 |  |
| ae | process | 99.76 | 99.76 | 99.75 | 99.77 | 9 |  |
| ae | system_all_cores | 8.5 | 8.5 | 8.33 | 9.66 | 60 | 1.02 |
| ae | busiest_core | 100.0 | 100.0 | 100.0 | 100.0 | 60 |  |
| full | thread | 99.01 | 99.01 | 98.98 | 99.02 | 9 |  |
| full | process | 99.09 | 99.09 | 99.07 | 99.11 | 9 |  |
| full | system_all_cores | 8.48 | 8.48 | 8.18 | 8.81 | 60 | 1.02 |
| full | busiest_core | 100.0 | 100.0 | 98.0 | 100.0 | 60 |  |

Three scopes, because they answer different questions: `thread` is what the Python loop itself burns, `process` adds the CUDA driver's worker threads, and `system_all_cores` / `busiest_core` come from the kernel's own `/proc/stat` accounting, which is the only one that sees the whole machine. Per-core samples are in `raw/cpu_util_samples.csv`.

## Power per phase (four INA3221 rails)

| phase | rail | mean mW | median | p5 | p95 | n |
|---|---|---|---|---|---|---|
| prefix | VDD_GPU_SOC | 18580.9 | 18717.0 | 18319.0 | 18717.0 | 196 |
| prefix | VDD_CPU_CV | 2788.0 | 2788.0 | 2788.0 | 2788.0 | 196 |
| prefix | VIN_SYS_5V0 | 6298.0 | 6360.0 | 6260.0 | 6360.0 | 196 |
| prefix | VDDQ_VDD2_1V8AO | 1711.4 | 1714.0 | 1714.0 | 1814.0 | 196 |
| prefix | module | 29378.3 | 29579.0 | 29081.0 | 29679.0 | 196 |
| ae | VDD_GPU_SOC | 8512.3 | 8363 | 8363.0 | 8761.0 | 83 |
| ae | VDD_CPU_CV | 2788.0 | 2788 | 2788.0 | 2788.0 | 83 |
| ae | VIN_SYS_5V0 | 5068.0 | 5040 | 5040.0 | 5141.0 | 83 |
| ae | VDDQ_VDD2_1V8AO | 915.5 | 907 | 907.0 | 907.0 | 83 |
| ae | module | 17283.7 | 17098 | 17098.0 | 17597.0 | 83 |
| full | VDD_GPU_SOC | 10716.4 | 10757 | 10358.0 | 10757.0 | 59 |
| full | VDD_CPU_CV | 2788.6 | 2789 | 2788.0 | 2789.0 | 59 |
| full | VIN_SYS_5V0 | 5340.3 | 5342 | 5342.0 | 5342.0 | 59 |
| full | VDDQ_VDD2_1V8AO | 1105.6 | 1109 | 1109.0 | 1109.0 | 59 |
| full | module | 19950.9 | 19997 | 19597.0 | 19997.0 | 59 |

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
python bench/mempressure.py --outdir results/20260924_E11e_smolvla_eager_mempressure --seconds 60
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
