# E11e — memory pressure and CPU (protocol §13.5, §13.6)

Run directory: `20260924_E11e_pi05_reduce-overhead_mempressure`  
Written by `bench/mempressure.py` at 2026-09-24T22:37:19Z.

## Answer — what a memory-bandwidth competitor does to the prefix

| load | duty % asked | duty % got, alone | duty % got, during | bg GB/s alone | bg GB/s during | % of 187.47 | median ms | p5 | p95 | n | inf/s | slowdown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 |  |  |  |  |  | 647.951 | 646.037 | 649.923 | 93 | 1.543 | 1.0 |
| gpu_hog_25 | 25 | 26.74 | 30.2 | 36.8 | 8.497 | 4.5 | 713.824 | 709.896 | 718.419 | 85 | 1.401 | 1.1017 |
| gpu_hog_50 | 50 | 51.37 | 60.14 | 72.59 | 16.324 | 8.7 | 786.956 | 782.687 | 792.286 | 77 | 1.270 | 1.2145 |
| gpu_hog_75 | 75 | 76.58 | 82.19 | 110.328 | 21.475 | 11.5 | 843.518 | 836.992 | 848.079 | 72 | 1.186 | 1.3018 |
| gpu_hog_100 | 100 | 99.99 | 99.99 | 146.278 | 24.49 | 13.1 | 878.014 | 875.143 | 883.308 | 69 | 1.139 | 1.3551 |
| cpu_hog_25 | 25 | 29.28 | 31.17 | 25.081 | 25.079 | 13.4 | 676.910 | 673.434 | 682.346 | 89 | 1.476 | 1.0447 |
| cpu_hog_50 | 50 | 63.3 | 58.47 | 43.166 | 39.512 | 21.1 | 704.634 | 697.331 | 707.673 | 86 | 1.421 | 1.0875 |
| cpu_hog_75 | 75 | 85.91 | 84.31 | 49.31 | 44.73 | 23.9 | 718.901 | 715.936 | 721.127 | 84 | 1.391 | 1.1095 |
| cpu_hog_100 | 100 | 99.99 | 99.98 | 54.055 | 47.659 | 25.4 | 726.923 | 725.007 | 728.979 | 83 | 1.375 | 1.1219 |
| gpu_spin_control | 100 | 99.99 | 99.99 | 21.616 | 0.686 |  | 756.397 | 753.129 | 760.425 | 80 | 1.322 | 1.1674 |
| cpu_spin_control | 100 | 99.99 | 99.99 | 188.201 | 187.843 |  | 650.753 | 648.314 | 653.187 | 93 | 1.537 | 1.0043 |

**Read the achieved duty, not the requested one.** The unit of work cannot be made arbitrarily small — a 32 MiB GPU chunk is ~0.4 ms against a 5 ms window at duty 25 — so a hog asked for 25 % delivers more. The achieved figure is measured from the time the hog actually spent working in each 20 ms period.

## CPU utilisation per phase

| phase | scope | pct | median | p5 | p95 | n | cores busy |
|---|---|---|---|---|---|---|---|
| prefix | thread | 89.02 | 90.2 | 90.18 | 90.42 | 9 |  |
| prefix | process | 89.12 | 90.3 | 90.27 | 90.52 | 9 |  |
| prefix | system_all_cores | 8.33 | 8.33 | 7.18 | 58.43 | 61 | 1.0 |
| prefix | busiest_core | 94.12 | 94.12 | 86.27 | 100.0 | 61 |  |
| ae | thread | 99.43 | 99.41 | 99.38 | 99.48 | 9 |  |
| ae | process | 99.51 | 99.51 | 99.47 | 99.56 | 9 |  |
| ae | system_all_cores | 8.5 | 8.5 | 8.19 | 9.31 | 60 | 1.02 |
| ae | busiest_core | 100.0 | 100.0 | 100.0 | 100.0 | 60 |  |
| full | thread | 5.94 | 5.95 | 5.88 | 5.99 | 8 |  |
| full | process | 6.02 | 6.03 | 5.95 | 6.06 | 8 |  |
| full | system_all_cores | 0.67 | 0.67 | 0.0 | 1.16 | 61 | 0.08 |
| full | busiest_core | 8.0 | 8.0 | 0.0 | 9.8 | 61 |  |

Three scopes, because they answer different questions: `thread` is what the Python loop itself burns, `process` adds the CUDA driver's worker threads, and `system_all_cores` / `busiest_core` come from the kernel's own `/proc/stat` accounting, which is the only one that sees the whole machine. Per-core samples are in `raw/cpu_util_samples.csv`.

## Power per phase (four INA3221 rails)

| phase | rail | mean mW | median | p5 | p95 | n |
|---|---|---|---|---|---|---|
| prefix | VDD_GPU_SOC | 36140.8 | 36211.0 | 36031.9 | 36225.0 | 32 |
| prefix | VDD_CPU_CV | 2400.5 | 2388.0 | 2388.0 | 2567.5 | 32 |
| prefix | VIN_SYS_5V0 | 10165.4 | 10197.0 | 10096.0 | 10197.0 | 32 |
| prefix | VDDQ_VDD2_1V8AO | 4227.7 | 4234.0 | 4234.0 | 4234.0 | 32 |
| prefix | module | 52934.4 | 53030.0 | 52749.9 | 53217.2 | 32 |
| ae | VDD_GPU_SOC | 14684.8 | 14337.0 | 14337.0 | 15650.5 | 78 |
| ae | VDD_CPU_CV | 2793.1 | 2788.0 | 2788.0 | 2788.0 | 78 |
| ae | VIN_SYS_5V0 | 6754.1 | 6663.0 | 6653.0 | 6880.1 | 78 |
| ae | VDDQ_VDD2_1V8AO | 2039.3 | 2016.0 | 2016.0 | 2031.1 | 78 |
| ae | module | 26271.2 | 25804.0 | 25794.0 | 27349.8 | 78 |
| full | VDD_GPU_SOC | 33725.2 | 33850 | 33367.2 | 34249.0 | 29 |
| full | VDD_CPU_CV | 1593.0 | 1593 | 1593.0 | 1593.0 | 29 |
| full | VIN_SYS_5V0 | 9723.3 | 9793 | 9631.4 | 9793.0 | 29 |
| full | VDDQ_VDD2_1V8AO | 3924.1 | 3931 | 3931.0 | 3931.0 | 29 |
| full | module | 48965.6 | 49167 | 48522.6 | 49566.0 | 29 |

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
python bench/mempressure.py --outdir results/20260924_E11e_pi05_reduce-overhead_mempressure --seconds 60
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
