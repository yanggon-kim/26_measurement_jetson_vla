# E11e — memory pressure and CPU (protocol §13.5, §13.6)

Run directory: `20260924_E11e_pi05_eager_mempressure`  
Written by `bench/mempressure.py` at 2026-09-24T22:12:25Z.

## Answer — what a memory-bandwidth competitor does to the prefix

| load | duty % asked | duty % got, alone | duty % got, during | bg GB/s alone | bg GB/s during | % of 187.47 | median ms | p5 | p95 | n | inf/s | slowdown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 |  |  |  |  |  | 2425.464 | 2423.092 | 2428.198 | 25 | 0.412 | 1.0 |
| gpu_hog_25 | 25 | 26.68 | 30.05 | 36.799 | 8.43 | 4.5 | 2666.377 | 2659.679 | 2674.884 | 23 | 0.375 | 1.0993 |
| gpu_hog_50 | 50 | 51.54 | 60.27 | 73.598 | 16.191 | 8.6 | 2932.526 | 2925.285 | 2942.574 | 21 | 0.341 | 1.2091 |
| gpu_hog_75 | 75 | 76.27 | 82.28 | 109.392 | 21.342 | 11.4 | 3139.362 | 3132.175 | 3150.966 | 20 | 0.319 | 1.2943 |
| gpu_hog_100 | 100 | 99.99 | 99.99 | 144.643 | 24.202 | 12.9 | 3274.344 | 3272.351 | 3275.407 | 19 | 0.305 | 1.35 |
| cpu_hog_25 | 25 | 32.9 | 31.2 | 30.088 | 26.611 | 14.2 | 2424.469 | 2422.179 | 2427.483 | 25 | 0.412 | 0.9996 |
| cpu_hog_50 | 50 | 57.42 | 57.38 | 46.576 | 43.06 | 23.0 | 2441.724 | 2437.797 | 2443.531 | 25 | 0.410 | 1.0067 |
| cpu_hog_75 | 75 | 82.03 | 85.37 | 50.389 | 49.021 | 26.1 | 2448.777 | 2446.830 | 2451.360 | 25 | 0.408 | 1.0096 |
| cpu_hog_100 | 100 | 99.99 | 99.99 | 53.906 | 52.219 | 27.9 | 2452.985 | 2451.983 | 2454.483 | 25 | 0.408 | 1.0113 |
| gpu_spin_control | 100 | 100.0 | 99.99 | 21.832 | 0.655 |  | 2824.432 | 2822.833 | 2826.001 | 22 | 0.354 | 1.1645 |
| cpu_spin_control | 100 | 99.99 | 99.99 | 189.939 | 189.76 |  | 2409.526 | 2407.477 | 2411.440 | 25 | 0.415 | 0.9934 |

**Read the achieved duty, not the requested one.** The unit of work cannot be made arbitrarily small — a 32 MiB GPU chunk is ~0.4 ms against a 5 ms window at duty 25 — so a hog asked for 25 % delivers more. The achieved figure is measured from the time the hog actually spent working in each 20 ms period.

## CPU utilisation per phase

| phase | scope | pct | median | p5 | p95 | n | cores busy |
|---|---|---|---|---|---|---|---|
| prefix | thread | 89.38 | 94.03 | 91.97 | 94.03 | 7 |  |
| prefix | process | 89.46 | 94.11 | 92.06 | 94.12 | 7 |  |
| prefix | system_all_cores | 8.49 | 8.49 | 5.78 | 8.98 | 67 | 1.02 |
| prefix | busiest_core | 100.0 | 100.0 | 70.59 | 100.0 | 67 |  |
| ae | thread | 99.7 | 99.7 | 99.69 | 99.71 | 9 |  |
| ae | process | 99.79 | 99.79 | 99.79 | 99.8 | 9 |  |
| ae | system_all_cores | 8.5 | 8.5 | 8.32 | 9.16 | 60 | 1.02 |
| ae | busiest_core | 100.0 | 100.0 | 100.0 | 100.0 | 60 |  |
| full | thread | 36.04 | 36.03 | 36.03 | 36.06 | 6 |  |
| full | process | 36.12 | 36.11 | 36.1 | 36.14 | 6 |  |
| full | system_all_cores | 2.66 | 2.66 | 0.0 | 8.33 | 63 | 0.32 |
| full | busiest_core | 23.53 | 23.53 | 0.0 | 99.2 | 63 |  |

Three scopes, because they answer different questions: `thread` is what the Python loop itself burns, `process` adds the CUDA driver's worker threads, and `system_all_cores` / `busiest_core` come from the kernel's own `/proc/stat` accounting, which is the only one that sees the whole machine. Per-core samples are in `raw/cpu_util_samples.csv`.

## Power per phase (four INA3221 rails)

| phase | rail | mean mW | median | p5 | p95 | n |
|---|---|---|---|---|---|---|
| prefix | VDD_GPU_SOC | 34192.3 | 34235 | 33757.4 | 34479.4 | 9 |
| prefix | VDD_CPU_CV | 2743.7 | 2788 | 2548.6 | 2788.0 | 9 |
| prefix | VIN_SYS_5V0 | 7729.1 | 7774 | 7673.0 | 7774.0 | 9 |
| prefix | VDDQ_VDD2_1V8AO | 2587.3 | 2621 | 2520.0 | 2621.0 | 9 |
| prefix | module | 47252.4 | 47418 | 46559.6 | 47662.4 | 9 |
| ae | VDD_GPU_SOC | 21717.6 | 21505 | 21505.0 | 22460.6 | 73 |
| ae | VDD_CPU_CV | 2967.9 | 2788 | 2788.0 | 3584.0 | 73 |
| ae | VIN_SYS_5V0 | 7051.8 | 7067 | 6966.0 | 7168.0 | 73 |
| ae | VDDQ_VDD2_1V8AO | 2136.3 | 2117 | 2117.0 | 2218.0 | 73 |
| ae | module | 33873.6 | 33477 | 33376.0 | 35072.6 | 73 |
| full | VDD_GPU_SOC | 27976.5 | 27877.0 | 27877.0 | 28275.0 | 8 |
| full | VDD_CPU_CV | 2140.6 | 1992.0 | 1991.0 | 2389.7 | 8 |
| full | VIN_SYS_5V0 | 7370.0 | 7370.0 | 7370.0 | 7370.0 | 8 |
| full | VDDQ_VDD2_1V8AO | 2318.0 | 2318.0 | 2318.0 | 2318.0 | 8 |
| full | module | 39805.1 | 39954.0 | 39556.3 | 39954.7 | 8 |

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
python bench/mempressure.py --outdir results/20260924_E11e_pi05_eager_mempressure --seconds 60
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
