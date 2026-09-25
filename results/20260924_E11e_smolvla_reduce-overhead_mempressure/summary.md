# E11e — memory pressure and CPU (protocol §13.5, §13.6)

Run directory: `20260924_E11e_smolvla_reduce-overhead_mempressure`  
Written by `bench/mempressure.py` at 2026-09-25T00:54:10Z.

## Answer — what a memory-bandwidth competitor does to the prefix

| load | duty % asked | duty % got, alone | duty % got, during | bg GB/s alone | bg GB/s during | % of 187.47 | median ms | p5 | p95 | n | inf/s | slowdown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 |  |  |  |  |  | 97.792 | 97.532 | 98.916 | 613 | 10.213 | 1.0 |
| gpu_hog_25 | 25 | 26.52 | 28.73 | 36.799 | 20.067 | 10.7 | 101.888 | 100.122 | 104.599 | 589 | 9.810 | 1.0419 |
| gpu_hog_50 | 50 | 50.71 | 53.31 | 70.722 | 34.72 | 18.5 | 105.701 | 104.221 | 108.698 | 566 | 9.423 | 1.0809 |
| gpu_hog_75 | 75 | 75.2 | 78.31 | 107.122 | 49.845 | 26.6 | 108.006 | 106.488 | 110.733 | 556 | 9.254 | 1.1044 |
| gpu_hog_100 | 100 | 99.99 | 99.99 | 146.395 | 62.401 | 33.3 | 110.327 | 109.552 | 114.342 | 543 | 9.038 | 1.1282 |
| cpu_hog_25 | 25 | 30.24 | 31.04 | 26.752 | 26.573 | 14.2 | 102.349 | 101.532 | 103.500 | 586 | 9.761 | 1.0466 |
| cpu_hog_50 | 50 | 63.72 | 59.2 | 32.686 | 39.784 | 21.2 | 104.509 | 102.980 | 106.000 | 575 | 9.570 | 1.0687 |
| cpu_hog_75 | 75 | 89.76 | 86.78 | 31.235 | 48.337 | 25.8 | 105.545 | 105.036 | 106.148 | 569 | 9.471 | 1.0793 |
| cpu_hog_100 | 100 | 99.99 | 99.98 | 28.556 | 51.088 | 27.3 | 106.630 | 105.702 | 115.356 | 557 | 9.268 | 1.0904 |
| gpu_spin_control | 100 | 99.99 | 99.99 | 21.797 | 3.484 |  | 104.108 | 103.236 | 106.847 | 575 | 9.569 | 1.0646 |
| cpu_spin_control | 100 | 99.99 | 99.99 | 190.713 | 188.271 |  | 98.079 | 97.775 | 98.597 | 612 | 10.189 | 1.0029 |

**Read the achieved duty, not the requested one.** The unit of work cannot be made arbitrarily small — a 32 MiB GPU chunk is ~0.4 ms against a 5 ms window at duty 25 — so a hog asked for 25 % delivers more. The achieved figure is measured from the time the hog actually spent working in each 20 ms period.

## CPU utilisation per phase

| phase | scope | pct | median | p5 | p95 | n | cores busy |
|---|---|---|---|---|---|---|---|
| prefix | thread | 97.89 | 97.94 | 97.77 | 97.98 | 9 |  |
| prefix | process | 98.0 | 98.06 | 97.86 | 98.08 | 9 |  |
| prefix | system_all_cores | 58.18 | 58.18 | 8.18 | 58.54 | 60 | 6.98 |
| prefix | busiest_core | 100.0 | 100.0 | 97.96 | 100.0 | 60 |  |
| ae | thread | 99.68 | 99.69 | 99.63 | 99.69 | 9 |  |
| ae | process | 99.76 | 99.77 | 99.7 | 99.78 | 9 |  |
| ae | system_all_cores | 8.5 | 8.5 | 8.33 | 9.35 | 60 | 1.02 |
| ae | busiest_core | 100.0 | 100.0 | 100.0 | 100.0 | 60 |  |
| full | thread | 96.76 | 99.89 | 99.88 | 99.9 | 9 |  |
| full | process | 96.84 | 99.97 | 99.96 | 99.98 | 9 |  |
| full | system_all_cores | 8.49 | 8.49 | 8.18 | 8.83 | 61 | 1.02 |
| full | busiest_core | 100.0 | 100.0 | 100.0 | 100.0 | 61 |  |

Three scopes, because they answer different questions: `thread` is what the Python loop itself burns, `process` adds the CUDA driver's worker threads, and `system_all_cores` / `busiest_core` come from the kernel's own `/proc/stat` accounting, which is the only one that sees the whole machine. Per-core samples are in `raw/cpu_util_samples.csv`.

## Power per phase (four INA3221 rails)

| phase | rail | mean mW | median | p5 | p95 | n |
|---|---|---|---|---|---|---|
| prefix | VDD_GPU_SOC | 19047.6 | 18725.0 | 18725.0 | 20059.3 | 194 |
| prefix | VDD_CPU_CV | 2770.6 | 2789.0 | 2789.0 | 2790.0 | 194 |
| prefix | VIN_SYS_5V0 | 6399.1 | 6360.0 | 6325.0 | 6562.0 | 194 |
| prefix | VDDQ_VDD2_1V8AO | 1739.9 | 1714.0 | 1714.0 | 1814.0 | 194 |
| prefix | module | 29957.3 | 29588.0 | 29489.0 | 31224.3 | 194 |
| ae | VDD_GPU_SOC | 8149.0 | 7971.0 | 7968.0 | 8785.1 | 80 |
| ae | VDD_CPU_CV | 2789.0 | 2789.0 | 2789.0 | 2789.0 | 80 |
| ae | VIN_SYS_5V0 | 5096.8 | 5040.0 | 5040.0 | 5146.0 | 80 |
| ae | VDDQ_VDD2_1V8AO | 915.8 | 907.0 | 907.0 | 907.0 | 80 |
| ae | module | 16950.7 | 16707.0 | 16704.0 | 17627.1 | 80 |
| full | VDD_GPU_SOC | 36284.0 | 38216 | 22085.0 | 38231.0 | 411 |
| full | VDD_CPU_CV | 2543.2 | 2389 | 2388.0 | 2788.0 | 411 |
| full | VIN_SYS_5V0 | 8647.3 | 8884 | 6865.0 | 8884.0 | 411 |
| full | VDDQ_VDD2_1V8AO | 3312.7 | 3427 | 2318.0 | 3427.0 | 411 |
| full | module | 50787.1 | 52816 | 33656.0 | 53330.0 | 411 |

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
python bench/mempressure.py --outdir results/20260924_E11e_smolvla_reduce-overhead_mempressure --seconds 60
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
