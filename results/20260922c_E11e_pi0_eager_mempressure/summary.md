# E11e — memory pressure and CPU (protocol §13.5, §13.6)

Run directory: `20260922c_E11e_pi0_eager_mempressure`  
Written by `bench/mempressure.py` at 2026-09-22T15:56:50Z.

## Answer — what a memory-bandwidth competitor does to the prefix

| load | duty % asked | duty % got, alone | duty % got, during | bg GB/s alone | bg GB/s during | % of 187.47 | median ms | p5 | p95 | n | inf/s | slowdown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 |  |  |  |  |  | 477.493 | 477.226 | 477.873 | 126 | 2.094 | 1.0 |
| gpu_hog_25 | 25 | 26.57 | 30.34 | 36.8 | 8.564 | 4.6 | 526.040 | 523.669 | 528.625 | 115 | 1.901 | 1.1017 |
| gpu_hog_50 | 50 | 50.82 | 59.99 | 71.927 | 16.259 | 8.7 | 577.402 | 573.746 | 581.661 | 104 | 1.731 | 1.2092 |
| gpu_hog_75 | 75 | 76.61 | 82.17 | 109.794 | 21.475 | 11.5 | 617.493 | 614.431 | 622.374 | 98 | 1.619 | 1.2932 |
| gpu_hog_100 | 100 | 99.99 | 99.99 | 146.076 | 24.518 | 13.1 | 643.461 | 642.840 | 644.413 | 94 | 1.554 | 1.3476 |
| cpu_hog_25 | 25 | 31.27 | 30.52 | 27.048 | 27.177 | 14.5 | 492.357 | 489.697 | 495.478 | 122 | 2.031 | 1.0311 |
| cpu_hog_50 | 50 | 63.83 | 57.89 | 32.742 | 40.016 | 21.3 | 504.799 | 502.129 | 507.176 | 119 | 1.981 | 1.0572 |
| cpu_hog_75 | 75 | 87.87 | 84.41 | 31.215 | 46.037 | 24.6 | 509.783 | 509.099 | 510.487 | 118 | 1.962 | 1.0676 |
| cpu_hog_100 | 100 | 99.99 | 99.98 | 28.166 | 49.255 | 26.3 | 511.356 | 510.669 | 512.558 | 118 | 1.956 | 1.0709 |
| gpu_spin_control | 100 | 100.0 | 99.99 | 21.831 | 0.686 |  | 554.640 | 553.924 | 555.331 | 109 | 1.803 | 1.1616 |
| cpu_spin_control | 100 |  |  |  |  |  | 473.792 | 473.459 | 474.413 | 127 | 2.110 | 0.9923 |

**Read the achieved duty, not the requested one.** The unit of work cannot be made arbitrarily small — a 32 MiB GPU chunk is ~0.4 ms against a 5 ms window at duty 25 — so a hog asked for 25 % delivers more. The achieved figure is measured from the time the hog actually spent working in each 20 ms period.

## CPU utilisation per phase

| phase | scope | pct | median | p5 | p95 | n | cores busy |
|---|---|---|---|---|---|---|---|
| prefix | thread | 69.38 | 69.94 | 68.92 | 69.95 | 9 |  |
| prefix | process | 69.46 | 70.03 | 69.01 | 70.04 | 9 |  |
| prefix | system_all_cores | 6.0 | 6.0 | 5.5 | 6.33 | 60 | 0.72 |
| prefix | busiest_core | 71.43 | 71.43 | 66.0 | 72.55 | 60 |  |
| ae | thread | 99.68 | 99.69 | 99.61 | 99.7 | 9 |  |
| ae | process | 99.76 | 99.77 | 99.7 | 99.77 | 9 |  |
| ae | system_all_cores | 8.49 | 8.49 | 8.33 | 8.96 | 60 | 1.02 |
| ae | busiest_core | 100.0 | 100.0 | 100.0 | 100.0 | 60 |  |
| full | thread | 55.98 | 55.98 | 55.95 | 56.0 | 8 |  |
| full | process | 56.06 | 56.06 | 56.03 | 56.08 | 8 |  |
| full | system_all_cores | 5.33 | 5.33 | 3.32 | 5.98 | 60 | 0.64 |
| full | busiest_core | 62.38 | 62.38 | 38.0 | 68.73 | 60 |  |

Three scopes, because they answer different questions: `thread` is what the Python loop itself burns, `process` adds the CUDA driver's worker threads, and `system_all_cores` / `busiest_core` come from the kernel's own `/proc/stat` accounting, which is the only one that sees the whole machine. Per-core samples are in `raw/cpu_util_samples.csv`.

## Power per phase (four INA3221 rails)

| phase | rail | mean mW | median | p5 | p95 | n |
|---|---|---|---|---|---|---|
| prefix | VDD_GPU_SOC | 39176.4 | 39410 | 38653.8 | 39410.0 | 43 |
| prefix | VDD_CPU_CV | 2388.2 | 2388 | 2388.0 | 2389.0 | 43 |
| prefix | VIN_SYS_5V0 | 8621.9 | 8683 | 8582.0 | 8683.0 | 43 |
| prefix | VDDQ_VDD2_1V8AO | 3211.9 | 3226 | 3226.0 | 3226.0 | 43 |
| prefix | module | 53398.4 | 53707 | 52849.8 | 53708.0 | 43 |
| ae | VDD_GPU_SOC | 11215.1 | 10757.0 | 10757.0 | 12470.0 | 88 |
| ae | VDD_CPU_CV | 2798.1 | 2789.0 | 2789.0 | 2789.0 | 88 |
| ae | VIN_SYS_5V0 | 5702.9 | 5645.0 | 5645.0 | 5846.0 | 88 |
| ae | VDDQ_VDD2_1V8AO | 1332.9 | 1310.0 | 1310.0 | 1375.6 | 88 |
| ae | module | 21049.0 | 20501.0 | 20501.0 | 22480.7 | 88 |
| full | VDD_GPU_SOC | 24208.2 | 24322 | 23923.0 | 24322.0 | 29 |
| full | VDD_CPU_CV | 2405.7 | 2392 | 2391.0 | 2392.0 | 29 |
| full | VIN_SYS_5V0 | 7185.4 | 7168 | 7168.0 | 7269.0 | 29 |
| full | VDDQ_VDD2_1V8AO | 2218.0 | 2218 | 2218.0 | 2218.0 | 29 |
| full | module | 36017.3 | 36100 | 35856.2 | 36201.0 | 29 |

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
python bench/mempressure.py --outdir results/20260922c_E11e_pi0_eager_mempressure --seconds 60
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
