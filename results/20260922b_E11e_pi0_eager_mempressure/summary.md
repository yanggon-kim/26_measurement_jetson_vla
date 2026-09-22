# E11e — memory pressure and CPU (protocol §13.5, §13.6)

Run directory: `20260922b_E11e_pi0_eager_mempressure`  
Written by `bench/mempressure.py` at 2026-09-22T14:22:45Z.

## Answer — what a memory-bandwidth competitor does to the prefix

| load | duty % | bg GB/s alone | bg GB/s during | % of 187.47 | median ms | p5 | p95 | n | inf/s | slowdown |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 |  |  |  | 476.983 | 474.574 | 479.836 | 126 | 2.096 | 1.0 |
| gpu_hog_25 | 25 | 53.527 | 26.764 | 14.3 | 589.133 | 586.073 | 594.075 | 102 | 1.696 | 1.2351 |
| gpu_hog_50 | 50 | 107.053 | 53.523 | 28.6 | 770.718 | 766.628 | 774.114 | 78 | 1.298 | 1.6158 |
| gpu_hog_75 | 75 | 133.817 | 60.242 | 32.1 | 841.366 | 790.942 | 880.149 | 72 | 1.194 | 1.7639 |
| gpu_hog_100 | 100 | 172.878 | 68.191 | 36.4 | 928.726 | 925.058 | 932.758 | 65 | 1.077 | 1.9471 |
| cpu_hog_25 | 25 | 54.294 | 49.638 | 26.5 | 513.117 | 511.506 | 515.216 | 117 | 1.949 | 1.0758 |
| cpu_hog_50 | 50 | 34.692 | 49.563 | 26.4 | 513.231 | 511.573 | 515.641 | 117 | 1.947 | 1.076 |
| cpu_hog_75 | 75 | 35.301 | 49.628 | 26.5 | 513.176 | 511.591 | 515.054 | 117 | 1.949 | 1.0759 |
| cpu_hog_100 | 100 | 29.365 | 49.572 | 26.4 | 512.481 | 511.553 | 515.044 | 117 | 1.950 | 1.0744 |
| gpu_spin_control | 100 | 22.269 | 0.687 |  | 555.999 | 555.093 | 557.707 | 108 | 1.798 | 1.1657 |
| cpu_spin_control | 100 | 190.203 | 184.724 |  | 489.072 | 488.975 | 489.239 | 123 | 2.044 | 1.0253 |

## CPU utilisation per phase

| phase | scope | pct | median | p5 | p95 | n | cores busy |
|---|---|---|---|---|---|---|---|
| prefix | thread | 69.83 | 70.0 | 69.56 | 70.88 | 9 |  |
| prefix | process | 69.92 | 70.1 | 69.66 | 70.98 | 9 |  |
| prefix | system_all_cores | 6.63 | 6.63 | 5.68 | 56.46 | 60 | 0.8 |
| prefix | busiest_core | 72.0 | 72.0 | 66.0 | 100.0 | 60 |  |
| ae | thread | 99.72 | 99.72 | 99.7 | 99.72 | 9 |  |
| ae | process | 99.81 | 99.81 | 99.8 | 99.82 | 9 |  |
| ae | system_all_cores | 8.5 | 8.5 | 8.33 | 9.02 | 60 | 1.02 |
| ae | busiest_core | 100.0 | 100.0 | 100.0 | 100.0 | 60 |  |
| full | thread | 56.43 | 56.43 | 56.41 | 56.46 | 8 |  |
| full | process | 56.52 | 56.51 | 56.49 | 56.56 | 8 |  |
| full | system_all_cores | 5.33 | 5.33 | 3.33 | 6.19 | 60 | 0.64 |
| full | busiest_core | 61.39 | 61.39 | 38.0 | 71.46 | 60 |  |

Three scopes, because they answer different questions: `thread` is what the Python loop itself burns, `process` adds the CUDA driver's worker threads, and `system_all_cores` / `busiest_core` come from the kernel's own `/proc/stat` accounting, which is the only one that sees the whole machine. Per-core samples are in `raw/cpu_util_samples.csv`.

## Power per phase (four INA3221 rails)

| phase | rail | mean mW | median | p5 | p95 | n |
|---|---|---|---|---|---|---|
| prefix | VDD_GPU_SOC | 39215.2 | 39394 | 39050.2 | 39753.8 | 43 |
| prefix | VDD_CPU_CV | 2388.0 | 2388 | 2388.0 | 2388.0 | 43 |
| prefix | VIN_SYS_5V0 | 8743.1 | 8784 | 8683.0 | 8797.0 | 43 |
| prefix | VDDQ_VDD2_1V8AO | 3312.0 | 3326 | 3326.0 | 3326.0 | 43 |
| prefix | module | 53658.3 | 53892 | 53448.5 | 54253.1 | 43 |
| ae | VDD_GPU_SOC | 11207.4 | 10757 | 10757.0 | 12589.6 | 85 |
| ae | VDD_CPU_CV | 2784.4 | 2789 | 2789.0 | 2789.0 | 85 |
| ae | VIN_SYS_5V0 | 5715.4 | 5654 | 5645.0 | 5936.8 | 85 |
| ae | VDDQ_VDD2_1V8AO | 1329.1 | 1310 | 1210.0 | 1390.8 | 85 |
| ae | module | 21036.1 | 20510 | 20410.0 | 22706.2 | 85 |
| full | VDD_GPU_SOC | 24120.3 | 24312 | 23674.6 | 24312.0 | 29 |
| full | VDD_CPU_CV | 2391.2 | 2391 | 2391.0 | 2392.0 | 29 |
| full | VIN_SYS_5V0 | 7244.6 | 7269 | 7168.0 | 7269.0 | 29 |
| full | VDDQ_VDD2_1V8AO | 2218.0 | 2218 | 2218.0 | 2218.0 | 29 |
| full | module | 35974.1 | 36190 | 35452.6 | 36196.0 | 29 |

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
python bench/mempressure.py --outdir results/20260922b_E11e_pi0_eager_mempressure --seconds 60
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
