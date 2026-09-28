# E11e - GR00T N1.7 memory pressure and CPU (eager)

Run directory: `20260928_E11e_groot_n17_eager_mempressure`  
Written by `bench/groot_phases.py` at 2026-09-28T21:48:37Z.

## Answer

| load_label | duty_pct_achieved_during | bg_GBps_alone | bg_GBps_during | median_ms | p5_ms | p95_ms | n | throughput_inf_per_s | slowdown |
|---|---|---|---|---|---|---|---|---|---|
| none |  |  |  | 154.9991 | 154.5188 | 156.181 | 387 | 6.446 | 1.0 |
| gpu_hog_25 | 30.57 | 36.799 | 16.322 | 171.041 | 169.2139 | 173.7095 | 351 | 5.8418 | 1.1035 |
| gpu_hog_50 | 54.7 | 71.187 | 30.242 | 187.3297 | 184.2581 | 190.191 | 321 | 5.3417 | 1.2086 |
| gpu_hog_75 | 79.38 | 108.522 | 42.352 | 201.4126 | 199.3282 | 209.1738 | 297 | 4.9398 | 1.2994 |
| gpu_hog_100 | 99.99 | 145.912 | 56.593 | 226.4092 | 225.2149 | 229.4029 | 265 | 4.4081 | 1.4607 |
| cpu_hog_25 | 30.87 | 30.057 | 26.713 | 169.1299 | 167.251 | 171.6455 | 355 | 5.9077 | 1.0912 |
| cpu_hog_50 | 57.62 | 32.242 | 39.347 | 175.7709 | 174.4607 | 177.5581 | 342 | 5.6849 | 1.134 |
| cpu_hog_75 | 82.78 | 30.724 | 43.195 | 178.5027 | 177.3665 | 180.5092 | 336 | 5.596 | 1.1516 |
| cpu_hog_100 | 99.98 | 30.517 | 46.7 | 180.6344 | 179.862 | 182.1498 | 332 | 5.5326 | 1.1654 |
| gpu_spin_control | 99.99 | 23.894 | 3.249 | 181.52 | 180.7683 | 182.815 | 331 | 5.5037 | 1.1711 |
| cpu_spin_control | 99.99 | 188.364 | 184.926 | 160.4801 | 159.1568 | 181.32 | 368 | 6.1169 | 1.0354 |

## CPU utilisation

| phase | scope | pct | median_pct | n | cores_busy |
|---|---|---|---|---|---|
| prefix | thread | 87.12 | 87.4 | 9 |  |
| prefix | process | 87.23 | 87.51 | 9 |  |
| prefix | system_all_cores | 56.84 | 56.84 | 60 | 6.82 |
| prefix | busiest_core | 100.0 | 100.0 | 60 |  |
| ae | thread | 99.6 | 99.67 | 9 |  |
| ae | process | 99.71 | 99.78 | 9 |  |
| ae | system_all_cores | 9.16 | 9.16 | 60 | 1.1 |
| ae | busiest_core | 100.0 | 100.0 | 60 |  |
| full | thread | 96.43 | 96.51 | 9 |  |
| full | process | 96.54 | 96.63 | 9 |  |
| full | system_all_cores | 9.25 | 9.25 | 60 | 1.11 |
| full | busiest_core | 97.96 | 97.96 | 60 |  |

## Power

| phase | rail | mean_mW | median_mW | p5_mW | p95_mW | n |
|---|---|---|---|---|---|---|
| prefix | VDD_GPU_SOC | 29942.1 | 29856.0 | 29458.0 | 30254.0 | 130 |
| prefix | VDD_CPU_CV | 2882.3 | 2788.0 | 2787.0 | 3186.0 | 130 |
| prefix | VIN_SYS_5V0 | 9088.3 | 9086.0 | 9086.0 | 9187.0 | 130 |
| prefix | VDDQ_VDD2_1V8AO | 3591.2 | 3623.0 | 3522.0 | 3623.0 | 130 |
| prefix | module | 45503.9 | 45750.0 | 44854.0 | 45851.0 | 130 |
| ae | VDD_GPU_SOC | 17334.8 | 17124 | 17124.0 | 17722.0 | 91 |
| ae | VDD_CPU_CV | 3103.0 | 3186 | 2788.0 | 3187.0 | 91 |
| ae | VIN_SYS_5V0 | 7560.1 | 7560 | 7459.0 | 7661.0 | 91 |
| ae | VDDQ_VDD2_1V8AO | 2517.1 | 2516 | 2415.0 | 2516.0 | 91 |
| ae | module | 30515.0 | 30386 | 29988.0 | 31085.0 | 91 |
| full | VDD_GPU_SOC | 21512.8 | 21514.0 | 21505.0 | 21514.0 | 54 |
| full | VDD_CPU_CV | 2788.9 | 2789.0 | 2788.0 | 2789.0 | 54 |
| full | VIN_SYS_5V0 | 8080.8 | 8064.0 | 8064.0 | 8165.0 | 54 |
| full | VDDQ_VDD2_1V8AO | 2818.0 | 2818.0 | 2818.0 | 2818.0 | 54 |
| full | module | 35200.5 | 35185.0 | 35176.0 | 35286.0 | 54 |

## Conditions

rung `eager`, N = 4, MAXN; foreground cores 0-5, hog cores 6-11; prefix = the whole backbone (vision inside), the same loop as E7. PROXY, NOT PIM (protocol §13.5).

## Reproduce

```
python bench/groot_phases.py --version n17 --checkpoint nvidia/GR00T-N1.7-3B --rung eager --tf32 on --e11 mempressure --seconds 60 --outdir results/20260928_E11e_groot_n17_eager_mempressure
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
