# E11e - GR00T N1.7 memory pressure and CPU (vlm_compile)

Run directory: `20260928_E11e_groot_n17_vlm_compile_mempressure`  
Written by `bench/groot_phases.py` at 2026-09-28T22:09:50Z.

## Answer

| load_label | duty_pct_achieved_during | bg_GBps_alone | bg_GBps_during | median_ms | p5_ms | p95_ms | n | throughput_inf_per_s | slowdown |
|---|---|---|---|---|---|---|---|---|---|
| none |  |  |  | 155.9845 | 155.2224 | 157.831 | 384 | 6.3994 | 1.0 |
| gpu_hog_25 | 29.61 | 36.799 | 16.191 | 170.5074 | 168.0397 | 173.0616 | 352 | 5.8639 | 1.0931 |
| gpu_hog_50 | 53.55 | 73.261 | 29.639 | 180.7093 | 178.4323 | 183.4849 | 332 | 5.528 | 1.1585 |
| gpu_hog_75 | 79.08 | 108.054 | 43.824 | 200.5714 | 197.8045 | 204.0099 | 299 | 4.9762 | 1.2858 |
| gpu_hog_100 | 99.99 | 146.475 | 56.467 | 213.7608 | 211.7381 | 218.3779 | 281 | 4.6693 | 1.3704 |
| cpu_hog_25 | 31.19 | 30.093 | 24.909 | 169.6518 | 167.3163 | 172.1557 | 354 | 5.8913 | 1.0876 |
| cpu_hog_50 | 57.28 | 31.681 | 39.244 | 177.2601 | 174.0826 | 179.1021 | 340 | 5.6525 | 1.1364 |
| cpu_hog_75 | 82.95 | 31.866 | 43.834 | 180.697 | 179.7894 | 182.7156 | 332 | 5.521 | 1.1584 |
| cpu_hog_100 | 99.98 | 30.261 | 47.011 | 183.1046 | 182.0464 | 184.9931 | 328 | 5.4568 | 1.1739 |
| gpu_spin_control | 99.99 | 23.536 | 3.605 | 180.2527 | 178.8123 | 182.3739 | 333 | 5.5423 | 1.1556 |
| cpu_spin_control | 99.99 | 188.918 | 183.867 | 161.418 | 159.9534 | 170.2541 | 369 | 6.1467 | 1.0348 |

## CPU utilisation

| phase | scope | pct | median_pct | n | cores_busy |
|---|---|---|---|---|---|
| prefix | thread | 87.33 | 87.82 | 9 |  |
| prefix | process | 87.44 | 87.93 | 9 |  |
| prefix | system_all_cores | 56.75 | 56.75 | 60 | 6.81 |
| prefix | busiest_core | 100.0 | 100.0 | 60 |  |
| ae | thread | 89.55 | 99.92 | 9 |  |
| ae | process | 89.62 | 99.99 | 9 |  |
| ae | system_all_cores | 8.41 | 8.41 | 66 | 1.01 |
| ae | busiest_core | 100.0 | 100.0 | 66 |  |
| full | thread | 57.86 | 57.86 | 9 |  |
| full | process | 57.96 | 57.96 | 9 |  |
| full | system_all_cores | 5.14 | 5.14 | 60 | 0.62 |
| full | busiest_core | 59.18 | 59.18 | 60 |  |

## Power

| phase | rail | mean_mW | median_mW | p5_mW | p95_mW | n |
|---|---|---|---|---|---|---|
| prefix | VDD_GPU_SOC | 29803.0 | 29868.0 | 29458.0 | 29868.0 | 132 |
| prefix | VDD_CPU_CV | 2827.0 | 2788.0 | 2787.0 | 3186.0 | 132 |
| prefix | VIN_SYS_5V0 | 9089.8 | 9086.0 | 8985.0 | 9187.0 | 132 |
| prefix | VDDQ_VDD2_1V8AO | 3527.4 | 3522.0 | 3522.0 | 3522.0 | 132 |
| prefix | module | 45247.2 | 45263.0 | 44854.0 | 45511.4 | 132 |
| ae | VDD_GPU_SOC | 35874.2 | 36609.0 | 31834.0 | 36623.0 | 296 |
| ae | VDD_CPU_CV | 2743.8 | 2787.0 | 2388.0 | 2787.0 | 296 |
| ae | VIN_SYS_5V0 | 11699.6 | 12014.0 | 9868.8 | 12115.0 | 296 |
| ae | VDDQ_VDD2_1V8AO | 5238.7 | 5435.0 | 4126.0 | 5435.0 | 296 |
| ae | module | 55556.3 | 56845.0 | 48613.8 | 56960.0 | 296 |
| full | VDD_GPU_SOC | 31282.0 | 31448.0 | 31050.0 | 31461.0 | 86 |
| full | VDD_CPU_CV | 2398.3 | 2389.0 | 2389.0 | 2389.0 | 86 |
| full | VIN_SYS_5V0 | 9874.0 | 9894.0 | 9793.0 | 9995.0 | 86 |
| full | VDDQ_VDD2_1V8AO | 4030.7 | 4026.0 | 4026.0 | 4026.0 | 86 |
| full | module | 47585.0 | 47669.0 | 47271.0 | 47871.0 | 86 |

## Conditions

rung `vlm_compile`, N = 4, MAXN; foreground cores 0-5, hog cores 6-11; prefix = the whole backbone (vision inside), the same loop as E7. PROXY, NOT PIM (protocol §13.5).

## Reproduce

```
python bench/groot_phases.py --version n17 --checkpoint nvidia/GR00T-N1.7-3B --rung vlm_compile --tf32 on --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs --e11 mempressure --seconds 60 --outdir results/20260928_E11e_groot_n17_vlm_compile_mempressure
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
