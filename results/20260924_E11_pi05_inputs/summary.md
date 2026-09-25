# E11 — the consolidated hand-off (protocol §13.8)

Run directory: `20260924_E11_pi05_inputs`  
Written by `bench/consolidate_e11.py` at 2026-09-25T03:34:52Z.

## Answer

`model_inputs.csv` carries **1159 rows** covering **27 of the 35** quantities §13.8 names, drawn from 7 source runs.

Checker (`docs/reference_scripts/check_model_inputs.py`, run in place, never adapted): **exit 0 — passes**

## Coverage

| quantity | rows |
|---|---:|
| t_preprocess | 0 |
| t_vision | 27 |
| t_vlm_prefix | 27 |
| t_postprocess | 0 |
| t_ae_step | 0 |
| t_ae_total | 27 |
| t_e2e | 54 |
| t_host_segment | 648 |
| n_kernels_host_segment | 0 |
| bytes_host_segment | 9 |
| flops_host_segment | 9 |
| t_host_layer | 24 |
| t_step_extra | 87 |
| t_launch_per_kernel | 0 |
| t_sync | 0 |
| xfer_latency | 0 |
| xfer_latency_fit | 0 |
| kv_bytes_per_layer | 2 |
| kv_bytes_total | 2 |
| t_kv_materialise | 15 |
| kv_copied_bytes_per_step | 1 |
| boundary_bytes_per_step | 4 |
| prefix_latency_fit | 3 |
| prefix_throughput | 24 |
| e2e_throughput | 2 |
| ae_throughput | 2 |
| prefix_slowdown | 22 |
| host_layer_slowdown | 4 |
| bg_bw_GBps | 40 |
| cpu_util | 24 |
| p_mean_W | 60 |
| gpu_idle_ae_total | 4 |
| gpu_idle_frac_ae_total | 4 |
| achieved_bw_GBps | 18 |
| achieved_tflops | 16 |

## Provenance

| part | run directory | rows |
|---|---|---:|
| E11a_pi05_hostseg | `20260924_E11a_pi05_hostseg` | 811 |
| E11c_pi05_handover | `20260924_E11c_pi05_handover` | 24 |
| E11d_pi05_eager_prefixsweep | `20260924_E11d_pi05_eager_prefixsweep` | 111 |
| E11d_pi05_reduce-overhead_prefixsweep | `20260924_E11d_pi05_reduce-overhead_prefixsweep` | 27 |
| E11e_pi05_eager_mempressure | `20260924_E11e_pi05_eager_mempressure` | 84 |
| E11e_pi05_reduce-overhead_mempressure | `20260924_E11e_pi05_reduce-overhead_mempressure` | 84 |
| E11f_pi05_pipelining | `20260924_E11f_pi05_pipelining` | 18 |

Rows restated from another run carry `from=<run dir>` in `variant` and `source_file` points at a copy of that run's tidy CSV under `raw/restated/<run dir>/`, never at a profiler report.

## Conditions

pi05 `lerobot/pi05_libero_base`, bf16, MAXN + jetson_clocks, P = 968, S = 50. Each row's own run directory holds its gate record, device state and raw files; this directory is derived and measures nothing itself.

## Reproduce

```
python bench/consolidate_e11.py --model pi05 --outdir results/20260924_E11_pi05_inputs
python docs/reference_scripts/check_model_inputs.py results/20260924_E11_pi05_inputs/model_inputs.csv
```

## Not measured

- t_preprocess: not measured — see reports/E11_pi05_hybrid_inputs.md
- t_postprocess: not measured — see reports/E11_pi05_hybrid_inputs.md
- t_ae_step: not measured — see reports/E11_pi05_hybrid_inputs.md
- n_kernels_host_segment: not measured — see reports/E11_pi05_hybrid_inputs.md
- t_launch_per_kernel: not measured — see reports/E11_pi05_hybrid_inputs.md
- t_sync: not measured — see reports/E11_pi05_hybrid_inputs.md
- xfer_latency: not measured — see reports/E11_pi05_hybrid_inputs.md
- xfer_latency_fit: not measured — see reports/E11_pi05_hybrid_inputs.md
