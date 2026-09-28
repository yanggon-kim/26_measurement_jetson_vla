# E11 — the consolidated hand-off (protocol §13.8)

Run directory: `20260928_E11_smolvla_inputs`  
Written by `bench/consolidate_e11.py` at 2026-09-28T22:30:29Z.

## Answer

`model_inputs.csv` carries **1129 rows** covering **31 of the 35** quantities §13.8 names, drawn from 9 source runs.

Checker (`docs/reference_scripts/check_model_inputs.py`, run in place, never adapted): **exit 0 — passes**

## Coverage

| quantity | rows |
|---|---:|
| t_preprocess | 0 |
| t_vision | 31 |
| t_vlm_prefix | 31 |
| t_postprocess | 0 |
| t_ae_step | 2 |
| t_ae_total | 31 |
| t_e2e | 66 |
| t_host_segment | 360 |
| n_kernels_host_segment | 37 |
| bytes_host_segment | 5 |
| flops_host_segment | 5 |
| t_host_layer | 24 |
| t_step_extra | 87 |
| t_launch_per_kernel | 38 |
| t_sync | 0 |
| xfer_latency | 168 |
| xfer_latency_fit | 0 |
| kv_bytes_per_layer | 2 |
| kv_bytes_total | 2 |
| t_kv_materialise | 15 |
| kv_copied_bytes_per_step | 1 |
| boundary_bytes_per_step | 5 |
| prefix_latency_fit | 15 |
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
| achieved_bw_GBps | 10 |
| achieved_tflops | 8 |

## Provenance

| part | run directory | rows |
|---|---|---:|
| E11a_smolvla_hostseg | `20260928_E11a_smolvla_hostseg` | 574 |
| E11c_smolvla_handover | `20260924_E11c_smolvla_handover` | 25 |
| E11c_smolvla_handover_xfer | `20260928_E11c_smolvla_handover_xfer` | 168 |
| E11d_smolvla_eager_prefixsweep | `20260924_E11d_smolvla_eager_prefixsweep` | 120 |
| E11d_smolvla_reduce-overhead_prefixsweep | `20260924_E11d_smolvla_reduce-overhead_prefixsweep` | 30 |
| E11e_smolvla_eager_mempressure | `20260924_E11e_smolvla_eager_mempressure` | 84 |
| E11e_smolvla_reduce-overhead_mempressure | `20260924_E11e_smolvla_reduce-overhead_mempressure` | 84 |
| E11f_smolvla_pipelining | `20260924_E11f_smolvla_pipelining` | 18 |
| E3/E4 region latencies (T006 N sweep) | `20260925_E3E4_smolvla_Nsweep` | 26 |

Rows restated from another run carry `from=<run dir>` in `variant` and `source_file` points at a copy of that run's tidy CSV under `raw/restated/<run dir>/`, never at a profiler report.

## Conditions

smolvla `lerobot/smolvla_base`, bf16, MAXN + jetson_clocks, P = ['113', '177'], S = 50. Each row's own run directory holds its gate record, device state and raw files; this directory is derived and measures nothing itself.

## Reproduce

```
python bench/consolidate_e11.py --model smolvla --outdir results/20260928_E11_smolvla_inputs
python docs/reference_scripts/check_model_inputs.py results/20260928_E11_smolvla_inputs/model_inputs.csv
```

## Not measured

- t_preprocess: not measured — see reports/E11_smolvla_hybrid_inputs.md
- t_postprocess: not measured — see reports/E11_smolvla_hybrid_inputs.md
- t_sync: not measured — see reports/E11_smolvla_hybrid_inputs.md
- xfer_latency_fit: not measured — see reports/E11_smolvla_hybrid_inputs.md
