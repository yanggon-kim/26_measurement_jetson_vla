# E11 — the consolidated hand-off (protocol §13.8)

Run directory: `20260922b_E11_pi0_inputs`  
Written by `bench/consolidate_e11.py` at 2026-09-22T17:12:29Z.

## Answer

`model_inputs.csv` carries **1564 rows** covering **33 of the 35** quantities §13.8 names, drawn from 13 source runs.

Checker (`docs/reference_scripts/check_model_inputs.py`, run in place, never adapted): **exit 0 — passes**

## Coverage

| quantity | rows |
|---|---:|
| t_preprocess | 0 |
| t_vision | 33 |
| t_vlm_prefix | 33 |
| t_postprocess | 0 |
| t_ae_step | 8 |
| t_ae_total | 33 |
| t_e2e | 66 |
| t_host_segment | 648 |
| n_kernels_host_segment | 61 |
| bytes_host_segment | 9 |
| flops_host_segment | 9 |
| t_host_layer | 24 |
| t_step_extra | 87 |
| t_launch_per_kernel | 62 |
| t_sync | 15 |
| xfer_latency | 216 |
| xfer_latency_fit | 12 |
| kv_bytes_per_layer | 2 |
| kv_bytes_total | 2 |
| t_kv_materialise | 15 |
| kv_copied_bytes_per_step | 1 |
| boundary_bytes_per_step | 5 |
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
| E11a_pi0_hostseg | `20260922b_E11a_pi0_hostseg` | 934 |
| E11b_xfer | `20260922b_E11b_xfer` | 243 |
| E11c_pi0_handover | `20260922b_E11c_pi0_handover` | 25 |
| E11d_pi0_eager_prefixsweep | `20260922b_E11d_pi0_eager_prefixsweep` | 111 |
| E11d_pi0_reduce-overhead_prefixsweep | `20260922b_E11d_pi0_reduce-overhead_prefixsweep` | 27 |
| E11e_pi0_eager_mempressure | `20260922d_E11e_pi0_eager_mempressure` | 84 |
| E11e_pi0_reduce-overhead_mempressure | `20260922d_E11e_pi0_reduce-overhead_mempressure` | 84 |
| E11f_pi0_pipelining | `20260922b_E11f_pi0_pipelining` | 18 |
| E3 eager N=4 | `20260921_E3_pi0_eager_N4` | 12 |
| E3 eager N=10 | `20260921_E3_pi0_eager_N10` | 12 |
| E4 reduce-overhead N=4 | `20260921_E4_pi0_reduce-overhead_N4` | 3 |
| E4 reduce-overhead N=10 | `20260921_E4_pi0_reduce-overhead_N10` | 3 |
| E4 slopes | `20260921_E4_pi0_summary.csv` | 8 |

Rows restated from another run carry `from=<run dir>` in `variant` and `source_file` points at a copy of that run's tidy CSV under `raw/restated/<run dir>/`, never at a profiler report.

## Conditions

π0 `lerobot/pi0_libero_finetuned_v044` @ `45dcc8fc`, bf16, MAXN + jetson_clocks, P = 816 processed (533 valid), S = 51. Each row's own run directory holds its gate record, device state and raw files; this directory is derived and measures nothing itself.

## Reproduce

```
python bench/consolidate_e11.py --outdir results/20260922b_E11_pi0_inputs
python docs/reference_scripts/check_model_inputs.py results/20260922b_E11_pi0_inputs/model_inputs.csv
```

## Not measured

- t_preprocess: not measured — see reports/E11_pi0_hybrid_inputs.md
- t_postprocess: not measured — see reports/E11_pi0_hybrid_inputs.md
