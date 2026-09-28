# E11 — the consolidated hand-off (protocol §13.8)

Run directory: `20260928_E11_groot_n17_inputs`  
Written by `bench/consolidate_e11.py` at 2026-09-28T22:44:45Z.

## Answer

`model_inputs.csv` carries **2201 rows** covering **32 of the 35** quantities §13.8 names, drawn from 13 source runs.

Checker (`docs/reference_scripts/check_model_inputs.py`, run in place, never adapted): **exit 0 — passes**

## Coverage

| quantity | rows |
|---|---:|
| t_preprocess | 12 |
| t_vision | 30 |
| t_vlm_prefix | 30 |
| t_postprocess | 0 |
| t_ae_step | 9 |
| t_ae_total | 30 |
| t_e2e | 30 |
| t_host_segment | 1008 |
| n_kernels_host_segment | 80 |
| bytes_host_segment | 28 |
| flops_host_segment | 28 |
| t_host_layer | 48 |
| t_step_extra | 291 |
| t_launch_per_kernel | 81 |
| t_sync | 0 |
| xfer_latency | 192 |
| xfer_latency_fit | 0 |
| kv_bytes_per_layer | 16 |
| kv_bytes_total | 2 |
| t_kv_materialise | 18 |
| kv_copied_bytes_per_step | 1 |
| boundary_bytes_per_step | 4 |
| prefix_latency_fit | 21 |
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
| achieved_bw_GBps | 28 |
| achieved_tflops | 28 |

## Provenance

| part | run directory | rows |
|---|---|---:|
| E11a_groot_n17_hostseg | `20260928_E11a_groot_n17_hostseg` | 1620 |
| E11c_groot_n17_handover | `20260928_E11c_groot_n17_handover` | 41 |
| E11c_groot_n17_handover_xfer | `20260928_E11c_groot_n17_handover_xfer` | 192 |
| E11d_groot_n17_eager_prefixsweep | `20260928_E11d_groot_n17_eager_prefixsweep` | 57 |
| E11d_groot_n17_vlm_compile_prefixsweep | `20260928_E11d_groot_n17_vlm_compile_prefixsweep` | 45 |
| E11e_groot_n17_eager_mempressure | `20260928_E11e_groot_n17_eager_mempressure` | 84 |
| E11e_groot_n17_vlm_compile_mempressure | `20260928_E11e_groot_n17_vlm_compile_mempressure` | 84 |
| E11f_groot_n17_eager_pipelining | `20260928_E11f_groot_n17_eager_pipelining` | 13 |
| E11f_groot_n17_vlm_compile_pipelining | `20260928_E11f_groot_n17_vlm_compile_pipelining` | 5 |
| E3/E4 phase split N=4 eager_tf32 | `20260925_E3E4_groot_n17_N4_eager_tf32` | 15 |
| E3/E4 phase split N=10 eager_tf32 | `20260925_E3E4_groot_n17_N10_eager_tf32` | 15 |
| E3/E4 phase split N=4 vlm_compile | `20260925_E3E4_groot_n17_N4_vlm_compile` | 15 |
| E3/E4 phase split N=10 vlm_compile | `20260925_E3E4_groot_n17_N10_vlm_compile` | 15 |

Rows restated from another run carry `from=<run dir>` in `variant` and `source_file` points at a copy of that run's tidy CSV under `raw/restated/<run dir>/`, never at a profiler report.

## Conditions

groot_n17 `nvidia/GR00T-N1.7-3B`, bf16, MAXN + jetson_clocks, P = ['270', '273'], S = 41. Each row's own run directory holds its gate record, device state and raw files; this directory is derived and measures nothing itself.

## Reproduce

```
python bench/consolidate_e11.py --model groot_n17 --outdir results/20260928_E11_groot_n17_inputs
python docs/reference_scripts/check_model_inputs.py results/20260928_E11_groot_n17_inputs/model_inputs.csv
```

## Not measured

- t_postprocess: not measured — see reports/E11_groot_n17_hybrid_inputs.md
- t_sync: not measured — see reports/E11_groot_n17_hybrid_inputs.md
- xfer_latency_fit: not measured — see reports/E11_groot_n17_hybrid_inputs.md
