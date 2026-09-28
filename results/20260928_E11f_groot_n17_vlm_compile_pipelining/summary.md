# E11f - GR00T N1.7 pipelining (vlm_compile)

Run directory: `20260928_E11f_groot_n17_vlm_compile_pipelining`  
Written by `bench/groot_phases.py` at 2026-09-28T22:26:27Z.

## Throughput

| phase | mode | median_ms | p5_ms | p95_ms | n | throughput_inf_per_s |
|---|---|---|---|---|---|---|
| prefix | vlm_compile | 154.2925 | 153.7889 | 154.7995 | 389 | 6.4814 |
| ae | vlm_compile | 81.4632 | 81.3357 | 81.7618 | 737 | 12.27 |
| full | vlm_compile | 236.4501 | 235.7499 | 237.1806 | 254 | 4.2289 |

## Concurrency (host sequence vs the prefix loop)

| sharing | condition | median_ms | p95_ms | max_ms | n | slowdown | slowdown_non_sleep |
|---|---|---|---|---|---|---|---|
| none | alone | 1411.6679 | 1412.4578 | 1414.6847 | 30 |  |  |
| two_streams | with_prefix_loop | 1554.3974 | 1555.2637 | 1555.4595 | 30 | 1.1011 | 2.084 |
| two_processes | with_prefix_loop | 1414.2163 | 1437.7547 | 1452.3507 | 30 | 1.0018 | 1.0194 |

## GPU idle inside the denoising loop

| mode | N | ae_total_ms_unprofiled | kernel_busy_ms | gpu_idle_frac_corrected | note |
|---|---|---|---|---|---|
| vlm_compile | 4 |  |  |  | not measured: the compiled action head replays as one CUDA graph, so there are no per-step NVTX ranges (as for the LeRobot compiled captures) |

## Conditions

rung `vlm_compile`, N = 4; host sequence = 32 DiT blocks x N of the host segments (all E11a segments except ('attention_fused', 'feedforward')), 2.0 ms pause between segments; two streams from one thread and a second process running the prefix loop (§13.7).

## Reproduce

```
python bench/groot_phases.py --version n17 --checkpoint nvidia/GR00T-N1.7-3B --rung vlm_compile --tf32 on --head-mode reduce-overhead --llm-mode max-autotune-no-cudagraphs --e11 pipelining --seconds 60 --outdir results/20260928_E11f_groot_n17_vlm_compile_pipelining
```

## Not measured

- not measured: the compiled action head replays as one CUDA graph, so there are no per-step NVTX ranges (as for the LeRobot compiled captures)
