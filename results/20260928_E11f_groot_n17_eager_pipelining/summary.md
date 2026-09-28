# E11f - GR00T N1.7 pipelining (eager)

Run directory: `20260928_E11f_groot_n17_eager_pipelining`  
Written by `bench/groot_phases.py` at 2026-09-28T22:18:05Z.

## Throughput

| phase | mode | median_ms | p5_ms | p95_ms | n | throughput_inf_per_s |
|---|---|---|---|---|---|---|
| prefix | eager | 157.1862 | 156.5973 | 158.255 | 382 | 6.3557 |
| ae | eager | 214.0748 | 213.261 | 215.3929 | 281 | 4.6683 |
| full | eager | 363.3691 | 362.3333 | 365.1151 | 166 | 2.7508 |

## Concurrency (host sequence vs the prefix loop)

| sharing | condition | median_ms | p95_ms | max_ms | n | slowdown | slowdown_non_sleep |
|---|---|---|---|---|---|---|---|
| none | alone | 1411.7874 | 1412.5559 | 1413.0332 | 30 |  |  |
| two_streams | with_prefix_loop | 1558.2662 | 1560.3148 | 1560.6603 | 30 | 1.1038 | 2.1115 |
| two_processes | with_prefix_loop | 1416.2268 | 1434.727 | 1444.4196 | 30 | 1.0031 | 1.0337 |

## GPU idle inside the denoising loop

| mode | N | ae_total_ms_unprofiled | kernel_busy_ms | gpu_idle_frac_corrected | note |
|---|---|---|---|---|---|
| eager | 4 | 214.8315 | 80.340032 | 0.626 |  |
| eager | 10 | 525.5481 | 202.088112 | 0.6155 |  |

## Conditions

rung `eager`, N = 4; host sequence = 32 DiT blocks x N of the host segments (all E11a segments except ('attention_fused', 'feedforward')), 2.0 ms pause between segments; two streams from one thread and a second process running the prefix loop (§13.7).

## Reproduce

```
python bench/groot_phases.py --version n17 --checkpoint nvidia/GR00T-N1.7-3B --rung eager --tf32 on --e11 pipelining --seconds 60 --outdir results/20260928_E11f_groot_n17_eager_pipelining
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
