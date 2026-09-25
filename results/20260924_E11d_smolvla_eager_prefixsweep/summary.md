# E11d — prefix latency vs its input (measurement plan, part 4)

Run directory: `20260924_E11d_smolvla_eager_prefixsweep`  
Written by `bench/prefix_sweep.py` at 2026-09-24T23:21:05Z.

## Answer — prefix latency vs prefix length

| mode | images | prompt | prompt tok | image tok | P processed | P valid | region | median ms | p5 | p95 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eager | 1 | short | 48 | 65 | 113 | 71 | vision | 16.730 | 16.664 | 16.795 | 30 |
| eager | 1 | short | 48 | 65 | 113 | 71 | vlm_prefix | 64.864 | 64.664 | 65.179 | 30 |
| eager | 1 | short | 48 | 65 | 113 | 71 | ae_step[0] | 61.333 | 61.107 | 61.530 | 30 |
| eager | 1 | short | 48 | 65 | 113 | 71 | ae_step[1] | 61.134 | 61.005 | 61.322 | 30 |
| eager | 1 | short | 48 | 65 | 113 | 71 | ae_step[2] | 61.206 | 60.987 | 61.410 | 30 |
| eager | 1 | short | 48 | 65 | 113 | 71 | ae_step[3] | 61.170 | 61.008 | 61.464 | 30 |
| eager | 1 | short | 48 | 65 | 113 | 71 | ae_total | 245.710 | 245.119 | 246.160 | 30 |
| eager | 1 | short | 48 | 65 | 113 | 71 | e2e | 334.358 | 333.803 | 335.083 | 30 |
| eager | 1 | typical | 48 | 65 | 113 | 87 | vision | 16.728 | 16.685 | 16.825 | 30 |
| eager | 1 | typical | 48 | 65 | 113 | 87 | vlm_prefix | 64.982 | 64.772 | 65.184 | 30 |
| eager | 1 | typical | 48 | 65 | 113 | 87 | ae_step[0] | 61.485 | 61.229 | 61.845 | 30 |
| eager | 1 | typical | 48 | 65 | 113 | 87 | ae_step[1] | 61.282 | 61.038 | 61.536 | 30 |
| eager | 1 | typical | 48 | 65 | 113 | 87 | ae_step[2] | 61.212 | 61.083 | 61.470 | 30 |
| eager | 1 | typical | 48 | 65 | 113 | 87 | ae_step[3] | 61.232 | 60.999 | 61.428 | 30 |
| eager | 1 | typical | 48 | 65 | 113 | 87 | ae_total | 246.120 | 245.374 | 246.699 | 30 |
| eager | 1 | typical | 48 | 65 | 113 | 87 | e2e | 334.852 | 334.054 | 335.467 | 30 |
| eager | 1 | long | 48 | 65 | 113 | 105 | vision | 16.735 | 16.692 | 16.819 | 30 |
| eager | 1 | long | 48 | 65 | 113 | 105 | vlm_prefix | 64.994 | 64.777 | 65.366 | 30 |
| eager | 1 | long | 48 | 65 | 113 | 105 | ae_step[0] | 61.472 | 61.325 | 61.767 | 30 |
| eager | 1 | long | 48 | 65 | 113 | 105 | ae_step[1] | 61.319 | 61.101 | 61.471 | 30 |
| eager | 1 | long | 48 | 65 | 113 | 105 | ae_step[2] | 61.380 | 61.181 | 61.578 | 30 |
| eager | 1 | long | 48 | 65 | 113 | 105 | ae_step[3] | 61.240 | 61.030 | 61.401 | 30 |
| eager | 1 | long | 48 | 65 | 113 | 105 | ae_total | 246.177 | 245.808 | 246.706 | 30 |
| eager | 1 | long | 48 | 65 | 113 | 105 | e2e | 334.992 | 334.436 | 335.784 | 30 |
| eager | 2 | short | 48 | 129 | 177 | 135 | vision | 33.117 | 33.020 | 33.331 | 30 |
| eager | 2 | short | 48 | 129 | 177 | 135 | vlm_prefix | 65.091 | 64.939 | 65.304 | 30 |
| eager | 2 | short | 48 | 129 | 177 | 135 | ae_step[0] | 61.490 | 61.293 | 61.806 | 30 |
| eager | 2 | short | 48 | 129 | 177 | 135 | ae_step[1] | 61.172 | 61.023 | 61.403 | 30 |
| eager | 2 | short | 48 | 129 | 177 | 135 | ae_step[2] | 61.189 | 60.983 | 61.486 | 30 |
| eager | 2 | short | 48 | 129 | 177 | 135 | ae_step[3] | 61.157 | 61.004 | 61.475 | 30 |
| eager | 2 | short | 48 | 129 | 177 | 135 | ae_total | 245.792 | 245.409 | 247.046 | 30 |
| eager | 2 | short | 48 | 129 | 177 | 135 | e2e | 351.574 | 351.010 | 352.933 | 30 |
| eager | 2 | typical | 48 | 129 | 177 | 151 | vision | 33.151 | 33.094 | 33.353 | 30 |
| eager | 2 | typical | 48 | 129 | 177 | 151 | vlm_prefix | 65.160 | 64.942 | 65.452 | 30 |
| eager | 2 | typical | 48 | 129 | 177 | 151 | ae_step[0] | 61.561 | 61.393 | 61.860 | 30 |
| eager | 2 | typical | 48 | 129 | 177 | 151 | ae_step[1] | 61.306 | 61.099 | 61.650 | 30 |
| eager | 2 | typical | 48 | 129 | 177 | 151 | ae_step[2] | 61.290 | 61.125 | 61.457 | 30 |
| eager | 2 | typical | 48 | 129 | 177 | 151 | ae_step[3] | 61.311 | 61.112 | 61.577 | 30 |
| eager | 2 | typical | 48 | 129 | 177 | 151 | ae_total | 246.294 | 245.871 | 246.851 | 30 |
| eager | 2 | typical | 48 | 129 | 177 | 151 | e2e | 352.262 | 351.537 | 352.990 | 30 |
| eager | 2 | long | 48 | 129 | 177 | 169 | vision | 33.161 | 33.086 | 33.359 | 30 |
| eager | 2 | long | 48 | 129 | 177 | 169 | vlm_prefix | 65.215 | 65.007 | 65.516 | 30 |
| eager | 2 | long | 48 | 129 | 177 | 169 | ae_step[0] | 61.540 | 61.391 | 61.773 | 30 |
| eager | 2 | long | 48 | 129 | 177 | 169 | ae_step[1] | 61.250 | 61.106 | 61.577 | 30 |
| eager | 2 | long | 48 | 129 | 177 | 169 | ae_step[2] | 61.263 | 61.064 | 61.490 | 30 |
| eager | 2 | long | 48 | 129 | 177 | 169 | ae_step[3] | 61.264 | 61.031 | 61.548 | 30 |
| eager | 2 | long | 48 | 129 | 177 | 169 | ae_total | 246.136 | 245.607 | 246.884 | 30 |
| eager | 2 | long | 48 | 129 | 177 | 169 | e2e | 352.106 | 351.444 | 353.308 | 30 |
| eager | 3 | short | 48 | 193 | 241 | 199 | vision | 49.413 | 49.287 | 49.536 | 30 |
| eager | 3 | short | 48 | 193 | 241 | 199 | vlm_prefix | 65.014 | 64.764 | 65.448 | 30 |
| eager | 3 | short | 48 | 193 | 241 | 199 | ae_step[0] | 61.660 | 61.377 | 61.924 | 30 |
| eager | 3 | short | 48 | 193 | 241 | 199 | ae_step[1] | 61.317 | 61.142 | 61.632 | 30 |
| eager | 3 | short | 48 | 193 | 241 | 199 | ae_step[2] | 61.374 | 61.181 | 61.587 | 30 |
| eager | 3 | short | 48 | 193 | 241 | 199 | ae_step[3] | 61.245 | 61.037 | 61.562 | 30 |
| eager | 3 | short | 48 | 193 | 241 | 199 | ae_total | 246.319 | 245.886 | 247.063 | 30 |
| eager | 3 | short | 48 | 193 | 241 | 199 | e2e | 368.852 | 368.136 | 370.098 | 30 |
| eager | 3 | typical | 48 | 193 | 241 | 215 | vision | 49.357 | 49.248 | 49.517 | 30 |
| eager | 3 | typical | 48 | 193 | 241 | 215 | vlm_prefix | 64.983 | 64.815 | 65.252 | 30 |
| eager | 3 | typical | 48 | 193 | 241 | 215 | ae_step[0] | 61.635 | 61.354 | 61.795 | 30 |
| eager | 3 | typical | 48 | 193 | 241 | 215 | ae_step[1] | 61.298 | 61.098 | 61.608 | 30 |
| eager | 3 | typical | 48 | 193 | 241 | 215 | ae_step[2] | 61.294 | 61.118 | 61.566 | 30 |
| eager | 3 | typical | 48 | 193 | 241 | 215 | ae_step[3] | 61.263 | 61.009 | 61.560 | 30 |
| eager | 3 | typical | 48 | 193 | 241 | 215 | ae_total | 246.346 | 245.599 | 246.881 | 30 |
| eager | 3 | typical | 48 | 193 | 241 | 215 | e2e | 368.800 | 367.870 | 369.345 | 30 |
| eager | 3 | long | 48 | 193 | 241 | 233 | vision | 49.345 | 49.196 | 49.565 | 30 |
| eager | 3 | long | 48 | 193 | 241 | 233 | vlm_prefix | 64.943 | 64.590 | 65.143 | 30 |
| eager | 3 | long | 48 | 193 | 241 | 233 | ae_step[0] | 61.569 | 61.340 | 61.862 | 30 |
| eager | 3 | long | 48 | 193 | 241 | 233 | ae_step[1] | 61.342 | 61.128 | 61.619 | 30 |
| eager | 3 | long | 48 | 193 | 241 | 233 | ae_step[2] | 61.404 | 61.163 | 61.607 | 30 |
| eager | 3 | long | 48 | 193 | 241 | 233 | ae_step[3] | 61.258 | 61.097 | 61.520 | 30 |
| eager | 3 | long | 48 | 193 | 241 | 233 | ae_total | 246.404 | 245.908 | 247.080 | 30 |
| eager | 3 | long | 48 | 193 | 241 | 233 | e2e | 368.698 | 368.025 | 369.605 | 30 |

## Fits

| fit | region | x | a (ms) | b | unit b | r2 | points | note |
|---|---|---|---|---|---|---|---|---|
| vision_vs_images | vision | n_images | 0.4517 | 16.31378 | ms_per_image | 1.0000 | 3 |  |
| vlm_prefix_vs_P | vlm_prefix | P_processed | 65.0406 | 6.17e-06 | ms_per_token | 0.0000 | 3 |  |
| ae_step_vs_P | ae_total | P_processed | 245.8877 | 0.00176836 | ms_per_token | 0.8023 | 3 |  |
| ae_step_vs_P | e2e | P_processed | 304.9746 | 0.26522367 | ms_per_token | 0.9999 | 3 |  |

## Conditions

π0, bf16, MAXN + jetson_clocks. **P_processed is constant at 816 for every point**: the processor always encodes three image slots and pads the prompt to 48 tokens, so what the sweep varies is how many of those 816 positions carry a valid mask (P_valid), not how many are computed. That is why prefix latency is flat.

## Reproduce

```
python bench/prefix_sweep.py --outdir-prefix results/20260924_E11d_smolvla
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
