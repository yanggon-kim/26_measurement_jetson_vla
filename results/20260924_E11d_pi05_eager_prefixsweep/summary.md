# E11d — prefix latency vs its input (measurement plan, part 4)

Run directory: `20260924_E11d_pi05_eager_prefixsweep`  
Written by `bench/prefix_sweep.py` at 2026-09-24T21:36:32Z.

## Answer — prefix latency vs prefix length

| mode | images | prompt | prompt tok | image tok | P processed | P valid | region | median ms | p5 | p95 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eager | 1 | short | 200 | 768 | 968 | 303 | vision | 241.287 | 241.205 | 241.358 | 30 |
| eager | 1 | short | 200 | 768 | 968 | 303 | vlm_prefix | 2179.549 | 2177.581 | 2181.438 | 30 |
| eager | 1 | short | 200 | 768 | 968 | 303 | ae_step[0] | 60.624 | 60.381 | 62.055 | 30 |
| eager | 1 | short | 200 | 768 | 968 | 303 | ae_step[1] | 59.567 | 59.318 | 60.957 | 30 |
| eager | 1 | short | 200 | 768 | 968 | 303 | ae_step[2] | 59.309 | 59.063 | 60.506 | 30 |
| eager | 1 | short | 200 | 768 | 968 | 303 | ae_step[3] | 59.326 | 59.119 | 60.677 | 30 |
| eager | 1 | short | 200 | 768 | 968 | 303 | ae_total | 239.642 | 239.005 | 244.005 | 30 |
| eager | 1 | short | 200 | 768 | 968 | 303 | e2e | 2667.652 | 2664.484 | 2672.101 | 30 |
| eager | 1 | typical | 200 | 768 | 968 | 318 | vision | 239.775 | 239.360 | 241.336 | 30 |
| eager | 1 | typical | 200 | 768 | 968 | 318 | vlm_prefix | 2172.252 | 2161.873 | 2181.230 | 30 |
| eager | 1 | typical | 200 | 768 | 968 | 318 | ae_step[0] | 60.624 | 60.304 | 61.156 | 30 |
| eager | 1 | typical | 200 | 768 | 968 | 318 | ae_step[1] | 59.546 | 59.368 | 59.783 | 30 |
| eager | 1 | typical | 200 | 768 | 968 | 318 | ae_step[2] | 59.317 | 59.182 | 59.679 | 30 |
| eager | 1 | typical | 200 | 768 | 968 | 318 | ae_step[3] | 59.325 | 59.157 | 59.786 | 30 |
| eager | 1 | typical | 200 | 768 | 968 | 318 | ae_total | 239.690 | 239.086 | 240.612 | 30 |
| eager | 1 | typical | 200 | 768 | 968 | 318 | e2e | 2657.939 | 2646.642 | 2668.767 | 30 |
| eager | 1 | long | 200 | 768 | 968 | 336 | vision | 239.261 | 239.174 | 239.485 | 30 |
| eager | 1 | long | 200 | 768 | 968 | 336 | vlm_prefix | 2162.202 | 2159.488 | 2164.612 | 30 |
| eager | 1 | long | 200 | 768 | 968 | 336 | ae_step[0] | 60.709 | 60.368 | 61.015 | 30 |
| eager | 1 | long | 200 | 768 | 968 | 336 | ae_step[1] | 59.586 | 59.345 | 59.848 | 30 |
| eager | 1 | long | 200 | 768 | 968 | 336 | ae_step[2] | 59.369 | 59.137 | 59.532 | 30 |
| eager | 1 | long | 200 | 768 | 968 | 336 | ae_step[3] | 59.411 | 59.295 | 59.566 | 30 |
| eager | 1 | long | 200 | 768 | 968 | 336 | ae_total | 240.000 | 239.143 | 240.377 | 30 |
| eager | 1 | long | 200 | 768 | 968 | 336 | e2e | 2647.412 | 2645.167 | 2650.108 | 30 |
| eager | 2 | short | 200 | 768 | 968 | 559 | vision | 239.129 | 239.085 | 239.217 | 30 |
| eager | 2 | short | 200 | 768 | 968 | 559 | vlm_prefix | 2160.963 | 2158.538 | 2164.063 | 30 |
| eager | 2 | short | 200 | 768 | 968 | 559 | ae_step[0] | 60.620 | 60.412 | 60.897 | 30 |
| eager | 2 | short | 200 | 768 | 968 | 559 | ae_step[1] | 59.572 | 59.445 | 59.785 | 30 |
| eager | 2 | short | 200 | 768 | 968 | 559 | ae_step[2] | 59.378 | 59.152 | 59.686 | 30 |
| eager | 2 | short | 200 | 768 | 968 | 559 | ae_step[3] | 59.380 | 59.246 | 59.580 | 30 |
| eager | 2 | short | 200 | 768 | 968 | 559 | ae_total | 239.796 | 239.431 | 240.370 | 30 |
| eager | 2 | short | 200 | 768 | 968 | 559 | e2e | 2646.167 | 2643.815 | 2649.737 | 30 |
| eager | 2 | typical | 200 | 768 | 968 | 574 | vision | 239.103 | 239.052 | 239.148 | 30 |
| eager | 2 | typical | 200 | 768 | 968 | 574 | vlm_prefix | 2160.909 | 2157.960 | 2163.313 | 30 |
| eager | 2 | typical | 200 | 768 | 968 | 574 | ae_step[0] | 60.691 | 60.463 | 61.066 | 30 |
| eager | 2 | typical | 200 | 768 | 968 | 574 | ae_step[1] | 59.645 | 59.509 | 59.853 | 30 |
| eager | 2 | typical | 200 | 768 | 968 | 574 | ae_step[2] | 59.445 | 59.308 | 59.773 | 30 |
| eager | 2 | typical | 200 | 768 | 968 | 574 | ae_step[3] | 59.507 | 59.325 | 59.825 | 30 |
| eager | 2 | typical | 200 | 768 | 968 | 574 | ae_total | 240.324 | 239.496 | 240.842 | 30 |
| eager | 2 | typical | 200 | 768 | 968 | 574 | e2e | 2646.837 | 2643.622 | 2649.881 | 30 |
| eager | 2 | long | 200 | 768 | 968 | 592 | vision | 239.127 | 239.089 | 239.176 | 30 |
| eager | 2 | long | 200 | 768 | 968 | 592 | vlm_prefix | 2161.352 | 2159.576 | 2163.459 | 30 |
| eager | 2 | long | 200 | 768 | 968 | 592 | ae_step[0] | 60.681 | 60.286 | 61.155 | 30 |
| eager | 2 | long | 200 | 768 | 968 | 592 | ae_step[1] | 59.650 | 59.464 | 59.910 | 30 |
| eager | 2 | long | 200 | 768 | 968 | 592 | ae_step[2] | 59.482 | 59.233 | 59.779 | 30 |
| eager | 2 | long | 200 | 768 | 968 | 592 | ae_step[3] | 59.479 | 59.236 | 59.988 | 30 |
| eager | 2 | long | 200 | 768 | 968 | 592 | ae_total | 239.995 | 239.508 | 241.391 | 30 |
| eager | 2 | long | 200 | 768 | 968 | 592 | e2e | 2647.022 | 2645.279 | 2649.707 | 30 |
| eager | 3 | short | 200 | 768 | 968 | 815 | vision | 239.139 | 239.095 | 239.218 | 30 |
| eager | 3 | short | 200 | 768 | 968 | 815 | vlm_prefix | 2161.862 | 2158.490 | 2163.414 | 30 |
| eager | 3 | short | 200 | 768 | 968 | 815 | ae_step[0] | 60.637 | 60.260 | 61.139 | 30 |
| eager | 3 | short | 200 | 768 | 968 | 815 | ae_step[1] | 59.505 | 59.286 | 60.121 | 30 |
| eager | 3 | short | 200 | 768 | 968 | 815 | ae_step[2] | 59.327 | 59.140 | 59.877 | 30 |
| eager | 3 | short | 200 | 768 | 968 | 815 | ae_step[3] | 59.272 | 59.099 | 59.675 | 30 |
| eager | 3 | short | 200 | 768 | 968 | 815 | ae_total | 239.607 | 238.846 | 241.620 | 30 |
| eager | 3 | short | 200 | 768 | 968 | 815 | e2e | 2647.307 | 2643.443 | 2650.716 | 30 |
| eager | 3 | typical | 200 | 768 | 968 | 830 | vision | 239.178 | 239.148 | 239.215 | 30 |
| eager | 3 | typical | 200 | 768 | 968 | 830 | vlm_prefix | 2162.198 | 2159.858 | 2164.145 | 30 |
| eager | 3 | typical | 200 | 768 | 968 | 830 | ae_step[0] | 60.547 | 60.209 | 60.873 | 30 |
| eager | 3 | typical | 200 | 768 | 968 | 830 | ae_step[1] | 59.538 | 59.452 | 59.714 | 30 |
| eager | 3 | typical | 200 | 768 | 968 | 830 | ae_step[2] | 59.422 | 59.275 | 59.633 | 30 |
| eager | 3 | typical | 200 | 768 | 968 | 830 | ae_step[3] | 59.359 | 59.229 | 59.668 | 30 |
| eager | 3 | typical | 200 | 768 | 968 | 830 | ae_total | 239.756 | 239.406 | 240.281 | 30 |
| eager | 3 | typical | 200 | 768 | 968 | 830 | e2e | 2647.702 | 2645.276 | 2649.863 | 30 |
| eager | 3 | long | 200 | 768 | 968 | 848 | vision | 239.142 | 239.106 | 239.202 | 30 |
| eager | 3 | long | 200 | 768 | 968 | 848 | vlm_prefix | 2161.029 | 2158.637 | 2163.306 | 30 |
| eager | 3 | long | 200 | 768 | 968 | 848 | ae_step[0] | 60.908 | 60.592 | 61.330 | 30 |
| eager | 3 | long | 200 | 768 | 968 | 848 | ae_step[1] | 59.725 | 59.521 | 60.175 | 30 |
| eager | 3 | long | 200 | 768 | 968 | 848 | ae_step[2] | 59.483 | 59.328 | 59.752 | 30 |
| eager | 3 | long | 200 | 768 | 968 | 848 | ae_step[3] | 59.433 | 59.221 | 59.717 | 30 |
| eager | 3 | long | 200 | 768 | 968 | 848 | ae_total | 240.387 | 239.731 | 241.463 | 30 |
| eager | 3 | long | 200 | 768 | 968 | 848 | e2e | 2647.726 | 2645.143 | 2649.993 | 30 |

## Fits

| fit | region | x | a (ms) | b | unit b | r2 | points | note |
|---|---|---|---|---|---|---|---|---|
| vision_vs_images | vision | n_images | 239.9813 | -0.31666 | ms_per_image | 0.7332 | 3 |  |
| vlm_prefix_vs_P | vlm_prefix | P_processed |  |  |  |  | 1 | x is constant at 968 across the sweep; no slope can be fitted, which is itself the result |
| ae_step_vs_P | ae_total | P_processed |  |  |  |  | 1 | x is constant at 968 across the sweep; no slope can be fitted, which is itself the result |
| ae_step_vs_P | e2e | P_processed |  |  |  |  | 1 | x is constant at 968 across the sweep; no slope can be fitted, which is itself the result |

## Conditions

π0, bf16, MAXN + jetson_clocks. **P_processed is constant at 816 for every point**: the processor always encodes three image slots and pads the prompt to 48 tokens, so what the sweep varies is how many of those 816 positions carry a valid mask (P_valid), not how many are computed. That is why prefix latency is flat.

## Reproduce

```
python bench/prefix_sweep.py --outdir-prefix results/20260924_E11d_pi05
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
