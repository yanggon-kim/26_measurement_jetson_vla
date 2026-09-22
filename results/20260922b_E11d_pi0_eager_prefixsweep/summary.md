# E11d — prefix latency vs its input (measurement plan, part 4)

Run directory: `20260922b_E11d_pi0_eager_prefixsweep`  
Written by `bench/prefix_sweep.py` at 2026-09-22T13:46:21Z.

## Answer — prefix latency vs prefix length

| mode | images | prompt | prompt tok | image tok | P processed | P valid | region | median ms | p5 | p95 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eager | 1 | short | 48 | 768 | 816 | 262 | vision | 241.228 | 241.079 | 241.350 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | vlm_prefix | 234.301 | 234.174 | 234.507 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | ae_step[0] | 57.632 | 57.351 | 57.952 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | ae_step[1] | 57.087 | 56.900 | 57.321 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | ae_step[2] | 56.931 | 56.746 | 57.213 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | ae_step[3] | 56.922 | 56.767 | 57.243 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | ae_step[4] | 56.915 | 56.716 | 57.241 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | ae_step[5] | 56.854 | 56.681 | 57.095 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | ae_step[6] | 56.915 | 56.716 | 57.210 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | ae_step[7] | 56.865 | 56.711 | 57.231 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | ae_step[8] | 56.849 | 56.715 | 57.193 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | ae_step[9] | 56.934 | 56.740 | 57.237 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | ae_total | 572.631 | 571.726 | 573.723 | 100 |
| eager | 1 | short | 48 | 768 | 816 | 262 | e2e | 1054.236 | 1053.246 | 1055.367 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | vision | 241.165 | 241.048 | 241.273 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | vlm_prefix | 234.291 | 234.180 | 234.464 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | ae_step[0] | 57.914 | 57.595 | 58.329 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | ae_step[1] | 57.321 | 57.164 | 57.752 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | ae_step[2] | 57.189 | 56.999 | 57.611 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | ae_step[3] | 57.227 | 56.983 | 57.618 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | ae_step[4] | 57.190 | 56.963 | 57.475 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | ae_step[5] | 57.106 | 56.909 | 57.619 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | ae_step[6] | 57.121 | 56.931 | 57.634 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | ae_step[7] | 57.126 | 56.969 | 57.550 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | ae_step[8] | 57.142 | 56.933 | 57.478 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | ae_step[9] | 57.129 | 56.889 | 57.488 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | ae_total | 575.506 | 573.734 | 577.073 | 100 |
| eager | 1 | typical | 48 | 768 | 816 | 277 | e2e | 1057.011 | 1055.239 | 1058.632 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | vision | 241.188 | 241.043 | 241.298 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | vlm_prefix | 234.316 | 234.209 | 234.448 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | ae_step[0] | 57.820 | 57.514 | 58.291 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | ae_step[1] | 57.260 | 57.012 | 57.569 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | ae_step[2] | 57.098 | 56.892 | 57.448 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | ae_step[3] | 57.114 | 56.963 | 57.448 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | ae_step[4] | 57.072 | 56.889 | 57.430 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | ae_step[5] | 57.050 | 56.826 | 57.386 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | ae_step[6] | 57.071 | 56.865 | 57.364 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | ae_step[7] | 57.033 | 56.822 | 57.368 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | ae_step[8] | 57.013 | 56.837 | 57.330 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | ae_step[9] | 57.058 | 56.887 | 57.366 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | ae_total | 574.202 | 573.052 | 576.374 | 100 |
| eager | 1 | long | 48 | 768 | 816 | 295 | e2e | 1055.775 | 1054.786 | 1057.969 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | vision | 240.640 | 240.124 | 241.256 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | vlm_prefix | 234.408 | 234.179 | 234.863 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | ae_step[0] | 57.839 | 57.570 | 58.115 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | ae_step[1] | 57.249 | 57.054 | 57.455 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | ae_step[2] | 57.173 | 57.016 | 57.468 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | ae_step[3] | 57.077 | 56.848 | 57.356 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | ae_step[4] | 57.106 | 56.898 | 57.326 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | ae_step[5] | 57.052 | 56.880 | 57.344 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | ae_step[6] | 57.081 | 56.903 | 57.324 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | ae_step[7] | 57.060 | 56.839 | 57.322 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | ae_step[8] | 57.096 | 56.898 | 57.408 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | ae_step[9] | 57.051 | 56.892 | 57.356 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | ae_total | 574.369 | 573.393 | 575.226 | 100 |
| eager | 2 | short | 48 | 768 | 816 | 518 | e2e | 1055.849 | 1054.546 | 1057.006 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | vision | 240.180 | 239.920 | 240.710 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | vlm_prefix | 233.296 | 232.738 | 234.627 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | ae_step[0] | 57.702 | 57.404 | 58.012 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | ae_step[1] | 57.138 | 56.899 | 57.460 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | ae_step[2] | 57.095 | 56.883 | 57.356 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | ae_step[3] | 57.019 | 56.814 | 57.248 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | ae_step[4] | 57.002 | 56.802 | 57.265 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | ae_step[5] | 56.943 | 56.740 | 57.172 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | ae_step[6] | 56.988 | 56.823 | 57.272 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | ae_step[7] | 56.909 | 56.734 | 57.217 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | ae_step[8] | 56.982 | 56.765 | 57.266 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | ae_step[9] | 56.933 | 56.788 | 57.185 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | ae_total | 573.212 | 572.374 | 574.303 | 100 |
| eager | 2 | typical | 48 | 768 | 816 | 533 | e2e | 1053.434 | 1051.972 | 1054.807 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | vision | 240.019 | 239.825 | 240.370 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | vlm_prefix | 233.142 | 232.766 | 234.494 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | ae_step[0] | 57.792 | 57.541 | 58.081 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | ae_step[1] | 57.203 | 57.008 | 57.477 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | ae_step[2] | 57.126 | 56.932 | 57.461 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | ae_step[3] | 57.069 | 56.867 | 57.320 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | ae_step[4] | 57.088 | 56.906 | 57.301 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | ae_step[5] | 57.048 | 56.848 | 57.295 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | ae_step[6] | 57.063 | 56.896 | 57.342 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | ae_step[7] | 57.005 | 56.830 | 57.243 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | ae_step[8] | 57.086 | 56.902 | 57.292 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | ae_step[9] | 57.021 | 56.840 | 57.286 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | ae_total | 574.071 | 573.243 | 575.020 | 100 |
| eager | 2 | long | 48 | 768 | 816 | 551 | e2e | 1053.427 | 1052.632 | 1055.122 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | vision | 239.891 | 239.709 | 240.238 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | vlm_prefix | 233.681 | 233.264 | 234.074 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | ae_step[0] | 57.806 | 57.549 | 58.179 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | ae_step[1] | 57.289 | 57.115 | 57.497 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | ae_step[2] | 57.210 | 57.012 | 57.496 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | ae_step[3] | 57.133 | 56.977 | 57.364 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | ae_step[4] | 57.141 | 56.969 | 57.377 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | ae_step[5] | 57.087 | 56.893 | 57.349 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | ae_step[6] | 57.136 | 56.940 | 57.377 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | ae_step[7] | 57.116 | 56.897 | 57.372 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | ae_step[8] | 57.153 | 56.969 | 57.464 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | ae_step[9] | 57.111 | 56.889 | 57.331 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | ae_total | 574.772 | 573.610 | 575.929 | 100 |
| eager | 3 | short | 48 | 768 | 816 | 774 | e2e | 1054.735 | 1053.649 | 1055.917 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | vision | 239.863 | 239.741 | 240.093 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | vlm_prefix | 233.716 | 233.334 | 234.221 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | ae_step[0] | 57.943 | 57.656 | 58.189 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | ae_step[1] | 57.356 | 57.145 | 57.633 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | ae_step[2] | 57.309 | 57.044 | 57.538 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | ae_step[3] | 57.230 | 57.020 | 57.492 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | ae_step[4] | 57.230 | 57.055 | 57.429 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | ae_step[5] | 57.181 | 57.015 | 57.396 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | ae_step[6] | 57.221 | 57.014 | 57.487 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | ae_step[7] | 57.161 | 56.948 | 57.402 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | ae_step[8] | 57.227 | 57.046 | 57.415 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | ae_step[9] | 57.185 | 57.001 | 57.538 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | ae_total | 575.531 | 574.681 | 576.548 | 100 |
| eager | 3 | typical | 48 | 768 | 816 | 789 | e2e | 1055.638 | 1054.743 | 1056.596 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | vision | 239.808 | 239.673 | 240.056 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | vlm_prefix | 234.781 | 234.292 | 235.446 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | ae_step[0] | 57.862 | 57.633 | 58.214 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | ae_step[1] | 57.334 | 57.090 | 57.694 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | ae_step[2] | 57.245 | 57.028 | 57.526 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | ae_step[3] | 57.176 | 56.940 | 57.514 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | ae_step[4] | 57.160 | 56.990 | 57.424 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | ae_step[5] | 57.127 | 56.880 | 57.418 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | ae_step[6] | 57.142 | 57.002 | 57.463 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | ae_step[7] | 57.074 | 56.888 | 57.463 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | ae_step[8] | 57.154 | 56.982 | 57.485 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | ae_step[9] | 57.113 | 56.909 | 57.414 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | ae_total | 574.905 | 573.903 | 576.237 | 100 |
| eager | 3 | long | 48 | 768 | 816 | 807 | e2e | 1055.972 | 1054.954 | 1057.603 | 100 |

## Fits

| fit | region | x | a (ms) | b | unit b | r2 | points | note |
|---|---|---|---|---|---|---|---|---|
| vision_vs_images | vision | n_images | 241.7351 | -0.66228 | ms_per_image | 0.9170 | 3 |  |
| vlm_prefix_vs_P | vlm_prefix | P_processed |  |  |  |  | 1 | x is constant at 816 across the sweep; no slope can be fitted, which is itself the result |
| ae_step_vs_P | ae_total | P_processed |  |  |  |  | 1 | x is constant at 816 across the sweep; no slope can be fitted, which is itself the result |
| ae_step_vs_P | e2e | P_processed |  |  |  |  | 1 | x is constant at 816 across the sweep; no slope can be fitted, which is itself the result |

## Conditions

π0, bf16, MAXN + jetson_clocks. **P_processed is constant at 816 for every point**: the processor always encodes three image slots and pads the prompt to 48 tokens, so what the sweep varies is how many of those 816 positions carry a valid mask (P_valid), not how many are computed. That is why prefix latency is flat.

## Reproduce

```
python bench/prefix_sweep.py --outdir-prefix results/20260922b_E11d_pi0
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
