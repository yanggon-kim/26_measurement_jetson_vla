# Note (2026-09-26)

The measured curve (`sm_sensitivity.csv`, 2/4/6/8 SMs requested inside MODE_30W's hardware 8) is
valid. The **printed verdict** of `bench/corun_green.py` ("8 SMs is 1.000x its 16-SM time ...
SM-INSENSITIVE") is **not**: the script compares its 8-SM point with the whole GPU, which at
MODE_30W is itself 8 SMs, so the ratio is 8-vs-8 = 1.000 by construction. Compare instead with
`results/20260925_E12_pi0_green_N10_MAXN_eager` after the GPU-clock ratio 1300.5 / 612 = 2.125
(and the CPU ratio 2.2016 / 1.728 = 1.27 for the host-bound eager expert):

| eager N=10 | MAXN green | MODE_30W green | ratio |
|---|---:|---:|---:|
| prefix 4 SMs | 1425.07 | 3091.61 | 2.169 |
| prefix 8 SMs | 779.79 | 1789.74 | 2.295 |
| expert 8 SMs | 587.56 | 710.20 | 1.209 |

A request for 2 SMs was granted 4 at this mode (`granted_sm`), recorded, not assumed.
