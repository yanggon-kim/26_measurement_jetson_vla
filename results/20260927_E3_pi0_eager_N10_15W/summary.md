# 20260927_E3_pi0_eager_N10_15W

* **What:** E3 — task T002
* **Conditions:** date 2026-09-28T03:44:54Z, power mode NV Power Mode: MODE_15W, model pi0 (LeRobot implementation)
* **Command:** `python bench/run_e3.py --steps 10 --warmup 20 --timed 100`

Headline — first rows of `latency_summary.csv` (42 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| ae_layer[0] | 5.83073616027832 | 5.8453113555908205 | 5.777566576004029 | 5.963075184822083 | 5.760608196258545 | 6.180992126464844 | 100 |
| ae_layer[10] | 5.046752214431763 | 5.059947838783264 | 4.991348934173584 | 5.143855881690979 | 4.9429121017456055 | 5.618720054626465 | 100 |
| ae_layer[11] | 5.059823989868164 | 5.068827543258667 | 4.999326181411743 | 5.1394449234008786 | 4.947231769561768 | 5.677567958831787 | 100 |
| ae_layer[12] | 5.02676796913147 | 5.042452173233032 | 4.976982474327087 | 5.1237646579742435 | 4.946656227111816 | 5.771552085876465 | 100 |
| ae_layer[13] | 5.0382080078125 | 5.044532494544983 | 4.979655909538269 | 5.103879928588867 | 4.934624195098877 | 5.451968193054199 | 100 |
| ae_layer[14] | 5.040927886962891 | 5.046549763679504 | 4.982391953468323 | 5.099918246269226 | 4.967967987060547 | 5.460544109344482 | 100 |

**Anything odd:** model_note: LeRobot's port, not openpi's — see env/README.md §10 for why; compile_mode_note: eager; the checkpoint ships compile_model=true with compile_mode=max-autotune and both were overridden BEFORE construction (PI0Pytorch.__init__ compiles immediately); notes: ["LeRobot's denoising loop is `for step in range(num_steps)` with a Python float time (flow_matching/euler loop). openpi's is `while time >= -dt/2` on CUDA tensors, which forces a CPU-GPU sync per ste

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
