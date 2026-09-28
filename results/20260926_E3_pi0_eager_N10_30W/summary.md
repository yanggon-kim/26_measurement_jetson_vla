# 20260926_E3_pi0_eager_N10_30W

* **What:** E3 — task T002
* **Conditions:** date 2026-09-26T20:07:08Z, power mode NV Power Mode: MODE_30W, model pi0 (LeRobot implementation)
* **Command:** `python bench/run_e3.py --steps 10 --warmup 20 --timed 100`

Headline — first rows of `latency_summary.csv` (42 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| ae_layer[0] | 3.3313599824905396 | 3.347177917957306 | 3.2922160387039185 | 3.442246472835541 | 3.2665278911590576 | 3.772416114807129 | 100 |
| ae_layer[10] | 3.169167995452881 | 3.17089631319046 | 3.1155041098594665 | 3.2126943707466125 | 3.0919361114501953 | 3.390239953994751 | 100 |
| ae_layer[11] | 3.1754720211029053 | 3.1719414448738097 | 3.1327199935913086 | 3.2057344555854796 | 3.0848960876464844 | 3.4040958881378174 | 100 |
| ae_layer[12] | 3.153216004371643 | 3.154406726360321 | 3.1212271094322204 | 3.1895071148872374 | 3.0938239097595215 | 3.3898561000823975 | 100 |
| ae_layer[13] | 3.1643359661102295 | 3.1658480095863344 | 3.1203999638557436 | 3.197748875617981 | 3.1111040115356445 | 3.4513280391693115 | 100 |
| ae_layer[14] | 3.161888003349304 | 3.163452799320221 | 3.1198288202285767 | 3.2022735357284544 | 3.0901119709014893 | 3.512255907058716 | 100 |

**Anything odd:** model_note: LeRobot's port, not openpi's — see env/README.md §10 for why; compile_mode_note: eager; the checkpoint ships compile_model=true with compile_mode=max-autotune and both were overridden BEFORE construction (PI0Pytorch.__init__ compiles immediately); notes: ["LeRobot's denoising loop is `for step in range(num_steps)` with a Python float time (flow_matching/euler loop). openpi's is `while time >= -dt/2` on CUDA tensors, which forces a CPU-GPU sync per ste

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
