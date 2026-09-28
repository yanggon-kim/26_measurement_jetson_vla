# 20260926_E3_pi0_eager_N4_30W

* **What:** E3 — task T002
* **Conditions:** date 2026-09-26T19:57:22Z, power mode NV Power Mode: MODE_30W, model pi0 (LeRobot implementation)
* **Command:** `python bench/run_e3.py --steps 4 --warmup 20 --timed 100`

Headline — first rows of `latency_summary.csv` (30 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| ae_layer[0] | 3.3562079668045044 | 3.3655046319961546 | 3.308356821537018 | 3.459073567390442 | 3.2847681045532227 | 3.5499839782714844 | 100 |
| ae_layer[10] | 3.2074719667434692 | 3.2078979277610777 | 3.163663959503174 | 3.246121621131897 | 3.1413440704345703 | 3.303328037261963 | 100 |
| ae_layer[11] | 3.201599955558777 | 3.2036243176460264 | 3.1593791127204893 | 3.2411888003349305 | 3.130143880844116 | 3.541856050491333 | 100 |
| ae_layer[12] | 3.191536068916321 | 3.192096002101898 | 3.1554415583610536 | 3.224239909648895 | 3.1175360679626465 | 3.3034238815307617 | 100 |
| ae_layer[13] | 3.2002720832824707 | 3.2002512001991272 | 3.1662030935287477 | 3.2353119015693665 | 3.130944013595581 | 3.251199960708618 | 100 |
| ae_layer[14] | 3.1936639547348022 | 3.193850872516632 | 3.1578784465789793 | 3.2272032260894776 | 3.136863946914673 | 3.2425599098205566 | 100 |

**Anything odd:** model_note: LeRobot's port, not openpi's — see env/README.md §10 for why; compile_mode_note: eager; the checkpoint ships compile_model=true with compile_mode=max-autotune and both were overridden BEFORE construction (PI0Pytorch.__init__ compiles immediately); notes: ["LeRobot's denoising loop is `for step in range(num_steps)` with a Python float time (flow_matching/euler loop). openpi's is `while time >= -dt/2` on CUDA tensors, which forces a CPU-GPU sync per ste

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
