# 20260927_E3_pi0_eager_N4_15W

* **What:** E3 — task T002
* **Conditions:** date 2026-09-28T03:28:21Z, power mode NV Power Mode: MODE_15W, model pi0 (LeRobot implementation)
* **Command:** `python bench/run_e3.py --steps 4 --warmup 20 --timed 100`

Headline — first rows of `latency_summary.csv` (30 rows):

| region | median_ms | mean_ms | p5_ms | p95_ms | min_ms | max_ms | n |
|---|---|---|---|---|---|---|---|
| ae_layer[0] | 5.7598559856414795 | 5.7610009574890135 | 5.66821448802948 | 5.860803389549256 | 5.630080223083496 | 6.081567764282227 | 100 |
| ae_layer[10] | 4.860207796096802 | 4.8800387287139895 | 4.775206303596496 | 4.965718245506286 | 4.7297282218933105 | 6.218368053436279 | 100 |
| ae_layer[11] | 4.848832130432129 | 4.869335680007935 | 4.782627272605896 | 4.930403041839599 | 4.774655818939209 | 6.256351947784424 | 100 |
| ae_layer[12] | 4.826095819473267 | 4.837815356254578 | 4.749108982086182 | 4.9023710012435915 | 4.71830415725708 | 6.105343818664551 | 100 |
| ae_layer[13] | 4.841232061386108 | 4.859809937477112 | 4.771422576904297 | 4.933617496490479 | 4.743584156036377 | 6.249216079711914 | 100 |
| ae_layer[14] | 4.834367990493774 | 4.847420501708984 | 4.761503911018371 | 4.928488087654114 | 4.728799819946289 | 5.954048156738281 | 100 |

**Anything odd:** model_note: LeRobot's port, not openpi's — see env/README.md §10 for why; compile_mode_note: eager; the checkpoint ships compile_model=true with compile_mode=max-autotune and both were overridden BEFORE construction (PI0Pytorch.__init__ compiles immediately); notes: ["LeRobot's denoising loop is `for step in range(num_steps)` with a Python float time (flow_matching/euler loop). openpi's is `while time >= -dt/2` on CUDA tensors, which forces a CPU-GPU sync per ste

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
