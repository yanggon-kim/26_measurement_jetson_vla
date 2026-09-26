# Note (2026-09-25)

Valid run: success 63/100, `episodes.csv` per episode, device gate in `meta.json` (MAXN, GPU
1300.5 MHz). **It has no per-step raw log** (`raw/` is empty): `bench/libero_eval.py` at bench
commit 0f2c1a3 did not write one, which T006's raw-file requirement asks for. The harness was
fixed in 0701d95 (additive logging only) and this exact command is re-run as
`20260925_E8_pi0_libero_spatial_fp32_N10_settle10_steplog`, which supplies the per-step log and a
same-seed repeatability check of this success count.

"fp32" in the name is the arm label of the plan: the checkpoint **as shipped**, no rung and no
cast. Its parameters are 3.61 B bf16 + 0.42 B fp32 (recorded by the fixed harness in the re-run).
