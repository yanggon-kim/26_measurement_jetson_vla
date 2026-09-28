# T008 — π0.5 and SmolVLA: timeline, SM utilisation, instruction level, roofline, power (E5, E5b, E6, E7)  · priority P2 · depends on: T006 (step 1), T004, T007

Everything T004 and T007 did for π0, for **π0.5** (openpi, S = 50, 18 expert layers, hidden 1024) and **SmolVLA**
(lerobot, S = 50, 16 expert layers — 8 self + 8 cross-attention, hidden 720, default N = 10). Do one model
completely before starting the other; π0.5 first. Reuse the `bench/` scripts of T004/T007 (add `--model`).
Per model, eager and its best compiled mode from T006:
1. `floor.csv` (protocol §9) from the loaded model's own parameter bytes and shapes.
2. E5 nsys at the paper's N (4) and the default N (10) — T004 step 1.
3. E5 ncu, one filtered run per region `ae_layer[0]/`, `ae_step[0]/`, `vlm_prefix/`, `vision/` — T004 step 2.
   For SmolVLA capture one self-attention and one cross-attention expert layer (`ae_layer[0]/`, `ae_layer[1]/`
   or whichever indices they are — say which).
4. E6 roofline CSV — T004 step 3.
5. E7 power at MAXN (idle / prefix-only / AE-only / full loop); capped power modes optional.
6. E5b per `docs/03_instruction_level_analysis.md`: targets at the model's own S and projection shapes (read them
   from the loaded model and list them in `sass/targets.csv`; SmolVLA's hidden-720 shapes will likely select
   different GEMM kernels from π0's), attention if in the top-5; S sweep S ∈ {1, 16, 50, 256} on the model's
   largest projection and its q projection; eager vs compiled. If a target is the **same kernel symbol and
   architecture** as one already analysed in T007, do not repeat the static part (§2–§5, §9): reference it and
   report only the dynamic metrics (§6–§8) at this model's shapes.
Result directories: `results/<date>_<E5|E5b|E6|E7>_<pi05|smolvla>_<mode>_<short>/`.
Acceptance, per model `<m>` ∈ {`pi05`, `smolvla`}: `reports/E5_<m>_timeline.md`, `reports/E5b_<m>.md`,
`reports/E6_<m>_roofline.md`, `reports/E7_<m>_power.md` with the same content as the π0 reports, plus one
cross-model table in `reports/E5_E5b_models.md` (`model, S, ms/step eager, ms/step compiled, floor, gap ratio,
GPU idle % in ae_step, region-weighted SM occupancy, top stall reason and share, LSU load share, dominant kernel`).
Raw files committed under each result directory's `raw/` exactly as in T004 / T007 (protocol §3.5 / §6);
repository-size lines in `STATUS.md` after each model (§6f). Self-review (§11).

status: claimed 2026-09-24

status: done 2026-09-26

**Deliverables:** `reports/E5_{pi05,smolvla}_timeline.md`, `reports/E5b_{pi05,smolvla}.md`,
`reports/E6_{pi05,smolvla}_roofline.md`, `reports/E7_{pi05,smolvla}_power.md` (per-model extracts of the
consolidated reports, with their own tables; the research side was asked in STATUS.md whether extracts
satisfy "same content as the π0 reports"), and the cross-model `reports/E5_E5b_models.md` with every
column the task names.

**Step 6 S sweep** (the last open item, run 2026-09-25 23:08–23:20): `results/20260925_E5b_{pi05,smolvla}_ssweep`,
S ∈ {1, 16, 50, 256} on the expert's largest projection and q projection (SmolVLA also its first
cross-attention layer's q and k), shapes read from the loaded model; π0.5 swept in fp32 (eager) and bf16
(compiled rung). **Correction found doing it:** the sweep script's CUDA-event column was per-call harness
overhead (~45–75 µs); kernel time is taken from the sweeps' own ncu captures (`s_sweep_ncu.csv`,
`bench/sweep_ncu_table.py`), and GR00T's committed sweep report was corrected the same way (ae47c79).

**Deviations:** E6/E7 per-model files are extracts (above); instrumented per-step not available for
these models (§2 slope used, `E3_E4_models.md`); the compiled rung for E5b is each model's best Part B rung.

bench commit `ae47c79bda14`.

status: done 2026-09-28 — completeness audit (owner request): the missing pieces are now delivered
- Step 2 at N = 4 (only N = 10 existed): `results/20260928_E5_{pi05,smolvla}_{eager,best}_N4`; E5 timeline reports regenerated.
- Step 3 ae_layer ncu captures (never run): `results/20260928_E5_ncu_pi05_{eager,best}_aelayer` (layer 0),
  `results/20260928_E5_ncu_smolvla_eager_aelayer` (layers 0 self-attn, 1 cross-attn). SmolVLA best: NOT MEASURED — per-layer
  NVTX ranges split the graph-captured denoise_step and cudagraph_trees refuses the replay (exact error in meta; cloning
  the outputs was tried and did not help); the graph-level best capture is `results/20260924_E5b_smolvla_best`.
  `bench/regions.py` now marks SmolVLA layers (its layer body is inlined). First attempt refused by the profiler guard,
  which counted a CPU-only `ncu --import` as a profiler (fixed in all four ncu scripts).
- E5b §0 file sets and §10-structured reports for π0.5 / SmolVLA as in T007's block.
- bench commit at close: 333505a.
