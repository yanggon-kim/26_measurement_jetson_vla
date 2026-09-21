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
Self-review (§11).
