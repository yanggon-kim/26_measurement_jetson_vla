# E12b M1 — hoist equivalence (compiled `bf16_both` path)

Compared: `expert_stage(..., e7_timesteps=False)` (hoist) against `e7_timesteps=True` (stock, the shipping per-step timestep), both through the compiled rung, same prefix cache and noise, on 5 distinct LIBERO observations with their own seeds. Floor: stock against itself with identical inputs.

* hoist vs stock: **PASS — bitwise identical on every frame**
* stock vs stock (run-to-run floor): bitwise
* power of the check: consecutive frames' actions differ by up to 5.4077 (max abs), so a hoist that changed the computation would be visible

Per-frame rows: `equivalence.csv`.
