# reference/orin — read-only snapshot of the Orin study

**Everything under this directory is Orin code and Orin numbers.** It was copied from branch `main` of this
repository at commit `4b22260` (2026-10-21). On `main`, an on-device agent ran it on a Jetson AGX Orin 64GB
(JetPack 6, CUDA 12, Python 3.10, `sm_87`, 16 SMs at MAXN).

- **Port it; do not run it unchanged.** Copy a script into the top-level `bench/` (or `env/`), adapt it to this
  device, and commit your version there. Every hard-coded Orin constant in it is wrong on Thor: peaks, ridge,
  SM count, clocks, rail names, sysfs paths, wheel URLs, `-arch=sm_87` and the Python ABI. Read the device's own
  values from T001 and the `hw_constants.json` of T003 instead.
- **Never edit this directory.** It is the comparison record. A bug found here is fixed in your port and noted in
  the report of the task that found it.
- **Orin numbers are for comparison only.** Quote one as "Orin, `reference/orin/<file>`" next to your Thor value,
  never as a Thor value.
- Paths such as `results/2026…` inside these files point to branch `main`, which this branch does not contain.
  Read them with `git fetch origin main` and `git show origin/main:<path>` if you need one. Never merge `main`
  into `thor`: it would add ~250 MB of Orin results.

| path | what it is | ported by |
|---|---|---|
| `bench/` | the whole Orin harness (163 files; `managed_alloc.so` left out, rebuild it from `managed_alloc.cu`). Key entry points: `model_registry.py` (the six models, their N field and wrap points), `regions.py` (NVTX + CUDA-event regions, installed after compilation), `opt_ladder.py` (the E4 ladder), `phase_loops.py` (prefix-only, expert-only and full loops), `ncu_target.py` / `nsys_target.py`, `parse_ncu.py`, `parse_nsys.py`, `roofline.py`, `micro_peak.py`, `micro_ae_kernels.py`, `power_core.py`, `libero_eval.py`, `e12b_*.py`, `e13*.py`, `e14*.py`, `finalize_raw.py` (hashing, compression, secrets scan), `fixtures/` (the LIBERO observations) | T001–T014 |
| `env/` | the Orin setup scripts: `setup.sh` (base venv), `setup_pi0.sh` (LeRobot; also used for π0.5 and SmolVLA), `setup_groot.sh` and `setup_groot_n1d5_n1d6.md` (GR00T N1.7 / N1.6 / N1.5), `setup_libero.sh`, `README.md` (every trap the Orin stack hit), `INSTALL_LOG.md` | T001 |
| `reports/` | the key Orin reports: `SYNTHESIS_all_measurements.md` (the cross-model picture), `E4i_best_combinations.md` (best rung per model), `E4d_phase_balance.md` (the ladder), `E4_PARTS_ABC_GUIDE.md` (Parts A, B, C explained), `E4g_c1_c2_bottlenecks.md`, `E4h_vlm_prefix_kernels.md`, `E5_timeline.md`, `E5_E5b_models.md`, `E5b_summary.md`, `E6_*_roofline.md`, `E1_E2_microbench.md`, `E0_platform.md`, `E3_pi0_eager.md`, `E4_pi0_compile.md`, `E3_E4_models.md`, `E3_E4_groot.md`, `E7_power_energy.md`, `E8_libero.md`, `E8s_libero_seeded.md`, `E11_models.md`, `E12b_pi0_compiled_corun.md`, `E13_pi0_bandwidth_steal.md`, `E14_prefix_savings.md`, `E14b_pi0_padtrim.md` | T009, T015 (comparison) |
| `specs/` | `E12_corun/`, `E12b_compiled_corun/` (the co-run specifications), and the Orin versions of the protocol and the E5b procedure (`orin_02_protocol.md`, `orin_03_instruction_level_analysis.md`) | T013; history only |
| `tasks/` | the Orin task files T001–T018 with the Orin agent's status blocks (what was tried, what failed) | every task (its "Orin counterpart") |
| `analysis/` | research-side Orin analyses: `thor_projection.{md,py}` (the static Orin → Thor projection that this branch tests), `prefix_root_cause.{md,csv}` (why the Orin VLM prefix is slow) | T009 |
