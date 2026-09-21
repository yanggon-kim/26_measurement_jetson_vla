# T001 — Bring-up and platform record (E0)  · priority P0 · depends on: —

Goal: a working, recorded software stack on the Jetson and the E0 platform record.
0. Check the prerequisites in `AGENTS.md` (JetPack 6.x, passwordless sudo, push access, disk). If JetPack is 5.x, record it and stop: the human must reflash.
1. Set MAXN + `jetson_clocks`; record everything in protocol §1 to `results/<date>_E0_platform/`.
2. Install PyTorch with CUDA for this JetPack (NVIDIA's wheel or container), torchvision, Triton if available.
   Verify: `torch.cuda.is_available()`, a bf16 GEMM runs, **`torch.compile` compiles and runs a small MLP**
   (say which backend/mode worked). Write `env/README.md` + `env/setup.sh` so the stack can be rebuilt.
3. Profilers (protocol §3). `ncu` (the Nsight Compute command-line profiler) is already installed; install `nsys`
   if JetPack did not (`apt`), and anything else you need — record installs in `env/`.
   a. `which ncu nsys cuobjdump nvdisasm c++filt`; `ncu --version`, `nsys --version`; save `ncu --help`,
      `nsys --help`, `nsys stats --help-reports`, `cuobjdump --help` to `env/tool_help/`. From `ncu --help` record
      in `env/README.md`: the metric sets offered on this Orin (`--set …`; we expect at least `basic`, `roofline`,
      `full`), and the exact spelling of the NVTX filter, kernel-name filter, count / skip, profile-from-start and
      import / CSV options.
   b. `nsys profile` on a toy script and `nsys stats` CSV export.
   c. **A toy filtered `ncu` run**: a script that warms up, then between `cudaProfilerStart/Stop` runs one bf16
      `torch.nn.functional.linear` with x `[51, 1024]`, W `[4096, 1024]` inside the NVTX range `toy`:
      `ncu --set full --nvtx --nvtx-include "toy/" --profile-from-start off -o results/<run>/raw/ncu_toy -f python bench/ncu_toy.py`
      (never unfiltered). Export both pages (`ncu --import results/<run>/raw/ncu_toy.ncu-rep --csv --page details`, and
      `--page raw`) into the E0 result directory, save the header and the distinct metric names to
      `ncu/columns.txt`, and state which metrics of protocol §3.3 are present, absent or empty on this device
      (SM occupancy, pipe utilisation, stall reasons, LSU request counters). Note the run time and report size.
      Commit the raw files of steps b and c (`.nsys-rep`, `.ncu-rep`, command logs) under the E0 directory's
      `raw/` per protocol §3.5 / §6 (install `zstd` if missing), and record in `STATUS.md` the uncompressed and
      `zstd -19` sizes of the toy `.nsys-rep` and `.ncu-rep`: they calibrate the size budget of later tasks.
      On `ERR_NVGPUCTRPERM` apply the fix in protocol §12.
   d. `hw_constants.json` (protocol §8): SM count (`multi_processor_count`), compute capability and the `sm_`
      architecture string (expected `sm_87` — confirm, e.g. from `cuobjdump --list-text` on the cuBLAS library
      PyTorch loads). Peaks are added by T005.
4. Verify `tegrastats` rails listed in protocol §4 exist on this carrier board; record the exact rail names.
Acceptance: E0 result directory (incl. the toy ncu exports, `ncu/columns.txt`, `hw_constants.json`, and `raw/`
with the toy `.nsys-rep` / `.ncu-rep` listed in `meta.json.raw_files`), repository-size lines in `STATUS.md`
(protocol §6f), `env/` reproducible
(incl. `env/tool_help/`), a line in `STATUS.md` with the versions that work and which ncu metrics this Orin exposes.

status: claimed 2026-09-21
status: done 2026-09-21 (step 1 excepted — see "deviation", below)

- Result directory: `results/20260921_E0_platform/`
- Report: `reports/E0_platform.md`
- Environment: `env/README.md`, `env/setup.sh`, `env/requirements_base.txt`, `env/tool_help/` (12 files)
- Harness commit: `f92a895` (the exact scripts that ran are pinned per file in
  `results/20260921_E0_platform/meta.json.bench_files_sha256`; `bench_commit` alone does not
  identify them, because HEAD at collection time predated the commit containing `bench/`)
- Repository size after this task: `STATUS.md` → "Repository size (protocol §6f)"

Deviations from the task:

1. **Step 1 (MAXN + `jetson_clocks`) is NOT met.** `sudo nvpmodel -m 0` answers
   `NVPM WARN: Reboot required for changing to this power mode: 0` and prompts for
   confirmation; `AGENTS.md` forbids confirming a reboot prompt, so it was declined and the
   device stayed in `MODE_30W` (id 2). In that mode 4 of 8 GPU TPCs are power-gated
   (CUDA sees 8 of 16 SMs) and the GPU is capped at 612 MHz, so **nothing in this run
   characterises the Orin**. `jetson_clocks` itself was verified to work and is applied.
   A human must reboot into MAXN; a second, authoritative `results/<date>_E0_platform_maxn/`
   then follows (≈ 5 min, all scripts exist). Recorded at the top of `STATUS.md`.
2. **`ncu` requires root on this device.** Protocol §12's `perf_event_paranoid=1` is not
   sufficient. Substitution recorded in `meta.json.notes`: every ncu run is
   `sudo $(which ncu) ...` (plain `sudo ncu` fails on `secure_path`).
3. **Protocol §3.3's DRAM metrics cannot be collected here at all.** `ncu --query-metrics`
   lists 4 426 metrics on `Orin (GA10B)`, none `dram__*`; an explicit `--metrics
   dram__bytes.sum` returns `n/a`. ncu's `Memory Throughput` on this device is the L2
   throughput. Proposed substitute (validated to 1.0032 on the read path, write path not
   validated): `lts__t_sectors_*_sysmem_*_lookup_miss.sum × 32 B`. Raised as question 1 for
   the research side in `STATUS.md`.
4. **Protocol §4's `VDDQ_VDD2_1V8AO` is not printed by `tegrastats`** on this carrier board;
   it is read from sysfs instead, and `bench/collect_platform.py` now samples all four rails
   there, so four-rail module power exists. That channel measures a constant 5 040 mV bus,
   not 1.8 V — raised as question 2.
5. **Two toy profiler captures were taken before `jetson_clocks` was applied**, so their GPU
   clock was not fixed and was not recorded. Carried into the MAXN re-run (`STATUS.md`).

Review: self-review plus an independent sub-agent pass (protocol §11). The sub-agent found
one fabricated process claim in an earlier draft of the report and nine smaller defects; all
are fixed and listed in `reports/E0_platform.md` §9.
