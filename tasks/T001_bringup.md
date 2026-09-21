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
      `ncu --set full --nvtx --nvtx-include "toy/" --profile-from-start off -o raw/<run>/ncu_toy -f python bench/ncu_toy.py`
      (never unfiltered). Export both pages (`ncu --import raw/<run>/ncu_toy.ncu-rep --csv --page details`, and
      `--page raw`) into the E0 result directory, save the header and the distinct metric names to
      `ncu/columns.txt`, and state which metrics of protocol §3.3 are present, absent or empty on this device
      (SM occupancy, pipe utilisation, stall reasons, LSU request counters). Note the run time and report size.
      On `ERR_NVGPUCTRPERM` apply the fix in protocol §12.
   d. `hw_constants.json` (protocol §8): SM count (`multi_processor_count`), compute capability and the `sm_`
      architecture string (expected `sm_87` — confirm, e.g. from `cuobjdump --list-text` on the cuBLAS library
      PyTorch loads). Peaks are added by T005.
4. Verify `tegrastats` rails listed in protocol §4 exist on this carrier board; record the exact rail names.
Acceptance: E0 result directory (incl. the toy ncu exports, `ncu/columns.txt`, `hw_constants.json`), `env/` reproducible
(incl. `env/tool_help/`), a line in `STATUS.md` with the versions that work and which ncu metrics this Orin exposes.

status: claimed 2026-09-21
