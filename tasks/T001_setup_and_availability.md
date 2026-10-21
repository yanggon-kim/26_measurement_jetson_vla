# T001 — Setup and availability: stack, tools, access, checkpoints, one inference per model · priority P0 · depends on: —

## Why
This is a new machine and a new software generation. Nothing about its stack is known in advance, and every later
task depends on what this one finds. Do everything you can alone (`docs/05_setup.md`); ask the human only for what
you cannot do, with exact commands; keep going with whatever does not depend on the answer.

## Orin counterpart
`reference/orin/tasks/T001_bringup.md` (and its status blocks: what failed there — no passwordless sudo, the board
booted in a capped mode, ncu needed root, no `dram__` counters, tegrastats missed the memory rail, nsys binaries
leaked a token), `reference/orin/env/README.md`, `reference/orin/reports/E0_platform.md`.

## Configuration
Any power mode (the toy runs below are labelled with the mode they ran in; nothing here is a performance result).
Repository: this branch, cloned with `git clone -b thor --single-branch`. Work outside the repository for
environments (`~/venvs/`), sources (`~/src/`) and checkpoints (the Hugging Face cache).

## Steps
0. **Orient.** `git status` (branch `thor`, clean), read `README.md`, `AGENTS.md`, `docs/00`–`05`. Copy the Orin
   fixtures so Thor sees the same inputs: `cp -r reference/orin/bench/fixtures bench/fixtures`; commit.
1. **Platform as found** (matrix rows A1–A5, A12): board, L4T, JetPack, CUDA, driver, Python, kernel, disk,
   network, CPU topology. Power mode: `nvpmodel -q --verbose` and `/etc/nvpmodel.conf` — export the mode table
   (id, name, CPU cores, GPU / EMC caps, TPC or GPC masks, power budget) to `nvpmodel_modes.csv`, and identify the
   **maximum-performance mode** (protocol §1). Do not switch. If the board is not in it: H5. Find and record the
   GPU frequency node (devfreq `cur_freq`, `min_freq`, `max_freq`, `available_frequencies`), the EMC rate node, the
   thermal zones and the fan control.
2. **sudo and git** (A6, A7). Passwordless sudo for each command class; repo-local git identity if none is set;
   push access (generate the deploy key yourself if needed). Missing pieces: H1, H3.
3. **The base environment and PyTorch** (A8). Try the routes of `docs/05_setup.md` A8 in order; keep the first that
   passes every check: `torch.cuda.is_available()`, the device's architecture in `get_arch_list()` (or JIT from PTX
   proven to work), a bf16 GEMM 4096³ that matches an fp32 reference to bf16 accuracy, `torch.compile` of a bf16
   MLP (1024 → 4096 → 1024, GeLU, input `[51, 1024]`) in `default`, `reduce-overhead`, `max-autotune`,
   `max-autotune-no-cudagraphs` (20 warm-up, 100 timed CUDA-event calls each, max relative error vs eager),
   `triton` importable. Record for each route tried: what was installed, what failed, the exact error. Read
   Inductor's `is_big_gpu` (the SM threshold in `torch/_inductor/utils.py` of the installed torch) and say whether
   it fires on this GPU. Write `env/setup_base.sh` (re-runnable) and `env/requirements_base.txt`.
4. **Profilers and binary tools** (A9, A10).
   a. `command -v nsys ncu cuobjdump nvdisasm c++filt zstd` and versions; install what is missing (`env/INSTALL_LOG.md`).
      Save under `env/tool_help/`: `nsys --help`, `nsys profile --help`, `nsys stats --help-reports`, `ncu --help`,
      `ncu --list-sets`, `ncu --list-sections`, `ncu --query-metrics` (full list), `cuobjdump --help`, `nvdisasm --help`.
      Record the exact spelling of: the NVTX filter, the kernel-name filter, `-c` / `-s`, `--profile-from-start`,
      graph-node profiling (`--graph-profiling node` or equivalent), `--clock-control` values, import / CSV export.
   b. **Toy nsys** (`bench/nsys_toy.py`): the toy GEMM below inside NVTX `toy`, protocol §3.1 command, `nsys stats`
      CSV export. Note run time, report size uncompressed and `zstd -19`.
   c. **Toy ncu** (`bench/ncu_toy.py`): warm up, then between `cudaProfilerStart/Stop` one bf16
      `torch.nn.functional.linear` with x `[51, 1024]`, W `[4096, 1024]` inside NVTX `toy`:
      `ncu --set full --nvtx --nvtx-include "toy/" --profile-from-start off -o results/<run>/raw/ncu_toy -f python bench/ncu_toy.py`.
      First **without sudo**; on a permission error, with root (protocol §3.2; needs H1's `SETENV` line). Record
      which worked (A10). Export both pages; `ncu/columns.txt` (header + distinct metric names).
   d. **Metric discovery** → `ncu/metric_presence.csv` (`metric, category, present, nonzero_on_toy, notes`): every
      metric of protocol §3.3, plus the candidates `ncu --query-metrics` lists for `dram__`, `lts__t_sectors`,
      `lts__d_sectors_fill`, `sm__pipe_tensor`, `sm__inst_executed_pipe_tensor`, `sm__ops_path_tensor`,
      `smsp__sass_thread_inst_executed_op_`, `smsp__average_warps_issue_stalled_`, `l1tex__t_requests_pipe_lsu`,
      `gpc__cycles_active`. Collect the candidates on the toy kernel with an explicit `--metrics` list (form D).
      Also find which metrics ncu's roofline sections use on this architecture (the section files listed by
      `ncu --list-sections`). State in `env/README.md`: are there DRAM counters; which tensor-pipe and tensor-FLOP
      metrics exist; on which page the stall reasons are.
   e. **Clock control.** With `jetson_clocks` applied (if sudo allows), compare ncu's `SM Frequency` under each
      `--clock-control` value with the locked GPU frequency; record which value to use (protocol §3.2). If clocks
      cannot be locked yet, record that and repeat this step once H1 is resolved.
   f. **ncu limits.** Profile a loop of 1,000 tiny kernels with a metric-only list (`-c 1000`) and record whether
      ncu stops early (the Orin stopped after 480 per process); and whether ncu sees kernels inside a CUDA-graph
      replay (capture the toy in a `torch.cuda.CUDAGraph`, profile with graph-node profiling).
   g. **Architecture string.** `torch.cuda.get_device_capability()` and `cuobjdump --list-text` on the cuBLAS(Lt)
      library torch loads: the `sm_` strings present. Record `gpu_arch` in `env/README.md`.
5. **Power monitor** (A11): every channel, its label and sysfs path, one 10 s sample at idle, one `tegrastats`
   line; the rail table of protocol §4 in `env/README.md`, with which rails form "module power" and whether a
   memory rail exists.
6. **nsys and ncu secrets re-test** (A17). On the Orin, `.nsys-rep` / `.sqlite` embedded the environment of
   processes nsys did not launch. Test it here with a canary, never a real secret:
   1. start an unrelated background process with a canary variable:
      `env THOR_CANARY_SECRET=canary_$(openssl rand -hex 16) sleep 900 &` and note the value in `scratch/` only;
   2. capture the toy nsys run **twice**: once in your normal shell, once under `env -u HF_TOKEN -u
      HUGGING_FACE_HUB_TOKEN -u GITHUB_TOKEN -u GH_TOKEN HF_HUB_OFFLINE=1`; export each `.sqlite`;
   3. scan each `.nsys-rep`, `.sqlite` and the toy `.ncu-rep` with `strings -n 8` for: the canary value, the values
      of every token-like variable in your own environment (`env | grep -iE 'token|secret|key|session|auth'`), the
      content of any token file you use, and the protocol §3.5 patterns. Record per file and per class: found /
      not found, and the byte offset only — never the matched bytes.
   Verdict: **clean** (no class found in any file) → nsys binaries may be committed: delete the two nsys lines
   from `.gitignore` in a commit of its own whose message cites this result directory; every later binary is still
   scanned per file (protocol §3.5). **Not clean** → they are never committed; leave `.gitignore` as it is; the
   CSV exports are the raw record. Either way, write the verdict into `env/README.md` and the `PROGRESS.md`
   ceilings line. Kill the canary process.
7. **Model access, code and environments** (A13–A15). Clone every pin of `docs/00_context.md` into `~/src/`. One
   environment per code base, built the Orin way but for this stack (`reference/orin/env/setup_pi0.sh`,
   `setup_groot.sh`, `setup_groot_n1d5_n1d6.md`, `setup_libero.sh`): the CUDA torch of step 3 first, the
   repository with `--no-deps`, then dependencies by hand, re-checking step 3's checks after every install. Download
   the six checkpoints at their revisions (record sha256 of each weight file and the total bytes) and everything
   their load path pulls (tokenizers, base VLM configs). The PaliGemma tokenizer: the ungated route first (A13),
   H2 only if it fails. LIBERO in its own environment, `MUJOCO_GL=egl`, one `libero_spatial` reset rendered. One
   `env/setup_<env>.sh` and `env/requirements_<env>.txt` per environment.
8. **One inference per model** (A16), smoke test, labelled *not a measurement*: each of the six models on its
   fixture, eager, its default N, then once with `torch.compile` of the denoising step: actions finite, their
   shape, P and S as processed, wall time of 5 calls after 3 warm-ups, peak memory. Verify real weights were loaded
   (the Orin's `assert_real_weights()` check: a LeRobot policy can silently random-initialise,
   `reference/orin/tasks/T002_pi0_eager.md` §8). A model that does not run is not skipped: record the error, try
   the nearest commit that runs (record why), and if nothing runs mark that model blocked in `PROGRESS.md` with the
   error, and continue with the others.
9. **Write it up.** `env/README.md` = the availability matrix of `docs/05_setup.md` §2 plus the facts below; the
   "Needs the human" boxes; `PROGRESS.md`; push.

**Facts T001 must leave in `env/README.md`** (each with its source command): board, L4T, JetPack, CUDA, driver,
Python, kernel; CPU cores online; RAM, swap; GPU name, compute capability, `gpu_arch`, SM count in the current mode;
the maximum-performance mode (id, name) and the current one; GPU frequency and EMC nodes; the chosen torch route,
versions of torch / torchvision / triton, and whether Inductor's SM threshold fires; nsys / ncu / cuobjdump versions;
ncu permission and clock-control setting; DRAM-byte, FLOP and tensor-pipe metric names available; stall-reason page;
ncu kernel limit and graph-node visibility; rail table; nsys-binary verdict; per model: environment, commits,
checkpoint revision, smoke result.

## Result directories and reports
- `results/<date>_E0_toolcheck/` — toy nsys and ncu (raw reports, both page exports, `columns.txt`,
  `metric_presence.csv`), `stack_verification.json` (step 3), tool versions, `nvpmodel_modes.csv`, the rail sample.
- `results/<date>_E0_secrets_test/` — step 6: per-file, per-class verdicts (no matched bytes), commands, verdict.
- `results/<date>_E0_smoke_models/` — step 8: one row per model and mode in `smoke.csv`, stdout logs, `meta.json`.
- `reports/E0_setup.md` — what works, what does not, what was tried, the human items, the facts list above.

## Acceptance
- `env/README.md` holds the full availability matrix and every fact of the list above, or the reason it is missing.
- Every requirement is `ok`, or `missing` / `blocked` with what was tried and an H-item with exact commands in both
  "Needs the human" boxes.
- A working CUDA PyTorch route with `torch.compile` verified in four modes, or the failure of every route documented.
- Toy nsys and toy ncu ran; `metric_presence.csv` answers the three questions of step 4d.
- The nsys-binary verdict is recorded, and `.gitignore` matches it.
- Each of the six models has a smoke result (pass, or the error and what was tried).
- `env/setup_*.sh`, `env/requirements_*.txt`, `env/INSTALL_LOG.md`, `env/tool_help/` committed.
- Repository-size lines in `STATUS.md`; independent review pass (protocol §11, P0).

## Push points
After step 0 (fixtures), after step 4 (tool check directory), after step 6 (verdict, and the `.gitignore` commit if
clean), after step 8, and at the close.

## Status (appended by the on-device agent)
