# 02 — Measurement protocol (binding, Thor)

Adapted from the Orin protocol (`reference/orin/specs/orin_02_protocol.md`, history only). Every device-specific
value is **detected in T001 or measured in T003** and read from there. Where this file names an Orin value, it is
labelled "Orin" and is not a requirement here.

## 1. Device state — record for every run, fix before every run
- **Power mode: recorded, never changed by you.** `nvpmodel -q --verbose` and `/etc/nvpmodel.conf` list the modes
  and their ids. T001 identifies the **maximum-performance mode** (named `MAXN` on Jetson boards so far — a hint;
  if no mode has that name, the mode with no CPU, GPU, EMC or TPC caps and the highest power budget, with the
  reasoning written into `env/README.md`). You never run `nvpmodel -m`, never use `nvpmodel --force` and never
  confirm a reboot prompt: on Jetson a change of the GPU configuration needs a reboot. Setting the mode is the
  human's (`docs/05_setup.md` §3, H5).
- **Gate at the start of every session and before every measured or profiled run:**
  1. the current mode is the maximum-performance mode recorded in `env/README.md`;
  2. `torch.cuda.get_device_properties(0).multi_processor_count` equals the SM count T001 recorded in that mode;
  3. `sudo jetson_clocks` (and `sudo jetson_clocks --fan`) has been applied **in this boot** before the run;
  4. the GPU's current frequency is read from the path T001 found (a devfreq `cur_freq` node or equivalent) and
     recorded next to the run; it is within 2 % of the locked frequency T002 recorded;
  5. no other GPU job is running and no stale `tegrastats` is alive (`pgrep -a tegrastats` empty).
  If (1) fails, add item H5 to "Needs the human" and run **no** latency, occupancy, bandwidth, FLOP/s or power
  measurement. If (3) fails for want of sudo, add H1 and do the same. Meanwhile do the work that does not depend on
  the GPU configuration: environments, downloads and conversions, fixtures, porting `bench/`, analytic floors and
  FLOP counts, `segments.csv` from model code, and harness dry runs (never reported as measurements).
- **Record per run** (`meta.json`, §5): mode id and name, SM count, GPU and EMC clocks (EMC from the path T001
  found, e.g. a BPMP debugfs `clk/emc/rate` node), `jetson_clocks --show`, `cat /etc/nv_tegra_release`,
  `dpkg -l | grep -E 'nvidia-jetpack|cuda-toolkit|tensorrt|cudnn'`, `nvcc --version`, the GPU driver version,
  Python, `torch` / `torchvision` / `triton` versions and `torch.version.cuda`, kernel version.
- Thermal: temperatures logged for every run (tegrastats or the thermal-zone nodes T001 found); a run with
  throttling or a clock below its locked value is marked `throttled: true`, then cool down and repeat.
- Nothing else running: no second benchmark, desktop session closed if possible, swap / zram state recorded.

## 2. Timing
- Batch 1, fixed seeds, fixed real inputs (`bench/fixtures/`, copied from `reference/orin/bench/fixtures/` in T001
  so that Thor and Orin see the same observation). Precision as the rung defines (§E4 of the plan).
- GPU time with CUDA events (`torch.cuda.Event(enable_timing=True)`) around each region, synchronised only at
  region boundaries; also wall clock end to end. Never time with a profiler attached: profile in separate runs.
- **≥ 20 warm-up iterations** (for `torch.compile`: until no recompilation is reported; log the compile time),
  then **≥ 100 timed iterations**. Report **median, mean, p5, p95, min, n**; keep every per-iteration sample.
- **Regions** (the same names as NVTX ranges): `preprocess`, `vision`, `vlm_prefix` (the prefill that builds the
  KV cache or the backbone features), `ae_total`, `ae_step[i]` per denoising step, `ae_layer[j]` inside one
  representative step, `postprocess`; "other" = e2e − the sum of the named regions. **Every model has separate
  `vision` and `vlm_prefix` regions.** For GR00T, wrap the vision-tower call and the LLM call *inside* the
  backbone's forward separately (on the Orin, a `vlm_prefix` range on `backbone.forward` silently contained
  vision). Validate once per model and mode: `vision` + `vlm_prefix` equals the enclosing call within 2 %, and the
  two kernel sets in nsys are disjoint. If the code fuses two regions so that they cannot be separated, say which.
- Regions are installed **after** compilation ("compile first, instrument second",
  `reference/orin/reports/E4_PARTS_ABC_GUIDE.md` §3): a region boundary inside a compiled graph becomes a graph
  break.
- Sweep N with the model's own parameter (`00_context.md`), never by editing loops.
- **Per-step time has two definitions; report both and label them.** (a) *Instrumented*: events around each step
  (valid in eager). (b) *Slope*: fit `ae_total` (or e2e) = a + b·N over the N sweep; b is ms per step. Under
  compilation and CUDA graphs (b) is primary.
- Known stack properties (graph breaks, host syncs in a denoising loop) are reported, not fixed; a patched loop is
  a separately labelled variant.

## 3. Profilers
Two profilers are required: Nsight Systems (`nsys`, the timeline) and Nsight Compute (`ncu`, per-kernel counters).
Order: CUDA-event timing (§2) → `torch.profiler` → `nsys` → `ncu`; never time with a profiler attached. Flag
names vary between versions: T001 saves `nsys --help`, `nsys profile --help`, `nsys stats --help-reports`,
`ncu --help`, `ncu --list-sets`, `ncu --list-sections`, `ncu --query-metrics` and `cuobjdump --help` under
`env/tool_help/`; if a flag below is rejected, use the equivalent the installed version documents and record the
substitution in `meta.json.notes`.

### 3.1 Nsight Systems — timeline
```
nsys profile -t cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi --capture-range-end=stop -f true \
     -o scratch/<run>/<name> python bench/<script>.py ...
```
with `torch.cuda.profiler.start()/stop()` around **1–3 steady-state inferences** after the §2 warm-up — never a
whole benchmark loop. `--capture-range-end=stop` keeps the target alive after the window (on the Orin the default
killed it at `cudaProfilerStop`). For graphed modes add `--cuda-graph-trace=node`, otherwise graph-replayed kernels
are invisible. Export and commit:
```
nsys stats -r cuda_gpu_trace,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum,nvtx_kern_sum,nvtx_gpu_proj_trace \
     -f csv -o results/<run>/nsys/<name> scratch/<run>/<name>.nsys-rep
```
Report names differ between versions: record the ones used. From the trace, per NVTX region: number of kernels,
summed kernel time, **GPU idle = region time − union of kernel intervals**, the distribution of gaps between
consecutive kernels, **every** kernel with its grid and block (the per-kernel table is never capped at top-N),
memcpy time. Idle fractions use the unprofiled region time of §2 as the denominator where the traced wall time is
inflated (on the Orin by ×1.53); say which denominator.

### 3.2 Nsight Compute — per-kernel counters
ncu replays each kernel many times; its wall clock is meaningless and **a latency never comes from ncu**.
- **Hard rule: never run `ncu --set full` unfiltered.** Every ncu command carries one NVTX range
  (`--nvtx --nvtx-include "<range>/"`) or one kernel-name regex (`-k "regex:..."`) and, where the filter can still
  match many launches, `-c <count>` and / or `-s <skip>`.
- Command forms:
  ```
  # (A) one NVTX range, capture window set by the script (recommended)
  ncu --set full --nvtx --nvtx-include "ae_step[0]/" --profile-from-start off -o results/<run>/raw/ncu_ae_step0 -f python bench/ncu_target.py ...
  # (B) kernels by name, at most 5 launches
  ncu --set full -k "regex:gemm|gemv|cutlass" -c 5 -o results/<run>/raw/ncu_gemm -f python bench/ncu_target.py ...
  # (C) a window of the kernel order taken from the nsys cuda_gpu_trace
  ncu --set full -s 700 -c 10 -o results/<run>/raw/ncu_s700_c10 -f python bench/ncu_target.py ...
  # (D) metrics only (cheap, for full coverage): no --set, an explicit list
  ncu --metrics <comma-separated list> --nvtx --nvtx-include "vlm_prefix/" --profile-from-start off -o ... python ...
  # export, both pages
  ncu --import <rep>.ncu-rep --csv --page details > results/<run>/ncu/<name>_details.csv
  ncu --import <rep>.ncu-rep --csv --page raw     > results/<run>/ncu/<name>_raw.csv
  ```
- **Full coverage.** Bottleneck and roofline work covers **every kernel of a part**, not the first 40 (the Orin's
  gap). If ncu stops after a fixed number of profiled kernels per process (the Orin died after 480), chunk the
  part with form (C) windows taken from the nsys kernel order, one process per chunk, and report the kernel-time
  coverage of what was captured (target ≥ 95 % of the part's kernel time).
- **Compiled and graphed runs.** Profile through the same compiled callables that are timed. For CUDA graphs use
  ncu's graph-node profiling (`--graph-profiling node` where the version offers it); if the captured kernel list
  is empty or incomplete, capture the same compilation without CUDA graphs, say so, and compare kernel names and
  grids with the nsys trace of the graphed run.
- **Target script** `bench/ncu_target.py`: loads model and fixture, runs the §2 warm-up outside the window, then
  exactly one inference (or one step) between `cudaProfilerStart()` and `cudaProfilerStop()` with the §2 NVTX
  ranges inside. It must run standalone first.
- **Permissions.** T001 finds out whether ncu runs unprivileged on this device. If it needs root, the profiled
  Python runs as root: pass the interpreter by absolute path and the environment explicitly
  (`sudo HOME=$HOME HF_HOME=… HF_HUB_OFFLINE=1 TORCHINDUCTOR_CACHE_DIR=$PWD/scratch/<run>/inductor_root
  TRITON_CACHE_DIR=$PWD/scratch/<run>/triton_root $(command -v ncu) …`, which needs the `SETENV` sudoers tag of
  `docs/05_setup.md` §3; never `sudo -E`, which carries token variables), keep root's caches in their own
  `scratch/` directories, and `chown -R` what the run wrote before committing.
- **Clock check, every ncu run.** Record ncu's `SM Frequency` and the GPU's current frequency in `meta.json`. If
  they differ by more than 2 %, the capture is not used: find out why (`--clock-control` setting, clocks not
  locked, throttling), fix, capture again. T001 decides which `--clock-control` value matches the locked clock.
- **Parse with `csv.DictReader`**, never by splitting on commas: kernel names contain commas.

### 3.3 What to keep from ncu (decided per device in T001; validated in T003)
T001 writes `results/<E0 run>/ncu/metric_presence.csv` (`metric, category, present, nonzero_on_toy, notes`) for
every metric named below and for the candidates found with `ncu --query-metrics | grep -iE '<pattern>'`. Use the
names this device exports; where a name differs or is absent, use the nearest exported one and say so. A column
that is zero on one kernel is not evidence that the counter works: re-check per kernel.

Keep, per kernel launch (`results/<run>/ncu/kernels.csv`, one row per launch, with its NVTX region, kernel name,
grid, block, registers, shared memory):
- **occupancy / activity:** `SM Busy`, `SM Active Cycles`, `Achieved Occupancy`, `Theoretical Occupancy`,
  `Achieved Active Warps Per SM`, the occupancy limiters (`launch__occupancy_limit_*`);
- **throughput:** `Duration` (never a latency), `Compute (SM) Throughput`, `Memory Throughput`, `L1/TEX Hit Rate`,
  `L2 Hit Rate`, `Executed Ipc Active`, and — if present — `DRAM Throughput`. Write in every table what `Memory
  Throughput` measures on this device (on the Orin it was the L2 throughput, not DRAM);
- **DRAM bytes (the roofline x-axis, measured).** If `dram__bytes_read.sum` / `dram__bytes_write.sum` (or
  `dram__sectors_*`) exist and are non-zero, use them. If not (the Orin had none), use the L2 counters the Orin
  validated: reads = `lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum` × 32 B (or L2 fills,
  `lts__d_sectors_fill_*`), writes = `lts__t_sectors_aperture_sysmem_op_write.sum` × 32 B (an upper bound).
  **Validate whichever you use** in T003 on ≥ 4 kernels with known traffic (a streaming copy, an axpy, a large
  GEMV, a large GEMM): measured ÷ analytic for reads and writes, in `validation.csv`. The **analytic** bytes are
  always reported next to the measured ones;
- **FLOPs (the roofline numerator, measured).** SIMT FP: `smsp__sass_thread_inst_executed_op_{fadd,fmul,ffma}_pred_on.sum`
  and the `h*` and `d*` variants (FLOPs = add + mul + 2 × fma). Tensor-core FLOPs: the metrics ncu's own roofline
  sections use on this architecture (find them with `ncu --list-sections` and the section files, and
  `--query-metrics | grep -i tensor`); if the device only counts tensor **instructions**, FLOPs = instructions ×
  FLOPs per instruction from the SASS opcode shape (T008), labelled *derived*. Validate in T003 on the E1 GEMMs per
  dtype: measured ÷ analytic FLOPs. If no tensor FLOP counter can be validated, the measured rows use SIMT FLOPs
  plus analytic tensor FLOPs and say so;
- **tensor-pipe utilisation** (an Orin gap): every exported `sm__pipe_tensor*` / `sm__inst_executed_pipe_tensor*`
  active-cycle and instruction column, named as exported, plus the per-pipe `sm__inst_executed_pipe_<p>`
  percentages for `lsu`, `fma`, `alu`, `xu`, `adu`, `cbu` and any half-precision pipe present;
- **warp-stall reasons:** all `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` columns (or the
  `details` page's `Stall …` rows, whichever this version exports), every reason kept; check once whether they sum
  to ≈ 1 per kernel and say what they sum to;
- **LSU request counters:** `l1tex__t_requests_pipe_lsu_mem_global_op_{ld,st,red,atom}.sum`,
  `l1tex__t_requests_pipe_lsu_mem_local_op_{ld,st}.sum`, `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st,atom}.sum`;
- for E5b: `gpc__cycles_active.avg.per_warp` or its derivation (`gpc__cycles_active.avg` ÷ `sm__warps_launched.sum`,
  labelled derived), `launch__grid_size`, `launch__block_size`, `launch__registers_per_thread`.

Aggregate per region (`results/<run>/ncu/region_utilisation.csv`:
`region, n_kernels_captured, kernel_time_ms, kernel_time_coverage, metric, kernel_weighted, region_weighted`):
`kernel_weighted` = Σ(durationₖ × metricₖ) ÷ Σ durationₖ; `region_weighted` = Σ(durationₖ × metricₖ) ÷ region time,
durations and region time from the **nsys** run of the same configuration (labelled derived). Classify each
kernel by ncu's own speed-of-light rule (its `SOLBottleneck` thresholds: compute-, memory- or latency-bound), **and**
by its measured intensity (FLOPs ÷ DRAM bytes) against the ridge of `hw_constants.json`; where the two disagree,
say so; neither is tuned to agree.

### 3.4 Other tools
`torch.profiler` (`record_shapes=True, with_flops=True`) as a cross-check and the fastest top-30 table;
`cuobjdump`, `nvdisasm`, `c++filt` for E5b (`03_instruction_level_analysis.md`).

### 3.5 Raw and derived files — both are committed
**We want as much raw data as possible.** A summary is written *in addition to* the raw file, never instead.
- **Raw, committed under `results/<run>/raw/`:** every `.ncu-rep` (including failed or partial ones that carry
  information); the full SASS and PTX dumps, `-res-usage` output and the analysed `.cubin` files (copied from the
  Inductor / Triton cache); the raw power and tegrastats logs; `torch.profiler` traces and tables; stdout / stderr
  of every profiler and benchmark command (`raw/<name>.stdout.log`); per-iteration samples; the full ncu CSV
  exports (both pages). The full `nsys stats` CSV exports go to `results/<run>/nsys/` (compressed above 1 MB).
- **nsys binaries (`.nsys-rep`, `.sqlite`) are decided by T001 step 6.** On the Orin they embedded the system-wide
  process environment, including a session token of the agent's harness, even under `env -i`; they were never
  committed there. Until T001 step 6 proves otherwise on this device, they stay in `scratch/<run>/`, listed in
  `raw_files` with `committed: false`, the reason and the regenerating command; `.gitignore` blocks them. If step 6
  is clean, they are committed (compressed, §6) and **every** one is still scanned before committing; a hit puts
  that file back to `committed: false`.
- **Derived, committed under `results/<run>/`:** `nsys/*.csv`, `ncu/columns.txt`, `ncu/kernels.csv`,
  `ncu/region_utilisation.csv`, `roofline.csv`, `kernel_roofline.csv`, the E5b inventories and excerpts,
  `power.csv`, `latency_summary.csv`, `summary.md`.
- **Every raw file is listed in `meta.json.raw_files`** with path, sha256 and bytes; for a compressed or split file
  also the command and the sha256 + bytes of the uncompressed original (hash it **before** compressing).
- **Never committed:** weights and checkpoints, datasets, caches (Hugging Face, Inductor, Triton, pip), virtual
  environments, containers, vendor libraries (record path, version and sha256 instead), credentials, tokens, keys.
  `bench/fixtures/` is the only model-input data in git.
- **Secrets check before committing raw files.** (1) Never pass a token on a command line. (2) Download in a
  separate step; run every profiled or logged command with token variables removed and the hub offline:
  `env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u GITHUB_TOKEN -u GH_TOKEN HF_HUB_OFFLINE=1 …`. (3) Scan each raw
  file before compressing:
  `strings -n 8 <file> | grep -nE 'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|sk-ant-[A-Za-z0-9_-]{20,}|BEGIN [A-Z ]*PRIVATE KEY'`,
  plus the literal content of every token file you use (e.g. `~/.cache/huggingface/token`) and of every
  token-like variable in your own environment (`env | grep -iE 'token|secret|key|session'`). A hit means the file
  is **not committed**: record the pattern class and byte offset only — **never the matched bytes** — fix the
  environment, capture again, note it in `STATUS.md`. Record `secrets_scan: "clean"` per committed file.
- `scratch/` (ignored) holds uncompressed originals, caches and over-size captures; keep originals there until the
  research side accepts the result.

## 4. Power and energy
- **Rails are detected in T001**, not assumed: every power-monitor channel under `/sys/bus/i2c/drivers/*/`,
  `/sys/class/hwmon/*/` (`name`, `in*_label`, `curr*_label`, `power*_label`) and what `tegrastats` prints. T001
  writes the rail table to `env/README.md`: name, sysfs path, what it powers (from the label and NVIDIA's
  documentation for this module; mark inferences `[ASSUMPTION]`), bus voltage at idle. Find a channel by its label,
  never by `hwmon` index (indices change across boots).
- Every power run samples **all rails from sysfs at ≥ 10 Hz** (tidy CSV `t_s, rail, mW, mV, mA`) and runs
  `tegrastats --interval 100` alongside for temperatures, GPU and EMC frequency and as a cross-check.
  `tegrastats --logfile` appends: delete the log before each run, and stop it by pid or `tegrastats --stop`
  (`reference/orin/bench/common.py: stop_tegrastats()`).
- **Module power** = the sum of the rails T001 identifies as the module's inputs, with that list stated in every
  report; also report each rail separately. A memory rail, if one exists, is reported as its above-idle delta per
  phase next to its absolute value, under its exported name and with what it includes; never "DRAM power" without
  those words. If no memory rail exists, say so.
- 60 s of idle baseline in the same mode before each power run. Phase loops (vision-only, prefix-only, expert-only,
  full) of ≥ 60 s back-to-back iterations; the sampler covers the **timed window only** (on the Orin a sampler
  started before warm-up understated compiled energies). Energy per inference = mean power × mean latency, total and
  above idle.

## 5. Result directory — one per run
`results/<YYYYMMDD>_<Exx>_<model>_<mode>_<short>/` containing:
- `meta.json`: `experiment, task, date_utc, model, model_repo, model_commit, checkpoint, checkpoint_revision,
  precision, rung, compile_mode, variant, num_denoise_steps, action_horizon, prefix_tokens, suffix_tokens, batch,
  n_warmup, n_timed, power_mode (id and name), jetson_clocks, sm_count, gpu_hz, emc_hz, gpu_arch, jetpack, l4t,
  cuda, driver, python, torch, torchvision, triton, hw_constants_source, bench_commit, command, throttled,
  raw_files:[…], notes`. No `null`: a field that does not apply says `"n/a: <reason>"`. Each `raw_files` entry:
  `{path, sha256, bytes, kind, produced_by, secrets_scan}`; compressed files add `compression: {tool, level,
  command}, original: {name, sha256, bytes}`; split files add `parts: [{path, sha256, bytes}], reassembled:
  {sha256, bytes}`; an uncommitted file has `committed: false`, the reason and the regenerating command. `kind` ∈
  `nsys-rep, nsys-sqlite, ncu-rep, ncu-csv, nsys-csv, sass, ptx, cubin, res-usage, power-log, tegrastats,
  torch-trace, stdout, other`. Profiler runs add `nsys_version, ncu_version, ncu_set, ncu_filter, ncu_metrics,
  cuobjdump_version, ncu_sm_frequency_hz`.
- Tidy CSVs, one row per observation, long format: `latency_samples.csv (region, iter, ms)`,
  `latency_summary.csv (region, median_ms, mean_ms, p5_ms, p95_ms, min_ms, n)`, `kernels_per_region.csv`,
  `floor.csv` (§9), `power.csv`, `ncu/*` (§3.3), `sass/*` (E5b), and for E11 `model_inputs.csv` (§13.8),
  `segments.csv`, `handover.csv`, `boundary.csv`.
- **Roofline CSVs (E6).** `roofline.csv`, one row per (model, mode, part, source):
  `model, mode, rung, part, flops, bytes, ms, intensity, achieved_TFLOPps, achieved_GBps, ceiling_TFLOPps,
  ceiling_GBps, ridge, pct_peak_compute, pct_peak_bw, bound, source, time_source, n_kernels, coverage`, with
  `source` ∈ {`analytic`, `ncu`}, `time_source` ∈ {`nsys_kernel_sum`, `region_cuda_event`}, the ceilings and ridge
  read from `hw_constants.json` for the dtype the part runs in. `kernel_roofline.csv`, one row per kernel launch:
  `model, mode, part, launch_idx, kernel, grid, block, op_class, M, N, K, dtype, flops_analytic, flops_ncu,
  bytes_analytic, bytes_ncu, ms_nsys, intensity_ncu, achieved_TFLOPps, achieved_GBps, pct_peak_compute,
  pct_peak_bw, sol_bound, stall_top`, joined from ncu to nsys on (launch order, kernel name, grid) — never by
  phase alone.
- `raw/`: the raw files of §3.5, compressed per §6, with `REASSEMBLE.txt` when a file was split.
- `summary.md`: five to fifteen lines — what ran, the headline table, anything odd.
Never overwrite a result directory; a re-run is a new directory and the old one is referenced.

## 6. Git hygiene and raw-file size rules (binding)
`git pull --rebase` before every push; small commits whose message names the experiment and task; never
force-push; never rewrite pushed history. A result directory goes in its own commit, raw files included.
**GitHub rejects any file over 100 MB, and the whole push with it. The limit here is 50 MB per committed file.**
a. **Design captures small**: nsys 1–3 inferences; ncu one filtered range or one kernel window; power and
   `torch.profiler` the measured window only. Never profile a warm-up, a compile or a whole benchmark loop.
b. **Compress** every raw file over 1 MB with `zstd -19 -T0` (`--long=27` above 100 MB, then record that
   `zstd -d --long=27` is needed); hash the original first; move the original to `scratch/<run>/`.
c. **Over 50 MB compressed:** shorten the capture first. Only if irreducible: `split -b 45M -d -a 3 <name>.zst
   <name>.zst.part` and a `REASSEMBLE.txt` with the commands and the hashes of the reassembled and decompressed
   files; verify the round trip. More than 8 parts for one capture: ask under "Questions for the research side".
d. No Git LFS.
e. nsys binaries: §3.5.
f. **Report sizes after every task** in `STATUS.md` under "Repository size": `du -sh results/`, `du -sh .git`,
   `git count-objects -vH`, the five largest committed files (`git ls-files -z | xargs -0 du -b | sort -nr | head -5`).
   If `size-pack` approaches ~2 GB, stop adding raw files, keep them in `scratch/` with `committed: false`, and ask.
g. Before every commit with raw files: `git status` shows nothing under `scratch/`, no weights, datasets or
   environments; `find results -size +50M` is empty; the secrets scan ran.

## 7. Where things go
Virtual environments and containers: outside the repository (e.g. `~/venvs/<model>`), rebuilt from `env/`. Model
checkouts and weights: outside the repository; repo URL, commit, checkpoint and revision go in `meta.json`. Scripts:
`bench/`. Profiles and traces: `results/<run>/raw/`. Caches and uncompressed originals: `scratch/<run>/`. Data from
analysis: `results/<run>/`. Prose: `reports/`. `analysis/` and `reference/` are not yours.

## 8. Hardware constants: derive them on this device
Every constant in a profiling guide, a spec sheet or `reference/orin/` belongs to another device or is a claim to
verify. `hw_constants.json` holds this device's values; T002 writes the static half to
`results/<date>_E0_platform/hw_constants.json`, T003 the complete file to
`results/<date>_E1_peak/hw_constants.json`. **Every later task cites the T003 file in
`meta.json.hw_constants_source`** and reads its ceilings from it. Fields (no value without a source):
```
device, gpu_name, compute_capability, gpu_arch (the cuobjdump string), sm_count, power_mode {id, name},
gpu_clock_locked_hz, emc_hz, l2_bytes, smem_per_sm_bytes, max_warps_per_sm, max_threads_per_sm, regs_per_sm,
memory {type, bus_width_bits, spec_GBps, spec_source},
measured_bw_GBps {copy, axpy, read, best, spread_ratio, n_processes},
measured_tflops {fp32, tf32, bf16, fp16, int8, fp8_record_only}  (each: value, shape, kernel name),
ridge_flop_per_byte {fp32, tf32, bf16, fp16}  = measured TFLOP/s ÷ measured best GB/s,
spec_tflops {…, spec_source} where NVIDIA publishes one (else "n/a: not published"),
counter_validation {dram_bytes_ratio_read, dram_bytes_ratio_write, flops_ratio_per_dtype, source_file},
source_runs []
```
Compare every result against the **measured** ceilings, and against the spec only where a spec exists (labelled).
Because bandwidth may vary between processes, a floor ratio quoted in a profiling session uses a peak measured in
the same session where one exists (`bench/micro_peak.py` is cheap: run it at the start of each profiling session).

## 9. Physical floor — compute it before profiling
For each part (`vision`, `vlm_prefix`, one `ae_step`, and `ae_total`) write `results/<run>/floor.csv`
(`model, region, bytes_read, flops, peak_GBps, peak_TFLOPps, floor_mem_ms, floor_compute_ms, floor_ms, bound, peak_source`):
`floor_mem_ms = bytes that must be read at least once ÷ peak bandwidth`, `floor_compute_ms = FLOPs ÷ peak
throughput`, `floor_ms = max(…)`, `bound` = the larger term. Bytes = Σ `p.numel() × p.element_size()` over the
parameters the part touches (from the loaded model), plus the KV cache an `ae_step` reads, plus activations if
not negligible. FLOPs = 2 × tokens × linear parameters touched + attention FLOPs, from the real S and P. Rows with
`peak_source = measured` (T003) and, where NVIDIA publishes one, `spec`. Every report states
`gap_ratio = measured ÷ floor` per part. Reading: < 1.5× near-optimal, 1.5–3× moderate, 3–10× substantial software
overhead, > 10× a structural problem.

## 10. Root-cause section of every E5 / E5b / C1 report: ask "why" three times
For the largest contributor to each part's gap, a chain of at least three "why" steps, each with a number and its
source file, ending at a fact that cannot be decomposed further (a shape, a data structure, a physical limit):

| step | observation | evidence (file, metric, value) |
|---|---|---|
| symptom | e.g. region-weighted SM occupancy is x % in `ae_step` | `ncu/region_utilisation.csv` |
| why 1 | e.g. the GPU is idle y % of the step; kernels average z µs | `nsys/…`, `kernels_per_region.csv` |
| why 2 | e.g. inside kernels, long-scoreboard stalls are w % and global loads dominate the LSU | `ncu/kernels.csv` |
| why 3 | e.g. M = 51 gives too few warps per SM to hide the load latency | `sass/…`, launch grid |
| root cause | one sentence | |

Then a gap table `factor, evidence, contribution_ms, share_of_gap`, shares summing to 100 % ± 10. The examples
show the form, not expected answers. Never write "memory-bound" or "launch-bound" without the chain.

## 11. Review before marking a task done
- **Every task — self-review.** Item by item against the acceptance list: every file exists under its name; every
  number in a report traces to a file in `results/`; `meta.json` complete (§5, no null) with hashes; n, power mode
  and `throttled` stated; measured / derived / not measured separated; deviations listed; every raw file committed
  or listed `committed: false` with a reason; §6g checks pass; sizes in `STATUS.md`; nothing outside your
  partition edited; `PROGRESS.md` updated. Append the filled checklist to the report under "Self-review".
- **P0 tasks (T001–T006) — independent pass** if your framework has sub-agents: give a fresh sub-agent only the
  task file, this protocol and the produced files; ask it for unmet acceptance items and numbers that cannot be
  traced. Fix, then mark done. Without sub-agents, review twice, the second time starting from the result files.

## 12. Profiling checklist and known pitfalls
Checklist per model: (1) one clean inference without a profiler; (2) parameters, bytes and FLOPs per part;
(3) `floor.csv`; (4) NVTX + CUDA-event timing; (5) `torch.profiler` top-30; (6) nsys + CSV export; (7) ncu per part
+ both page exports; (8) SASS of the dominant kernels; (9) roofline rows; (10) root-cause chain; (11) self-review.

| symptom | cause / fix (the Orin case in brackets where it was learned there) |
|---|---|
| ncu runs for hours, report ≥ 1 GB | unfiltered `--set full`; filter (§3.2), reduce `-c`, narrow the range |
| push rejected, "exceeds GitHub's file size limit" | a raw file was committed unsplit; undo the unpushed commit (`git reset --soft HEAD~1`), shorten / compress / split (§6), commit again |
| occupancy reads 0 % | collected with a set that lacks the Occupancy section; use `--set full` or add the section |
| `ERR_NVGPUCTRPERM` / "insufficient privileges" | ncu needs root here: §3.2 permissions, `docs/05_setup.md` H1 |
| `sudo ncu`: command not found | sudo's `secure_path`; `sudo $(command -v ncu)` |
| target dies at `cudaProfilerStop` (exit 143) | add `--capture-range-end=stop` |
| ncu stops after N kernels in one process | chunk with `-s` / `-c` windows, one process per chunk [Orin: 480] |
| ncu sees no kernels inside CUDA-graph replays | `--graph-profiling node`; else the same compilation without graphs, labelled (§3.2) |
| a compiled model shows many graph breaks and eager-like times | region context managers inside the compiled region; compile first, instrument second (§2) |
| "Not enough SMs to use max_autotune_gemm mode" | Inductor's `is_big_gpu` SM threshold; report it, and measure the labelled forced-autotune variant (`all_compile_autotune`) [Orin: fired in every mode] |
| `torch.cuda.is_available()` false after a `pip install` | pip replaced the CUDA wheel with a CPU wheel; reinstall the CUDA wheels by URL with `--no-deps`, install model repos with `--no-deps`, re-check after every install step [Orin] |
| `CUDA error: no kernel image is available` | the wheel was not built for this architecture; use a wheel or container built for it (T001) |
| power log has two sessions in it | `tegrastats --logfile` appends; delete first |
| tegrastats keeps running after the harness exits | `Popen(["sudo","tegrastats",…]).terminate()` signals sudo, not tegrastats; stop by pid or `--stop` [Orin] |
| every call ~30 % slower under `torch.profiler`, even after it detached | CUPTI attach cost; count kernels and time them in separate processes [Orin] |
| bandwidth peak differs between processes at constant clocks | measure the peak in the same session and quote the spread (§8) [Orin: up to 1.19×] |
| `clone()` of a contiguous tensor shows no kernel in ncu | it runs on the copy engine; use an elementwise kernel with the same traffic [Orin] |
| ncu shows full L2 misses on a working set that fits in L2 | ncu restores device state between replay passes; L2 retention across launches is invisible under ncu |
| CUDA-event timing of one small kernel looks too slow | per-call event + sync overhead (Orin: 45–75 µs); time small kernels in amortised loops or graph replays |
| a kernel's time saving after a shape change | integer alignment can switch the kernel (Orin: M ≡ 0 mod 8); check bytes and kernel names before calling it less work |
| GPU time 0 or negative, first iteration 2–5× slower | `time.time()` or no warm-up: §2 |
| ncu CSV columns misaligned | kernel names contain commas: `csv.DictReader` |
| `flash-attn` will not build | `pip install wheel`, then `--no-build-isolation`; long aarch64 build; else PyTorch SDPA, recorded |
| matplotlib fails headless | `matplotlib.use("Agg")` |

## 13. E11 — inputs for the hybrid GPU + PIM model: methods and the `model_inputs.csv` schema
E11 measures what a system model needs when vision and the VLM prefix stay on the GPU and the action expert's large
matrix products run in memory. §1–§6 apply. If an API named here does not exist in your PyTorch build, use the
equivalent it offers and record the substitution in `meta.json.notes`.

### 13.1 Micro-timing of small GPU work
A host segment takes tens of µs to a few ms and in the target system runs **alone**. Time every small piece of
work as: `wall_isolated` (`synchronize`; `perf_counter`; run once; `synchronize`; `perf_counter`) — the primary
number — repeated with a pause `gap_ms` ∈ {0, 2} before the first `synchronize()` (`variant = gap_ms=<g>`);
`gpu_event` (CUDA events, calls back to back: pure device time); and `wall_launch` (`perf_counter` around the call
without the trailing synchronise: the CPU cost of issuing). ≥ 20 warm-ups, then **≥ 1,000 timed calls** per
(segment, style, timing); keep every sample (`raw/segment_samples.csv.zst`: `segment, exec_style, timing, variant,
iter, ms`). Execution styles (`exec_style`): `eager` (the model's own code); `compiled`
(`torch.compile(fn, dynamic=False)`, default mode, no CUDA graphs; a failure is recorded with its error and the row
marked `not measured`); `cudagraph` (`torch.cuda.CUDAGraph()` on static inputs; one graph per segment, and one graph
of the whole per-layer host sequence as a lower bound; say in `variant` whether the captured body was eager or
compiled). Kernels per call: one `torch.profiler` run over 10 calls per (segment, style); kernels per call = CUDA
kernel events ÷ 10; commit the table and trace under `raw/`; plus one small nsys capture of ten consecutive
per-layer host sequences per style. Inputs are **real**: capture each segment's inputs with forward hooks during
one real denoising step at layer 0 and a middle layer (saved to `scratch/`, not committed) and replay on them; use
the model's own modules; re-implement only what the code does inline, line for line, citing file and line.

### 13.2 What a host segment is, and `segments.csv`
A **host segment** is a maximal run of consecutive operations of one expert layer that are **neither a large linear
layer** (a weight matrix applied to the S suffix tokens: q / k / v / o, MLP gate / up / down, or a fused form)
**nor an attention matrix product** (q·Kᵀ, probs·V). Small linears on a single conditioning vector (an adaptive
norm's dense layer) belong to the host segment they sit in. Derive the list **from the model code**, per model:
`segments.csv` = `model, segment, order_in_layer, layer_kind, source_ref, ops, inputs, outputs, bytes_read,
bytes_written, flops, flops_formula, in_model_code, notes` (`source_ref` = `file:line-line` at the model commit;
`inputs` / `outputs` = `name:shape:dtype` lists; `layer_kind` separates layer types; `in_model_code` = `yes` or
`mapping_only`). Per-step extras (suffix / time embedding, final norm, output projection, Euler update) are listed
with `order_in_layer = -1`. π0's names: `rmsnorm1`, `rope`, `softmax` (+ variant `dtype=fp32_in_fp16_out`),
`residual_rmsnorm2`, `gelu_gate_mul_up`, `partials_residual` (`mapping_only`: `partials.sum(0) + x` over 4
partials), `residual`, `attention_gpu`, `cast_scores_fp32_fp16`; extras `suffix_embed`, `final_norm_out_proj`,
`euler_update`. Other models: name by analogy, one list per layer kind (`reference/orin/bench/host_segments.py`,
`groot_segments.py`).

### 13.3 Transfers and synchronisation on unified memory
Thor's CPU and GPU share one DRAM, but software still copies. Sizes {4 KB, 16 KB, 64 KB, 256 KB, 1 MB, 2 MB,
4 MB, 16 MB} (KB = 1024 B) plus the model's exact boundary sizes, both directions, three methods: `pageable`
(`copy_` with an ordinary CPU tensor), `pinned` (`pin_memory=True`, `copy_(…, non_blocking=True)` +
`synchronize()`), `zero_copy` (one allocation both address: `cudaMallocManaged` or mapped pinned host memory, by
whatever route works — a small extension, `ctypes` into `libcudart`, a pluggable allocator; the Orin's
`reference/orin/bench/managed_alloc.cu` is one; time CPU write, GPU kernel read, synchronise, CPU read, and the same
kernel on a device tensor). `wall_isolated`, ≥ 20 warm-ups, ≥ 200 timed per point (≥ 1,000 at ≤ 64 KB), all samples
kept (`raw/xfer_samples.csv.zst`). Fit **t = a + b·bytes** per (method, direction) on per-size medians: `fit_a`
(ms), `fit_b` (ms per MB, MB = 10⁶ B), `fit_r2`, 1/b as GB/s. Synchronisation costs (≥ 1,000 samples each):
`synchronize_idle`, `event_synchronize_done`, `event_query`, `tiny_kernel_roundtrip`, `stream_handoff`; and whether
the waiting CPU spins (process CPU time ÷ wall time during a long synchronise).

### 13.4 The hand-over object (prefix → action expert)
After the prefix ran on the fixture, inspect what the expert receives (a KV cache for π0 / π0.5 / SmolVLA; for
GR00T, whatever its backbone hands the DiT — say what it is). `handover.csv` = `model, layer, tensor, shape, dtype,
stride, contiguous, bytes, valid_tokens, padded_tokens, notes`, one row per tensor per layer plus `TOTAL`; valid vs
padded positions distinguished. Time (`gpu_event`, `wall_isolated`) the materialisation of the whole object as one
contiguous buffer `[layers, 2, P, kv_heads, head_dim]` (`variant = layout=stacked`) and with V transposed
(`layout=v_transposed`). From code **and** run time (`data_ptr()` across steps, copy kernels in the nsys trace of
`ae_step`), answer in `kv_reuse.md`: computed once per inference; referenced or re-copied per step; suffix K/V
appended and dropped; bytes copied per step. Per-step boundary tensors in `boundary.csv` = `model, tensor,
direction, shape, dtype, bytes, per, notes` (`per` ∈ {`inference`, `step`, `layer`}).

### 13.5 Background memory load (a proxy — read the limits)
`bench/bw_hog.py`, a **separate process**, duty cycle d ∈ {25, 50, 75, 100} % (period 20 ms): `gpu_hog`
(`dst.copy_(src)`, two 256 MiB device tensors), `cpu_hog` (multi-threaded CPU copy on cores the foreground does not
use, pinned with `taskset`; record both core sets); controls at d = 100 %: `gpu_spin_control` (small GEMMs),
`cpu_spin_control` (in-cache arithmetic). The hog logs its achieved rate once per second (`raw/bw_hog_<label>.log`);
report it alone and during the run and as a % of the E1-measured peak. **Limits, repeated in the report:** this is a
proxy for memory-side activity on the same DRAM; a real PIM occupies DRAM channels through the command bus, while a
software hog competes at the memory controller and in the caches and also takes GPU time slices or CPU cores. The
result is the sensitivity of the GPU work to lost bandwidth, not a measurement of PIM interference.

### 13.6 CPU utilisation per phase
In a separate run: (a) per region, the launching thread's CPU time (`time.thread_time()`) and the process CPU time
(`time.process_time()`) at region boundaries, summed over ≥ 100 iterations, ÷ region wall time; (b) per-core
utilisation from `/proc/stat` deltas over phase-only loops of ≥ 30 s. Commit `raw/cpu_util_samples.csv`. State the
cores online and the spin-wait finding of §13.3.

### 13.7 Concurrency (pipelining)
Throughput = completed inferences ÷ wall time of a back-to-back loop ≥ 60 s. For concurrent cases say how the two
activities share the GPU: two processes, or one process with two CUDA streams (try a high-priority stream for the
short work and record whether it is accepted). Report median, p95 and max of the short activity alone and under
concurrency. **Two-stream rule:** both activities are issued from **one** Python thread onto two streams (or the
long one as graph replays), never from a second Python thread (GIL contention). The two-process result is primary.

### 13.8 `model_inputs.csv` — the machine-readable hand-off (every E11 result directory has one)
Long format, UTF-8, header exactly:
```
model,model_commit,bench_commit,quantity,key,variant,value,unit,stat,n,exec_style,timing,power_mode,precision,P,S,N,source_file
```
| column | content |
|---|---|
| `model` | `pi0`, `pi05`, `smolvla`, `groot_n17`, `groot_n16`, `groot_n15`, or `none` for model-independent rows |
| `model_commit`, `bench_commit` | model repository commit; commit of `bench/` that produced the number |
| `quantity` | one name from the table below; a quantity that does not fit is proposed in `STATUS.md` |
| `key` | the sub-item (segment, method, phase, rail, load label); empty if none |
| `variant` | `name=value` pairs separated by `;` (`gap_ms=2`, `layer=0`, `size_bytes=65536`, `n_images=2`, `compile_mode=…`, `graph=per_segment`, `sharing=two_processes`, `from=<run dir>`); empty if none |
| `value`, `unit` | a number and its unit: `ms`, `bytes`, `flops`, `count`, `ratio`, `pct`, `W`, `GBps` (10⁹ B/s), `TFLOPps`, `inf_per_s`, `ms_per_MB`, `ms_per_token`, `ms_per_image` |
| `stat` | `median`, `mean`, `p5`, `p95`, `min`, `max`, `value`, `fit_a`, `fit_b`, `fit_r2` |
| `n` | samples behind the statistic (points for a fit; 1 for `value`) |
| `exec_style` | `eager`, `compiled`, `cudagraph`, or empty |
| `timing` | `wall_isolated`, `gpu_event`, `wall_launch`, `wall_loop`, `slope`, `analytic`, or empty |
| `power_mode` | the nvpmodel mode name and `+jetson_clocks` if applied, e.g. `<mode name>+jetson_clocks` |
| `precision` | `bf16`, `fp16`, `fp32` |
| `P`, `S`, `N` | prefix tokens, suffix tokens, denoising steps; empty when not applicable |
| `source_file` | path relative to this result directory of the file the value came from; it must exist here. To restate another run's number, copy its tidy CSV to `raw/restated/<run dir>/` and point there, with `from=<run dir>` |

Report at least `median`, `p5`, `p95` for every timed quantity. Quantities (`key` values in parentheses):

| quantity | unit | meaning |
|---|---|---|
| `t_preprocess`, `t_vision`, `t_vlm_prefix`, `t_postprocess`, `t_ae_step`, `t_ae_total`, `t_e2e` | ms | §2 regions; `t_e2e` = wall clock of one inference |
| `t_host_segment` (segment of `segments.csv`; also `cast_scores_fp32_fp16`) | ms | one host segment, §13.1 |
| `n_kernels_host_segment`, `bytes_host_segment`, `flops_host_segment` (segment) | count, bytes, flops | kernels per call; analytic bytes; analytic FLOPs |
| `t_host_layer` (`sum_of_segments`, `sequence`, `whole_layer_graph`; layer kind in `variant`) | ms | all host segments of one layer |
| `t_step_extra` (`suffix_embed`, `final_norm_out_proj`, `euler_update`, other names, `total`) | ms | per-step work outside the layer stack |
| `t_launch_per_kernel` | ms | `wall_launch` ÷ kernels per call, eager; median over segments, and per segment |
| `t_sync` (`synchronize_idle`, `event_synchronize_done`, `event_query`, `tiny_kernel_roundtrip`, `stream_handoff`) | ms | §13.3 |
| `xfer_latency` (`<method>:<direction>`; `size_bytes` in `variant`) | ms | one transfer |
| `xfer_latency_fit` (`<method>:<direction>`) | ms, ms_per_MB, ratio | t = a + b·bytes |
| `kv_bytes_per_layer`, `kv_bytes_total` (`valid` or `padded`) | bytes | §13.4 |
| `t_kv_materialise` (`stacked`, `v_transposed`) | ms | §13.4 |
| `kv_copied_bytes_per_step` | bytes | prefix KV bytes re-copied per step (0 if referenced) |
| `boundary_bytes_per_step` (tensor name, and `total`) | bytes | per-step boundary traffic |
| `prefix_latency_fit` (`vision_vs_images`, `vlm_prefix_vs_P`, `ae_step_vs_P`) | ms, ms_per_image or ms_per_token, ratio | E11 part 4 |
| `prefix_throughput`, `e2e_throughput`, `ae_throughput` | inf_per_s | back-to-back loops |
| `prefix_slowdown` (`gpu_hog_25` … `gpu_hog_100`, `cpu_hog_25` … `cpu_hog_100`, `gpu_spin_control`, `cpu_spin_control`, `host_segments`) | ratio | prefix latency under load ÷ alone |
| `host_layer_slowdown` (load label, incl. `prefix_loop`) | ratio | `t_host_layer` under load ÷ alone |
| `bg_bw_GBps` (load label; `alone` / `during` in `variant`) | GBps | achieved background rate |
| `cpu_util` (`vision`, `vlm_prefix`, `ae_total`, `e2e`; `scope=…` in `variant`) | pct | §13.6 |
| `p_mean_W` (`<phase>/<rail>`, rail names of `env/README.md` and `module`) | W | mean power of a phase loop |
| `gpu_idle_ae_total`, `gpu_idle_frac_ae_total` | ms, ratio | `ae_total` − union of kernel intervals (nsys) |
| `achieved_bw_GBps`, `achieved_tflops` (segment, region, or an E1 kernel) | GBps, TFLOPps | analytic bytes or FLOPs ÷ `gpu_event` time |

Check every file with `python docs/reference_scripts/check_model_inputs.py results/<run>/model_inputs.csv`; it
must exit 0.

## 14. E12 — co-running the prefix and the action expert on one GPU
Specification: `reference/orin/specs/E12_corun/README.md`, with the Thor adaptations of T013. Binding:
- **Definitions.** `A_iso`, `B_iso`: prefix (vision + VLM prefix) and expert (all N steps) stage times, each alone
  through the same harness, stream and code path as the co-run. `A_co`, `B_co`: the same while co-running in the
  steady-state pipeline, timed on the stage's own stream after any gating cross-stream wait. `T_serial`: both back
  to back on one stream. `T_pipe`: wall time of a steady-state pipeline loop ÷ frames. `s = B_co / B_iso`,
  `p = A_co / A_iso`, `r = A_iso / B_iso`, `O = T_pipe / max(A_iso, B_iso)`. All from unprofiled runs.
- **Same session, bracketed** (isolated → co-run → isolated), one power mode.
- **Fresh KV, verified**: pipelined vs serial actions frame by frame with distinct observations and noise seeds,
  in eager and in the compiled path that is timed, with a stale negative control that must fail. A mechanism whose
  actions fail is invalid, never a speed-up.
- **Baseline unchanged**: π0 at `best_bf16`; a mechanism needing a different compilation is a labelled variant
  with its own isolated times.
- §13.7's issuing rule; unavailable mechanisms (green contexts, MIG, MPS, `libsmctrl`, …) recorded with the version
  check, the exact call and the error under `raw/`, summarised in `mechanisms.csv`.

## 15. E12b — the compiled co-run, whole GPU and inside SM partitions
Specification: `reference/orin/specs/E12b_compiled_corun/README.md`, with T013's adaptations; §14 applies.
- **Fair configuration**: `best_bf16` + the timestep hoist (N timestep tensors built once), after its compiled
  equivalence check. `A_part`, `B_part`, `A_co,part`, `B_co,part`, `T_pipe,part`, `p_part`, `s_part`,
  `φ = (T_serial − T_pipe) / (T_serial − max(A_iso, B_iso))`.
- Every co-run number is from the compiled path; a harness change needed for correctness (own graph capture,
  separate pools, double-buffered stage I/O) is a labelled variant with its own isolated times and equivalence.
- **No silent NaN**: every action tensor of every timed, checked and energy run is finite-checked without a host
  sync inside the timed loop; a non-finite run is invalid and reported, never dropped.
- **Partitions**: the sizes the driver grants are recorded per run; confinement verified through the launch path
  that is timed (a CUDA graph obeys the partition it was **captured** in); a bandwidth hog used as a control runs
  in the same process on a partition stream. Splits are derived from this device's SM count and the partition
  granularity the driver reports.
- **Provenance**: every row and `meta.json` carry a variant label naming every change from `best_bf16`; no null.
- Done only when every acceptance item of the specification (as adapted in T013) holds, or the status line names
  the one that does not and why.

## 16. E13 / E14 — prefix under stolen bandwidth, prefix savings
Specification: `reference/orin/tasks/T016_prefix_bandwidth_and_followups.md` steps 1 and 3, with T014's adaptations.
- `σ(x) = A(prefix | load at x GB/s) / A(prefix | reference)`, same session, bracketed. The **primary** load takes
  **no SMs** (copy engine between pinned host and device buffers, or a pinned multi-threaded CPU copy); show it
  launches no GPU kernel. An SM-taking load is a secondary arm only. Rates are reported alone and during the prefix.
- Per-kernel DRAM bytes with an explicit metric list (§3.2 form D), never a `--set` that requests counters this
  device lacks.
- **Variants are never the baseline** (`nopad`, `keep1`, `kvonce`, …): each has its own same-session reference and
  an equivalence result (bitwise where exact). Which variant enters a baseline is the research side's decision.
- Probing the memory-channel mapping is read-only: no boot, device-tree or kernel-parameter change, no reboot.
- §15's finite check and provenance rules apply.

## 17. E14b — π0 prefix padding trim, decomposed
Specification: `reference/orin/tasks/T017_pi0_padding_trim.md`; §16 and §15 apply.
- Labels `pad` (reference), `nocam`, `trimprompt`, `nopad` (= `nocam` + `trimprompt`); `nocam` never creates the
  empty camera (no dummy image, no encoder call, no prefix tokens).
- **Gate before timing**: each variant matches `pad` in eager at relative error ‖Δ‖₂ / ‖a_pad‖₂ ≤ 1 × 10⁻⁴ on every
  input of the fixed set before it is timed or profiled.
- Same session, bracketed, in the order the specification fixes; DRAM bytes by §16's method.
- **Pass criterion fixed in advance**: compiled, variant vs `pad`, same inputs and seeded noise: bitwise; or PASS
  when the max-abs and relative differences are no larger than those between the shipped model and `best_bf16` on
  the same inputs; otherwise FAIL, reported. No edits to installed model sources: variants are harness overrides in
  `bench/`.
