# 02 — Measurement protocol (binding)

## 1. Device state — record for every run, fix before every run
- Power mode: `sudo nvpmodel -q --verbose` lists the modes and their ids on this board (on AGX Orin typically
  0 = MAXN, 1 = 15 W, 2 = 30 W, 3 = 50 W — **confirm, do not assume**). Default for all experiments: **MAXN**
  followed by `sudo jetson_clocks`. E7 repeats selected points at the other modes. Switching between MAXN and a
  capped mode can require a reboot: do all MAXN work first, run the mode sweep as the **last** item of a session,
  write "next: resume E7 at mode <id>" in `STATUS.md` before switching, and if a reboot is demanded stop and leave
  it to the human (never confirm the prompt yourself).
- Save `sudo jetson_clocks --show`, `cat /etc/nv_tegra_release`, `dpkg -l | grep -E 'nvidia-jetpack|cuda-toolkit|tensorrt|cudnn'`,
  `nvcc --version`, `python -c "import torch;print(torch.__version__, torch.version.cuda)"`, kernel version, and the EMC
  (memory) clock: `sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/rate` (expect 3,199 MHz = LPDDR5-6400).
- Thermal: log temperatures with tegrastats; if any run shows throttling or a clock below its locked value,
  mark it `throttled: true`, cool down, repeat. Use the fan at maximum (`jetson_clocks --fan`).
- Nothing else running: no desktop session if possible, no second benchmark, swap/zram state recorded.

## 2. Timing
- Batch size 1, bf16 weights and activations unless a task says otherwise, fixed seeds, fixed inputs
  (one real LIBERO observation — camera images, robot state, language prompt — saved once to `bench/fixtures/`
  as PNG + JSON, not random tensors, so token counts are real; take it from the published LIBERO dataset on
  Hugging Face so that this does not depend on installing the simulator, and record the dataset, episode and frame).
- GPU time with CUDA events (`torch.cuda.Event(enable_timing=True)`) around each region, synchronised only at
  region boundaries; also wall-clock end to end. Do not time with profilers attached: profile in separate runs.
- ≥ 20 warm-up iterations (for `torch.compile`: until no recompilation is reported, and log the compile time),
  then ≥ 100 timed iterations. (Profiling guides that say "3 warm-ups" describe profiler runs on desktop GPUs;
  here the rule is ≥ 20 / ≥ 100 for every timed run, and ≥ 20 warm-ups before every profiler capture window.) Report **median, mean, p5, p95, min, n**. Keep the per-iteration samples.
- Regions (use the same names as NVTX ranges): `preprocess`, `vision`, `vlm_prefix` (prefill that builds the
  KV cache), `ae_total`, `ae_step[i]` for each denoising step, and inside one representative step
  `ae_layer[j]`; `postprocess`. If the code fuses regions so they cannot be separated, say which.
- Sweep the number of denoising steps with the model's own parameter, not by editing loops: in openpi's PyTorch
  model it is the `num_steps` argument of `sample_actions`; in lerobot it is `num_steps` in the policy config.
- **Per-step time has two definitions; report both and label them.** (a) *Instrumented*: CUDA events / NVTX around
  each denoising step — valid in eager mode. (b) *Slope*: run the full N sweep and fit total action-expert (or
  end-to-end) latency = a + b·N; b is the ms per denoising step, a the fixed cost. Under `torch.compile` and CUDA
  graphs, events inside the compiled region cause graph breaks or disappear, so (b) is the primary number there;
  additionally measure a variant that compiles only the single-step function (`denoise_step`) and loops in Python.
- Known property, to be reported and **not fixed**: openpi's denoising loop tests a GPU tensor each iteration
  (`while time >= -dt/2`), which forces a CPU–GPU sync per step and a graph break. That is part of the software
  stack we are characterising. If you also measure a patched loop, report it as a separate, labelled variant.

## 3. Profilers
Two profilers are **required**: Nsight Systems (`nsys`, the timeline) and Nsight Compute (`ncu`, per-kernel
hardware metrics). Use them in this order — CUDA-event timing (§2) → `torch.profiler` → `nsys` → `ncu` — and never
time with a profiler attached. Flag names vary between versions: run `nsys --help`, `ncu --help` once, save both
outputs under `env/tool_help/`, and if a flag below is rejected, use the equivalent your version documents and
record the substitution in `meta.json.notes`.

### 3.1 Nsight Systems (`nsys`) — timeline
```
nsys profile -t cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi -f true \
     -o results/<run>/raw/<name> python bench/<script>.py ...
```
with `torch.cuda.profiler.start()/stop()` around **1–3 steady-state inferences** (after the §2 warm-up, which
stays outside the window) — never a whole benchmark loop: the `.nsys-rep` is committed (§3.5) and must stay
small (§6). Known flag problems: overwrite is
`-f true` (`--force-overwrite` fails to parse on some versions); `--cuda-memory-usage=true` is useful but some
versions reject it as ambiguous — drop it if so. Then export and commit the CSVs:
```
nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum,nvtx_kern_sum,nvtx_gpu_proj_trace,cuda_kern_exec_sum \
     -f csv -o results/<run>/nsys results/<run>/raw/<name>.nsys-rep
```
and the full kernel trace for one inference (`-r cuda_gpu_trace`). `nvtx_sum` covers CPU ranges only; the
kernels-per-region attribution comes from `nvtx_kern_sum` / `nvtx_gpu_proj_trace`. Report names differ between
nsys versions: run `nsys stats --help-reports` and record the version and the names you used. For compiled runs
add `--cuda-graph-trace=node`, otherwise kernels replayed inside CUDA graphs are invisible. From the trace
compute, per NVTX region: number of kernels, summed kernel time, **GPU idle time = region time − union of kernel
intervals**, the distribution of gaps between consecutive kernels, top-20 kernels by time with their launch
grids, memcpy time.

### 3.2 Nsight Compute (`ncu`) — per-kernel SM occupancy, pipe utilisation, stalls
`ncu` is already installed on this Jetson (T001 verifies it). It replays every captured kernel many times
(17 or more) to collect hardware metrics, so it is **very slow** and its wall-clock is meaningless: a 50 ms
inference can take 5–30 minutes. Latency always comes from §2, never from an ncu run.

**Hard rule: never run `ncu --set full` unfiltered.** Every ncu command carries (a) one NVTX range
(`--nvtx --nvtx-include "<range>/"`) or one kernel-name regex (`-k "regex:..."`), and (b) where the filter can
still match many launches, the limits `-c <count>` (capture at most this many kernels) and/or `-s <skip>` (skip
the first N). An unfiltered `--set full` run takes hours and writes a report of 1 GB or more — which also could not be
committed (§6).

Command forms (the only ones we rely on):
```
# (A) one NVTX range, capture window controlled by the script (RECOMMENDED)
ncu --set full --nvtx --nvtx-include "ae_step[0]/" --profile-from-start off \
    -o results/<run>/raw/ncu_ae_step0 -f  python bench/ncu_target.py --model <model> --num-steps <N> ...

# (B) kernels by name, at most 5 launches
ncu --set full -k "regex:gemm|gemv|cutlass" -c 5 -o results/<run>/raw/ncu_gemm -f  python bench/ncu_target.py ...

# (C) skip the first N kernels, capture M (use with the kernel order taken from the nsys cuda_gpu_trace)
ncu --set full -s 700 -c 10 -o results/<run>/raw/ncu_s700_c10 -f  python bench/ncu_target.py ...

# export for scripted analysis — two pages
ncu --import results/<run>/raw/ncu_ae_step0.ncu-rep --csv --page details > results/<run>/ncu/ae_step0_details.csv
ncu --import results/<run>/raw/ncu_ae_step0.ncu-rep --csv --page raw     > results/<run>/ncu/ae_step0_raw.csv
```
`-o` is the output path without extension (ncu writes `<path>.ncu-rep`); `-f` overwrites. The long form of `-k`
appears as `--kernel-name` in some versions; check `ncu --help`. Both exported pages are committed **in full**
(every row and column; compressed per §6 when large) next to the `.ncu-rep` itself. (`--page raw` is the name of
an ncu export page; the `raw/` directory of §3.5 is a different thing.)

- **Region names.** The NVTX range passed to `--nvtx-include` is the §2 region name followed by `/`:
  `"vision/"`, `"vlm_prefix/"`, `"ae_step[0]/"`, `"ae_layer[0]/"`. Ranges are nested
  (`ae_total` ⊃ `ae_step[i]` ⊃ `ae_layer[j]`); if your ncu version needs the full nested path or mishandles
  the brackets, find the accepted syntax in `ncu --help`, and if necessary add alias ranges without brackets
  (`ae_step_0`, `ae_layer_0`) in `bench/` — record what you used. One range per ncu run.
- **Metric sets.** `--set basic` (fast: duration, throughput), `--set roofline` (medium: SM and memory
  throughput), `--set full` (slow: adds occupancy, stalls, IPC, pipes). Occupancy reads 0 % under
  `roofline`, so the utilisation work uses `full`. T001 lists the sets this Orin actually offers.
- **Restricting the capture with the cudaProfiler API.** The target script (`bench/ncu_target.py`, written by
  you) loads the model, runs the §2 warm-up **outside** the capture window, then runs exactly **one** inference
  (or one AE step) between
  `torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart()` and
  `torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()`, with the §2 NVTX ranges inside. With
  `--profile-from-start off`, ncu honours these markers. The same script must run without ncu (to check it
  works) before any ncu run.
- **Work from small to large.** First the toy run of T001; then `ae_layer[0]/` (one layer); then
  `ae_step[0]/` (one step: expect several hundred kernels — if a run exceeds ~60 minutes, stop it, and cover the
  step with form (B) on the top-20 kernels by time from nsys plus form (C) windows, saying in the report what
  fraction of the step's kernel time the captured kernels cover). `vision/` and `vlm_prefix/` likewise.
- **Compiled runs.** Whether ncu can see kernels replayed from CUDA graphs is not documented: try the best
  compiled mode; if the captured kernel list is empty or incomplete, capture under
  `max-autotune-no-cudagraphs` instead and say so.
- **Permissions.** `ERR_NVGPUCTRPERM` means the counters are locked:
  `sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'` (a runtime setting, not a boot change; record it
  in `env/`). On Tegra some counters may still be missing: record exactly which metrics come back empty.
- **Parse with `csv.DictReader`**, never by splitting on commas: kernel names contain commas.
- If, and only if, ncu exposes no usable counters on this device after the steps above, document the failure
  and fall back to **analytic FLOPs and bytes per kernel (from tensor shapes) ÷ measured kernel time**.

### 3.3 What to keep from the ncu export (utilisation = SM occupancy)
**GPU utilisation in this project means ncu's streaming-multiprocessor (SM) occupancy/activity metrics**, per
kernel and aggregated per region. tegrastats `GR3D_FREQ` is only a coarse cross-check (§4).

The `details` page is long-format, one row per (kernel launch, metric), with columns `Kernel Name`, `ID`,
`Metric Name`, `Metric Value`, `Metric Unit`, `Section Name`. The `raw` page is wide, one row per kernel launch
keyed by `ID`, one column per raw counter. Column and metric names below are the ones documented for desktop
GPUs; **on the Orin, save the exported header and the distinct `Metric Name` list to
`results/<run>/ncu/columns.txt`** and, where a name differs or is absent, use what the device exports and say so.

Keep, per kernel launch (`results/<run>/ncu/kernels.csv`, one row per launch, plus the kernel's NVTX region,
grid and block size):
- occupancy / activity: `SM Busy`, `SM Active Cycles`, `Achieved Occupancy`, `Theoretical Occupancy`,
  `Achieved Active Warps Per SM`, and from the raw page `sm__warps_active.avg.pct_of_peak_sustained_active`;
- throughput: `Duration`, `Compute (SM) Throughput`, `Memory Throughput`, `DRAM Throughput`, `DRAM Utilization`,
  `Mem Busy`, `L2 Hit Rate`, `L1/TEX Hit Rate`, `Executed Ipc Active`, `Executed Ipc Elapsed`;
- **per-pipe utilisation** (raw page): `sm__inst_executed_pipe_<p>.avg.pct_of_peak_sustained_active` for
  `<p>` ∈ {`lsu`, `fma`, `alu`, `xu`, `adu`, `cbu`, `fp16`} and `...pct_of_peak_sustained_elapsed` for
  {`lsu`, `fma`, `alu`, `xu`}. Pipes have different peak issue rates: the percentages do not sum to 100 and
  are not comparable across pipes as if they were instruction counts;
- **warp-stall reasons**: every `Stall …` metric exported — `Stall Long Scoreboard`, `Stall Short Scoreboard`,
  `Stall Wait`, `Stall Barrier`, `Stall Not Selected`, `Stall Memory Dependency`, `Stall Memory Throttle`,
  `Stall Math Pipe Throttle`, `Stall Misc`, and any others present (some versions name them
  `long_scoreboard`, `short_scoreboard`, `mio_throttle`, `lg_throttle`, `barrier`, `not_selected`, `wait`,
  `dispatch_stall`, `drain`, `selected`, `tex_throttle`, `sleeping`);
- LSU request counters for the load/store split (raw page): `l1tex__t_requests_pipe_lsu_mem_global_op_{ld,st,red,atom}.sum`,
  `l1tex__t_requests_pipe_lsu_mem_local_op_{ld,st}.sum`, `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st,atom}.sum`;
- for E5b: `gpc__cycles_active.avg.per_warp`, and the launch metrics `launch__grid_size`, `launch__block_size`,
  `launch__registers_per_thread`, `launch__occupancy_limit_{blocks,registers,shared_mem,warps}` if exported.

Aggregate per region (`results/<run>/ncu/region_utilisation.csv`, columns
`region, n_kernels_captured, kernel_time_ms, kernel_time_coverage, metric, kernel_weighted, region_weighted`):
- `kernel_weighted` = Σ(durationₖ × metricₖ) ÷ Σ durationₖ over the captured kernels of the region — what the
  SMs do *while a kernel runs*;
- `region_weighted` = Σ(durationₖ × metricₖ) ÷ region time, with durations and region time from the **nsys**
  run of the same configuration (idle gaps count as 0) — what the GPU does over the whole region. This is the
  number to compare with the characterization paper's "SM utilisation 22 %". Label it *derived* and give
  `kernel_time_coverage` (captured kernel time ÷ all kernel time in the region).
Classify each kernel: memory-bound if `Memory Throughput` > 1.5 × `Compute (SM) Throughput`, compute-bound if
the reverse, else balanced; report the share of region kernel time in each class.

### 3.4 Other tools
- `torch.profiler` with `record_shapes=True, with_flops=True` is a welcome cross-check and the fastest way to
  get the first top-30 kernel table; export `key_averages().table` as CSV.
- `cuobjdump`, `nvdisasm`, `c++filt` for the instruction-level analysis (E5b): see
  `docs/03_instruction_level_analysis.md`.

### 3.5 Raw and derived files — both are committed
**We want as much raw data as possible.** Prefer committing a raw file over summarising it; a summary is
written *in addition to* the raw file, never instead of it. The research side re-analyses raw files that the
device agent never thought to summarise.

- **Raw, committed under `results/<run>/raw/`** (tracked; raw and derived stay in the same result directory):
  - `.nsys-rep`, and the `.sqlite` that nsys derives from it (rule in §6e);
  - `.ncu-rep` of every ncu run, including failed or partial ones that still carry information;
  - the **full** SASS and PTX dumps from `cuobjdump`, `-res-usage` output, and the extracted / JIT-compiled
    `.cubin` files of the kernels analysed (copy them from the Inductor / Triton cache into `raw/cubin/`; the
    cache directory itself is scratch and is not committed);
  - the raw tegrastats log (and raw INA3221 samples if you read sysfs);
  - `torch.profiler` Chrome-trace JSON and its `key_averages` table;
  - stdout / stderr of every profiler and benchmark command (`raw/<name>.stdout.log`);
  - the **full** ncu CSV exports (both pages, all rows and columns) — these may live in `ncu/` as before; there
    is no longer a "filter to the kept columns above 5 MB" rule: keep the full export, compressed;
  - per-iteration latency samples (`latency_samples.csv`, §5) — every sample, never only the summary.
- **Derived, committed** under `results/<run>/`: `nsys/*.csv`; `ncu/columns.txt`, `ncu/kernels.csv`,
  `ncu/region_utilisation.csv`; for E5b the instruction-inventory CSVs, annotated hot-loop excerpts and
  cross-check tables; `power.csv`; `latency_summary.csv`; `summary.md`.
- **Every raw file is listed in `meta.json.raw_files`** with path (relative to the result directory), sha256
  and bytes; for a compressed or split file also the compression command and the sha256 + bytes of the
  uncompressed original (schema in §5). Compute the hash of the original **before** compressing.
- **Size rules are in §6** (design captures small, compress, split above 50 MB). They are binding: GitHub
  rejects a push that contains any file over 100 MB, and the whole push fails, not just that file.
- **Never committed, whatever a task seems to imply:** model weights and checkpoints (`*.safetensors`, `*.ckpt`,
  `*.pt`, `*.pth`, `checkpoints/`), datasets, caches (Hugging Face, Inductor, Triton, pip), virtual environments,
  vendor libraries (`libcublas*.so` etc. — record path, version and sha256 instead), and credentials or tokens.
  `bench/fixtures/` (one PNG + JSON LIBERO observation) is the only model-input data in git.
- **Secrets check before committing raw files.** Profiler reports and logs can record the environment and the
  command line of the profiled process. (1) Never pass a token on a command line. (2) Download models in a
  separate step; run every profiled or logged command with the token variables removed and the hub offline:
  `env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u GITHUB_TOKEN -u GH_TOKEN HF_HUB_OFFLINE=1 nsys profile ...`.
  (3) Before compressing, scan each raw file:
  `strings -n 8 <file> | grep -nE 'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY'`
  and also grep for the literal content of any token file you use (e.g. `~/.cache/huggingface/token`). A hit
  means the file is **not committed**: delete it, fix the environment, capture again, and note it in
  `STATUS.md`. Record `secrets_scan: "clean"` per file in `meta.json.raw_files`.
- A temporary, untracked `scratch/` directory (ignored) is where uncompressed profiler output, caches and
  over-size captures live before they are compressed into `results/<run>/raw/`. Keep the uncompressed originals
  in `scratch/<run>/` until the research side says the result is accepted.

## 4. Power and energy
- `sudo tegrastats --interval 100 --logfile results/<run>/raw/tegrastats.log` (10 Hz) or read the INA3221 rails directly
  from sysfs at ≥ 10 Hz if finer. Rails to keep: `VDD_GPU_SOC`, `VDD_CPU_CV`, `VIN_SYS_5V0`, `VDDQ_VDD2_1V8AO`
  (the memory rail — we care about it specifically). Convert to a tidy CSV (`t_s, rail, mW`). Always report each
  rail separately **and** their sum as "module power" (`VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0 + VDDQ_VDD2_1V8AO`;
  check the rail list of this carrier board and say if it differs). tegrastats `GR3D_FREQ` % is logged in the
  same file as a **coarse cross-check only**: it says whether the GPU had work queued, not how much of the GPU
  the work used. The utilisation metric of this project is ncu's SM occupancy (§3.3). Both the raw
  tegrastats log (`raw/tegrastats.log`, compressed per §6 if large) and the tidy CSV are committed.
- Measure 60 s of idle baseline in the same power mode before each power run.
- Because one inference is shorter than the sampling period, measure power on a **loop of back-to-back
  inferences ≥ 60 s**, and separately on loops that run only the prefix and only the action expert, so the
  phases can be attributed. Energy per inference = mean power × mean latency; report both total and
  above-idle. State the sampling rate and that rail readings are averaged by the sensor.

## 5. Result directory — one per run
`results/<YYYYMMDD>_<Exx>_<model>_<mode>_<short>/` containing:
- `meta.json`: `experiment, task, date_utc, model, model_repo, model_commit, checkpoint, precision, compile_mode,
  num_denoise_steps, action_horizon, prefix_tokens, suffix_tokens, batch, n_warmup, n_timed, power_mode,
  jetson_clocks, emc_hz, gpu_hz, jetpack, l4t, cuda, torch, torchvision, triton, bench_commit, command,
  throttled, raw_files:[…], notes`. Each `raw_files` entry:
  `{path, sha256, bytes, kind, produced_by, secrets_scan}` for a file committed as-is, plus, for a compressed
  file, `compression: {tool, level, command}, original: {name, sha256, bytes}`, plus, for a split file,
  `parts: [{path, sha256, bytes}], reassembled: {sha256, bytes}` (§6). `path` is relative to the result
  directory; `kind` ∈ `nsys-rep, nsys-sqlite, ncu-rep, ncu-csv, sass, ptx, cubin, res-usage, tegrastats,
  torch-trace, stdout, other`; `produced_by` is the exact command. A raw file that was produced but could not
  be committed (over the limits of §6 after shortening, or failed the secrets scan) is still listed, with
  `committed: false`, the reason, and the command that regenerates it.
- tidy CSVs, one row per observation, long format: `latency_samples.csv (region, iter, ms)`,
  `latency_summary.csv (region, median_ms, mean_ms, p5_ms, p95_ms, min_ms, n)`, plus `nsys/*.csv`, `power.csv`,
  `kernels_per_region.csv`, `floor.csv` (§9), `ncu/*` (§3.3) and `sass/*` (E5b, `docs/03_instruction_level_analysis.md`)
  when applicable. For profiler runs add to `meta.json`: `nsys_version`, `ncu_version`, `ncu_set`, `ncu_filter`
  (the exact `--nvtx-include` / `-k` / `-c` / `-s` used), `gpu_arch` (e.g. `sm_87`), `cuobjdump_version`.
- `raw/`: the raw files of §3.5 (profiler reports, dumps, logs, traces), compressed per §6, with
  `REASSEMBLE.txt` when a file was split.
- `summary.md`: five to fifteen lines — what was run, the headline table, anything odd.
Never overwrite a result directory; a re-run is a new directory and the old one is referenced.

## 6. Git hygiene and raw-file size rules (binding)
`git pull --rebase` before every push; small commits with a message that names the experiment and task; never
force-push; never rewrite pushed history. Commit a result directory in its own commit (raw files included), so
that a push that fails on size can be fixed without touching other work.

Raw files are committed (§3.5), so size is managed, not forbidden. **GitHub rejects any file over 100 MB (the
whole push fails) and warns above 50 MB.** The working limit here is **50 MB per committed file**.

a. **Design captures to be small.** nsys: a bounded window with the cudaProfiler API around 1–3 steady-state
   inferences (§3.1). ncu: one NVTX range or one kernel filter, with `-c` / `-s` limits (§3.2). tegrastats and
   `torch.profiler`: the measured window only. Never profile a whole benchmark loop, a warm-up or a compile.
b. **Compress before committing.** Any raw file over 1 MB is committed compressed: `zstd -19 -T0 <file>`
   (preferred; add `--long=27` for files over 100 MB and then record that `zstd -d --long=27` is needed) or
   `xz -9 <file>` if zstd cannot be installed (`sudo apt install zstd`; record it in `env/`). Hash the original
   first (§3.5). Commit `<name>.zst`, not the original (move the uncompressed original to `scratch/<run>/` so it
   cannot be staged by accident; this also applies to a full ncu CSV export in `ncu/`); files under 1 MB and text the research side will read
   directly (`summary.md`, tidy CSVs, annotated excerpts) stay uncompressed.
c. **If a compressed file is still over 50 MB:** first shorten the capture (fewer inferences, a narrower NVTX
   range, smaller `-c`) and capture again. Only if it cannot be made smaller, split it:
   `split -b 45M -d -a 3 <name>.zst <name>.zst.part` → `<name>.zst.part000`, `…part001`, …, and write
   `results/<run>/raw/REASSEMBLE.txt` with, per split file, the exact commands
   `cat <name>.zst.part* > <name>.zst && zstd -d <name>.zst` and the sha256 + bytes of the reassembled `.zst`
   and of the decompressed original. Verify the round trip yourself before committing. Commit the parts, not
   the unsplit file. More than 8 parts (≈ 360 MB) for one capture means the capture is badly designed: go back
   to (a) and, if it is truly irreducible, ask under "Questions for the research side" before committing it.
d. **No Git LFS** unless a task says so.
e. **The nsys `.sqlite`** is derivable from the `.nsys-rep` (`nsys export --type sqlite` / first `nsys stats`
   call): commit it only if it is under 50 MB compressed; otherwise record the regenerating command in
   `meta.json.raw_files` (`committed: false`) and keep it in `scratch/`.
f. **Report sizes after every task** in `STATUS.md` under "Repository size": the output of `du -sh results/`,
   `du -sh .git`, `git count-objects -vH`, and the five largest committed files
   (`git ls-files -z | xargs -0 du -b | sort -nr | head -5`). **If the repository (`size-pack`) approaches
   ~2 GB, stop adding raw files**: keep them in `scratch/`, list them with `committed: false`, and ask under
   "Questions for the research side" how to proceed.
g. Before every commit that contains raw files: `git status` shows nothing under `scratch/`, no weights,
   datasets or environments; `find results -size +50M` is empty; the secrets scan of §3.5 was run.

## 7. Where things go (mapping from common profiling-project layouts to this repository)
Profiling guides usually assume one project directory with `venv/ models/ scripts/ profiles/ traces/ analysis/`.
Here the write partition of `README.md` decides instead:

| usual place | here |
|---|---|
| `venv/` | an untracked virtual environment (one per model) outside git or under the ignored `.venv/`; how to rebuild it, with exact versions, goes in `env/` |
| `models/` (source checkouts, weights) | untracked checkout paths outside the repository; repo URL, commit, checkpoint name and sha256 go in `meta.json` |
| `scripts/` | `bench/` |
| `profiles/` (`.nsys-rep`, `.ncu-rep`) and `traces/` (Chrome JSON) | `results/<run>/raw/` — **committed**, compressed per §6; path + sha256 + bytes in `meta.json.raw_files` |
| temporary files, caches (Inductor, Triton), uncompressed originals | `scratch/<run>/` — ignored, never committed |
| `analysis/` data (CSV, JSON, inventories) | `results/<run>/` |
| `analysis/` prose (reports, root-cause write-ups) | `reports/` |

The repository's own `analysis/` directory belongs to the research side; do not write there.

## 8. Hardware constants: derive them on this device
Every hardware constant printed in a published profiling guide belongs to another GPU. A typical example is an
RTX 4070 Ti SUPER: 44.1 TFLOP/s bf16, 672 GB/s, 66 SMs, ridge ≈ 66 FLOP/B, 48 warps per SM, DRAM latency
≈ 400 ns. **None of these applies to the Orin.** Derive the Orin's own values and record them in
`results/<date>_E0_platform/hw_constants.json` (E0/E1, tasks T001/T005):
- SM count: `torch.cuda.get_device_properties(0).multi_processor_count`; compute capability and the
  architecture string used by `cuobjdump` (we expect 8.7 → `sm_87`; confirm from the device, do not assume);
- peak memory bandwidth and peak bf16 / fp16 / fp32 / int8 throughput: measured in E1;
- ridge point = measured peak FLOP/s ÷ measured peak bytes/s;
- maximum warps per SM and any other per-SM limit ncu reports for this GPU.
Our expectation, **to be verified, not assumed**: 204.8 GB/s (LPDDR5-6400, 256-bit), ≈ 42 TFLOP/s bf16, ridge
≈ 208 FLOP/B. Report every comparison against both the expected and the measured peak. The per-instruction
cycle costs in `docs/03_instruction_level_analysis.md` also come from other architectures and are labelled so.

**Where a guide, a tool default or a habit differs from this protocol, this protocol wins**: ≥ 20 warm-ups and
≥ 100 timed iterations, CUDA-event timing, the §2 regions, the §5 result schema.

## 9. Physical floor — compute it before profiling
Before profiling a model, compute for each phase (`vision`, `vlm_prefix`, one `ae_step`) the time below which no
software can go, and write it to `results/<run>/floor.csv`
(`model, region, bytes_read, flops, peak_GBps, peak_TFLOPps, floor_mem_ms, floor_compute_ms, floor_ms, bound, peak_source`):
```
floor_mem_ms     = bytes that must be read at least once ÷ peak memory bandwidth
floor_compute_ms = FLOPs ÷ peak compute throughput
floor_ms         = max(floor_mem_ms, floor_compute_ms);  bound = whichever term is larger
```
- bytes = Σ `p.numel() × p.element_size()` over the parameters the phase touches (count them from the loaded
  model, per region), plus the KV cache read by an `ae_step` (2 × layers × KV heads × head dim × prefix tokens ×
  bytes per element), plus activations if they are not negligible;
- FLOPs = 2 × (tokens) × (parameters in linear layers touched) + attention FLOPs, from the real shapes (S, P);
- two rows per region: `peak_source = spec` (204.8 GB/s, 42 TFLOP/s) and `peak_source = measured` (E1).
Cross-check for π0: 623 MB of expert weights (+ KV ≈ 629 MB) ÷ 204.8 GB/s = 3.15 ms per step; 38.7 GFLOPs ÷
42 TFLOP/s = 0.92 ms; so the step is memory-bound with a 3.15 ms floor. Every report then states
`gap_ratio = measured ÷ floor` per region and decomposes the gap (§10). Reading: < 1.5× near-optimal, 1.5–3×
moderate, 3–10× substantial software overhead, > 10× a structural problem (launch-bound, idle GPU, too little
parallelism).

## 10. Root-cause section of every E5 / E5b report: ask "why" three times
Profiler metrics are symptoms. For the largest contributor to the gap, write a chain of at least three
"why" steps, each backed by a number and its source, ending at a fact that cannot be decomposed further (a
shape, a data structure, a physical limit):

| step | observation | evidence (file, metric, value) |
|---|---|---|
| symptom | e.g. region-weighted SM occupancy is x % in `ae_step` | `ncu/region_utilisation.csv` |
| why 1 | e.g. the GPU is idle y % of the step; kernels average z µs | `nsys/…`, `kernels_per_region.csv` |
| why 2 | e.g. inside kernels, `Stall Long Scoreboard` is w % and LSU-load dominates the pipes | `ncu/kernels.csv` |
| why 3 | e.g. the inner loop of the GEMM loads W rows per token; S = 51 gives too few warps per SM | `sass/…`, launch grid |
| root cause | one sentence | |

Then a gap table: `factor, evidence, contribution_ms, share_of_gap`, shares summing to 100 % ± 10. The examples
above are illustrations of the *form*, not expected answers: if the evidence points elsewhere (compute-bound,
launch-bound, CPU-bound), follow it. Do not write "memory-bound" or "launch-bound" without the chain.

## 11. Review before marking a task done
- **Every task — self-review.** Re-read the task's acceptance criteria and check, item by item: every
  requested file exists under the requested name; every number in the report traces to a file in `results/`;
  `meta.json` is complete (§5) and `raw_files` hashes are filled; n, power mode and `throttled` are stated;
  measured / derived / not-measured are separated; deviations are listed; **every raw file the run produced
  is committed under `results/<run>/raw/` and listed in `raw_files`** (or listed with `committed: false` and a
  reason); the §6g checks pass (no file over 50 MB, no weights / datasets / environments / secrets staged,
  secrets scan clean); sizes reported in `STATUS.md` (§6f); nothing on the research side was edited. Append the filled checklist to the report under
  "Self-review".
- **P0 tasks (T001–T003) — independent pass, if your framework supports sub-agents.** Give a fresh sub-agent
  only the task file, this protocol and the produced files; ask it to list unmet acceptance criteria and numbers
  that cannot be traced. Fix, then mark done. If sub-agents are not available, do the self-review twice, the
  second time starting from the result files rather than from the report. No heavier review cycle is wanted.

## 12. Profiling checklist and known pitfalls
Checklist per model: (1) one clean inference without any profiler; (2) parameters, bytes and FLOPs per region;
(3) `floor.csv`; (4) NVTX + CUDA-event timing (E3/E4); (5) `torch.profiler` top-30; (6) `nsys` + CSV export
(E5); (7) `ncu` on one region at a time + both page exports (E5); (8) instruction-level analysis of the dominant
kernels (E5b); (9) roofline points (E6); (10) root-cause chain (§10); (11) self-review (§11).

| symptom | cause / fix |
|---|---|
| `ncu` runs "forever", report is 1 GB+ | unfiltered `--set full`; always filter (§3.2), reduce `-c`, narrow the NVTX range |
| `git push` rejected: "exceeds GitHub's file size limit of 100.00 MB" | a raw file was committed unsplit. The commit is local, so undo it (`git reset --soft HEAD~1` — allowed because it was never pushed), then shorten / compress / split per §6 and commit again |
| ncu occupancy shows 0 % | collected with `--set roofline`; use `--set full` |
| `ERR_NVGPUCTRPERM` | `sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'` |
| nsys "option parsing failure" | use `-f true`, not `--force-overwrite`; drop `--cuda-memory-usage` if it is reported ambiguous |
| ncu CSV columns misaligned | kernel names contain commas: parse with `csv.DictReader` |
| `flash-attn` will not build | `pip install wheel` first, then `pip install flash-attn --no-build-isolation`; needs ~5 GB of temporary space and a torch whose CUDA version matches `nvcc`; on aarch64 expect a long source build — if it fails, run with PyTorch SDPA and record that |
| `CUDA error: no kernel image` | the torch wheel was not built for this GPU architecture; use the JetPack wheel |
| dependency conflicts between a model repo and the Jetson torch | `pip install --no-deps -e <repo>`, then add dependencies by hand; record in `env/` |
| GPU timing is 0 or negative, or first iteration is 2–5× slower | `time.time()` was used / no warm-up: §2 |
| profiled run slower than the timed run | expected; never mix profiling and timing runs |
| kernel names unreadable | `c++filt`; use NVTX ranges and `--nvtx-include` to tell phases apart |
| matplotlib fails headless | `matplotlib.use("Agg")` |

