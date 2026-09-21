# 02 — Measurement protocol (binding)

## 1. Device state — record for every run, fix before every run
- Power mode: `sudo nvpmodel -q --verbose` lists the modes and their ids. **Confirmed on this board by T001**
  (`results/20260921_E0_platform/nvpmodel_modes.csv`): 0 = `MAXN` (12 CPU cores, no TPC power-gated, GPU and EMC
  uncapped), 1 = `MODE_15W` (4 cores, 5 of 8 TPCs gated, GPU ≤ 408 MHz, EMC ≤ 2,133 MHz), 2 = `MODE_30W` (8 cores,
  4 of 8 TPCs gated, GPU ≤ 612 MHz), 3 = `MODE_50W` (12 cores, no TPC gated, GPU ≤ 816 MHz). The board booted in
  `MODE_30W`, and `nvpmodel -m 0` from there demands a reboot. Default for all experiments: **MAXN** followed by
  `sudo jetson_clocks`. **The human sets MAXN once (it persists across reboots); you never confirm a reboot prompt
  and never use `nvpmodel --force`.**
- **Gate at the start of every session and before every measured or profiled run:** `nvpmodel -q` reports `MAXN`,
  `torch.cuda.get_device_properties(0).multi_processor_count` equals the MAXN value recorded by T013 (16 is derived
  from `TPC_PG_MASK = 0`, not yet observed), `sudo jetson_clocks` has been applied **before** the run (it does not
  persist across reboots), and the GPU `devfreq` `cur_freq` is recorded next to the run (`bench/gpu_clock_probe.py`).
  If the mode is not MAXN, no latency, occupancy, bandwidth, FLOP/s or power task starts: write it at the top of
  `STATUS.md` and do only work that does not depend on the GPU configuration (environment set-up, checkpoint
  download and conversion, fixtures, harness code, analytic `floor.csv`, `segments.csv` derived from model code).
- **Power-mode sweep (E7).** A mode change that alters the set of power-gated TPCs needs a reboot, so the sweep
  cannot run inside one session. Rule: all MAXN work first. `MODE_50W` has the same TPC mask as MAXN: as the
  **last** item of the last MAXN session try `sudo nvpmodel -m 3`; if it asks for a reboot, decline and record
  that; if it switches, re-apply `jetson_clocks`, measure, then return with `sudo nvpmodel -m 0` (if *that* asks
  for a reboot, decline, write it at the top of `STATUS.md` and stop — the human restores MAXN). `MODE_30W` and
  `MODE_15W` are measured only in a session the human has started in that mode after a reboot, announced by a
  task; never request them yourself. Write "next: resume E7 at mode <id>" in `STATUS.md` before any switch.
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
small (§6). With nsys 2024.5.4 on this device (T001) `--capture-range=cudaProfilerApi` defaults to
`--capture-range-end=stop-shutdown`: the target is killed (exit 143) at `cudaProfilerStop` and everything the script
would print or write afterwards is lost — add **`--capture-range-end=stop`** whenever the script must run to
completion. Known flag problems: overwrite is
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
  `roofline`, so the utilisation work uses `full`. This Orin offers `basic`, `detailed`, `full`, `nvlink`,
  `pmsampling`, `roofline` (`env/tool_help/ncu_list_sets.txt`); `full` is the only set with Occupancy,
  WarpStateStats and SchedulerStats together.
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
- **Permissions — on this device `ncu` needs root (T001).** `perf_event_paranoid = 1` is not sufficient: every
  collecting run is `sudo $(which ncu) …` (plain `sudo ncu` fails, sudo's `secure_path` lacks the CUDA `bin`), and
  the root-owned `.ncu-rep` is `chown`ed back before the unprivileged `--import`. Consequences to handle and
  record: the profiled Python then runs **as root** — give it the venv interpreter by absolute path and pass the
  environment it needs explicitly (one workable pattern: `sudo env HOME=$HOME HF_HOME=… HF_HUB_OFFLINE=1
  TORCHINDUCTOR_CACHE_DIR=$PWD/scratch/<run>/inductor_root TRITON_CACHE_DIR=$PWD/scratch/<run>/triton_root $(which ncu) …`;
  never `sudo -E`, which would carry token variables, §3.5); keep root's compile caches in their own `scratch/`
  directories so that no root-owned file lands in your user caches or in the repository; `chown -R` what the run
  wrote before committing. Record the exact command in `meta.json`.
- **Clock control.** ncu's `--clock-control` defaults to `base`, which pins the SM clock (305.70 MHz in T001) below
  what the workload sees; the other value this version offers is `none` (`env/tool_help/ncu_help.txt`). ncu
  `Duration` is never a latency (§2 gives latency), but clock also changes the *ratios* we care about: at a lower
  SM clock a fixed memory latency costs fewer cycles, so memory stalls are under-stated. T013 runs the toy kernel
  with both values at MAXN + `jetson_clocks`. From then on use **`--clock-control none` with `jetson_clocks`
  applied** if T013 shows that ncu's `SM Frequency` then equals the locked GPU frequency and the metrics are
  populated; otherwise stay with `base` and state in every report which clock the ncu metrics were taken at. Always
  record ncu's `SM Frequency` and the `devfreq` `cur_freq` per run.
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
T001 checked every name below on this Orin (`results/20260921_E0_platform/ncu/metric_presence.csv`: 52 present, 14
present but zero on the toy kernel, 9 absent); the differences are marked **[Orin]** and are binding. A column that
is zero for one kernel is not evidence that the counter works: re-check `present_zero` metrics per kernel.

Keep, per kernel launch (`results/<run>/ncu/kernels.csv`, one row per launch, plus the kernel's NVTX region,
grid and block size):
- occupancy / activity: `SM Busy`, `SM Active Cycles`, `Achieved Occupancy`, `Theoretical Occupancy`,
  `Achieved Active Warps Per SM`, and from the raw page `sm__warps_active.avg.pct_of_peak_sustained_active`;
- throughput: `Duration`, `Compute (SM) Throughput`, `Memory Throughput`, `Mem Busy`, `L2 Hit Rate`,
  `L1/TEX Hit Rate`, `Executed Ipc Active`, `Executed Ipc Elapsed`.
  **[Orin] There are no DRAM counters**: `DRAM Throughput`, `DRAM Utilization` and every `dram__*` metric are absent
  (`--query-metrics` lists none; an explicit `--metrics dram__…` returns `n/a`; placeholder columns read 0). Never
  quote them. **[Orin] `Memory Throughput` is the L2 throughput** (it equals `lts__throughput…` on the raw page), not
  DRAM utilisation: write it as "Memory (L2) Throughput" in every table and never read it as off-chip bandwidth use;
- **[Orin] off-chip bytes (substitute for the DRAM counters; answer to question 1 of 2026-09-21).** Keep, per
  kernel, the L2 sysmem-miss sector counters of the raw page — `lts__t_sectors_*_sysmem_*_lookup_miss.sum`, read and
  write operations separately, and the all-operations total — and report `offchip_read_bytes = read-miss sectors ×
  32 B`, labelled ***measured (L2 sysmem-miss sectors)***. It is accepted for E5 / E5b / E6 under these conditions:
  (1) the **analytic** bytes (from tensor shapes) are always reported next to it and stay the primary x-axis input of
  the roofline; (2) it is a **read-traffic** estimator until E1 validates the write path (T005 step "substitute
  validation"); (3) achieved GB/s = these bytes ÷ the kernel's **nsys** duration of the same configuration, never
  ÷ ncu `Duration`; (4) the 32 B sector size and the validation ratio are cited from the run that established them;
- **per-pipe utilisation** (raw page): `sm__inst_executed_pipe_<p>.avg.pct_of_peak_sustained_active` for
  `<p>` ∈ {`lsu`, `fma`, `alu`, `xu`, `adu`, `cbu`, `fp16`} and `...pct_of_peak_sustained_elapsed` for
  {`lsu`, `fma`, `alu`, `xu`}. **[Orin]** the `fp16` pipe column does not exist — half-precision tensor work appears
  in the `sm__pipe_tensor_op_hmma_cycles_active.*` columns: keep those instead and name them as exported; the
  `_elapsed` variants exist only for `lsu` and `xu` (use `_active` for `fma`, `alu`). Pipes have different peak issue rates: the percentages do not sum to 100 and
  are not comparable across pipes as if they were instruction counts;
- **warp-stall reasons**. **[Orin] they are on the `raw` page only**, as
  `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` with `<reason>` ∈ {`barrier`,
  `branch_resolving`, `dispatch_stall`, `drain`, `imc_miss`, `lg_throttle`, `long_scoreboard`, `math_pipe_throttle`,
  `membar`, `mio_throttle`, `misc`, `no_instruction`, `not_selected`, `selected`, `short_scoreboard`, `sleeping`,
  `tex_throttle`, `wait`} — keep all 18 and check in the first model run whether they sum to ≈ 1 per kernel (say
  what they sum to; do not renormalise silently). The `details` page has no `Stall …` rows here. For reference, the
  desktop names are: every `Stall …` metric exported — `Stall Long Scoreboard`, `Stall Short Scoreboard`,
  `Stall Wait`, `Stall Barrier`, `Stall Not Selected`, `Stall Memory Dependency`, `Stall Memory Throttle`,
  `Stall Math Pipe Throttle`, `Stall Misc`, and any others present (some versions name them
  `long_scoreboard`, `short_scoreboard`, `mio_throttle`, `lg_throttle`, `barrier`, `not_selected`, `wait`,
  `dispatch_stall`, `drain`, `selected`, `tex_throttle`, `sleeping`);
- LSU request counters for the load/store split (raw page): `l1tex__t_requests_pipe_lsu_mem_global_op_{ld,st,red,atom}.sum`,
  `l1tex__t_requests_pipe_lsu_mem_local_op_{ld,st}.sum`, `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st,atom}.sum`;
- for E5b: `gpc__cycles_active.avg.per_warp` (**[Orin]** absent: derive it as `gpc__cycles_active.avg` ÷
  `sm__warps_launched.sum` if both are exported, label it *derived*, else *not measurable*), and the launch metrics `launch__grid_size`, `launch__block_size`,
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
the reverse, else balanced; report the share of region kernel time in each class. **[Orin]** because `Memory
Throughput` is L2 throughput here, call the classes "L2-throughput-dominated / SM-throughput-dominated / balanced"
and give, next to them, the classification that does not depend on that counter: analytic intensity (FLOPs ÷
analytic bytes) against the Orin ridge point (§8). Where the two disagree, say so; neither is tuned to agree.

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
- **On this carrier board `tegrastats` prints only three of the four rails (T001): `VDDQ_VDD2_1V8AO`, the memory
  rail, exists only in sysfs** (second INA3221, `/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon*/in2_*`; find the
  channel by its `in<i>_label`, the `hwmon` index can change across boots). Every power run therefore samples **all
  four rails from sysfs at ≥ 10 Hz** (`bench/common.py: ina3221_rails()`, tidy CSV `t_s, rail, mW, mV, mA`) and runs
  tegrastats alongside for temperatures, `GR3D_FREQ`, `EMC_FREQ` and as a cross-check of the three shared rails.
  `tegrastats --logfile` **appends**: delete the log before each run.
- `sudo tegrastats --interval 100 --logfile results/<run>/raw/tegrastats.log` (10 Hz). Rails to keep: `VDD_GPU_SOC`, `VDD_CPU_CV`, `VIN_SYS_5V0`, `VDDQ_VDD2_1V8AO`
  (the memory rail — we care about it specifically). Convert to a tidy CSV (`t_s, rail, mW`). Always report each
  rail separately **and** their sum as "module power" (`VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0 + VDDQ_VDD2_1V8AO`;
  check the rail list of this carrier board and say if it differs). tegrastats `GR3D_FREQ` % is logged in the
  same file as a **coarse cross-check only**: it says whether the GPU had work queued, not how much of the GPU
  the work used. The utilisation metric of this project is ncu's SM occupancy (§3.3). Both the raw
  tegrastats log (`raw/tegrastats.log`, compressed per §6 if large) and the tidy CSV are committed.
- **What the memory-rail reading is (answer to question 2 of 2026-09-21; no calibration step is needed).** Each
  INA3221 channel reports the bus voltage *at its shunt*, which sits at the **input** of the regulator(s) of that
  rail group, not at the load: `VDD_GPU_SOC` reads ≈ 19.9 V and `VDDQ_VDD2_1V8AO` ≈ 5.04 V in T001 for that reason.
  `mW = mV × mA` is therefore a real power: the input power of the regulators that make LPDDR5 VDDQ and VDD2 **and**
  the 1.8 V always-on rail `[ASSUMPTION: read from the rail's name and the bus voltage, not from the carrier
  schematic]`. Do not rescale it to 1.8 V. Report it under the name `VDDQ_VDD2_1V8AO (regulator input)`. It is an
  **upper bound** on DRAM power (it includes regulator loss and non-memory 1.8 V loads), so the quantity to quote
  for memory activity is the **above-idle delta** of this rail per phase, next to its absolute value; never call
  it "DRAM power" without those words.
- **Is the module sum double-counting? Check once, in E1 (T005).** The memory regulators' 5 V input may or may not
  be downstream of the `VIN_SYS_5V0` shunt. Log the four rails at idle, during the E1 memory-copy loop and during
  the E1 large-GEMM loop, and report ΔVIN_SYS_5V0 and ΔVDDQ_VDD2_1V8AO (above idle) for each: if `VIN_SYS_5V0` rises
  by about the memory-rail delta or more whenever the memory rail rises, say that nesting is likely. Until then
  "module power" stays the four-rail sum (NVIDIA's convention for this module, `[ASSUMPTION]`), and every table
  also gives the three-rail sum without the memory rail, so either reading can be recomputed.
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
  when applicable; for E11 runs `model_inputs.csv` (schema in §13.8, required in every E11 result directory) and
  `segments.csv`, `handover.csv`, `boundary.csv` (§13.2, §13.4). For profiler runs add to `meta.json`: `nsys_version`, `ncu_version`, `ncu_set`, `ncu_filter`
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
- SM count: `torch.cuda.get_device_properties(0).multi_processor_count` — **it depends on the power mode** (8 in
  `MODE_30W`, where 4 of 8 TPCs are power-gated; T001): record it per mode, as T001's `hw_constants.json` does, and
  take the MAXN value from an observation at MAXN (T013), not from the mask; compute capability and the
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
| `ERR_NVGPUCTRPERM`, or ncu: "Insufficient privileges to launch app for profiling" | on this device ncu needs root: `sudo $(which ncu) …`, then `chown` the report (§3.2); `perf_event_paranoid=1` alone is not enough |
| `sudo ncu`: command not found | sudo's `secure_path`; use `sudo $(which ncu)` |
| script output missing after an nsys run, exit code 143 | nsys killed the target at `cudaProfilerStop`; add `--capture-range-end=stop` (§3.1) |
| tegrastats log has two sessions in it | `--logfile` appends; delete the file first (§4) |
| memory rail missing from tegrastats | expected on this board; sample the four rails from sysfs (§4) |
| `torch.cuda.is_available()` false after a `pip install` in a model environment | pip replaced the Jetson CUDA wheel with PyPI's CPU-only `torch…+cpu` (it prefers a manylinux wheel over the index's `linux_aarch64` one). Install the three CUDA wheels by full URL with `--no-deps` (`env/setup.sh`), install model repositories with `--no-deps`, and **re-check `torch.__version__`, `torch.version.cuda` and `is_available()` after every install step** |
| Inductor logs "Not enough SMs to use max_autotune_gemm mode" | Inductor's `is_big_gpu` check (`torch/_inductor/utils.py`) compares `multi_processor_count` with a fixed minimum (68 for CUDA devices in the PyTorch sources the research side can read; read the value in the installed file and record it). The Orin has at most 16, so **expect this at MAXN too**: `max-autotune` then runs without Triton GEMM templates (ATen / cuBLAS GEMM, other autotuning and CUDA graphs still apply). It is a property of the stack: report it; see T003 for the labelled override variant |
| SM count is 8, GPU stops at 612 MHz | the board is in `MODE_30W`; §1 gate — no measurement until the human has set MAXN |
| nsys "option parsing failure" | use `-f true`, not `--force-overwrite`; drop `--cuda-memory-usage` if it is reported ambiguous |
| ncu CSV columns misaligned | kernel names contain commas: parse with `csv.DictReader` |
| `flash-attn` will not build | `pip install wheel` first, then `pip install flash-attn --no-build-isolation`; needs ~5 GB of temporary space and a torch whose CUDA version matches `nvcc`; on aarch64 expect a long source build — if it fails, run with PyTorch SDPA and record that |
| `CUDA error: no kernel image` | the torch wheel was not built for this GPU architecture; use the JetPack wheel |
| dependency conflicts between a model repo and the Jetson torch | `pip install --no-deps -e <repo>`, then add dependencies by hand; record in `env/` |
| GPU timing is 0 or negative, or first iteration is 2–5× slower | `time.time()` was used / no warm-up: §2 |
| profiled run slower than the timed run | expected; never mix profiling and timing runs |
| kernel names unreadable | `c++filt`; use NVTX ranges and `--nvtx-include` to tell phases apart |
| matplotlib fails headless | `matplotlib.use("Agg")` |


## 13. E11 — inputs for the hybrid GPU + PIM model: methods and the `model_inputs.csv` schema
E11 (`01_measurement_plan.md`) measures what a system model needs when the vision encoder and the VLM prefix stay
on the GPU and the action expert's large matrix products run in memory. Everything in §1–§6 applies (device
state, warm-up, raw files, sizes). The rules below are specific to E11. If an API named here does not exist in
your PyTorch build, use the equivalent it offers and record the substitution in `meta.json.notes`.

### 13.1 Micro-timing of small GPU work (host segments, casts, KV materialisation)
A host segment takes tens of microseconds to a few milliseconds, and in the target system it runs **alone**: the
GPU is idle before it, the CPU launches it, and the CPU waits for the result before the memory side can go on.
Therefore every small piece of work is timed in two ways, and both are reported (`timing` column, §13.8):
- `wall_isolated` — `torch.cuda.synchronize()`; `t0 = time.perf_counter()`; run the work once;
  `torch.cuda.synchronize()`; `t1`. This is launch + execution + completion wait from an idle GPU. It is the
  **primary** number. Repeat it with a pause between calls, `gap_ms` ∈ {0, 2} (`time.sleep(0.002)` before the
  first `synchronize()`), because the memory side works for about a millisecond between two segments and an
  idle GPU may answer more slowly; label the rows `variant = gap_ms=<g>`.
- `gpu_event` — CUDA events around the work with calls issued back to back (as in E2): pure device time.
- additionally `wall_launch` — `perf_counter` around the call **without** the trailing synchronise: the CPU cost of
  issuing the work.
≥ 20 warm-up calls, then **≥ 1,000 timed calls** per (segment, style, timing); keep every sample
(`raw/segment_samples.csv.zst`: `segment, exec_style, timing, variant, iter, ms`).

Three execution styles (`exec_style`):
- `eager` — the model's own PyTorch code, called from Python.
- `compiled` — `torch.compile(fn, dynamic=False)` of the segment function (default mode, no CUDA graphs, so that
  fusion and the kernel count stay visible); log the compile time; if `torch.compile` fails for a segment,
  record the error text and mark the row `not measured`.
- `cudagraph` — capture with `torch.cuda.CUDAGraph()` / `with torch.cuda.graph(g):` on static input tensors and
  time `g.replay()`. Two graph granularities: **one graph per segment** (how the target system would really run
  it, because memory-side work separates the segments), and **one graph of the whole per-layer host sequence**
  (all segments of one layer in order, outputs of one feeding the next where the model does so; a lower bound).
  Say in `variant` whether the captured body was the eager or the compiled function.

Kernels per call: one `torch.profiler` run (CPU + CUDA activities) over 10 calls per (segment, style);
kernels per call = CUDA kernel events ÷ 10; commit the `key_averages` table and the trace under `raw/`. Also one
small `nsys` capture (§3.1) of ten consecutive per-layer host sequences per style.

Inputs are **real**: register forward hooks (or wrap the functions) during one real denoising step of the
fixture observation, save the input tensors of each segment at layer 0 and at a middle layer to `scratch/`
(not committed), and replay the segment on them. Record shape, dtype, device and contiguity of every input and
output in `segments.csv` (§13.2). Use the model's own module instances and functions; re-implement only what the
model code does inline, line for line, and cite the file and line.

### 13.2 What a host segment is, and `segments.csv`
A **host segment** is a maximal run of consecutive operations of one action-expert layer that are **neither a
large linear layer** (a matrix product with a weight matrix applied to the S suffix tokens: q / k / v / o, MLP
gate / up / down, or a fused form of them) **nor an attention matrix product** (scores = q·Kᵀ, context =
probs·V). Large linear layers and attention products are what moves into memory; every segment between two of
them stays on the GPU. Small linears whose input is a single conditioning vector (for example the dense layer of
an adaptive norm) count as part of the host segment they belong to. Derive the list **from the model code**, per
model, and write it to `segments.csv`:
`model, segment, order_in_layer, layer_kind, source_ref, ops, inputs, outputs, bytes_read, bytes_written, flops,
flops_formula, in_model_code, notes` — `source_ref` = `file:line-line` at the model commit; `inputs` / `outputs` =
`name:shape:dtype` lists; `layer_kind` distinguishes layer types where a model has several (self-attention /
cross-attention, DiT block variants); `in_model_code` = `yes`, or `mapping_only` for a segment that exists only in
the memory mapping (below). Bytes and FLOPs are analytic, from the shapes, with the formula written out.
Per-step extras (everything in one denoising step outside the layer stack that is not a large linear layer on
the suffix tokens: suffix / time embedding, final norm, output projection, Euler update) are listed in the same
file with `order_in_layer = -1`.

### 13.3 Transfers and synchronisation on unified memory
The Orin's CPU and GPU share one DRAM, but software still "copies". Measure the latency of moving a buffer of
each size in {4 KB, 16 KB, 64 KB, 256 KB, 1 MB, 2 MB, 4 MB, 16 MB} (KB = 1024 B) **plus the exact boundary sizes
of the model** (E11 part 3), in both directions where the method has a direction, with three methods:
- `pageable` — `gpu.copy_(cpu)` / `cpu.copy_(gpu)` with an ordinary CPU tensor;
- `pinned` — the CPU tensor allocated with `pin_memory=True`, `copy_(…, non_blocking=True)` followed by
  `torch.cuda.synchronize()`;
- `zero_copy` — **one allocation that both the CPU and the GPU address, with no copy**: CUDA managed memory
  (`cudaMallocManaged`) or mapped pinned host memory (`cudaHostAlloc` with the mapped flag +
  `cudaHostGetDevicePointer`). PyTorch has no single documented call for this; candidate routes are a few lines
  of C++/CUDA extension, `ctypes` calls into `libcudart`, a pluggable allocator, or CuPy managed memory shared
  with torch through DLPack. Use whichever works on this stack and record exactly what you used. Time each
  part: CPU writes the buffer (`numpy` copy into it), a GPU kernel reads it (the fp32→fp16 cast of §13.1 or a
  `sum()`), synchronise, CPU reads the result — and the same GPU kernel on an ordinary device tensor, so that the
  penalty of the shared allocation is visible. If no route works, document what failed and mark it `not measured`.
Timing: `wall_isolated` (§13.1), ≥ 20 warm-ups, ≥ 200 timed per point (≥ 1,000 for sizes ≤ 64 KB), every sample
kept (`raw/xfer_samples.csv.zst`). Per (method, direction) fit **t = a + b·bytes** by least squares on the
per-size medians: report `fit_a` (ms), `fit_b` (ms per MB, MB = 10⁶ B), `fit_r2`, and 1/b as GB/s. The fixed term
`a` is the point of this experiment; a peak GB/s alone is not enough.
Synchronisation costs, `wall_isolated`, ≥ 1,000 samples each: `synchronize_idle` (`torch.cuda.synchronize()` with
nothing queued), `event_synchronize_done` (`Event.synchronize()` on an event that already completed),
`event_query` (`Event.query()`), `tiny_kernel_roundtrip` (one 1-element add + synchronise), and `stream_handoff`
(work on stream A, `event.record()`, stream B `wait_event`, a tiny kernel on B, synchronise — minus the same
two kernels on one stream). Record whether the CPU **spins** while it waits (process CPU time ÷ wall time during
a long synchronise): a spinning wait shows up as 100 % CPU utilisation in §13.6 and must not be read as work.

### 13.4 The hand-over object (prefix → action expert)
After the prefix ran on the fixture observation, inspect the object the action expert receives (a KV cache for
π0 / π0.5 / SmolVLA; whatever the model passes for GR00T — say what it is). Write `handover.csv`:
`model, layer, tensor, shape, dtype, stride, contiguous, bytes, valid_tokens, padded_tokens, notes`, one row per
tensor per layer, plus a `TOTAL` row. Distinguish **valid** prefix tokens from padded / masked positions that
still occupy the cache. Time (`gpu_event` and `wall_isolated`, §13.1) the materialisation of the whole object as
one contiguous buffer `[layers, 2, P, kv_heads, head_dim]` in the cache's dtype (`variant = layout=stacked`), and the
same with V transposed to `[…, head_dim, P]` (`variant = layout=v_transposed`; the memory mapping stores V by
dimension). Establish, from the code **and** at run time, how the denoising steps use the object, and write the
answer with evidence (file:line, `data_ptr()` of the cache tensors across steps, `torch.cat` / memcpy kernels in the
nsys trace of `ae_step`) to `kv_reuse.md`: is the prefix KV computed once per inference; does each step reference
it or copy it (e.g. a concatenation with the suffix K/V that re-copies the prefix); is the suffix K/V appended
and then dropped; bytes copied per step because of this. Also list the tensors that cross the prefix → expert
boundary **per denoising step** (noisy actions, state, time, the step's output) in `boundary.csv`:
`model, tensor, direction, shape, dtype, bytes, per, notes` with `per` ∈ {`inference`, `step`, `layer`}.

### 13.5 Background memory load (a proxy — read the limits)
`bench/bw_hog.py`, a **separate process**, two kinds of load, each with a duty cycle d ∈ {25, 50, 75, 100} %
(period 20 ms: copy for d·20 ms, sleep for the rest):
- `gpu_hog` — `dst.copy_(src)` between two 256 MiB device tensors;
- `cpu_hog` — a multi-threaded CPU copy (e.g. `numpy.copyto` on 256 MiB arrays in k worker processes, or the
  STREAM benchmark built with OpenMP — whichever you can run; record it), pinned with `taskset` to cores the
  foreground process does not use (pin the foreground too, and record both core sets).
Two **controls** at d = 100 %: `gpu_spin_control` (a compute-bound GPU load with a small memory footprint — small
GEMMs — at the same duty cycle) and `cpu_spin_control` (the same worker processes on the same cores doing
arithmetic that stays in cache). They separate "the GPU / CPU was taken" from "the memory bandwidth was taken".
The hog prints its achieved copy rate once per second (same GB/s definition as E1's `clone()` row; say which) to
`raw/bw_hog_<label>.log`; report the achieved rate alone and during the foreground run, and as a percentage of the
E1-measured peak (of the 204.8 GB/s spec, marked, if E1 has not run). Log tegrastats during every run (§4); if
this tegrastats prints a memory-controller utilisation next to `EMC_FREQ`, keep it.
**Limits, to be repeated in the report:** this is a proxy for memory-side activity on the same DRAM. A real PIM
occupies DRAM channels through the command bus and blocks ordinary accesses to them; a software hog instead
competes at the memory controller and in the caches, and additionally takes GPU time slices (`gpu_hog`) or CPU
cores (`cpu_hog`). The controls bound those side effects; the result is the sensitivity of the GPU work to
lost memory bandwidth, not a measurement of PIM interference.

### 13.6 CPU utilisation per phase
The CPU that launches GPU kernels would also drive the memory-side command stream. In a **separate run** (not
the latency run): (a) per region, the launching thread's CPU time (`time.thread_time()`) and the process CPU time
(`time.process_time()`) read at the region boundaries, summed over ≥ 100 iterations, ÷ region wall time;
(b) system-wide per-core utilisation from `/proc/stat` deltas over phase-only loops of ≥ 30 s (`vision`-only if
separable, prefix-only, action-expert-only, full inference). Commit the raw deltas (`raw/cpu_util_samples.csv`).
State the number of cores online and the spin-wait finding of §13.3.

### 13.7 Concurrency (pipelining)
Throughput = completed inferences ÷ wall time of a back-to-back loop ≥ 60 s. For concurrent cases say how the two
activities share the GPU: **two processes**, or **one process with two CUDA streams** (`torch.cuda.Stream`; try a
high-priority stream for the short work and record whether the priority argument is accepted). Report the
latency distribution (median, p95, max) of the short activity alone and under concurrency, not only the mean.

### 13.8 `model_inputs.csv` — the machine-readable hand-off (every E11 result directory has one)
Long format, one row per (quantity, key, variant, statistic), UTF-8, comma-separated, header exactly:
```
model,model_commit,bench_commit,quantity,key,variant,value,unit,stat,n,exec_style,timing,power_mode,precision,P,S,N,source_file
```
| column | content |
|---|---|
| `model` | `pi0`, `pi05`, `smolvla`, `groot_n17`, or `none` for model-independent rows (transfers, synchronisation) |
| `model_commit`, `bench_commit` | model repository commit; commit of `bench/` that produced the number |
| `quantity` | one name from the table below — nothing else; a quantity that does not fit is proposed in `STATUS.md` and reported in `summary.md` meanwhile |
| `key` | the sub-item (segment name, method, phase, rail, load label); empty if the quantity has none |
| `variant` | `name=value` pairs separated by `;` for everything else that distinguishes the row (`gap_ms=2`, `layer=0`, `size_bytes=65536`, `n_images=2`, `prompt_tokens=17`, `compile_mode=max-autotune`, `graph=per_segment`, `sharing=two_processes`, `from=<run dir>`); empty if none. Same spelling everywhere |
| `value`, `unit` | a number (no thousands separators) and its unit: `ms`, `bytes`, `flops`, `count`, `ratio`, `pct`, `W`, `GBps` (10⁹ B/s), `TFLOPps`, `inf_per_s`, `ms_per_MB`, `ms_per_token`, `ms_per_image` |
| `stat` | `median`, `mean`, `p5`, `p95`, `min`, `max`, `value` (an exact or single number), `fit_a`, `fit_b`, `fit_r2` |
| `n` | samples behind the statistic (points for a fit; 1 for `value`) |
| `exec_style` | `eager`, `compiled`, `cudagraph`, or empty |
| `timing` | `wall_isolated`, `gpu_event`, `wall_launch`, `wall_loop` (a back-to-back loop), `slope` (§2), `analytic`, or empty |
| `power_mode` | the nvpmodel mode name and `+jetson_clocks` if set, e.g. `MAXN+jetson_clocks` |
| `precision` | `bf16`, `fp16`, `fp32` (weights / activations of the measured work) |
| `P`, `S`, `N` | prefix tokens, suffix tokens, denoising steps of the run; empty when not applicable |
| `source_file` | path **relative to this result directory** of the file the value was computed from; it must exist in this directory. To restate a number from another run, copy that run's tidy CSV (never its profiler reports) to `raw/restated/<run dir>/` and point there, with `from=<run dir>` in `variant` |

Report at least `median`, `p5`, `p95` for every timed quantity. Quantities (`key` values in parentheses):

| quantity | unit | meaning |
|---|---|---|
| `t_preprocess`, `t_vision`, `t_vlm_prefix`, `t_postprocess`, `t_ae_step`, `t_ae_total`, `t_e2e` | ms | §2 regions; `t_e2e` = wall clock of one inference |
| `t_host_segment` (segment name of `segments.csv`; also `cast_scores_fp32_fp16`) | ms | one host segment, §13.1 |
| `n_kernels_host_segment`, `bytes_host_segment`, `flops_host_segment` (segment) | count, bytes, flops | kernels per call; analytic bytes read + written; analytic FLOPs |
| `t_host_layer` (`sum_of_segments`, `sequence`, `whole_layer_graph`; layer kind in `variant`) | ms | all host segments of one layer: sum of the isolated segments; the sequence issued back to back with one final synchronise; one CUDA-graph replay |
| `t_step_extra` (`suffix_embed`, `final_norm_out_proj`, `euler_update`, other names from `segments.csv`, `total`) | ms | per-step work outside the layer stack |
| `t_launch_per_kernel` | ms | `wall_launch` of a segment ÷ its kernels per call, eager; median over segments, and per segment with the segment in `key` |
| `t_sync` (`synchronize_idle`, `event_synchronize_done`, `event_query`, `tiny_kernel_roundtrip`, `stream_handoff`) | ms | §13.3 |
| `xfer_latency` (`<method>:<direction>`, e.g. `pinned:h2d`, `pageable:d2h`, `zero_copy:roundtrip`; `size_bytes` in `variant`) | ms | one transfer of that size |
| `xfer_latency_fit` (`<method>:<direction>`) | ms (`fit_a`), ms_per_MB (`fit_b`), ratio (`fit_r2`) | t = a + b·bytes |
| `kv_bytes_per_layer`, `kv_bytes_total` (`valid` or `padded`) | bytes | the hand-over object, §13.4 |
| `t_kv_materialise` (`stacked`, `v_transposed`) | ms | §13.4 |
| `kv_copied_bytes_per_step` | bytes | prefix KV bytes re-copied inside one denoising step (0 if referenced) |
| `boundary_bytes_per_step` (tensor name from `boundary.csv`, and `total`) | bytes | tensors crossing the prefix → expert boundary per denoising step |
| `prefix_latency_fit` (`vision_vs_images`, `vlm_prefix_vs_P`, `ae_step_vs_P`) | ms (`fit_a`), ms_per_image or ms_per_token (`fit_b`), ratio (`fit_r2`) | E11 part 4 |
| `prefix_throughput`, `e2e_throughput`, `ae_throughput` | inf_per_s | back-to-back loops: prefix only, full inference, action expert only |
| `prefix_slowdown` (load label: `gpu_hog_25` … `gpu_hog_100`, `cpu_hog_25` … `cpu_hog_100`, `gpu_spin_control`, `cpu_spin_control`, `host_segments`) | ratio | prefix-only median latency under that load ÷ alone |
| `host_layer_slowdown` (load label, incl. `prefix_loop`) | ratio | `t_host_layer` under that load ÷ alone |
| `bg_bw_GBps` (load label; `alone` or `during` in `variant`) | GBps | achieved rate of the background load |
| `cpu_util` (phase: `vision`, `vlm_prefix`, `ae_total`, `e2e`; `scope=thread`, `scope=process`, `scope=core<i>` or `scope=all_cores_mean` in `variant`) | pct | §13.6 |
| `p_mean_W` (`<phase>/<rail>`, rail names of §4 and `module`) | W | mean power of a phase loop |
| `gpu_idle_ae_total`, `gpu_idle_frac_ae_total` | ms, ratio | `ae_total` region time − union of kernel intervals (nsys, §3.1): GPU time free for the next frame's prefix in the ordinary run |
| `achieved_bw_GBps`, `achieved_tflops` (segment, region, or an E1 kernel name) | GBps, TFLOPps | analytic bytes or FLOPs ÷ `gpu_event` time |

Check the file with `python docs/reference_scripts/check_model_inputs.py results/<run>/model_inputs.csv` (standard
library only, runs anywhere) before committing; it must exit 0. It verifies the header, the enumerations, that
`value` parses, and that every `source_file` exists.
