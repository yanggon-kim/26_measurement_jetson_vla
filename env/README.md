# env — how this Jetson is set up

Produced by **T001 (E0)**, 2026-09-21. Rebuild with `bash env/setup.sh`.
Everything below was read off this device; nothing is copied from a spec sheet or a guide.

---

## 1. Platform as found

| item | value | where from |
|---|---|---|
| board | NVIDIA Jetson AGX Orin Developer Kit (`tegra234`) | `/proc/device-tree/model` |
| L4T | **R36.5.0** (GCID 43688277, 2026-01-16) | `/etc/nv_tegra_release` |
| JetPack | **6.2.3+b81** (`nvidia-jetpack`) | `dpkg -l` |
| CUDA toolkit | **12.6.68** (`cuda-toolkit-12-6` 12.6.11-1) | `nvcc --version` |
| cuDNN | 9.3.0.75 | `dpkg -l` |
| TensorRT | 10.3.0.30 | `dpkg -l` |
| kernel | 5.15.185-tegra (`oot` variant) | `uname -r` |
| system Python | 3.10.12 | `python3 --version` |
| RAM | 61 GiB (62 828 MB), 8 × 3.8 G zram swap, no disk swap | `free -h`, `swapon --show` |
| NVMe | 1.8 T total, 1.7 T free at bring-up | `df -h /` |
| EMC clock | **3 199 000 000 Hz** = 3 199 MHz — LPDDR5-6400, as expected | `/sys/kernel/debug/bpmp/debug/clk/emc/rate` |
| GPU | Orin, **compute capability 8.7 → `sm_87`** | torch / `cuobjdump` |
| `nvidia-smi` | present and works on this JetPack | — |

**JetPack 6.x prerequisite: satisfied.** JetPack 6.2.3 / L4T r36.5, CUDA 12.6, Python 3.10.
No reflash needed. `torch.compile` works (§4).

### Prerequisite that was missing
**Passwordless `sudo` was not configured** (`sudo -n` fails; the user is in group `sudo`
but a password is required). The human supplied the password for this session and it is
used through a `SUDO_ASKPASS` helper kept outside the repository. This is *not* persistent:
a future session needs either passwordless sudo or the password again. See `STATUS.md`.

---

## 2. Power mode — now at MAXN (after a human-authorised reboot)

| | |
|---|---|
| mode at bring-up | `MODE_30W` (id 2) — the board's configured default (`PM_CONFIG DEFAULT=2`) |
| **mode now** | **MAXN (id 0)**, persisted (`/var/lib/nvpmodel/status` = `pmode:0000`) |
| how | switching to MAXN **requires a reboot** on this board: `nvpmodel -m 0` prints `NVPM WARN: Reboot required for changing to this power mode: 0` and prompts. Per `AGENTS.md` the agent declined it; the human ran the command and confirmed the reboot on 2026-09-21. |
| note | declining that prompt does **not** stage the change — `nvpmodel` reports `optMask is 1, no request for power mode` and `/var/lib/nvpmodel/status` is left untouched. Answer `YES`, or use `--force`. |

The modes of this board (`/etc/nvpmodel.conf`, exported to
`results/20260921_E0_platform/nvpmodel_modes.csv`) — the protocol's expected id mapping is
**confirmed**:

| id | name | CPU cores | `TPC_PG_MASK` | active GPU TPCs | GPU max | EMC max |
|---|---|---|---|---|---|---|
| 0 | MAXN | 12 | 0 | 8 (→ 16 SMs) | uncapped | uncapped |
| 1 | MODE_15W | 4 | 248 | 1 | 408 MHz | capped |
| 2 | MODE_30W | 8 | 240 | **4 (→ 8 SMs)** | 612 MHz | uncapped |
| 3 | MODE_50W | 8 | 0 | 8 (→ 16 SMs) | 816 MHz | uncapped |

**At MAXN** (`results/20260921_E0_platform_maxn/`): `TPC_PG_MASK` 0, **8 of 8 TPCs**, CUDA
reports **16 SMs**, GPU locked at **1 300.5 MHz** (confirmed under load), 12 CPU cores, EMC
3 199 MHz. Idle module power 11 012 mW, tj 41.9 °C.

**What MODE_30W cost, for reference** (`results/20260921_E0_platform/`): 4 of 8 TPCs,
**8 SMs**, GPU capped at 612 MHz, 8 CPU cores, idle module power 8 522 mW. The same toy
bf16 GEMM took **185.4 µs** there against **54.6 µs** at MAXN — 3.4×. **No latency,
bandwidth, occupancy or power number from the 30 W run belongs on the ladder of
`docs/00_context.md`**; that directory is a tool-and-stack verification record only.

`jetson_clocks` works in both modes and is now applied **before** every capture, with
`jetson_clocks --fan`. It does **not** survive a reboot.

---

## 3. Python environments

**One environment per model** (`docs/00_context.md`). This is the *base* environment, used
for E0/E1/E2 and all profiler work; model environments come with T002 onwards.

| | |
|---|---|
| path | `~/venvs/jetson_base` (outside the repository, untracked) |
| python | 3.10.12 (system) |
| exact pins | `env/requirements_base.txt` |

### Installed, and why

| what | version | why / note |
|---|---|---|
| apt `python3-pip` | 22.0.2 | pip was **not installed** on this image |
| apt `python3-setuptools`, `python3-wheel` | 59.6.0, 0.37.1 | pip dependencies |
| apt `zstd` | already present | raw-file compression (protocol §6b) |
| `torch` | **2.8.0** (JetPack wheel, CUDA 12.6, `sm_87`) | — |
| `torchvision` | 0.23.0 (JetPack wheel) | — |
| `triton` | 3.4.0 (JetPack wheel) | Inductor backend |
| `numpy` | 1.26.4 | `<2`: the JetPack torch wheel is built against the numpy 1.x ABI |

`python3-venv`, `python3-dev` and `libopenblas-dev` were already installed.

### The trap that cost a reinstall — read before touching the environment
The Jetson index serves `...-linux_aarch64.whl`. pip ranks a **manylinux** wheel of the same
version higher, so

```bash
pip install torch==2.8.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
                         --extra-index-url https://pypi.org/simple      # WRONG
```

silently installs PyPI's **`2.8.0+cpu`** (`torch.cuda.is_available() → False`).
Install the three CUDA wheels **by full URL with `--no-deps`**, as `env/setup.sh` does.
The URL form that works is `https://pypi.jetson-ai-lab.io/jp6/cu126/+f/<a>/<b>/<wheel>`;
the `../../+f/...` hrefs on the index pages do **not** resolve.

---

## 4. `torch.compile` — works, every mode

`bench/verify_stack.py` → `results/20260921_E0_platform/stack_verification.json`.
bf16 MLP (1024 → 4096 → 1024, GeLU), input `[51, 1024]`: 20 warm-up calls, then 100 timed
CUDA-event calls per mode (700 samples in all, kept in
`results/20260921_E0_platform/latency_samples_compile_probe.csv`), compared against eager:

| backend / mode | compiles | first call incl. compile | max rel. err vs eager |
|---|---|---|---|
| `inductor` (no mode) | yes | 2.83 s | 4.7e-3 |
| `inductor` `default` | yes | 0.09 s | 4.7e-3 |
| `inductor` `reduce-overhead` | yes | 0.68 s | 4.7e-3 |
| `inductor` `max-autotune` | yes | 2.34 s | 4.7e-3 |
| `inductor` `max-autotune-no-cudagraphs` | yes | 0.60 s | 4.7e-3 |
| `aot_eager` | yes | 0.12 s | 0 |
| `eager` | yes | 0.06 s | 0 |

The 4.7e-3 relative difference is Inductor's bf16 GeLU/`addmm` fusion, not a failure
(bf16 has ~3 decimal digits). A plain bf16 2048³ GEMM differs from an fp32 reference by
4.2e-3, the same order.

### ⚠ `max-autotune` never uses Triton GEMM templates on this GPU
With a cold Inductor cache the run logs **`torch/_inductor/utils.py:1446] [0/0] Not enough
SMs to use max_autotune_gemm mode`**. This was first read as an artefact of the 8-SM
`MODE_30W` state — **that was wrong**: it still fires at MAXN with 16 SMs
(`results/20260921_E0_platform_maxn/raw/verify_stack.stdout.log`, line 3). The cause is a
hard-coded threshold in `torch/_inductor/utils.py`:

```python
min_sms = 16 if device.type == "xpu" else 68  # 3080
avail_sms = prop.multi_processor_count
if avail_sms < min_sms:
    log.warning("Not enough SMs to use max_autotune_gemm mode", ...)
    return False
```

Orin has **16** SMs against a required **68**, in every power mode. So `max-autotune` falls
back to ATen/cuBLAS for GEMMs — the autotune log compares only `addmm` against
`bias_addmm`, both ATen. **E4 must read "max-autotune" on this device as "default + CUDA
graphs + ATen autotuning", not as template autotuning.** No Jetson Orin will ever clear
that threshold.

Also verified to run: bf16, fp16, fp32 matmul and int8 via `torch._int_mm`
(Orin has INT8 tensor cores and no FP8).

---

## 5. Profilers

| tool | path | version |
|---|---|---|
| `ncu` | `/usr/local/cuda/bin/ncu` | **2024.3.1.0** (build 34702747) |
| `nsys` | `/usr/local/bin/nsys` | **2024.5.4.34-245434855735v0** |
| `cuobjdump` | `/usr/local/cuda/bin/cuobjdump` | CUDA 12.6.68 |
| `nvdisasm` | `/usr/local/cuda/bin/nvdisasm` | CUDA 12.6.68 |
| `c++filt` | `/usr/bin/c++filt` | binutils 2.38 |

Both shipped with JetPack; nothing had to be installed.
Full help output: `env/tool_help/` — 12 files (`ncu_help.txt`, `nsys_help.txt`,
`nsys_profile_help.txt`, `nsys_stats_help.txt`, `nsys_stats_help_reports.txt`,
`ncu_list_sets.txt`, `ncu_list_sections.txt`, **`ncu_query_metrics.txt`**,
`cuobjdump_help.txt`, `nvdisasm_help.txt`, `ncu_version.txt`, `nsys_version.txt`,
`cxxfilt_help.txt`).

### Nsight Compute needs **root** on this device
`perf_event_paranoid = 1` (protocol §12) is **not sufficient** here. With it applied, an
unprivileged `ncu` still fails with

```
==WARNING== Insufficient privileges to launch app for profiling. Launch app with root privileges
```

so every `ncu` invocation must be `sudo <absolute path to ncu> ...`. Two details:

* `sudo ncu` fails with `sudo: ncu: command not found` — `secure_path` drops
  `/usr/local/cuda/bin`. Use `sudo $(which ncu)`.
* the `.ncu-rep` is then owned by root; `chown` it before exporting as the normal user.

`ncu --import ... --csv` needs no privileges.

### `--set` values this Orin offers (`ncu --list-sets`)
The protocol expects at least `basic`, `roofline`, `full`; all three exist, plus two more.

| set | sections | estimated metrics |
|---|---|---|
| `basic` (default) | LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution | 145 |
| `detailed` | + ComputeWorkloadAnalysis, MemoryWorkloadAnalysis(+Chart), SourceCounters, SpeedOfLight_RooflineChart | 460 |
| **`full`** | + InstructionStats, MemoryWorkloadAnalysis_Tables, NumaAffinity, Nvlink*, **PmSampling**, **SchedulerStats**, **WarpStateStats** | **614** |
| `nvlink` | Nvlink, Nvlink_Tables, Nvlink_Topology | 52 |
| `pmsampling` | PmSampling, PmSampling_WarpStates | 80 |
| `roofline` | SpeedOfLight + 4 hierarchical roofline charts, WorkloadDistribution | 2841 |

24 sections are available in total (`env/tool_help/ncu_list_sections.txt`).
Occupancy and the warp-stall statistics are only in `full` — use `--set full`, filtered.

### Exact flag spellings on ncu 2024.3.1 (all as the protocol assumes)

| purpose | flag |
|---|---|
| enable NVTX | `--nvtx` |
| NVTX filter | `--nvtx-include arg` / `--nvtx-exclude arg` |
| kernel-name filter | `-k [ --kernel-name ] arg` (also `--kernel-name-base arg (=function)`) |
| capture at most N | `-c [ --launch-count ] arg` |
| skip first N | `-s [ --launch-skip ] arg (=0)` (also `--launch-skip-before-match`) |
| honour cudaProfilerStart/Stop | `--profile-from-start arg (=1)` → pass `off` |
| metric set | `--set arg` |
| single section | `--section arg` |
| import a report | `-i [ --import ] arg` |
| CSV output | `--csv` |
| page selection | `--page arg (=details)` — `details`, `raw`, … |
| overwrite | `-f` |
| clock behaviour | `--clock-control arg (=base)` ← **see §6** |
| cache behaviour | `--cache-control arg (=all)` |
| replay | `--replay-mode arg (=kernel)` |

### nsys
`-f true` for overwrite works (`--force-overwrite` was not needed).
All seven report types of protocol §3.1 exist and ran
(`cuda_gpu_kern_sum, cuda_gpu_mem_time_sum, cuda_api_sum, nvtx_sum, nvtx_kern_sum,
nvtx_gpu_proj_trace, cuda_kern_exec_sum`), plus `cuda_gpu_trace`; full list in
`env/tool_help/nsys_stats_help_reports.txt`.

One thing to know: with `--capture-range=cudaProfilerApi` the default
`--capture-range-end=stop-shutdown` **kills the target** at `cudaProfilerStop`, so the
script's own output after the window is lost (the process exits 143). Add
`--capture-range-end=stop` when the script must finish.

---

## 6. What ncu can and cannot measure on this Orin

From the toy `--set full` run (`results/20260921_E0_platform/ncu/`,
inventory in `ncu/columns.txt`, verdicts in `ncu/metric_presence.csv`).
One kernel, `--set full` → **8.30 s wall clock, 30–31 replay passes, 7.07 MB report
(92 KB as `zstd -19`), 1 117 raw columns**.

**Verdicts: 52 present, 14 present-but-zero, 9 absent.** The `present_zero` verdict is the
"empty" case T001 asks about, made precise: the column exists and every value is exactly 0.
Most of the 14 are legitimately zero *for this kernel* (no local-memory traffic, no atomics,
no shared-memory atomics, no `tex_throttle` / `membar` / `sleeping` stalls, an unused `xu`
pipe, a 0 % L1/TEX hit rate on a streaming weight read). **Three are not: every `dram__*`.**

**Present with non-zero values** — all of protocol §3.3's occupancy metrics (`SM Busy`,
`SM Active Cycles`, `Achieved Occupancy`, `Theoretical Occupancy`,
`Achieved Active Warps Per SM`, `sm__warps_active.avg.pct_of_peak_sustained_active`),
throughput (`Duration`, `Compute (SM) Throughput`, `Memory Throughput`, `Mem Busy`,
`L2 Hit Rate`, both `Executed Ipc`), the LSU counters this kernel exercises, the
launch/occupancy-limit metrics, and **15 of the 18 warp-stall reasons**.

**Names that differ from the protocol's:** the stall reasons are **not** exported as
`Stall Long Scoreboard` etc. on the `details` page — that page has a `Warp State Statistics`
section but no `Stall …` metric rows. They are on the **`raw`** page as
`smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`, for
`barrier, branch_resolving, dispatch_stall, drain, imc_miss, lg_throttle, long_scoreboard,
math_pipe_throttle, membar, mio_throttle, misc, no_instruction, not_selected, selected,
short_scoreboard, sleeping, tex_throttle, wait`. **E5/E5b must read the raw page for stalls.**

**Absent (9):**

| metric | note |
|---|---|
| `DRAM Throughput`, `DRAM Utilization`, `dram__bytes{,_read,_write}.sum` | see below |
| `sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active` | no fp16 pipe counter; bf16/fp16 MMA shows in `sm__pipe_tensor_op_hmma_cycles_active.*` |
| `sm__inst_executed_pipe_{fma,alu}.avg.pct_of_peak_sustained_elapsed` | the `..._active` variants exist and are non-zero |
| `gpc__cycles_active.avg.per_warp` | derive if needed |

### ⚠ This Tegra exposes no DRAM counters at all
Three independent checks:

1. **`ncu --query-metrics` on `Device Orin (GA10B)` lists 4 426 metrics and not one is
   `dram__*`** (`env/tool_help/ncu_query_metrics.txt`). It lists 1 859 `lts__*` metrics.
2. **An explicit `--metrics dram__bytes.sum,dram__bytes_read.sum,dram__throughput...`**
   profiles the kernel fine and returns all three as **`n/a`** with an empty unit
   (`results/20260921_E0_platform/raw/ncu_dram_probe.log`).
3. In the `--set full` export the six `dram__*` placeholder columns all read **0**, and
   `DRAM Throughput` / `DRAM Utilization` are absent from the `details` page.

On Orin the LPDDR5 sits behind the GPU L2 as *sysmem* and its monitors are not exposed to ncu.
**ncu's `Memory Throughput` on this device is the L2 throughput** — 76.55 % for the toy
kernel, equal to two decimal places to `L2 Cache Throughput` and to the raw-page
`lts__throughput.avg.pct_of_peak_sustained_elapsed` (76.553465) — **not DRAM utilisation.**

**Substitute.** Off-chip *read* traffic is recoverable from the L2 sysmem-miss sectors
(32 B per sector):

| quantity | value |
|---|---|
| `lts__t_sectors_srcunit_tex_aperture_sysmem_op_read_lookup_miss.sum` | 266 264 → **8.5204 MB** |
| analytic compulsory reads: W (4096×1024×2 B) + x (51×1024×2 B) | **8.4931 MB** |
| **reads measured ÷ reads analytic** | **1.0032** |
| all sysmem misses, `lts__t_sectors_aperture_sysmem_lookup_miss.sum` | 266 491 → 8.5277 MB |

Validated on **one** cold, read-dominated kernel whose 8.5 MB working set is about twice the
L2. The **write path is not validated**: the kernel's 0.42 MB output never appears as a
sysmem miss (consistent with a write-back L2 that has not evicted it).

**Rule for E5/E6:** off-chip read bytes = `lts__t_sectors_*_sysmem_*_lookup_miss.sum × 32 B`,
labelled *measured (L2 sysmem-miss sectors)*, never as `dram__*`.

### ⚠ ncu pins the clocks, so its durations are not latencies
`--clock-control` defaults to `base`: in the toy run ncu reported **`SM Frequency
305.70 MHz`** and `Duration 195.74 µs`. A separate probe (`bench/gpu_clock_probe.py` →
`results/20260921_E0_platform/raw/gpu_clock_under_load.txt`) shows the GPU runs at
**612 MHz under load** in this mode, so ncu profiles at about half the clock the workload
sees. Protocol §3.2 already says latency comes from CUDA events only; this is the mechanism.
(nsys measured the same kernel at a 185.4 µs median, but that capture predates
`jetson_clocks` and its clock was not recorded, so the pair is not a measurement of the
size of the effect — and this kernel is dominated by LPDDR5 traffic at an EMC clock that
did not change.)

---

## 7. Power rails — one wanted rail is missing from `tegrastats`

`results/20260921_E0_platform/tegrastats_rails.csv`.

| rail (protocol §4) | in `tegrastats` | in sysfs INA3221 |
|---|---|---|
| `VDD_GPU_SOC` | yes | `/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_*` |
| `VDD_CPU_CV` | yes | `.../1-0040/hwmon/hwmon3/in2_*` |
| `VIN_SYS_5V0` | yes | `.../1-0040/hwmon/hwmon3/in3_*` |
| **`VDDQ_VDD2_1V8AO`** (the memory rail) | **no** | **yes** — `.../1-0041/hwmon/hwmon4/in2_*` |

`tegrastats` on this carrier board prints only the three rails of the first INA3221
(`1-0040`). The memory rail lives on the second INA3221 (`1-0041`) and **must be read from
sysfs**. `bench/common.py:ina3221_rails()` reads all four
(`mW = in<i>_input[mV] × curr<i>_input[mA] / 1000`), and `bench/collect_platform.py`
samples them at 10 Hz alongside tegrastats into `power_idle_sysfs.csv` — so protocol §4's
**four-rail module power is available**: **8 522 mW** at idle (MODE_30W, clocks locked),
against 8 117 mW for the three rails tegrastats prints.

Two things worth knowing before E7:

* **sysfs and tegrastats agree** once the clocks are locked: `VDD_GPU_SOC` reads
  **3 587 mW from both**, flat across 587 samples (min 3 585, max 3 588).
* **`VDDQ_VDD2_1V8AO` is measured on a ~5 V bus.** Its `in2_input` is constant at
  **5 040 mV** over all 563 samples while `curr2_input` varies 80–140 mA → 403–706 mW. The
  channel therefore monitors the input to the memory regulator, not the 1.8 V rail its name
  suggests. The product is a real power, but **whether it is the memory power the research
  side wants is not established** — settle this before E7 quotes memory energy.

`tegrastats` needs **no** sudo on this board, and it **appends** to `--logfile`: delete the
file first or a re-run silently concatenates two sessions. `GR3D_FREQ` is logged alongside
as the coarse cross-check protocol §4 allows.

## 8. Session-level settings that do not survive a reboot

```bash
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'   # protocol §12
sudo jetson_clocks                                            # lock clocks
sudo jetson_clocks --fan                                      # fan to maximum
```

Caches are kept out of the repository:
`TRITON_CACHE_DIR=$PWD/scratch/<task>/triton_cache`,
`TORCHINDUCTOR_CACHE_DIR=$PWD/scratch/<task>/inductor_cache`.

## 9. Not installed yet
`jax`, `flash-attn` (wheels 2.8.2/2.8.3 exist on the Jetson index for cp310),
`torchao` (0.13.0 exists), `openpi`, `lerobot`, `Isaac-GR00T`, LIBERO. These belong to
T002 and later and go into their own environments.
