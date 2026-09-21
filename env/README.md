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

## 2. Power mode — the device is **not** at MAXN

| | |
|---|---|
| mode at bring-up | **`MODE_30W` (id 2)** — the board's configured default (`PM_CONFIG DEFAULT=2`) |
| protocol requirement | MAXN (id 0) + `jetson_clocks` for every experiment |
| blocker | switching to MAXN **requires a reboot** on this board: `nvpmodel -m 0` prints `NVPM WARN: Reboot required for changing to this power mode: 0` and prompts. Per `AGENTS.md` the prompt was declined, so the mode did not change. |

The modes of this board (`/etc/nvpmodel.conf`, exported to
`results/20260921_E0_platform/nvpmodel_modes.csv`) — the protocol's expected id mapping is
**confirmed**:

| id | name | CPU cores | `TPC_PG_MASK` | active GPU TPCs | GPU max | EMC max |
|---|---|---|---|---|---|---|
| 0 | MAXN | 12 | 0 | 8 (→ 16 SMs) | uncapped | uncapped |
| 1 | MODE_15W | 4 | 248 | 1 | 408 MHz | capped |
| 2 | MODE_30W | 8 | 240 | **4 (→ 8 SMs)** | 612 MHz | uncapped |
| 3 | MODE_50W | 8 | 0 | 8 (→ 16 SMs) | 816 MHz | uncapped |

**Consequence, and it is large:** in `MODE_30W` the GPU runs with **half its SMs power-gated
and its clock capped at 612 MHz**. `torch.cuda.get_device_properties(0).multi_processor_count`
reports **8**, not 16, and `jetson_clocks --show` prints `Active GPU TPCs: 4`. Inductor
refuses `max-autotune` GEMM templates with *"Not enough SMs to use max_autotune_gemm mode"*.
**No latency, bandwidth, occupancy or power number taken in this mode belongs on the
ladder of `docs/00_context.md`.** Everything in `results/20260921_E0_platform/` is a
tool-and-stack verification record, not a measurement of the Orin's capability.

`jetson_clocks` itself works (verified, `raw/jetson_clocks_show_locked_30W.txt`): it locked
all 8 online CPUs to 1 728 MHz, the GPU to 612/612 MHz and EMC to 3 199 MHz, and
`jetson_clocks --fan` works.

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
bf16 MLP (1024 → 4096 → 1024, GeLU), input `[51, 1024]`, 20 steady-state calls after the
first, compared against eager:

| backend / mode | compiles | first call incl. compile | max rel. err vs eager |
|---|---|---|---|
| `inductor` (no mode) | yes | 3.05 s | 4.7e-3 |
| `inductor` `default` | yes | 0.10 s | 4.7e-3 |
| `inductor` `reduce-overhead` | yes | 0.70 s | 4.7e-3 |
| `inductor` `max-autotune` | yes | 1.97 s | 4.7e-3 |
| `inductor` `max-autotune-no-cudagraphs` | yes | 0.65 s | 4.7e-3 |
| `aot_eager` | yes | 0.12 s | 0 |
| `eager` | yes | 0.07 s | 0 |

The 4.7e-3 relative difference is Inductor's bf16 GeLU/`addmm` fusion, not a failure
(bf16 has ~3 decimal digits). A plain bf16 2048³ GEMM differs from an fp32 reference by
4.2e-3, the same order.

`max-autotune` emits **`Not enough SMs to use max_autotune_gemm mode`** — a consequence of
the 8-SM `MODE_30W` state, not of the stack. Re-verify at MAXN.

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
Full help output: `env/tool_help/` (`ncu_help.txt`, `nsys_help.txt`,
`nsys_profile_help.txt`, `nsys_stats_help.txt`, `nsys_stats_help_reports.txt`,
`ncu_list_sets.txt`, `ncu_list_sections.txt`, `cuobjdump_help.txt`, `nvdisasm_help.txt`).

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
One kernel, `--set full` → **9 s, 30 replay passes, 7.07 MB report, 1 117 raw columns**.

**Present and populated** — all of protocol §3.3's occupancy metrics (`SM Busy`,
`SM Active Cycles`, `Achieved Occupancy`, `Theoretical Occupancy`,
`Achieved Active Warps Per SM`, `sm__warps_active.avg.pct_of_peak_sustained_active`),
throughput (`Duration`, `Compute (SM) Throughput`, `Memory Throughput`, `Mem Busy`,
`L2 Hit Rate`, `L1/TEX Hit Rate`, both `Executed Ipc`), the LSU request counters, the
launch/occupancy-limit metrics, and **all 18 warp-stall reasons**.

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
| `DRAM Throughput`, `DRAM Utilization` | see below — this is the important one |
| `sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active` | no fp16 pipe counter; bf16/fp16 MMA shows in `sm__pipe_tensor_op_hmma_cycles_active.*` |
| `sm__inst_executed_pipe_{fma,alu}.avg.pct_of_peak_sustained_elapsed` | the `..._active` variants exist and are populated |
| `gpc__cycles_active.avg.per_warp` | derive if needed |
| `dram__bytes{,_read,_write}.sum` | not exported at all |

### ⚠ ncu cannot see DRAM traffic on this Tegra
Every `dram__*` counter that *is* exported reads **exactly 0**:
`dram__cycles_active.sum = 0`, `dram__cycles_elapsed.sum = 0`,
`dram__bytes_read.sum.pct_of_peak_sustained_elapsed = 0`. On Orin the LPDDR5 is behind the
GPU L2 as *sysmem*, and the DRAM performance monitors are not exposed to ncu.
**ncu's `Memory Throughput` on this device is the L2 throughput** — for the toy kernel it is
76.55 %, byte-identical to `lts__throughput.avg.pct_of_peak_sustained_elapsed` — **not DRAM
utilisation.** Reading it as DRAM utilisation in E5/E6 would be wrong.

**Substitute, and it checks out.** Off-chip traffic is measurable from the L2 sysmem-miss
sectors (32 B per sector):

| quantity | value |
|---|---|
| `lts__t_sectors_aperture_sysmem_lookup_miss.sum` | 266 491 sectors → **8.53 MB** |
| of which reads (`..._srcunit_tex_aperture_sysmem_op_read_lookup_miss.sum`) | 266 264 → **8.52 MB** |
| analytic bytes that *must* be read: W (4096×1024×2 B) + x (51×1024×2 B) | **8.49 MB** |
| measured ÷ analytic (reads) | **1.004** |

So `lts__t_sectors_aperture_sysmem_*.sum × 32 B` is a sound measured proxy for off-chip
bytes on this device, to 0.4 % on a kernel whose traffic is known. Use it for E6 and record
it as *measured (L2 sysmem-miss sectors)*, never as `dram__*`.

### ⚠ ncu locks the clocks, so its durations are not latencies
`--clock-control` defaults to `base`: in the toy run ncu reported `SM Frequency 305.70 MHz`
and `Duration 195.74 µs`, while the same kernel under nsys at the (30 W-capped) 612 MHz ran
in **186 µs** — and both are far from what MAXN will give. Protocol §3.2 already says
latency comes from CUDA events only; this is the mechanism.

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
sysfs**; `bench/common.py:ina3221_rails()` does that. E7's "module power" therefore needs
both sources. `curr<i>_input` (mA) × `in<i>_input` (mV) gives mW; for `VDD_GPU_SOC` this
agreed with the `tegrastats` figure at idle (3.19 W vs 2.79 W, sensor-averaged differently).
The `in2_input` voltage reported for `VDDQ_VDD2_1V8AO` is a bus voltage, not 1.8 V —
**E7 must confirm the scaling of this rail before quoting memory power.**

`tegrastats` needs **no** sudo on this board. `GR3D_FREQ` is logged alongside as the coarse
cross-check protocol §4 allows.

---

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
