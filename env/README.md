# env — how this Thor was set up (on-device agent, T001, 2026-10-22)

Setup scripts: `setup_base.sh` (tool chain + `~/venvs/base`), `setup_lerobot.sh` (π0, π0.5, SmolVLA),
`setup_groot17.sh`, `setup_groot16.sh`, `setup_groot15.sh`, `setup_libero.sh`; frozen pins in `requirements_*.txt`;
every installation in `INSTALL_LOG.md` and `INSTALL_LOG_groot_libero.md`; profiler help texts in `tool_help/`.
**Before any bench/ command:** `source env/thor_env.sh` (CUDA 13 tools on `PATH`, Triton → CUDA 13 `ptxas`).
**Root:** the agent has no sudo. Root-only reads and profiler runs use `env/rootdo.sh` (docker `--privileged`
+ `nsenter` into the host; allow-listed commands only; approved by the owner on 2026-10-22).

## 1. Facts (each with its source command)

| fact | value | source |
|---|---|---|
| board | NVIDIA Jetson AGX Thor Developer Kit (SoC family tegra264) | `cat /proc/device-tree/model`; `jetson_clocks --show` |
| L4T | R38 (release), REVISION 2.2, GCID 42205042, 2025-09-25; `nvidia-l4t-core` 38.2.2-20250925153837 | `cat /etc/nv_tegra_release`; `dpkg -l` |
| JetPack | the `nvidia-jetpack` meta-package is **not installed** (apt candidate 7.0-b128, repo `r38.2`); L4T r38.2 is the JetPack 7.0 line | `results/20261022_E0_addenda/raw/apt_jetpack.txt` |
| CUDA | driver CUDA 13.0 (`nvidia-smi`); **no toolkit on the image**: CUDA 13.0 tools unpacked into `~/opt/cuda-root` (nvcc 13.0.48); torch's own runtime 13.0 | `nvidia-smi`; `nvcc --version`; `torch.version.cuda` |
| driver | 580.00, NVIDIA UNIX Open Kernel Module for aarch64, 2025-09-25 (`nvidia.ko`, `nvidia_uvm`) | `/proc/driver/nvidia/version`; `lsmod` |
| Python | 3.12.3 (system); `ensurepip` missing → `uv` 0.13.5 makes the venvs | `python3 --version` |
| kernel | 6.8.12-tegra #1 SMP PREEMPT (2025-09-25) | `uname -a` |
| CPU | 14 cores online (0–13), Arm (Neoverse-V3AE class, `lscpu` model name blank), 2.601 GHz max, schedutil | `lscpu`; `jetson_clocks --show` |
| RAM, swap | 125,772 MB total (`total_memory` 131,881,750,528 B seen by CUDA); swap 0 B | `free -h`; `swapon --show` |
| disk | 937 GB NVMe; 185 GB free after the checkpoints (49.3 GB of pinned snapshot files) | `results/20261022_E0_addenda/raw/thermal_fan_disk.txt`; `checkpoint_files.csv` |
| network | github.com, pypi.org, download.pytorch.org, huggingface.co, pypi.jetson-ai-lab.io, nvcr.io: all HTTP 200 | `results/20261022_E0_toolcheck/raw/network_check.txt` |
| GPU | "NVIDIA Thor", ncu chip name **GB10B**, compute capability **11.0**, `gpu_arch` **`sm_110`** | `torch.cuda.get_device_properties`; `ncu --query-metrics`; `cuobjdump --list-elf` on cuBLASLt (1,140 `sm_110` cubins; `--list-text`: 5,668 `sm_110` + 2 `sm_110a`, `results/20261022_E0_addenda/raw/cublaslt_list_text_sm_strings.txt`) |
| SM count (MAXN) | **20** | `multi_processor_count` |
| GPU caches / limits | L2 **32 MiB** (33,554,432 B; hint said 50 MB — corrected here); smem/SM 233,472 B (opt-in/block 232,448 B); max threads/SM 1,536 (= 48 warps); regs/SM 65,536 | `results/20261022_E0_addenda/raw/gpu_properties.json` |
| power modes | `0 MAXN` (no caps) and `1 120W` (GPU ≤ 1.386 GHz, video ≤ 1.557 GHz, CPU ≤ 2.601 GHz, EMC ≤ 4.266 GHz); **default at boot is 120W** (`PM_CONFIG DEFAULT=1`) | `/etc/nvpmodel.conf` → `results/20261022_E0_toolcheck/nvpmodel_modes.csv` |
| maximum-performance mode | **id 0, MAXN** (no CPU, GPU, EMC or TPC caps; same `GPU_PG_MASK` 64 as 120W) — **current mode: MAXN** | `nvpmodel -q --verbose` |
| GPU frequency node | `/sys/class/devfreq/gpu-gpc-0/{cur_freq,min_freq,max_freq,available_frequencies}`; table 54 – **1575 MHz** in 9 MHz steps, nvpmodel floor 314 MHz; governor `nvhost_podgov`. (`gpu-nvd-0` = video, up to 1692 MHz) | `results/20261022_E0_addenda/raw/devfreq.txt` |
| EMC rate node | `/sys/kernel/debug/bpmp/debug/clk/emc/rate` (root): **4,266,000,000 Hz**; EMC range 665.6 – 4266 MHz | `env/rootdo.sh cat …`; `jetson_clocks --show` |
| clock lock | `env/rootdo.sh jetson_clocks` → GPU min = max = 1575 MHz, CPU 2.601 GHz, EMC 4266 MHz `FreqOverride=1`; `--fan` → PWM 255 | `results/20261022_E0_toolcheck/raw/jetson_clocks_show_locked.txt` |
| thermal zones | `tj-thermal`, `gpu-thermal` (one read before the clock lock returned EAGAIN — not saved; readable after it), `cpu-thermal`, `soc012-thermal`, `soc345-thermal` (`/sys/class/thermal/thermal_zone0..4`); tegrastats prints cpu/tj/soc012/gpu/soc345 | `grep . /sys/class/thermal/thermal_zone*/type` |
| fan | `/sys/class/hwmon/<pwmfan>/pwm1` (hwmon name `pwmfan`), controlled by `nvfancontrol` until `jetson_clocks --fan` | `jetson_clocks --show` |
| torch route | **(a)** PyTorch SBSA wheels: torch **2.10.0+cu130**, torchvision **0.25.0+cu130**, triton **3.6.0**; passes every A8 check **only with** Triton pointed at CUDA 13.0 `ptxas` (bundled `ptxas-blackwell` 12.9 rejects `sm_110a`) | `results/20261022_E0_toolcheck/stack_verification.json` |
| Inductor `is_big_gpu` | threshold **68** SMs (`torch/_inductor/utils.py:1676`), device 20 → **fires**: "Not enough SMs to use max_autotune_gemm mode" (logged once per process) | same file |
| nsys / ncu / cuobjdump | Nsight Systems **2025.3.2.367**; Nsight Compute **2025.3.0.0** (build 36273991); cuobjdump / nvdisasm **13.0.39** | `--version` |
| ncu permission | unprivileged → `ERR_NVGPUCTRPERM`; **runs as root via `env/rootdo.sh ncu …`** (clean environment) | `results/20261022_E0_toolcheck/raw/ncu_toy.*` |
| ncu clock control | use **`--clock-control none` with `jetson_clocks` applied**: SM 1.5735 GHz vs locked 1.575 GHz. `base` also read 1.5731 GHz when locked but **0.899 GHz when unlocked**, i.e. it does not hold a fixed clock here | `raw/ncu_toy_clock_*` |
| DRAM-byte metrics | **no `dram__*` metric** on GB10B. Reads: `lts__d_sectors_fill_sysmem.sum` (266,233) or `lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum` (266,139) vs 265,408 analytic sectors on the toy (1.003×). Writes: `lts__t_sectors_aperture_sysmem_op_write.sum` (13,453 vs 13,056). × 32 B. `mcc__dram_throughput_*` (memory controller, %) exist but read **0** on the toy. T003 validates on ≥ 4 kernels | `results/20261022_E0_toolcheck/ncu/metric_presence.csv` |
| FLOP metrics | SIMT: `smsp__sass_thread_inst_executed_op_{f,h,d}{add,mul,fma}_pred_on.sum` (all present). Tensor: `sm__ops_path_tensor_src_<src>_dst_<dst>.sum` and per-op `sm__ops_path_tensor_op_{hmma,utchmma,utcimma,utcqmma,utcomma,…}_*` — on the toy `…_src_bf16_dst_fp32.sum` = 469,762,048 = 2·56·4096·1024 (counts the M-padded tile: M 51 → 56); the cuBLAS kernel issues `utchmma` (tcgen05 / tensor memory) | same |
| tensor-pipe metrics | `sm__pipe_tensor_cycles_active`, `sm__pipe_tensor_subpipe_{hmma,imma}_cycles_active`, `sm__inst_executed_pipe_tensor{,_subpipe_hmma,_subpipe_imma}`, `sm__inst_executed_pipe_tc{,_scope_1cta,_scope_2cta}`, `sm__inst_executed_pipe_tma`, `sm__inst_executed_pipe_tmem` | same |
| stall reasons | **raw page**: 17 `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` columns (also the details page's WarpStateStats section under `--set full`); they sum to 84.3 on the toy — warps stalled per issued instruction, not fractions | same |
| ncu kernel limit | **none hit**: 1,000 of 1,000 kernels profiled in one process (Orin stopped at 480) | `raw/ncu_limit_1000_tiny.*` |
| graph-node visibility | **yes**: `--graph-profiling node` (default) profiles each kernel node of a CUDA-graph replay | `raw/ncu_graph_node.*` |
| nsys binaries | **NOT CLEAN** — `.nsys-rep` / `.sqlite` contain the environment of processes nsys did not launch (an unrelated process's canary; the agent harness's session token). Never committed; `.gitignore` unchanged; CSV exports are the record. ncu reports: clean | `results/20261022_E0_secrets_test/` |
| shared device | other users run long sessions on this Thor (including GPU work: VDD_GPU 6.7–30.8 W in the 10 s sysfs sample, 36.7 W instantaneous in tegrastats, with nothing of ours running). Gate item 5 must be checked before every run | `who`; `raw/power_idle.csv` |

**ncu's own roofline sections** (`results/20261022_E0_toolcheck/ncu/roofline_section_metrics.csv`, 366 metric
references in 5 section files) take the DRAM roof and point from `dram__bytes.sum.{peak_sustained,per_second}` and
`dram__cycles_elapsed.avg.per_second`, which **do not exist on GB10B**: ncu's built-in DRAM roofline cannot be drawn
on this device. T006 uses the L2 sysmem counters above, validated in T003. The tensor roofline uses the
`sm__ops_path_tensor_*` family. Every column the toy `--set full` capture exported: `ncu/columns.txt`.

### Exact flag spellings (ncu 2025.3.0, nsys 2025.3.2; `env/tool_help/`)
| purpose | spelling |
|---|---|
| NVTX filter (ncu) | `--nvtx --nvtx-include "<range>/"` (with `--profile-from-start off` and `cudaProfilerStart/Stop` in the script) |
| kernel-name filter (ncu) | `-k regex:<pattern>` (`--kernel-name`); `--kernel-name-base {function,demangled,mangled}` |
| launch count / skip (ncu) | `-c <n>` (`--launch-count`), `-s <n>` (`--launch-skip`) |
| capture window (ncu) | `--profile-from-start off` |
| CUDA-graph nodes (ncu) | `--graph-profiling node` (default) or `graph` |
| clock control (ncu) | `--clock-control {base,none,reset}`; use `none` with `jetson_clocks` |
| metrics only (ncu) | `--metrics <comma list>`; sets `--set {basic,detailed,full,nvlink,pmsampling,roofline}`; `--section <name>` |
| import / CSV export (ncu) | `ncu --import <f>.ncu-rep --csv --page {details,raw}` |
| capture range (nsys) | `--capture-range=cudaProfilerApi --capture-range-end=stop`; graphs: `--cuda-graph-trace=node` |
| stats / export (nsys) | `nsys stats -r cuda_gpu_trace,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum,nvtx_kern_sum,nvtx_gpu_proj_trace -f csv -o <prefix> <rep>` (all seven exist in 2025.3.2); `nsys export --type sqlite` |

## 2. Rail table (protocol §4)

| rail | chip / sysfs (find by label, never by index) | what it powers | 10 s idle sample, clocks locked, GPU shared: min / median / max / mean (W) |
|---|---|---|---|
| `VIN` | ina238, hwmon `name` = ina238, `label` = VIN: `in1_input` (mV), `curr1_input` (mA), `power1_input` (µW, measured) | module input (bus ≈ 28.0 V at idle) — **module power** [ASSUMPTION from the label and topology: the one ina238 channel upstream of the three ina3221 rails] | 32.4 / 33.4 / 62.7 / 37.8 |
| `VDD_GPU` | ina3221, `in1_label` = VDD_GPU: `in1_input`, `curr1_input` | GPU rail (bus ≈ 19.76 V) | 6.7 / 7.1 / 30.8 / 10.0 (tegrastats instantaneous max 36.7 W: other users' GPU work) |
| `VDD_CPU_SOC_MSS` | ina3221, `in2_label`: `in2_input`, `curr2_input` | CPU + SoC + **MSS (memory subsystem)** (bus ≈ 19.76 V) — the nearest thing to a memory rail; **no separate DRAM rail exists** | 8.7 / 8.7 / 9.5 / 8.9 |
| `VIN_SYS_5V0` | ina3221, `in3_label`: `in3_input`, `curr3_input` | 5 V system supply (bus ≈ 5.04 V) | 5.8 / 5.8 / 6.1 / 5.9 |

ina3221 mW is derived (bus mV × mA); ina238 power is read directly. `tegrastats` (root, via `env/rootdo.sh`) prints
the same four rails plus `GR3D_FREQ` per GPC (3 entries), `EMC_FREQ` and temperatures. Unprivileged `tegrastats`
omits `GR3D_FREQ` and `EMC_FREQ`. Run it in the foreground and stop it with `tegrastats --stop`: a `--start` daemon
dies with its docker container. Samples: `results/20261022_E0_toolcheck/raw/power_idle.csv`, `raw/tegrastats_idle.log`.

## 3. Availability matrix (docs/05_setup.md §2)

| # | item | check | expected | found | status | action taken | H |
|---|---|---|---|---|---|---|---|
| A1 | board, L4T, JetPack | device-tree, nv_tegra_release, dpkg | Thor, r38+ | Thor devkit, L4T R38.2.2; JetPack meta-package not installed | ok | — | — |
| A2 | CUDA, driver | nvcc, nvidia-smi | CUDA 13.x | driver 580.00 / CUDA 13.0; no toolkit | ok | CUDA 13.0 tools unpacked from the JetPack apt repo without root | — |
| A3 | Python | python3, venv, pip | 3.12 | 3.12.3; no ensurepip | ok | `uv` | — |
| A4 | disk | df | ≥ 150 GB | 185 GB free after the downloads | ok | — | — |
| A5 | network | curl -sI | reachable | all 200 | ok | — | — |
| A6 | sudo | sudo -n | passwordless | **password required** | ok via route | owner chose docker: `env/rootdo.sh` (allow-listed root commands) | — |
| A7 | git | identity, push | push works | identity set (global, by the owner), deploy key with write access | ok | — | — |
| A8 | PyTorch for sm_110 | arch list, bf16 GEMM, compile ×4, triton | first passing route | route (a) torch 2.10.0+cu130 | ok | Triton → CUDA 13 ptxas (`thor_env.sh`) | — |
| A9 | profilers, binary tools | command -v | present | none present | ok | unpacked: ncu 2025.3.0, nsys 2025.3.2, cuobjdump/nvdisasm 13.0.39; zstd, c++filt present | — |
| A10 | ncu permission | toy ncu | unprivileged or root | ERR_NVGPUCTRPERM unprivileged; works as root | ok | `env/rootdo.sh ncu` | — |
| A11 | power monitor, tegrastats | hwmon, tegrastats | rail table | 4 rails, tegrastats ok | ok | — | — |
| A12 | clocks, power mode | nvpmodel, jetson_clocks, devfreq, EMC | max-perf mode | MAXN (id 0) current; jetson_clocks works via rootdo | ok | — | — |
| A13 | Hugging Face | downloads | public; PaliGemma tokenizer ungated route | 6 checkpoints public ✓; ungated tokenizer ✓ (sha256 match); **N1.7's backbone `nvidia/Cosmos-Reason2-2B` gated** | **blocked** (N1.7 only) | ungated tokenizer + local `tokenizer_config.json` (21 valid tokens = Orin's 533 − 512) | **H2** |
| A14 | model code | clone + checkout | pinned commits | all five at their pins | ok | — | — |
| A15 | LIBERO | MUJOCO_GL=egl import + render | renders | renders on the GPU (EGL) | ok | mujoco 2.3.7 built from source; X11 headers via `apt-get download` | — |
| A16 | one inference per model | smoke | finite actions | π0, π0.5, SmolVLA, N1.6, N1.5 pass; N1.7 blocked by A13 | partial | — | H2 |
| A17 | nsys secrets | canary test | verdict | NOT CLEAN | ok (verdict recorded) | `.gitignore` unchanged | — |

## 4. Per model

| model | env | code commit | checkpoint @ revision | smoke (`results/20261022_E0_smoke_models/`) |
|---|---|---|---|---|
| π0 | `~/venvs/lerobot` (lerobot 0.4.5 editable, transformers 5.3.0) | lerobot f0d2b37bebddf7e8852cb624712da0ad564601b6 | `lerobot/pi0_libero_finetuned_v044` @ 45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f | pass: P 816 (533 valid), S 51, actions [1, 50, 7] |
| π0.5 | same | same | `lerobot/pi05_libero_base` @ a217bfd3b14673cf2ce597e69997ab21866438dd (fp32) | pass: P 968 (574), S 50 |
| SmolVLA | same (+ `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @ 7b375e1b73b11138ff12fe22c8f2822d8fe03467) | same | `lerobot/smolvla_base` @ d9f33c94a60fb382c90dea2164c96845bd955e28 | pass: P 177 (151), S 50, actions [1, 50, 6] |
| GR00T N1.7 | `~/venvs/groot17` (transformers 4.57.6, flash-attn 2.8.4) | Isaac-GR00T 376ba890cff8c9de64d71d982772a9c36185fdd7 | `nvidia/GR00T-N1.7-3B` @ 2fc962b973bccdd5d8ce4f67cc63b264d6886495 | **blocked**: gated backbone (H2) |
| GR00T N1.6 | `~/venvs/groot16` (transformers 4.51.3, flash-attn 2.8.4) | n1d6 9b37aa1ce69c73c6d165233fa88128283bba4508 | `nvidia/GR00T-N1.6-3B` @ d0814e7ecb19202e7c8468b46098b0b7ef3a6d61 | pass (`robocasa_panda_omron`): P 284, S 51 (`results/20261022_E0_addenda/`) |
| GR00T N1.5 | `~/venvs/groot15` (own env; torch 2.10 instead of N1.5's 2.5.1/2.8.0, no cu130 build of those) | n1d5 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96 | `nvidia/GR00T-N1.5-3B` @ 869830fc749c35f34771aa5209f923ac57e4564e | pass (`oxe_droid`): P 814, S 49 |
| LIBERO | `~/venvs/libero` (mujoco 2.3.7, robosuite 1.4.1) | LIBERO 8f1084e3132a39270c3a13ebe37270a43ece2a01 | — | render pass |

Every checkpoint file with its bytes and sha256: `results/20261022_E0_addenda/checkpoint_files.csv` (116 files, 49,260,960,405 B).

**Load pitfalls to keep (harness):**
- lerobot's `from_pretrained` drops the `revision` argument, and offline it returns a random model. Load from
  the pinned snapshot directory (`bench/lerobot_common.py`).
- The ungated PaliGemma `tokenizer.model` needs `sentencepiece` to convert.
- LIBERO's init states need `torch.serialization.add_safe_globals` on torch ≥ 2.6.
- The GR00T checkouts are on `sys.path` through `.pth` files; run GR00T scripts from outside the checkout.
