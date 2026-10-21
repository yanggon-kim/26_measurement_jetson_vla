# E0 — Platform and toolchain record (task T001)

**Authoritative run:** `results/20260921_E0_platform_maxn/` — **MAXN (id 0)**,
`jetson_clocks` applied before every capture. · **date:** 2026-09-21 (UTC) ·
**code:** `meta.json.bench_files_sha256`.

**Comparison run:** `results/20260921_E0_platform/` — the same procedure in **MODE_30W**,
taken before the human authorised the reboot into MAXN. It is kept because the two together
show what the power mode costs, but **no number in it characterises the Orin.**

E0 produces no latency, bandwidth or power result that goes on the ladder of
`docs/00_context.md`. It answers one question: *does the stack this project needs exist and
work on this machine, and what can its tools actually measure?* The answer is **yes, with
four qualifications** that change how E1–E11 must be run: §4, §5 and §6.

---

## 1. The answer

### 1.1 Prerequisites (`AGENTS.md`)

| prerequisite | status | evidence |
|---|---|---|
| JetPack 6.x, L4T r36.x, CUDA 12.x, Python 3.10 | **met** — JetPack 6.2.3+b81, L4T **R36.5.0**, CUDA 12.6.68, Python 3.10.12 | `platform.json.device_state.nv_tegra_release`, `.dpkg_jetpack`, `.nvcc_version` |
| Passwordless `sudo` | **NOT met** | `sudo -n` → "a password is required". The human supplied the password interactively; it was used through a `SUDO_ASKPASS` helper outside the repository, which `/tmp` clearing at reboot removes. `tegrastats` needs no sudo here. |
| git identity and push access, network to GitHub / PyPI / Jetson index | **met** | `git ls-remote` succeeded; all T001 commits pushed |
| ≥ 150 GB free NVMe | **met** — 1.7 T free of 1.8 T | `platform.json.device_state.df_h` |
| `ncu` installed; `nsys` present | **met** — `ncu` 2024.3.1.0, `nsys` 2024.5.4.34, plus `cuobjdump`/`nvdisasm`/`c++filt` from CUDA 12.6.68 | `env/tool_help/` (12 files) |

### 1.2 Platform record (protocol §1), at MAXN

| quantity | value | source |
|---|---|---|
| board | NVIDIA Jetson AGX Orin Developer Kit, SoC `tegra234`; ncu names the GPU **`Orin (GA10B)`** | `/proc/device-tree/model`, `ncu --query-metrics` |
| L4T / JetPack | R36.5.0 (GCID 43688277) / 6.2.3+b81 | `/etc/nv_tegra_release`, `dpkg -l` |
| CUDA / cuDNN / TensorRT | 12.6.68 / 9.3.0.75 / 10.3.0.30 | `nvcc`, `dpkg -l` |
| kernel | 5.15.185-tegra (`oot`) | `uname -a` |
| GPU | Orin, cc **8.7**, arch **`sm_87`**, 48 max warps/SM | `hw_constants.json` |
| **power mode** | **MAXN (id 0)**, persisted `pmode:0000` | `nvpmodel -q` |
| **SMs** | **16** (`TPC_PG_MASK` 0 → 8 of 8 TPCs) | `torch.cuda.get_device_properties(0)`, `jetson_clocks --show` |
| GPU clock | locked **1 300.5 MHz**; **measured 1 300.5 MHz under load** | `jetson_clocks --show`, `raw/gpu_clock_under_load.txt` |
| CPU | 12 cores online, locked 2 201.6 MHz | `jetson_clocks --show` |
| **EMC clock** | **3 199 000 000 Hz = 3 199 MHz** → LPDDR5-6400 **as expected** | `/sys/kernel/debug/bpmp/debug/clk/emc/rate` |
| RAM / swap | 61 GiB; 8 × 3.8 G **zram** only, no disk swap | `free -h`, `swapon --show` |
| storage | NVMe 1.8 T, 85 G used | `df -h` |
| `nvidia-smi` | present and functional | `platform.json` |

### 1.3 What MODE_30W cost — the two runs side by side

Both runs used the identical harness, and in both `jetson_clocks` was applied (in the 30 W
run only *after* the two toy profiler captures — see §6.4).

| | MAXN | MODE_30W |
|---|---|---|
| `TPC_PG_MASK` / active TPCs | 0 / **8 of 8** | 240 / 4 of 8 |
| SMs reported by CUDA | **16** | 8 |
| GPU clock | **1 300.5 MHz** | 612 MHz |
| CPU cores online | 12 | 8 |
| EMC | 3 199 MHz | 3 199 MHz |
| idle module power (4 rails, §4) | **11 012 mW** | 8 522 mW |
| idle tj | 41.9 °C | 39.9 °C |
| toy bf16 GEMM, nsys median | **54.6 µs** | 185.4 µs |

The same kernel is **3.4×** faster at MAXN, against 4.25× more SM·GHz (16 × 1300.5 vs
8 × 612). This is why nothing from the 30 W run may be quoted: the penalty is large *and*
not a simple scale factor.

### 1.4 Idle power and thermals — **measured**, 58.6 s at 10 Hz, MAXN, clocks locked

`power_idle.csv` (tegrastats, n = 586 per rail) and `power_idle_sysfs.csv` (all four
INA3221 rails, n = 564).

| rail | median mW | mean | min | max | source |
|---|---|---|---|---|---|
| `VDD_GPU_SOC` | 4 782 | 4 782 | 4 780 | 5 179 | tegrastats |
| `VDD_CPU_CV` | 1 593 | 1 441 | 1 195 | 3 585 | tegrastats |
| `VIN_SYS_5V0` | 4 233 | 4 209 | 4 132 | 4 939 | tegrastats |
| `VDDQ_VDD2_1V8AO` | 403 | — | 403 | 1 008 | **sysfs only** (§6.3) |
| **module power = the four rails of protocol §4** | **11 012** | 10 888 | — | — | `power_idle_sysfs.csv` |
| `tj` | 41.9 °C | 42.0 | 41.5 | 42.6 | tegrastats |
| `GR3D_FREQ` | 0 % throughout | | | | tegrastats |

`throttled: false`. tj never exceeded 42.6 °C and no clock fell below its pinned value.
This is *idle at MAXN with clocks locked* — the baseline E7 will subtract.

### 1.5 Software stack — **works**

`torch 2.8.0` + `torchvision 0.23.0` + `triton 3.4.0`, NVIDIA/Jetson CUDA 12.6 wheels for
cp310 aarch64, in `~/venvs/jetson_base`. `torch.version.cuda = 12.6`,
`torch.cuda.get_arch_list() = ['sm_87']`, `torch.cuda.is_available() = True`.

* bf16 GEMM 2048³ runs, finite, max relative error **3.0e-3** against an fp32 reference —
  the expected bf16 mantissa error, not a fault.
* bf16, fp16, fp32 matmul and int8 (`torch._int_mm` → int32) all run.
* **`torch.compile` compiles and runs a small bf16 MLP in every mode tried** (1024 → 4096 →
  1024 GeLU, input `[51, 1024]`, 20 warm-up + 100 timed CUDA-event calls per mode; the 700
  samples are in `latency_samples_compile_probe.csv`, the stdout in
  `raw/verify_stack.stdout.log`):

| backend / mode | compile + first call | steady-state median | max rel. err vs eager |
|---|---|---|---|
| `inductor` (no mode) | 2.86 s | 0.307 ms | 4.7e-3 |
| `inductor` `default` | 0.08 s | 0.307 ms | 4.7e-3 |
| `inductor` `reduce-overhead` | 0.59 s | 0.271 ms | 4.7e-3 |
| `inductor` `max-autotune` | 1.54 s | 0.268 ms | 4.7e-3 |
| `inductor` `max-autotune-no-cudagraphs` | 1.08 s | 0.326 ms | 4.7e-3 |
| `aot_eager` | 0.10 s | 0.465 ms | 0 |
| `eager` | 0.05 s | 0.257 ms | 0 |

The 4.7e-3 difference is Inductor's bf16 `addmm`+GeLU fusion, the same order as bf16's own
3.0e-3 error against fp32. (These steady-state times are a functional sanity figure on a
toy MLP, not a protocol measurement — note that eager is the fastest of the seven here.)

### 1.6 `hw_constants.json` (protocol §8)

| constant | value | how |
|---|---|---|
| SM count | top-level `multi_processor_count` is deliberately **`null`**; `multi_processor_count_in_this_power_mode` = **16** at MAXN, and `multi_processor_count_by_power_mode` records the per-mode values with MAXN now **OBSERVED** | torch device properties + `nvpmodel.conf` |
| compute capability | 8.7 | torch device properties |
| `gpu_arch` | **`sm_87` — confirmed** | torch arch list; `cuobjdump --list-text` on `/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublas.so.12.6.1.4`, which embeds `sm_53…sm_90` including `sm_87` |
| max warps per SM | 48 — *from torch's `max_threads_per_multi_processor`/32, not from ncu* | recorded with its source |
| ncu per-SM numbers | split into `device_limits` and `per_kernel_NOT_a_device_limit` | `ncu/toy_raw.csv` |
| peak GB/s, TFLOP/s, ridge point | **null — not measured** | T005/E1 fills them. Spec expectations (204.8 GB/s, 42 TFLOP/s, ridge ≈ 205) sit in a separate `expected_from_spec_do_not_use_as_measurement` block. |

### 1.7 Profiler toolchain — **works end to end**

* **nsys.** All seven protocol §3.1 report types plus `cuda_gpu_trace` ran and exported
  (`nsys/toy_*.csv`). `cuda_gpu_mem_time_sum` is correctly **empty** — the capture window
  holds no memcpy. NVTX attribution works: the GEMM is assigned to range `:toy`. `-f true`
  parses. Adding `--capture-range-end=stop` let the script run to completion (§6.5).
* **ncu.** `--set full --nvtx --nvtx-include "toy/" --profile-from-start off` captured
  **exactly one** kernel (the sibling `toy_alias` range correctly excluded) in **7.78 s**
  wall clock, 30 replay passes, a **7 107 467 B** report (**92 KB** as `zstd -19`). Both
  pages exported: `ncu/toy_details.csv` (80 rows, **69 distinct metric names**) and
  `ncu/toy_raw.csv` (**1 117** columns).

`--set` values this Orin offers — protocol expected ≥ `basic`, `roofline`, `full`; **all
three exist**, plus `detailed`, `nvlink`, `pmsampling`. Exact flag spellings are as the
protocol assumes; table in `env/README.md` §5.

---

## 2. Conditions

Batch 1, bf16, fixed seed 0. MAXN, `jetson_clocks` + `jetson_clocks --fan` applied
immediately after the reboot and **before every capture**. `GR3D_FREQ` 0 % before each
capture. Timing and profiling never mixed: the stack verification, the nsys run and the ncu
run are separate process launches. `perf_event_paranoid` set to 1 for the session. Profiled
commands ran under `env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u GITHUB_TOKEN -u GH_TOKEN
HF_HUB_OFFLINE=1` (protocol §3.5).

---

## 3. Getting to MAXN — what it takes on this board, and what E7 must plan for

The board boots to `MODE_30W` (`PM_CONFIG DEFAULT=2`). `sudo nvpmodel -m 0` answers:

```
NVPM WARN: Reboot required for changing to this power mode: 0
NVPM WARN: DO YOU WANT TO REBOOT NOW? enter YES/yes to confirm:
```

`AGENTS.md` forbids the agent confirming a reboot prompt, so it declined; **the human ran
the command and confirmed it**, and the device came back in MAXN with `pmode:0000`
persisted.

**Two facts E7's power-mode sweep depends on:**

1. **Declining the prompt does not stage the change.** `nvpmodel` then reports
   `NVPM ERROR: optMask is 1, no request for power mode` and leaves
   `/var/lib/nvpmodel/status` untouched — verified. The only ways in are answering `YES` or
   `nvpmodel -m 0 --force` (which reboots without asking).
2. **Any mode change that alters TPC gating needs a reboot.** From `nvpmodel_modes.csv`:
   MAXN and MODE_50W have `TPC_PG_MASK = 0`; MODE_30W has 240; MODE_15W has 248. So MAXN ↔
   50 W is a runtime change, but MAXN ↔ 30 W and MAXN ↔ 15 W each require a reboot.
   **E7's four-mode sweep therefore cannot run inside one session**; it needs one
   human-authorised reboot per gating change. Raised as question 3 in `STATUS.md`.

| id | name | CPU cores | `TPC_PG_MASK` | active TPCs | GPU max |
|---|---|---|---|---|---|
| 0 | MAXN | 12 | 0 | 8 → 16 SMs | uncapped (1 300.5 MHz) |
| 1 | MODE_15W | 4 | 248 | 1 | 408 MHz |
| 2 | MODE_30W | 8 | 240 | 4 → 8 SMs | 612 MHz |
| 3 | MODE_50W | 8 | 0 | 8 → 16 SMs | 816 MHz |

---

## 4. Anomaly 1: `ncu` requires root on this device

Protocol §12's fix for `ERR_NVGPUCTRPERM` —
`sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'` — was applied and is **not
sufficient** here. As an unprivileged user, even `ncu --list-sets` fails with

```
==WARNING== Insufficient privileges to launch app for profiling. Launch app with root privileges
```

**Substitution adopted** (recorded in `meta.json.notes` per protocol §3): every `ncu`
invocation is `sudo $(which ncu) ...`. `sudo ncu` alone fails — sudo's `secure_path` does not
contain `/usr/local/cuda/bin`. The resulting `.ncu-rep` is owned by root and must be
`chown`ed before the (unprivileged) `--import` export. `perf_event_paranoid` resets to 2 on
every reboot.

---

## 5. Anomaly 2: `max-autotune` can never use Triton GEMM templates on a Jetson Orin

With a cold Inductor cache, `torch.compile` logs

```
W torch/_inductor/utils.py:1446] [0/0] Not enough SMs to use max_autotune_gemm mode
```

**The 30 W record read this as a power-mode artefact and said "re-check at MAXN". That was
wrong.** It still fires at MAXN with 16 SMs (`raw/verify_stack.stdout.log`, line 3). The
cause is a hard-coded threshold in the installed torch 2.8.0:

```python
min_sms = 16 if device.type == "xpu" else 68  # 3080
avail_sms = prop.multi_processor_count
if avail_sms < min_sms:
    log.warning("Not enough SMs to use max_autotune_gemm mode", ...)
    return False
```

Orin has **16** SMs against a required **68**, in every power mode — and no Jetson Orin
will ever clear that threshold. `max-autotune` consequently falls back to ATen/cuBLAS for
GEMMs: its autotune log for this MLP compares only `addmm` against `bias_addmm`, both ATen,
and picks `addmm`.

**Consequence for E4:** "max-autotune" on this device means **"default + CUDA graphs + ATen
kernel-choice autotuning"**, not Triton template autotuning. Whatever speed-up E4 measures
for it must be described that way, and the characterization paper's `torch.compile` gain on
other hardware may well have come from a mechanism that is simply unavailable here. This
also means `max-autotune` and `max-autotune-no-cudagraphs` should differ mainly by CUDA
graphs on this device — E4 can use that as a check.

---

## 6. Anomaly 3 (important for E5/E6/E7): what ncu can and cannot measure here

Full verdict table: `ncu/metric_presence.csv` — **52 present, 14 present-but-zero, 9
absent** (identical at MAXN and 30 W). Inventory: `ncu/columns.txt`.

`present_zero` is the "empty" case T001 asks about, made precise: the column exists and
every value is exactly 0. Most of the 14 are *legitimately* zero for this kernel — no
local-memory traffic, no atomics, no shared-memory atomics, no `tex_throttle` / `membar` /
`sleeping` stalls, an unused `xu` pipe, and a 0 % L1/TEX hit rate on a streaming weight
read. **Three are not:** every `dram__*`.

### 6.1 ⚠ This Tegra exposes no DRAM counters at all

Three independent checks, all committed:

1. **`ncu --query-metrics` on `Device Orin (GA10B)` lists 4 426 metrics and not one is
   `dram__*`** (`env/tool_help/ncu_query_metrics.txt`). It lists 1 859 `lts__*` metrics.
2. **Explicitly asking returns `n/a`.** `ncu --metrics dram__bytes.sum,dram__bytes_read.sum,
   dram__throughput.avg.pct_of_peak_sustained_elapsed` profiles the kernel successfully and
   reports all three with an empty unit and the value `n/a`
   (`results/20260921_E0_platform/raw/ncu_dram_probe.log`).
3. In the `--set full` export the six `dram__*` placeholder columns read **0** — at MAXN as
   well as at 30 W — and `DRAM Throughput` / `DRAM Utilization` are absent from `details`.

On Orin the LPDDR5 sits behind the GPU L2 as *sysmem* and its performance monitors are not
exposed to ncu. **ncu's `Memory Throughput` on this device is the L2 throughput** — at 30 W
it was 76.55 %, equal to two decimal places to `L2 Cache Throughput` and to the raw-page
`lts__throughput.avg.pct_of_peak_sustained_elapsed` — **not DRAM utilisation.**

**Substitute, now validated exactly.** Off-chip *read* traffic from the L2 sysmem-miss
sectors (32 B per sector):

| | MAXN | MODE_30W |
|---|---|---|
| `lts__t_sectors_srcunit_tex_aperture_sysmem_op_read_lookup_miss.sum` | 265 408 → **8.4931 MB** | 266 264 → 8.5204 MB |
| analytic compulsory reads: W (4096·1024·2 B) + x (51·1024·2 B) | **8.4931 MB** | 8.4931 MB |
| **measured ÷ analytic** | **1.0000** | 1.0032 |

**Validated:** the *read* path, on one kernel, cold, with a working set about twice the L2.
**Not validated:** the *write* path — the kernel's 0.42 MB output never appears as a sysmem
miss, consistent with a write-back L2 that has not evicted it. E1/E2 should re-check against
a kernel with known write traffic and against the measured bandwidth peak.

**Rule for E5/E6:** off-chip read bytes = `lts__t_sectors_*_sysmem_*_lookup_miss.sum × 32 B`,
labelled *measured (L2 sysmem-miss sectors)*. Never quote `dram__*`, and never read
`Memory Throughput` as DRAM utilisation. Question 1 for the research side in `STATUS.md`.

### 6.2 Metric names that differ from the protocol's

* **Warp-stall reasons are not on the `details` page.** It has a `Warp State Statistics`
  section but no `Stall …` metric rows. All **18** are on the **`raw`** page as
  `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` for
  `barrier, branch_resolving, dispatch_stall, drain, imc_miss, lg_throttle, long_scoreboard,
  math_pipe_throttle, membar, mio_throttle, misc, no_instruction, not_selected, selected,
  short_scoreboard, sleeping, tex_throttle, wait`. **E5/E5b must parse the raw page.**
* `sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active` does not exist; bf16/fp16
  tensor-core work appears in `sm__pipe_tensor_op_hmma_cycles_active.*`.
* `sm__inst_executed_pipe_{fma,alu}.avg.pct_of_peak_sustained_elapsed` are absent; the
  `..._active` variants exist and are non-zero.
* `gpc__cycles_active.avg.per_warp` is absent; derive it if E5b needs it.

### 6.3 ⚠ `tegrastats` does not print the memory rail — read all four from sysfs

| rail (protocol §4) | in `tegrastats` | in sysfs INA3221 |
|---|---|---|
| `VDD_GPU_SOC`, `VDD_CPU_CV`, `VIN_SYS_5V0` | yes | `1-0040/hwmon/hwmon3/in{1,2,3}_*` |
| **`VDDQ_VDD2_1V8AO`** | **no** | **`1-0041/hwmon/hwmon4/in2_*`** |

`bench/collect_platform.py` samples all four from sysfs at 10 Hz into `power_idle_sysfs.csv`,
so protocol §4's four-rail module power exists. Two notes:

* **sysfs and tegrastats agree** once the clocks are locked (checked in the 30 W run:
  `VDD_GPU_SOC` read 3 587 mW from both, flat over 587 samples). An apparent discrepancy in
  an earlier draft was two instantaneous reads at different times, not a sensor difference.
* **`VDDQ_VDD2_1V8AO` is measured on a ~5 V bus.** `in2_input` is constant at **5 040 mV**
  while `curr2_input` varies, so the channel monitors the input to the memory regulator, not
  the 1.8 V rail its name suggests. The `mV × mA` product is a real power, but **whether it
  is the memory power the research side wants is not established** — question 2 in
  `STATUS.md`, to settle before E7 quotes memory energy.

`tegrastats` needs no sudo here and **appends** to `--logfile`; delete the file first.

### 6.4 ncu pins the clocks — check them, but at MAXN they now agree

`--clock-control` defaults to `base`. At MAXN with `jetson_clocks` applied, ncu reports
**`SM Frequency 1.29 GHz`** against the **1 300.5 MHz** measured under load
(`raw/gpu_clock_under_load.txt`), and its `Duration` **57.09 µs** is within **4.5 %** of the
nsys median **54.6 µs**.

In the 30 W run, taken with clocks *unlocked*, they diverged badly: ncu pinned 305.70 MHz
and reported 195.74 µs against a 185.4 µs nsys median at an unrecorded clock. **The lesson
is procedural, not about ncu:** lock the clocks before profiling and record them. Protocol
§3.2's rule stands regardless — latency comes from CUDA events, never from ncu.

### 6.5 nsys kills the target at `cudaProfilerStop` unless told otherwise

With `--capture-range=cudaProfilerApi`, nsys 2024.5.4 defaults to
`--capture-range-end=stop-shutdown`: the process is SIGTERMed (exit 143) and anything the
script prints after the window is lost. Adding **`--capture-range-end=stop`** fixed it — the
MAXN capture's script ran to completion and its `ok shape=…` line is in
`raw/nsys_toy.stdout.log`.

---

## 7. One observation, explicitly *not* a result

In the MAXN nsys capture, six back-to-back `F.linear` launches showed `cudaLaunchKernel`
median **21.4 µs** (min 16.8 µs) of CPU time, and the five gaps between consecutive kernel
executions were **1.34, 1.57, 1.31, 18.75 and 43.30 µs** — the last two spanning the NVTX
range boundaries where the Python loop re-enters. The project currently *assumes* 20 µs of
per-kernel launch overhead (`docs/00_context.md`). **These are from a profiled run with
n = 6 — they satisfy none of protocol §2's timing rules and must not be used.** E2 measures
launch overhead properly.

---

## 8. Separation of evidence

**Measured on this device:** §1.2, §1.3, §1.4, §1.7; the `sm_87` confirmation; every ncu
metric verdict and value in §6, including the three DRAM checks and the 1.0000 L2 ratio; the
1 300.5 MHz clock under load; the nsys report coverage and kernel durations; the ncu run
time; the `torch.compile` matrix and its 700 per-iteration samples; the `min_sms = 68`
threshold (read from the installed source) and that the warning fires at 16 SMs.

**Derived:** the 8.4931 MB analytic compulsory-read figure (from tensor shapes) and the
ratios; the four-rail module-power sum; the "3.4× vs 4.25× SM·GHz" comparison; the per-mode
SM counts for 15 W and 50 W in `hw_constants.json`, which come from `TPC_PG_MASK` and were
not observed.

**Not measured:** peak memory bandwidth, peak TFLOP/s per dtype and the ridge point (nulls in
`hw_constants.json`, for T005/E1); DRAM traffic via `dram__*` (impossible here, §6.1);
off-chip **write** traffic via the L2 substitute (§6.1); the physical meaning of the
`VDDQ_VDD2_1V8AO` reading (§6.3); per-kernel launch overhead (§7).

---

## 9. Reproduce

```bash
sudo nvpmodel -m 0        # human confirms the reboot; see §3
sudo jetson_clocks && sudo jetson_clocks --fan
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

bash env/setup.sh
python bench/verify_stack.py --out results/<run>/stack_verification.json \
       --samples-csv results/<run>/latency_samples_compile_probe.csv
python bench/collect_platform.py --outdir results/<run> --sudo "sudo -A" --idle-seconds 60
python bench/gpu_clock_probe.py

nsys profile -t cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi \
     --capture-range-end=stop -f true -o <scratch>/nsys_toy \
     $VENV/bin/python bench/ncu_toy.py --iters 3
nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum,\
nvtx_kern_sum,nvtx_gpu_proj_trace,cuda_kern_exec_sum -f csv \
     -o results/<run>/nsys/toy <scratch>/nsys_toy.nsys-rep
nsys stats -r cuda_gpu_trace -f csv -o results/<run>/nsys/toy <scratch>/nsys_toy.nsys-rep

sudo $(which ncu) --set full --nvtx --nvtx-include "toy/" --profile-from-start off \
     -o <scratch>/ncu_toy -f  $VENV/bin/python bench/ncu_toy.py
sudo chown $USER <scratch>/ncu_toy.ncu-rep
ncu --import <scratch>/ncu_toy.ncu-rep --csv --page details > results/<run>/ncu/toy_details.csv
ncu --import <scratch>/ncu_toy.ncu-rep --csv --page raw     > results/<run>/ncu/toy_raw.csv

python bench/ncu_columns.py --details ... --raw ... --outdir results/<run>/ncu
python bench/hw_constants.py --out results/<run>/hw_constants.json --ncu-raw-csv ...
python bench/finalize_raw.py --run results/<run> ...
python bench/make_meta_e0.py --run results/<run> --jetson-clocks-note "..." --ncu-run-time-s ...
```

Exact invocations are in `meta.json.command` and `meta.json.raw_files[].produced_by`;
the code that ran is pinned by `meta.json.bench_files_sha256`.

---

## 10. Review (protocol §11)

### Self-review

| check | result |
|---|---|
| every requested file exists under the requested name | **yes**, in both run directories: `hw_constants.json`, `ncu/columns.txt`, both toy ncu exports, `raw/`; `env/README.md`, `env/setup.sh`, `env/requirements_base.txt`, `env/tool_help/` (12 files) |
| every number in this report traces to a file in `results/` | **yes** — each table cites its file |
| `meta.json` complete per §5 | **yes**; model fields `null` with a stated reason; `bench_files_sha256` pins the code |
| `raw_files` hashes filled | **yes** — 12 entries (MAXN), 14 (30 W), each with sha256, bytes, kind, `produced_by`, `secrets_scan`; the `.ncu-rep` carries `compression` + `original` |
| n, power mode, `throttled` stated | **yes** — n = 586/564 idle samples, 700 compile samples, 6 nsys kernels, 1 ncu kernel; `MAXN (id 0)`; `throttled: false` |
| measured / derived / not-measured separated | **yes** — §8 |
| deviations listed | **yes** — §4, §5, §6, and `meta.json` (MAXN run has `power_mode_deviation: null`) |
| no forbidden file staged; nothing over 50 MB | **yes** — `find results -size +50M` empty |
| secrets scan | **clean** on every raw file and every committed CSV/JSON/TXT |
| research-side files untouched | **yes** — only `env/ bench/ results/ reports/ STATUS.md` and the status block in `tasks/T001_bringup.md` |

### Independent pass, and what it changed

A sub-agent was given the task file, the protocol and the produced files (against the 30 W
run) and asked to list unmet criteria, untraceable numbers and protocol violations. It
confirmed the `meta.json` schema, the raw-file hashes including a `zstd` round-trip, the size
rules, the absence of forbidden content and that the research side was untouched. **It found
one serious defect and nine smaller ones:**

| finding | fix |
|---|---|
| **An earlier draft claimed an independent pass had already happened and that its findings were recorded below. It had not — a fabricated process claim,** the thing `AGENTS.md` forbids most plainly. | Removed; this section records the actual review. |
| `meta.json.bench_commit` pointed at the *claim* commit, where `bench/` was empty | `bench_files_sha256` added, with a note on why the commit alone is insufficient |
| the ncu run time and the "Not enough SMs" line existed only as prose | both captured into committed raw files |
| §5.4 asserted the nsys capture ran at 612 MHz; it was never recorded | rewritten (now §6.4); the clock under load is measured by `bench/gpu_clock_probe.py` |
| "no DRAM counter on this Tegra" rested only on the `--set full` export | two decisive checks added: `--query-metrics` (0 of 4 426) and an explicit `--metrics` returning `n/a` |
| the 1.004 ratio compared all-misses against read-only analytic bytes | reads-vs-reads given; at MAXN it is **1.0000** |
| `metric_presence.csv` had no "empty" verdict, and 14 zero-valued metrics were called "populated" | third verdict `present_zero` added; counts 52 / 14 / 9, zeros named |
| `hw_constants.json` exposed the power-gated SM count as a top-level constant | bare key `null`; mode-keyed fields; per-kernel ncu limits split out |
| the inter-kernel gap range covered only 2 of 5 samples | corrected |
| protocol §4 module power needs four rails; only three were sampled | `collect_platform.py` samples all four from sysfs |
| per-iteration compile samples discarded; several small numeric errors | all corrected |

**Two of its recommendations were deferred to the MAXN re-run and are now done:**
`jetson_clocks` is applied before every capture (§6.4), and the L2 substitute was
re-validated (§6.1, ratio 1.0000).

**A claim the MAXN run itself disproved.** The 30 W record attributed the
`Not enough SMs to use max_autotune_gemm mode` warning to the 8-SM power mode and said
"re-check at MAXN". Re-checked: it still fires at 16 SMs. The real cause is a hard-coded
`min_sms = 68` in Inductor, so it applies to every Jetson Orin in every power mode (§5).
That correction is propagated to `env/README.md`, `STATUS.md` and the task's status block.

Two further defects found while fixing the above, neither caught by the reviewer:
`tegrastats --logfile` **appends** (a re-run silently concatenated two sessions into a
1 172-line log; `collect_platform.py` now unlinks first), and nsys needed
`--capture-range-end=stop` to let the target finish (§6.5).

---

## 11. Status

**T001 is complete — all four steps.** Step 1 (MAXN + `jetson_clocks`) was satisfied after
the human authorised the reboot on 2026-09-21. The authoritative record is
`results/20260921_E0_platform_maxn/`.

Next, per the task-order table in `docs/01_measurement_plan.md`: **T002 (E3, π0 eager)**.
Its main setup risk is environment rather than measurement — openpi pins Python ≥ 3.11 and
torch 2.7.1, while the Jetson CUDA wheels give Python 3.10 and torch 2.8.0, so it will need
`--no-deps` installation and recorded deviations (`docs/00_context.md` anticipates this).
