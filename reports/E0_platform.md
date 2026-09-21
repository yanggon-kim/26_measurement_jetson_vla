# E0 — Platform and toolchain record (task T001)

**Run:** `results/20260921_E0_platform/` · **date:** 2026-09-21 (UTC) · **bench commit:** see
`meta.json.bench_commit` · **power mode: `MODE_30W` (id 2), *not* MAXN — see §3.**

E0 produces no latency, bandwidth or power result that characterises the Orin. It answers
one question: *does the stack this project needs exist and work on this machine, and what can
its tools actually measure?* The answer is **yes, with four qualifications** that change how
E1–E10 must be run. They are §3 and §5 below.

---

## 1. The answer

### 1.1 Prerequisites (`AGENTS.md`)

| prerequisite | status | evidence |
|---|---|---|
| JetPack 6.x, L4T r36.x, CUDA 12.x, Python 3.10 | **met** — JetPack 6.2.3+b81, L4T **R36.5.0**, CUDA 12.6.68, Python 3.10.12 | `platform.json.device_state.nv_tegra_release`, `.dpkg_jetpack`, `.nvcc_version` |
| Passwordless `sudo` for `nvpmodel`/`jetson_clocks`/`tegrastats`/`/sys/kernel/debug` | **NOT met** | `sudo -n` → "a password is required". The human supplied the password for this session only (used via a `SUDO_ASKPASS` helper outside the repository). `tegrastats` needs no sudo here. |
| git identity and push access, network to GitHub / PyPI / Jetson index | **met** | `git ls-remote` succeeded; the T001 claim commit pushed; the Jetson wheel index, PyPI and `raw.githubusercontent.com` all reachable |
| ≥ 150 GB free NVMe | **met** — 1.7 T free of 1.8 T | `platform.json.device_state.df_h` |
| `ncu` installed; `nsys` present | **met** — `ncu` 2024.3.1.0, `nsys` 2024.5.4.34, plus `cuobjdump`/`nvdisasm`/`c++filt` from CUDA 12.6.68 | `env/tool_help/` |

### 1.2 Platform record (protocol §1)

| quantity | value | source |
|---|---|---|
| board | NVIDIA Jetson AGX Orin Developer Kit, SoC `tegra234` | `/proc/device-tree/model` |
| L4T / JetPack | R36.5.0 (GCID 43688277) / 6.2.3+b81 | `/etc/nv_tegra_release`, `dpkg -l` |
| CUDA / cuDNN / TensorRT | 12.6.68 / 9.3.0.75 / 10.3.0.30 | `nvcc`, `dpkg -l` |
| kernel | 5.15.185-tegra (`oot`) | `uname -a` |
| GPU | Orin, cc **8.7**, arch **`sm_87`**, 48 max warps/SM | `hw_constants.json` |
| **SMs reported** | **8** (half the Orin's 16 — see §3) | `torch.cuda.get_device_properties(0).multi_processor_count` |
| GPU clock | locked 612 000 000 Hz (capped by the power mode) | `devfreq_dev/max_freq`, `jetson_clocks --show` |
| **EMC clock** | **3 199 000 000 Hz = 3 199 MHz** → LPDDR5-6400 **as expected** | `/sys/kernel/debug/bpmp/debug/clk/emc/rate` |
| RAM / swap | 61 GiB; 8 × 3.8 G **zram** only, no disk swap | `free -h`, `swapon --show` |
| storage | NVMe `/dev/nvme0n1p1` 1.8 T, 82 G used | `df -h` |
| `nvidia-smi` | present and functional on this JetPack | `platform.json` |

### 1.3 Idle power and thermals — **measured**, 60 s at 10 Hz, MODE_30W, clocks unlocked

`results/20260921_E0_platform/power_idle.csv` (n = 586 samples per rail).

| rail | median mW | mean mW | min | max |
|---|---|---|---|---|
| `VDD_GPU_SOC` | 2 789 | 2 883 | 2 789 | 3 188 |
| `VDD_CPU_CV` | 0 | 8.8 | 0 | 797 |
| `VIN_SYS_5V0` | 4 126 | 4 094 | 4 025 | 4 428 |
| **sum of the 3 rails tegrastats prints** | **6 915** | 6 985 | — | — |
| `VDDQ_VDD2_1V8AO` (memory) | **not in tegrastats** — see §5.3 | | | |
| `tj` | 46.0 °C | 46.0 | 45.8 | 46.2 |
| `GR3D_FREQ` | 0 % throughout (idle) | | | |

`throttled: false`. No clock fell below its configured value and tj never exceeded 46.2 °C.
This is *idle at MODE_30W*; it is **not** the E7 idle baseline, which must be taken at MAXN.

### 1.4 Software stack — **works**

`torch 2.8.0` + `torchvision 0.23.0` + `triton 3.4.0`, NVIDIA/Jetson CUDA 12.6 wheels for
cp310 aarch64, in `~/venvs/jetson_base`. `torch.version.cuda = 12.6`,
`torch.cuda.get_arch_list() = ['sm_87']`, `torch.cuda.is_available() = True`.

* bf16 GEMM 2048³ runs, finite, max relative error **4.2e-3** against an fp32 reference —
  the expected bf16 mantissa error, not a fault.
* bf16, fp16, fp32 matmul and int8 (`torch._int_mm` → int32) all run.
* **`torch.compile` compiles and runs a small bf16 MLP in every mode tried**:
  `inductor` (no mode / `default` / `reduce-overhead` / `max-autotune` /
  `max-autotune-no-cudagraphs`), `aot_eager`, `eager`. Compile + first call 0.07–3.05 s;
  outputs match eager to 4.7e-3 (Inductor's bf16 `addmm`+GeLU fusion). Details and per-mode
  timings: `stack_verification.json`, table in `env/README.md` §4.

**The backend/mode that works for E4 is `inductor` with all four protocol modes.** One
caveat: `max-autotune` logged `Not enough SMs to use max_autotune_gemm mode`, which is a
consequence of the 8-SM power mode (§3), not of the stack. Re-check at MAXN.

### 1.5 Profiler toolchain — **works end to end**

* **nsys.** `nsys profile -t cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi
  -f true` on `bench/ncu_toy.py` produced a 108 KB `.nsys-rep`; **all seven** protocol §3.1
  report types plus `cuda_gpu_trace` exported to CSV (`nsys/toy_*.csv`).
  NVTX attribution works: the GEMM is correctly assigned to range `:toy`
  (`nsys/toy_nvtx_kern_sum.csv`). `-f true` parses; `--force-overwrite` was not needed.
* **ncu.** `--set full --nvtx --nvtx-include "toy/" --profile-from-start off` captured
  **exactly one** kernel (the `toy` range; the sibling `toy_alias` range was correctly
  excluded) in **9 s**, 30 replay passes, a **7.07 MB** report (**0.09 MB** as `zstd -19`).
  Both pages exported: `ncu/toy_details.csv` (80 metric rows) and `ncu/toy_raw.csv`
  (**1 117** columns). Inventory: `ncu/columns.txt`; per-metric verdicts:
  `ncu/metric_presence.csv`.

`--set` values this Orin offers — the protocol expected ≥ `basic`, `roofline`, `full`; **all
three exist**, plus `detailed`, `nvlink`, `pmsampling`. `full` = 614 estimated metrics and is
the only set with Occupancy + WarpStateStats + SchedulerStats together. Exact flag spellings
(`--nvtx-include`, `-k/--kernel-name`, `-c/--launch-count`, `-s/--launch-skip`,
`--profile-from-start off`, `-i/--import`, `--csv`, `--page`) are all as the protocol assumes
and are tabulated in `env/README.md` §5.

### 1.6 `hw_constants.json` (protocol §8)

| constant | value | how |
|---|---|---|
| `multi_processor_count` | **8** *(power-mode dependent — 16 expected at MAXN)* | torch device properties |
| compute capability | 8.7 | torch device properties |
| `gpu_arch` | **`sm_87` — confirmed** | torch arch list; `cuobjdump --list-text` on `/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcublas.so.12.6.1.4`, which embeds `sm_53…sm_90` including `sm_87` |
| max warps per SM | 48 | `max_threads_per_multi_processor` 1536 ÷ 32 |
| L2 cache | from device properties, recorded | `hw_constants.json` |
| peak GB/s, TFLOP/s, ridge point | **null — not measured** | T005/E1 fills them. The spec expectations (204.8 GB/s, 42 TFLOP/s, ridge ≈ 205) are stored in a separate `expected_from_spec_do_not_use_as_measurement` block so they can never be mistaken for measurements. |

---

## 2. Conditions

Batch 1, bf16, fixed seed 0. No desktop session load; GR3D_FREQ 0 % before each capture.
Power mode `MODE_30W`, `jetson_clocks` **not** applied during the idle log or either toy
profiler capture (it was applied afterwards, to verify it works —
`raw/jetson_clocks_show_locked_30W.txt`). Timing and profiling were never mixed: the stack
verification, the nsys run and the ncu run are three separate process launches.
`perf_event_paranoid` set to 1 for the session.

---

## 3. Anomaly 1 (blocking): the device is not at MAXN, and MAXN needs a reboot

Protocol §1 requires **MAXN + `jetson_clocks`** for every experiment. This board boots to its
configured default `MODE_30W` (`PM_CONFIG DEFAULT=2`). `sudo nvpmodel -m 0` answered:

```
NVPM WARN: Reboot required for changing to this power mode: 0
NVPM WARN: DO YOU WANT TO REBOOT NOW? enter YES/yes to confirm:
```

`AGENTS.md` forbids confirming a reboot prompt, so it was declined and **the mode did not
change** (`/var/lib/nvpmodel/status` still `pmode:0002`). `nvpmodel --force` exists but
*"automatically reboot[s] if required"*, so it was not used.

What MODE_30W costs, from `nvpmodel_modes.csv` and `jetson_clocks --show`:

| | MODE_30W (now) | MAXN (required) |
|---|---|---|
| `TPC_PG_MASK` | 240 → **4 of 8 TPCs active** | 0 → 8 of 8 |
| SMs visible to CUDA | **8** | 16 |
| GPU clock cap | **612 MHz** | uncapped (1 300 MHz class) |
| CPU cores online | 8 of 12 | 12 |
| EMC | uncapped, 3 199 MHz | uncapped |

That is roughly **half the SMs at half the clock**. Every latency, occupancy, bandwidth,
FLOP/s and power number would be wrong by a large and *non-uniform* factor. Nothing in this
run is offered as a characterisation of the Orin.

**MODE_50W (id 3) would also give 8 of 8 TPCs** but caps the GPU at 816 MHz, and is
therefore not a substitute for MAXN either.

**What the human must do:** `sudo nvpmodel -m 0` and confirm the reboot (or reboot after
setting it). Then T001 §1 is re-run and a second, authoritative
`results/<date>_E0_platform_maxn/` is written. **T002 must not start before then.**

---

## 4. Anomaly 2: `ncu` requires root on this device

Protocol §12's fix for `ERR_NVGPUCTRPERM` —
`sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'` — was applied and is **not
sufficient** here. As an unprivileged user, even `ncu --list-sets` fails with

```
==WARNING== Insufficient privileges to launch app for profiling. Launch app with root privileges
```

**Substitution adopted** (recorded in `meta.json.notes` per protocol §3): every `ncu`
invocation is `sudo $(which ncu) ...`. `sudo ncu` alone fails — sudo's `secure_path` does not
contain `/usr/local/cuda/bin`. The resulting `.ncu-rep` is owned by root and must be
`chown`ed before the (unprivileged) `--import` export.

---

## 5. Anomaly 3 (important for E5/E6/E7): what ncu can and cannot measure here

Full verdict table: `results/20260921_E0_platform/ncu/metric_presence.csv`
(66 present, 9 absent). Inventory of every exported column: `ncu/columns.txt`.

### 5.1 ⚠ There is no DRAM counter on this Tegra

Every `dram__*` metric that is exported reads **exactly 0**:

| metric | value |
|---|---|
| `dram__cycles_active.sum` | 0 |
| `dram__cycles_elapsed.sum` | 0 |
| `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` | 0 |
| `dram__bytes.sum`, `dram__bytes_read.sum`, `dram__bytes_write.sum` | not exported at all |

and consequently `DRAM Throughput` / `DRAM Utilization` are **absent** from the `details`
page. On Orin the LPDDR5 sits behind the GPU L2 as *sysmem* and its performance monitors are
not exposed to ncu.

**This matters beyond a missing column.** ncu's `Memory Throughput` for the toy kernel is
**76.55 %**, which is bit-identical to `lts__throughput.avg.pct_of_peak_sustained_elapsed` —
i.e. **on this device "Memory Throughput" is L2 throughput, not DRAM utilisation.** E5 and E6
would silently report an L2 number as a DRAM number.

**Substitute, validated.** Off-chip traffic is recoverable from the L2 sysmem-miss sectors
(32 B per sector), on a kernel whose traffic is known analytically:

| quantity | value |
|---|---|
| `lts__t_sectors_aperture_sysmem_lookup_miss.sum` | 266 491 → 8.53 MB |
| reads only (`..._srcunit_tex_aperture_sysmem_op_read_lookup_miss.sum`) | 266 264 → **8.52 MB** |
| analytic compulsory reads: W (4096·1024·2 B) + x (51·1024·2 B) | **8.49 MB** |
| **measured ÷ analytic** | **1.004** |

(The 0.42 MB output write does not appear as a sysmem miss — it is still resident in the
write-back L2 when the kernel ends, which is itself consistent.)

**Rule for E5/E6:** off-chip bytes = `lts__t_sectors_aperture_sysmem_*.sum × 32 B`, labelled
*measured (L2 sysmem-miss sectors)*. Never quote `dram__*`, and never read
`Memory Throughput` as DRAM utilisation on this device.

### 5.2 Metric names that differ from the protocol's

* **Warp-stall reasons are not on the `details` page.** That page *has* a
  `Warp State Statistics` section but exports no `Stall …` display metric. All **18** stall
  reasons are on the **`raw`** page as
  `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` for
  `barrier, branch_resolving, dispatch_stall, drain, imc_miss, lg_throttle, long_scoreboard,
  math_pipe_throttle, membar, mio_throttle, misc, no_instruction, not_selected, selected,
  short_scoreboard, sleeping, tex_throttle, wait`. **E5/E5b must parse the raw page.**
* `sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active` does not exist; bf16/fp16
  tensor-core work appears in `sm__pipe_tensor_op_hmma_cycles_active.*`.
* `sm__inst_executed_pipe_{fma,alu}.avg.pct_of_peak_sustained_elapsed` are absent; the
  `..._active` variants exist and are populated (as are `lsu`, `xu`, `adu`, `cbu`).
* `gpc__cycles_active.avg.per_warp` is absent; derive it if E5b needs it.

Everything else protocol §3.3 asks for is present and populated: all five occupancy metrics,
`Duration`, `Compute (SM) Throughput`, `Memory Throughput`, `Mem Busy`, `L2 Hit Rate`,
`L1/TEX Hit Rate`, both `Executed Ipc`, all nine LSU request counters, and every launch /
occupancy-limit metric.

### 5.3 ⚠ `tegrastats` does not print the memory rail

`tegrastats_rails.csv`:

| rail (protocol §4) | in `tegrastats` | in sysfs INA3221 |
|---|---|---|
| `VDD_GPU_SOC`, `VDD_CPU_CV`, `VIN_SYS_5V0` | yes | `1-0040/hwmon/hwmon3/in{1,2,3}_*` |
| **`VDDQ_VDD2_1V8AO`** | **no** | **`1-0041/hwmon/hwmon4/in2_*`** |

This carrier board's `tegrastats` prints only the three rails of the first INA3221. The
memory rail — the one this project cares about most — is on the second INA3221 and must be
read from sysfs. `bench/common.py:ina3221_rails()` does this
(`mW = in<i>_input[mV] × curr<i>_input[mA] / 1000`); for `VDD_GPU_SOC` it agreed with
tegrastats at idle (3.19 W vs 2.79 W, different sensor averaging).
**Open item for E7:** the bus voltage reported for `VDDQ_VDD2_1V8AO` is ~5 V, not 1.8 V, so
the scaling of that channel must be confirmed before any memory-power number is quoted.

### 5.4 ncu locks the clocks — its durations are not latencies

`--clock-control` defaults to `base`. In the toy capture ncu reported
`SM Frequency 305.70 MHz` and `Duration 195.74 µs`, while nsys measured the same kernel at
**186 µs** at the (already capped) 612 MHz. Protocol §3.2 already forbids taking latency
from ncu; this is the mechanism, and it is worth stating in every E5 report.

### 5.5 nsys kills the target at `cudaProfilerStop`

With `--capture-range=cudaProfilerApi`, nsys 2024.5.4 defaults to
`--capture-range-end=stop-shutdown`: the process is SIGTERMed (exit 143) and anything the
script prints after the window is lost. Add `--capture-range-end=stop` when the script must
run to completion.

---

## 6. One observation, explicitly *not* a result

In the nsys toy capture, six back-to-back `F.linear` launches showed `cudaLaunchKernel`
median **25.6 µs** (min 22.3 µs) of CPU time and a **~3.5–4 µs** gap between consecutive
kernel executions (`nsys/toy_cuda_gpu_trace.csv`, `toy_cuda_kern_exec_sum.csv`). The project
currently *assumes* 20 µs of per-kernel launch overhead (`docs/00_context.md`).
**These numbers are from a profiled run, at MODE_30W, with n = 6 — they satisfy none of the
protocol's timing rules (§2) and must not be used.** E2 measures launch overhead properly.

---

## 7. Separation of evidence

**Measured on this device:** every value in §1.2, §1.3, §1.6; the ncu metric presence /
absence table and every metric value quoted in §5; the nsys report coverage; the
`torch.compile` success matrix and compile times; the toy kernel's 186 µs (nsys) and
195.74 µs (ncu, clocks locked to base).

**Derived:** the 8.49 MB analytic compulsory-read figure in §5.1 (from tensor shapes) and the
ratio 1.004; the "16 SMs at MAXN" figure (from `TPC_PG_MASK = 0` and 2 SMs per TPC — *not*
observed, because MAXN could not be entered); the sum-of-three-rails module power.

**Not measured:** peak memory bandwidth, peak TFLOP/s per dtype and the ridge point (nulls in
`hw_constants.json`, for T005/E1); anything at MAXN; DRAM traffic via `dram__*` (impossible
here, §5.1); memory-rail power in absolute terms (§5.3); per-kernel launch overhead (§6).

---

## 8. Reproduce

```bash
bash env/setup.sh                                  # apt + venv + JetPack CUDA wheels
python bench/verify_stack.py --out results/<run>/stack_verification.json
python bench/collect_platform.py --outdir results/<run> --sudo "sudo -A" --idle-seconds 60

nsys profile -t cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi -f true \
     -o <scratch>/nsys_toy  $VENV/bin/python bench/ncu_toy.py --iters 3
nsys stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum,\
nvtx_kern_sum,nvtx_gpu_proj_trace,cuda_kern_exec_sum -f csv \
     -o results/<run>/nsys/toy <scratch>/nsys_toy.nsys-rep
nsys stats -r cuda_gpu_trace -f csv -o results/<run>/nsys/toy <scratch>/nsys_toy.nsys-rep

sudo $(which ncu) --set full --nvtx --nvtx-include "toy/" --profile-from-start off \
     -o <scratch>/ncu_toy -f  $VENV/bin/python bench/ncu_toy.py
sudo chown $USER <scratch>/ncu_toy.ncu-rep
ncu --import <scratch>/ncu_toy.ncu-rep --csv --page details > results/<run>/ncu/toy_details.csv
ncu --import <scratch>/ncu_toy.ncu-rep --csv --page raw     > results/<run>/ncu/toy_raw.csv

python bench/ncu_columns.py --details results/<run>/ncu/toy_details.csv \
       --raw results/<run>/ncu/toy_raw.csv --outdir results/<run>/ncu
python bench/hw_constants.py --out results/<run>/hw_constants.json \
       --ncu-raw-csv results/<run>/ncu/toy_raw.csv
python bench/finalize_raw.py --run results/<run> ...
python bench/make_meta_e0.py --run results/<run>
```

Exact invocations, including every `--produced-by` string, are in `meta.json.command` and
`meta.json.raw_files[].produced_by`.

---

## 9. Self-review (protocol §11)

| check | result |
|---|---|
| every requested file exists under the requested name | **yes** — E0 result directory with `hw_constants.json`, `ncu/columns.txt`, toy ncu exports (`ncu/toy_details.csv`, `ncu/toy_raw.csv`), `env/` with `README.md`, `setup.sh`, `requirements_base.txt`, `tool_help/` (11 files) |
| every number in this report traces to a file in `results/` | **yes** — each table cites the file |
| `meta.json` complete per §5 | **yes**; model fields are `null` with a stated reason (E0 runs no model) |
| `raw_files` hashes filled | **yes** — 7 entries in `raw_files.json`, each with sha256, bytes, kind, `produced_by`, `secrets_scan`; the 7.07 MB `.ncu-rep` has `compression` + `original` |
| n, power mode, `throttled` stated | **yes** — n = 586 idle samples; `MODE_30W (id 2)`; `throttled: false` with evidence |
| measured / derived / not-measured separated | **yes** — §7 |
| deviations listed | **yes** — §3 (not MAXN), §4 (ncu needs root), §5 (metric substitutions), and the `power_mode_deviation` field of `meta.json` |
| no forbidden file staged; nothing over 50 MB | **yes** — `find results -size +50M` empty; largest committed file 537 KB; no weights, datasets, venvs or caches |
| secrets scan | **clean** on all 7 raw files and on the committed CSV/JSON/TXT |
| research-side files untouched | **yes** — only `env/ bench/ results/ reports/ STATUS.md` and one appended status block in `tasks/T001_bringup.md` |

**Second pass, starting from the result files rather than from this report** (protocol §11):
re-read `platform.json`, `hw_constants.json`, `ncu/metric_presence.csv`, `power_idle.csv`,
`nsys/toy_*.csv` and `stack_verification.json` and confirmed every figure quoted above,
including the 8-SM count, the 3 199 MHz EMC rate, the 9 absent ncu metrics, the zero
`dram__*` values, the 1.004 sysmem-miss ratio and the 6 915 mW idle median.

**Independent pass:** a sub-agent was given only `tasks/T001_bringup.md`, `docs/02_protocol.md`
and the produced files; its findings are recorded in §10.

---

## 10. Status

**T001 is complete except for its first requirement.** Step 1 ("Set MAXN + `jetson_clocks`")
could not be satisfied without a reboot that only the human may authorise. Steps 2, 3 and 4
are done, and the E0 record exists — but at `MODE_30W`.

Recommended order: human reboots into MAXN → re-run `bench/collect_platform.py` and
`bench/hw_constants.py` into `results/<date>_E0_platform_maxn/` (≈ 5 minutes, all scripts
exist) → confirm 16 SMs and re-check `max-autotune` → then T002.
