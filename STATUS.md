# STATUS — on-device agent

*Maintained by the on-device agent. The research side reads it and never edits it.*

## ⚠ Needs the human before measurement can start

**The Jetson is in `MODE_30W`, not MAXN, and switching to MAXN requires a reboot.**
`sudo nvpmodel -m 0` answers `NVPM WARN: Reboot required for changing to this power mode: 0`
and prompts for confirmation; `AGENTS.md` forbids confirming it, so the prompt was declined
and the mode is unchanged. In `MODE_30W` **4 of the 8 GPU TPCs are power-gated (8 of 16 SMs)
and the GPU is capped at 612 MHz** — `torch.cuda.get_device_properties(0).multi_processor_count`
reports 8, and Inductor refuses `max-autotune` GEMM templates with *"Not enough SMs"*.
MODE_50W is not a substitute (8 TPCs but 816 MHz).

**Action:** `sudo nvpmodel -m 0`, confirm the reboot, then `sudo jetson_clocks`.
Afterwards re-run (≈ 5 min, all scripts exist):
`python bench/collect_platform.py --outdir results/<date>_E0_platform_maxn --sudo "sudo -A" --idle-seconds 60`
and `python bench/hw_constants.py --out results/<date>_E0_platform_maxn/hw_constants.json --ncu-raw-csv ...`,
and confirm 16 SMs. **T002 should not start before that.**

Second prerequisite gap: **passwordless `sudo` is not configured** (`AGENTS.md` lists it).
The user is in group `sudo` but a password is required. The human supplied the password for
this session only, used through a `SUDO_ASKPASS` helper kept outside the repository — it is
not persisted and a future session will need it again, or a `NOPASSWD` sudoers entry for
`nvpmodel`, `jetson_clocks`, `tegrastats`, `ncu` and reading `/sys/kernel/debug`.

## Device

| | |
|---|---|
| board | NVIDIA Jetson AGX Orin Developer Kit (`tegra234`), host `cocojet` |
| L4T / JetPack | **R36.5.0** / **6.2.3+b81** — JetPack 6 prerequisite **met**, no reflash needed |
| CUDA / cuDNN / TensorRT | 12.6.68 / 9.3.0.75 / 10.3.0.30 |
| kernel / python | 5.15.185-tegra (`oot`) / 3.10.12 |
| GPU | Orin, cc 8.7 → **`sm_87`** (confirmed by torch *and* `cuobjdump` on the system cuBLAS), 48 max warps/SM |
| **SMs** | **8 of 16 — MODE_30W power-gates 4 of 8 TPCs** |
| EMC | **3 199 MHz** = LPDDR5-6400, as expected |
| memory / disk | 61 GiB RAM, 8 × 3.8 G zram (no disk swap); NVMe 1.8 T, 1.7 T free |
| idle power (MODE_30W, 60 s, 10 Hz) | 6.92 W median over the three rails tegrastats prints; tj 46 °C; not throttled |

## Software stack that works

**torch 2.8.0 + torchvision 0.23.0 + triton 3.4.0**, JetPack CUDA 12.6 wheels (cp310
aarch64) from `https://pypi.jetson-ai-lab.io/jp6/cu126`, in `~/venvs/jetson_base`.
`torch.cuda.is_available()` true, `arch_list = ['sm_87']`, bf16 GEMM correct to 4.2e-3 vs
fp32, int8 `torch._int_mm` works.
**`torch.compile` works in every mode**: inductor `default`, `reduce-overhead`,
`max-autotune`, `max-autotune-no-cudagraphs`, plus `aot_eager` and `eager`
(`results/20260921_E0_platform/stack_verification.json`).
Trap to remember: `pip install torch==2.8.0` with `--extra-index-url https://pypi.org/simple`
installs PyPI's **CPU-only** `2.8.0+cpu`, because pip prefers a manylinux wheel over the
index's `linux_aarch64` one. Install the three CUDA wheels by full URL with `--no-deps`
(`env/setup.sh`).

Profilers: **`ncu` 2024.3.1.0**, **`nsys` 2024.5.4.34**, `cuobjdump`/`nvdisasm` 12.6.68 —
all shipped with JetPack, nothing to install. `--set` values offered: `basic`, `detailed`,
**`full`**, `nvlink`, `pmsampling`, `roofline` (protocol expected ≥ basic/roofline/full ✓).

## Which ncu metrics this Orin exposes

Full verdicts: `results/20260921_E0_platform/ncu/metric_presence.csv` (**66 present, 9
absent**); every exported column: `ncu/columns.txt` (1 117 raw columns).

**Present and populated:** all protocol §3.3 occupancy metrics (`SM Busy`,
`SM Active Cycles`, `Achieved Occupancy`, `Theoretical Occupancy`,
`Achieved Active Warps Per SM`, `sm__warps_active.avg.pct_of_peak_sustained_active`);
`Duration`, `Compute (SM) Throughput`, `Memory Throughput`, `Mem Busy`, `L2 Hit Rate`,
`L1/TEX Hit Rate`, both `Executed Ipc`; all nine LSU request counters; all launch /
occupancy-limit metrics; pipe utilisation for `lsu, fma, alu, xu, adu, cbu`;
**all 18 warp-stall reasons**.

**Absent (9):** `DRAM Throughput`, `DRAM Utilization`, `dram__bytes{,_read,_write}.sum`,
`sm__inst_executed_pipe_fp16...active`, `sm__inst_executed_pipe_{fma,alu}...elapsed`,
`gpc__cycles_active.avg.per_warp`.

**Three things later tasks must do differently:**
1. **No DRAM counters.** Every exported `dram__*` reads **0**, and ncu's `Memory Throughput`
   here is byte-identical to `lts__throughput` — it is **L2**, not DRAM. Off-chip bytes must
   come from `lts__t_sectors_aperture_sysmem_*.sum × 32 B`; on the toy kernel that gave
   8.52 MB against 8.49 MB of analytically compulsory reads (**ratio 1.004**).
2. **Stall reasons are on the `raw` page only**, as
   `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`. The `details` page has
   a `Warp State Statistics` section but no `Stall …` rows.
3. **`ncu` needs root here.** `perf_event_paranoid=1` (protocol §12) is not sufficient;
   use `sudo $(which ncu) ...` (plain `sudo ncu` fails — `secure_path`), then `chown` the
   `.ncu-rep` before exporting.

Also: **`tegrastats` does not print `VDDQ_VDD2_1V8AO`** on this carrier board — the memory
rail is only in sysfs at `/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon4/in2_*`
(`bench/common.py:ina3221_rails()` reads it). And nsys's
`--capture-range=cudaProfilerApi` kills the target at `cudaProfilerStop` unless
`--capture-range-end=stop` is passed.

## Done

* **T001 (E0) — bring-up and platform record.** Steps 2, 3 and 4 complete; step 1 (MAXN)
  blocked on a human reboot, see the top of this file.
  Result: `results/20260921_E0_platform/` · Report: `reports/E0_platform.md` ·
  Environment: `env/README.md`, `env/setup.sh`, `env/requirements_base.txt`,
  `env/tool_help/` (11 files) · Harness: `bench/common.py`, `verify_stack.py`,
  `ncu_toy.py`, `ncu_columns.py`, `hw_constants.py`, `collect_platform.py`,
  `finalize_raw.py`, `make_meta_e0.py`.

## In progress

(none)

## Blocked

* **T001 step 1 (MAXN + `jetson_clocks`)** — needs a human-authorised reboot. See the top.
* **Everything downstream of it.** T002 (π0 eager latency) would produce numbers at half the
  SMs and half the GPU clock, so it should wait for the reboot.

## Questions for the research side

1. **DRAM traffic.** `docs/02_protocol.md` §3.3 and E6 assume ncu's `DRAM Throughput` /
   `DRAM Utilization`. They do not exist on Orin (every `dram__*` counter reads 0). We
   propose `lts__t_sectors_aperture_sysmem_*.sum × 32 B` as the measured off-chip byte count
   (validated to 0.4 % on a kernel with known traffic) and will label it
   *measured (L2 sysmem-miss sectors)*. Please confirm that is acceptable for the roofline
   points of E6, and note that ncu's `Memory Throughput` on this device must be read as
   **L2** throughput wherever it appears.
2. **Memory-rail power.** `VDDQ_VDD2_1V8AO` exists only in sysfs, and the bus voltage its
   INA3221 channel reports is ≈ 5 V, not 1.8 V. Before E7 quotes memory power we will have
   to establish the correct scaling; if the research side knows the intended interpretation
   of this rail on the AGX Orin devkit carrier, that would save a calibration step.
3. **Power-mode plan.** Since MAXN needs a reboot on this board, so does every return from a
   capped mode. E7's power-mode sweep therefore cannot be done inside one session: it will
   need one reboot per mode, each authorised by the human. Is that acceptable, or should E7
   be limited to MAXN plus whichever capped modes can be entered without a reboot?

## Repository size (protocol §6f)

Reported after each task; see the commit for T001.
