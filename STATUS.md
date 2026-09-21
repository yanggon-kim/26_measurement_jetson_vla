# STATUS — on-device agent

*Maintained by the on-device agent. The research side reads it and never edits it.*

## Device is at MAXN — measurement can proceed

The human rebooted into **MAXN (id 0)** on 2026-09-21 and it is persisted
(`/var/lib/nvpmodel/status` = `pmode:0000`). `jetson_clocks` + `--fan` are applied and are
now run **before every capture**, per protocol §1. CUDA reports **16 SMs**, `TPC_PG_MASK` 0,
GPU locked at **1 300.5 MHz** (confirmed under load), 12 CPU cores, EMC 3 199 MHz.

The authoritative E0 record is **`results/20260921_E0_platform_maxn/`**.
`results/20260921_E0_platform/` is the earlier MODE_30W run, kept only for comparison; **no
number in it characterises the Orin.**

**Three things do not survive a reboot** and must be re-applied at the start of every session:
```bash
sudo jetson_clocks && sudo jetson_clocks --fan
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'   # ncu counters
```
plus the sudo password itself — **passwordless `sudo` is still not configured**
(`AGENTS.md` lists it as a prerequisite). The user is in group `sudo` but a password is
required; it was supplied interactively and used through a `SUDO_ASKPASS` helper kept
outside the repository, which `/tmp` clearing at reboot removes. A `NOPASSWD` sudoers entry
for `nvpmodel`, `jetson_clocks`, `tegrastats`, `ncu` and reading `/sys/kernel/debug` would
remove this friction.

Note for E7's power-mode sweep: **every mode change that alters TPC gating needs a reboot**,
and declining the prompt does not stage it (`nvpmodel` reports `optMask is 1, no request for
power mode` and leaves the persisted mode untouched). Answer `YES`, or use `--force`.

## Device

| | |
|---|---|
| board | NVIDIA Jetson AGX Orin Developer Kit (`tegra234`), host `cocojet` |
| L4T / JetPack | **R36.5.0** / **6.2.3+b81** — JetPack 6 prerequisite **met**, no reflash needed |
| CUDA / cuDNN / TensorRT | 12.6.68 / 9.3.0.75 / 10.3.0.30 |
| kernel / python | 5.15.185-tegra (`oot`) / 3.10.12 |
| GPU | Orin, cc 8.7 → **`sm_87`** (confirmed by torch *and* `cuobjdump` on the system cuBLAS), 48 max warps/SM |
| **SMs** | **16** at MAXN (`TPC_PG_MASK` 0, 8 of 8 TPCs); was 8 in MODE_30W |
| GPU clock | **1 300.5 MHz** locked and confirmed under load; was 612 MHz in MODE_30W |
| EMC | **3 199 MHz** = LPDDR5-6400, as expected |
| memory / disk | 61 GiB RAM, 8 × 3.8 G zram (no disk swap); NVMe 1.8 T, 1.7 T free |
| idle power (MAXN, clocks locked, 58.6 s @ 10 Hz) | **module power 11 012 mW median** (the four rails of protocol §4, from sysfs); tj 41.9 °C; not throttled. MODE_30W gave 8 522 mW / 39.9 °C. |

## Software stack that works

**torch 2.8.0 + torchvision 0.23.0 + triton 3.4.0**, JetPack CUDA 12.6 wheels (cp310
aarch64) from `https://pypi.jetson-ai-lab.io/jp6/cu126`, in `~/venvs/jetson_base`.
`torch.cuda.is_available()` true, `arch_list = ['sm_87']`, bf16 GEMM correct to 4.2e-3 vs
fp32, int8 `torch._int_mm` works.
**`torch.compile` works in every mode**: inductor `default`, `reduce-overhead`,
`max-autotune`, `max-autotune-no-cudagraphs`, plus `aot_eager` and `eager`
(`results/20260921_E0_platform_maxn/stack_verification.json`).
**But `max-autotune` never uses Triton GEMM templates on an Orin** — see point 5 below.
Trap to remember: `pip install torch==2.8.0` with `--extra-index-url https://pypi.org/simple`
installs PyPI's **CPU-only** `2.8.0+cpu`, because pip prefers a manylinux wheel over the
index's `linux_aarch64` one. Install the three CUDA wheels by full URL with `--no-deps`
(`env/setup.sh`).

Profilers: **`ncu` 2024.3.1.0**, **`nsys` 2024.5.4.34**, `cuobjdump`/`nvdisasm` 12.6.68 —
all shipped with JetPack, nothing to install. `--set` values offered: `basic`, `detailed`,
**`full`**, `nvlink`, `pmsampling`, `roofline` (protocol expected ≥ basic/roofline/full ✓).

## Which ncu metrics this Orin exposes

Full verdicts: `results/20260921_E0_platform_maxn/ncu/metric_presence.csv` — **52 present,
14 present-but-zero, 9 absent**; every exported column: `ncu/columns.txt` (1 117 raw
columns). `present_zero` is the "empty" case T001 asks about, made precise: the column
exists and every value is exactly 0.

**Present with non-zero values:** all protocol §3.3 occupancy metrics (`SM Busy`,
`SM Active Cycles`, `Achieved Occupancy`, `Theoretical Occupancy`,
`Achieved Active Warps Per SM`, `sm__warps_active.avg.pct_of_peak_sustained_active`);
`Duration`, `Compute (SM) Throughput`, `Memory Throughput`, `Mem Busy`, `L2 Hit Rate`,
both `Executed Ipc`; the LSU global/shared counters this kernel exercises; all launch /
occupancy-limit metrics; pipe utilisation for `lsu` (21.6 %), `alu` (13.5 %), `adu`,
`cbu`, `fma`; **15 of the 18 warp-stall reasons**.

**Present but zero on the toy kernel (14):** mostly legitimate for this kernel — the local
and atomic LSU counters, shared-atom wavefronts, the `xu` pipe, `L1/TEX Hit Rate`, and the
`membar` / `sleeping` / `tex_throttle` stalls. **Three are not legitimate: every
`dram__*`.** A zero on one kernel is not evidence a counter works; re-check per kernel.

**Absent (9):** `DRAM Throughput`, `DRAM Utilization`, `dram__bytes{,_read,_write}.sum`,
`sm__inst_executed_pipe_fp16...active`, `sm__inst_executed_pipe_{fma,alu}...elapsed`,
`gpc__cycles_active.avg.per_warp`.

**Five things later tasks must do differently:**
1. **No DRAM counters — established three ways.** `ncu --query-metrics` lists **4 426
   metrics on `Orin (GA10B)` and none is `dram__*`** (`env/tool_help/ncu_query_metrics.txt`);
   an explicit `--metrics dram__bytes.sum,...` returns **`n/a`** (`raw/ncu_dram_probe.log`);
   and the `--set full` placeholder columns all read 0. ncu's `Memory Throughput` here
   equals `lts__throughput` to two decimal places — it is **L2**, not DRAM. Off-chip *read*
   bytes must come from `lts__t_sectors_*_sysmem_*_lookup_miss.sum × 32 B`; on the toy
   kernel at MAXN that gave **8.4931 MB against 8.4931 MB** of analytically compulsory
   reads — **ratio 1.0000** (the MODE_30W run gave 1.0032). The **write** path is still not
   validated: the kernel's 0.42 MB output never appears as a sysmem miss.
2. **Stall reasons are on the `raw` page only**, as
   `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`. The `details` page has
   a `Warp State Statistics` section but no `Stall …` rows.
3. **`ncu` needs root here.** `perf_event_paranoid=1` (protocol §12) is not sufficient;
   use `sudo $(which ncu) ...` (plain `sudo ncu` fails — `secure_path`), then `chown` the
   `.ncu-rep` before exporting.
4. **Check ncu's clock against the real one every time.** `--clock-control base` pins the
   SM clock. At MAXN with `jetson_clocks` applied the two agree (ncu 1.29 GHz vs 1 300.5 MHz
   measured, `Duration` 57.09 µs vs nsys 54.6 µs, +4.5 %). In the MODE_30W run, taken with
   unlocked clocks, they did not (305.70 MHz vs 612 MHz). ncu `Duration` is still never a
   latency (protocol §3.2) — but lock the clocks first and the gap is small.
5. **`max-autotune` cannot use Triton GEMM templates on any Jetson Orin.**
   `torch/_inductor/utils.py` hard-codes `min_sms = 68` (`# 3080`) for CUDA; Orin has 16, so
   the check fails **in every power mode** — this was initially mis-attributed to MODE_30W
   and is disproved by the MAXN run (`raw/verify_stack.stdout.log` line 3). `max-autotune`
   falls back to ATen/cuBLAS; its autotune log compares only `addmm` vs `bias_addmm`.
   **E4 must present "max-autotune" here as "default + CUDA graphs + ATen autotuning".**

Also: **`tegrastats` does not print `VDDQ_VDD2_1V8AO`** on this carrier board — the memory
rail is only in sysfs at `/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon4/in2_*`.
`bench/collect_platform.py` now samples all four rails there at 10 Hz
(`power_idle_sysfs.csv`), so protocol §4 module power is available. That channel reads a
**constant 5 040 mV bus**, not 1.8 V — see question 2. `tegrastats` also **appends** to
`--logfile`; delete the file first. And nsys's `--capture-range=cudaProfilerApi` kills the
target at `cudaProfilerStop` unless `--capture-range-end=stop` is passed.

## Done

* **T001 (E0) — bring-up and platform record. COMPLETE, all four steps.** Step 1 (MAXN +
  `jetson_clocks`) was satisfied after the human authorised the reboot on 2026-09-21.
  Result: **`results/20260921_E0_platform_maxn/`** (authoritative), with
  `results/20260921_E0_platform/` kept as the MODE_30W comparison ·
  Report: `reports/E0_platform.md` ·
  Environment: `env/README.md`, `env/setup.sh`, `env/requirements_base.txt`,
  `env/tool_help/` (12 files) · Harness: `bench/common.py`, `verify_stack.py`,
  `ncu_toy.py`, `ncu_columns.py`, `hw_constants.py`, `collect_platform.py`,
  `gpu_clock_probe.py`, `finalize_raw.py`, `make_meta_e0.py`.
  An independent sub-agent pass (protocol §11) reviewed the run; it found one fabricated
  process claim in the report and nine smaller defects, all fixed and listed in
  `reports/E0_platform.md` §9.

### Carried into the MAXN re-run (accepted, not yet done)
* Apply `sudo jetson_clocks` **before** every capture, not after, and record
  `devfreq cur_freq` alongside each one (`bench/gpu_clock_probe.py`). In this run the two
  toy profiler captures were taken with `schedutil` free to scale 306–612 MHz, and that
  clock was not recorded.
* Re-validate the L2 sysmem-miss substitute for off-chip **write** traffic, and against the
  measured bandwidth peak from E1. Only the read path is validated so far.
* Confirm 16 SMs and re-check `max-autotune` (it is refused at 8 SMs).

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

After T001 (2026-09-21):

```
du -sh results/   1.5M
du -sh .git       1.9M
git count-objects -vH:  count: 176   size-pack: 62.49 KiB
```

Five largest committed files — all far under the 50 MB limit, nothing split:

| bytes | file |
|---|---|
| 794 544 | `env/tool_help/ncu_query_metrics.txt` (every metric ncu offers on this GPU) |
| 536 576 | `results/20260921_E0_platform/raw/nsys_toy.sqlite` |
| 190 947 | `results/20260921_E0_platform/raw/tegrastats_idle.log` |
| 108 402 | `results/20260921_E0_platform/raw/nsys_toy.nsys-rep` |
| 92 396 | `results/20260921_E0_platform/raw/ncu_toy.ncu-rep.zst` (7.07 MB → 0.09 MB, `zstd -19`) |

`find results -size +50M` is empty.

### Size budget calibration from the toy captures (T001 step 3c)

| file | capture | uncompressed | `zstd -19 -T0` | ratio |
|---|---|---|---|---|
| `nsys_toy.nsys-rep` | 3 inferences of one `F.linear`, 6 kernels | 108 402 B | 41 708 B | 2.6× |
| `nsys_toy.sqlite` | derived from the above | 536 576 B | 53 989 B | 9.9× |
| `ncu_toy.ncu-rep` | **1 kernel**, `--set full`, 30 replay passes | **7 072 411 B** | **92 396 B** | **76.5×** |

Only the `.ncu-rep` was committed compressed (protocol §6b: > 1 MB); the other two are
under 1 MB and are committed as-is.

**What this implies for later tasks.** The ncu report is ≈ **7 MB per captured kernel**
uncompressed at `--set full`, and compresses ~76× because it is mostly sparse. A π0
`ae_step[0]/` capture is expected to hold several hundred kernels → **order 1–2 GB
uncompressed, order 20–30 MB as `zstd -19`** — committable, but only just, and the
uncompressed intermediate needs room in `scratch/`. Budget `-c` limits accordingly and
check before committing. `bench/finalize_raw.py` compresses above 1 MB and splits above
50 MB with a verified `REASSEMBLE.txt`.

The nsys side is cheap by comparison (≈ 18 KB per kernel uncompressed here, and dominated
by fixed overhead), so the 1–3 inference capture window of protocol §3.1 is not the
constraint; ncu is.
