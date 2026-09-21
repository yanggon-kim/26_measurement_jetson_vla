# E0 — platform record (T001), 2026-09-21

Bring-up of the Jetson AGX Orin 64GB: JetPack **6.2.3** / L4T **R36.5.0**, CUDA **12.6.68**,
cuDNN 9.3.0, TensorRT 10.3.0, kernel 5.15.185-tegra, Python 3.10.12, 1.7 T free NVMe,
EMC **3 199 MHz** (LPDDR5-6400, as expected). GPU **Orin, cc 8.7 → `sm_87`** (confirmed three
ways: torch arch list, device properties, `cuobjdump --list-text` on the system cuBLAS
`libcublas.so.12.6.1.4`, which contains an `sm_87` cubin).

Stack: **torch 2.8.0 + torchvision 0.23.0 + triton 3.4.0**, JetPack CUDA wheels in
`~/venvs/jetson_base`. `torch.cuda.is_available()` true; bf16 GEMM correct (4.2e-3 vs fp32);
**`torch.compile` works in every mode** — `default`, `reduce-overhead`, `max-autotune`,
`max-autotune-no-cudagraphs`, plus `aot_eager`/`eager`. Profilers `ncu` 2024.3.1 and
`nsys` 2024.5.4 both work; the full nsys report set and both ncu export pages were produced
from a filtered toy run (one kernel, `--set full`, 9 s, 30 passes, 7.07 MB report).

| headline | value |
|---|---|
| power mode during this run | **MODE_30W (id 2)** — **not** the protocol's MAXN |
| GPU as configured | **4 of 8 TPCs active → 8 of 16 SMs**, clock capped 612 MHz |
| idle module power (3 tegrastats rails, 60 s, 10 Hz) | median **6.92 W**, tj 46 °C, no throttling |
| ncu `--set` values offered | `basic`, `detailed`, `full`, `nvlink`, `pmsampling`, `roofline` |
| ncu metrics of protocol §3.3 | 66 present, **9 absent** (`ncu/metric_presence.csv`) |

**Three findings that change how later experiments must be run.**
1. **MAXN needs a reboot** (`NVPM WARN: Reboot required for changing to this power mode: 0`).
   The prompt was declined per `AGENTS.md`. Nothing here characterises the Orin; E0 must be
   repeated at MAXN + `jetson_clocks` before any measurement is quoted.
2. **`ncu` cannot see DRAM on this Tegra** — every `dram__*` counter reads 0, and ncu's
   "Memory Throughput" equals `lts__throughput` (L2). Substitute:
   `lts__t_sectors_aperture_sysmem_*.sum × 32 B`, which matched the toy kernel's known
   8.49 MB of compulsory reads to **1.004**. Warp-stall reasons are on the `raw` page only.
3. **`tegrastats` does not print `VDDQ_VDD2_1V8AO`** (the memory rail E7 cares about most);
   it must be read from sysfs INA3221 `1-0041/hwmon/hwmon4/in2_*`.

Also: `ncu` needs **root** here (`perf_event_paranoid=1` is not enough), and passwordless
sudo — an `AGENTS.md` prerequisite — was not configured.
