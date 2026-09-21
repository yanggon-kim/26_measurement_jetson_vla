# E0 — platform record (T001), 2026-09-21

Bring-up of the Jetson AGX Orin 64GB: JetPack **6.2.3** / L4T **R36.5.0**, CUDA **12.6.68**,
cuDNN 9.3.0, TensorRT 10.3.0, Python 3.10.12, 1.7 T free NVMe, EMC **3 199 MHz**
(LPDDR5-6400, as expected), GPU **cc 8.7 → `sm_87`** (confirmed by torch *and* `cuobjdump`
on the system cuBLAS). Stack: **torch 2.8.0 + torchvision 0.23.0 + triton 3.4.0** (JetPack
CUDA wheels); bf16 GEMM correct to 4.2e-3; **`torch.compile` works in every inductor mode**.
`ncu` 2024.3.1 and `nsys` 2024.5.4 both verified end to end on a filtered toy run (1 kernel,
`--set full`, 8.30 s, 31 passes, 7.07 MB report → 92 KB with `zstd -19`).

| headline | value |
|---|---|
| power mode | **MODE_30W (id 2)** — **not** the protocol's MAXN |
| GPU as configured | **4 of 8 TPCs → 8 of 16 SMs**, 612 MHz (measured under load) |
| idle module power, 4 rails of §4, 58.7 s @ 10 Hz | median **8 522 mW**; tj 39.9 °C; not throttled |
| ncu `--set` values offered | `basic`, `detailed`, `full`, `nvlink`, `pmsampling`, `roofline` |
| ncu metrics of protocol §3.3 | **52 present, 14 present-but-zero, 9 absent** (`ncu/metric_presence.csv`) |

**Four findings that change how later experiments must be run.**
1. **MAXN needs a reboot** (`NVPM WARN: Reboot required…`), declined per `AGENTS.md`. Nothing
   here characterises the Orin; E0 must be repeated at MAXN before any number is quoted.
2. **This Tegra exposes no DRAM counters.** `ncu --query-metrics` lists 4 426 metrics on
   `Orin (GA10B)` and **none** is `dram__*`; an explicit request returns `n/a`; and ncu's
   `Memory Throughput` equals `lts__throughput` — it is **L2**. Substitute:
   `lts__t_sectors_*_sysmem_*_lookup_miss.sum × 32 B`, which matched the toy kernel's
   8.4931 MB of compulsory reads to **1.0032**. Read path only; writes unvalidated.
3. **`tegrastats` omits `VDDQ_VDD2_1V8AO`**, the memory rail. All four rails are now sampled
   from sysfs (`power_idle_sysfs.csv`), so §4 module power is available — but that channel
   measures a constant 5 040 mV bus, so its meaning must be settled before E7.
4. **`ncu` needs root** here (`perf_event_paranoid=1` is not enough), and it pins the SM
   clock to 305.70 MHz while the GPU runs at 612 MHz under load — its `Duration` is not a
   latency.

Warp-stall reasons are on the ncu `raw` page only. Details: `reports/E0_platform.md`.
