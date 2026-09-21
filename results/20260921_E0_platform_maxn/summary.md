# E0 — platform record at **MAXN** (T001), 2026-09-21

The authoritative E0 record. The human rebooted into MAXN (id 0) and `jetson_clocks` +
`jetson_clocks --fan` were applied **before every capture**, as protocol §1 requires. This
supersedes `results/20260921_E0_platform/`, which was taken in MODE_30W and is kept only as
the pre-reboot comparison.

| quantity | MAXN (this run) | MODE_30W (previous) |
|---|---|---|
| `TPC_PG_MASK` / active TPCs | **0 / 8 of 8** | 240 / 4 of 8 |
| SMs reported by CUDA | **16** | 8 |
| GPU clock (locked, and measured under load) | **1 300.5 MHz** | 612 MHz |
| CPU cores online | 12 | 8 |
| EMC | 3 199 MHz | 3 199 MHz |
| idle module power, 4 rails of §4, 58.6 s @ 10 Hz | **11 012 mW** median | 8 522 mW |
| idle tj | 41.9 °C | 39.9 °C |
| toy GEMM, nsys median (n = 6) | **54.6 µs** | 185.4 µs |
| toy GEMM, ncu `Duration` | 57.09 µs @ 1.29 GHz | 195.74 µs @ 305.70 MHz |

Platform otherwise as before: JetPack **6.2.3** / L4T **R36.5.0**, CUDA **12.6.68**,
cuDNN 9.3.0, TensorRT 10.3.0, Python 3.10.12, EMC 3 199 MHz (LPDDR5-6400), GPU cc 8.7 →
**`sm_87`**. Stack torch 2.8.0 + torchvision 0.23.0 + triton 3.4.0; `torch.compile` works in
every inductor mode. `throttled: false`.

**Four findings, two of them corrections to the 30 W record.**

1. **`max-autotune` will never use Triton GEMM templates on any Jetson Orin.** The
   `Not enough SMs to use max_autotune_gemm mode` warning **still fires at MAXN with 16 SMs**
   (`raw/verify_stack.stdout.log` line 3). It is not a power-mode artefact, as the 30 W
   record assumed: `torch/_inductor/utils.py:1443` hard-codes `min_sms = 68` (`# 3080`) for
   CUDA devices, and Orin has 16. `max-autotune` therefore falls back to ATen/cuBLAS —
   the autotune log compares only `addmm` vs `bias_addmm`. **E4 must interpret
   "max-autotune" on this device as "default + CUDA graphs + ATen autotuning".**
2. **The L2 sysmem-miss substitute for DRAM traffic is now exact.** At MAXN
   `lts__t_sectors_srcunit_tex_aperture_sysmem_op_read_lookup_miss.sum` = 265 408 sectors →
   **8.4931 MB**, against 8.4931 MB of analytically compulsory reads — **ratio 1.0000**
   (30 W gave 1.0032). Every `dram__*` counter still reads 0, so the substitute stands.
3. **ncu's clock now matches the workload's.** With `jetson_clocks` applied at MAXN, ncu's
   `--clock-control base` reports **1.29 GHz** against the 1 300.5 MHz the GPU actually runs
   at (`raw/gpu_clock_under_load.txt`), and its `Duration` 57.09 µs is within 4.5 % of the
   nsys median 54.6 µs. The large divergence seen at 30 W was a consequence of profiling
   with unlocked clocks. ncu `Duration` is still not a latency (protocol §3.2), but it is no
   longer off by 2×.
4. **More SMs made this kernel *less* efficient, not more.** Occupancy is unchanged at
   ~16.4 % (register-limited: 158 regs/thread), but `Waves Per SM` fell from 2 to 1 and
   `SM Busy` from 59.8 % to 24.7 % — the same 32-block grid now spreads over 16 SMs instead
   of 8. This is the skinny-kernel behaviour the project is about, visible already in a toy.

ncu metric availability is unchanged from the 30 W run: **52 present, 14 present-but-zero,
9 absent** (`ncu/metric_presence.csv`); warp-stall reasons on the `raw` page only;
`tegrastats` still omits `VDDQ_VDD2_1V8AO`. Details: `reports/E0_platform.md`.
