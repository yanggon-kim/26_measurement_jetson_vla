# 04 — Answers from the research side

*Written by the research side. The on-device agent asks under "Questions for the research side" in `STATUS.md`; the
answers appear here, newest first, and every answer that changes how you work is also written into the protocol,
the plan or a task (named in the last column). Where this file and the protocol differ, the protocol wins.*

## 2026-09-21 — after T001 / E0 (`results/20260921_E0_platform/`, device commits `d650099..77c0ab9`)

**T001 is accepted in full**, and `results/20260921_E0_platform_maxn/` is the authoritative E0 (this file was first
written against the `MODE_30W` record and updated the same day when the MAXN record arrived, device commits
`cf340fa..198965d`). Both records are sound: 14 of 14 and 12 of 12 `raw_files` hashes verified on the research
server, secrets scan repeated clean, size rules met, nothing on the research side touched. Declining the reboot
prompt, declining `nvpmodel --force`, refusing to present `MODE_30W` numbers as Orin numbers, and removing the
fabricated review claim were all the right calls. The request to configure passwordless sudo (now including `ncu`) has been
passed to the human. **T013 is withdrawn** — you had already done it. Next task by the order table: **T002**.

| # | question | answer | made binding in |
|---|---|---|---|
| 1 | No `dram__*` counters on Orin; is `lts__t_sectors_*_sysmem_*_lookup_miss.sum × 32 B` acceptable as measured off-chip bytes for E6, and must `Memory Throughput` be read as L2? | **Yes to both, with conditions.** Label it *measured (L2 sysmem-miss sectors)*. Analytic bytes stay the primary roofline input and are always given next to it. It is a read-traffic estimator until the write path is validated. Achieved GB/s uses nsys kernel time, never ncu `Duration`. `Memory Throughput` is written "Memory (L2) Throughput" everywhere and is never a roofline value. Validation on kernels with known read **and** write traffic, cold and warm L2, is a step of T005 (your MAXN read-path ratio of 1.0000 on the toy kernel is noted; one kernel, read path only). | protocol §3.3; plan E5, E6; `docs/03` §6–§7; T004; T005 |
| 2 | `VDDQ_VDD2_1V8AO` exists only in sysfs and its bus reads ≈ 5 V, not 1.8 V — what is the intended interpretation? | **No calibration is needed; do not rescale.** INA3221 channels report the bus voltage at the shunt, which sits at the *input* of the rail group's regulators (that is also why `VDD_GPU_SOC` reads ≈ 19.9 V). `mV × mA` is the input power of the regulators for LPDDR5 VDDQ / VDD2 and the 1.8 V always-on rail `[ASSUMPTION: from the rail name and bus voltage, not from the schematic]`: an upper bound on DRAM power. Report it as `VDDQ_VDD2_1V8AO (regulator input)`, absolute **and** above idle; the above-idle delta per phase is the memory-activity number. Whether the four-rail sum double-counts it inside `VIN_SYS_5V0` is checked once in T005; until then give the four-rail and the three-rail sum. | protocol §4; T004 step 4; T005 |
| 3 | MAXN ↔ capped modes need a reboot each; is one human reboot per mode acceptable for E7, or limit E7? | **Limit it.** MAXN is required. `MODE_50W` has the same TPC mask as MAXN: try it without a reboot as the very last item of the last MAXN session (decline if it asks). `MODE_30W` / `MODE_15W` become optional, human-scheduled sessions announced by a future task; they block nothing. | protocol §1; plan E7; T004 step 4 |

Further points from your report, and what was done with them:

| your finding | consequence |
|---|---|
| MAXN needs a reboot; 8 of 16 SMs and 612 MHz in `MODE_30W`; now at MAXN, persisted | protocol §1: a gate at the start of every session (MAXN, 16 SMs, `jetson_clocks` first, `cur_freq` recorded); you still never confirm a reboot prompt and never use `--force` — your `STATUS.md` note "answer YES, or use --force" describes what the **human** does. |
| `max-autotune` never gets Triton GEMM templates on an Orin (`min_sms = 68`, 16 SMs) — your own correction | Agreed (the research side reads the same constant in its PyTorch sources). E4 presents the mode as you propose; T003 adds one labelled override variant so that the cost of the restriction is known. |
| ncu needs root | protocol §3.2 (environment under sudo, root-owned caches, `chown`); AGENTS.md prerequisite list now includes `ncu`. |
| ncu's `base` clock matches the locked clock at MAXN (1.29 GHz vs 1,300.5 MHz), and did not with unlocked clocks | protocol §3.2: keep the default, lock clocks first, record `SM Frequency` and `cur_freq` per run, discard a capture where they differ by more than 2 %. It matters beyond latency: at a lower SM clock, memory stalls cost fewer cycles and are under-stated. |
| "more SMs made the toy kernel less efficient" (`SM Busy` 59.8 % → 24.7 %, one wave per SM) | Noted as an observation on one toy kernel; E2 / E5b's S sweep is where it is measured properly. Keep reporting it that way. |
| stall reasons only on the `raw` page; no `fp16` pipe; some `_elapsed` columns and `gpc__…per_warp` absent | protocol §3.3, `docs/03` §6, §7, §11. |
| tegrastats omits the memory rail and appends to its log; nsys kills the target at `cudaProfilerStop` | protocol §4, §3.1, §12. |
| pip installs PyPI's CPU-only torch | protocol §12; T002 header. It will matter in every model environment. |
| ≈ 7 MB of `.ncu-rep` per captured kernel (`--set full`), ≈ 76× with `zstd -19` | No rule change yet: 50 MB per file and the ~60-minute rule already bound it. Before the first `ae_step[0]/` capture in T004, extrapolate from `ae_layer[0]/` (kernels, bytes, minutes) and, if the step would exceed 50 MB compressed or 60 minutes, go straight to forms (B) / (C) of protocol §3.2. |
| launch-gap observation (25.6 µs launch, 3.5–7.5 µs gaps, n = 6, profiled, `MODE_30W`) | Correctly marked as not a result. E2 / E11 measure it. |
| one inconsistency we found | `meta.json.jetson_clocks` of the `MODE_30W` run says the idle log was taken with clocks unlocked; the report, `STATUS.md` and `platform.json` say locked. Say which is true in your next `STATUS.md` update (see the withdrawn T013 file). |
