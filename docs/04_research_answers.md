# 04 — Answers from the research side

*Written by the research side. The on-device agent asks under "Questions for the research side" in `STATUS.md`; the
answers appear here, newest first, and every answer that changes how you work is also written into the protocol,
the plan or a task (named in the last column). Where this file and the protocol differ, the protocol wins.*

## 2026-09-25 — after T014's MAXN return (device commit `b97da62`)

**T014 found real things and did not answer its question.** The research side's review is `analysis/E12_review.md`.
Accepted as measured: the eager co-run (`T_pipe` = `A_iso + B_iso`, fresh KV verified bitwise with a stale control
that fails), the 1 + 2N host synchronisations of shipping π0, the E7 reconciliation (the per-step timestep tensor
is 56.07 of 167.09 ms of the compiled expert; 111.02 ms hoisted), green contexts working on CUDA 12.6 with a passing
confinement control, and the two-process OOM. Not accepted: "MAXN scope complete" — the compiled co-run, the nsys
timeline, four Step-3 mechanisms, `equivalence.*` and `mechanisms.csv` are missing, and green contexts were never
run with both stages at once. The follow-up is **T015 / E12b** (`docs/E12b_compiled_corun/README.md`).

| # | finding | answer | made binding in |
|---|---|---|---|
| 1 | Compiled `bf16_both` corrupts silently when pipelined | Right not to report it. Diagnose and repair it (own capture, separate pools, double-buffered stage I/O, or better); if it resists, the evidence is a result | E12b §6 M2, §10 row 6 |
| 2 | `issue/wall` = 1.0000 read as "host-bound" | The metric is ≈ 1 for any loop with a short GPU tail, GPU-bound too; use a per-frame host-issue record and the nsys timeline | E12b §6 M2 step 4 |
| 3 | Rows labelled `baseline_eager` over two harness changes; green-context `meta.json` with null `bench_commit` / `power_mode` / `sm_count` | Every row carries a variant label naming every change; no null in those fields | protocol §15; E12b §5 NN4, §11 |
| 4 | Step-4 prediction scales the eager expert by the GPU-clock ratio | The eager expert is host-bound and `MODE_30W` slows the CPU too; predict with a GPU-bound quantity (GEMM control, compiled stages) and record CPU clocks; please restate it in `STATUS.md` before that session | E12b §14 |
| 5 | Timestep hoist as an optimisation of shipping π0 | Measured in E12b M1 with a compiled equivalence check; it becomes the fair expert only if that passes. `cropcache` stays out (P2 not approved) | E12b §6 M1, §13 |

## 2026-09-22 — after T002 / T003 / T005 / T010 (device commits `c1c7fe7..e6a0af7`)

**T002 (E3), T003 (E4), T005 (E1/E2) are accepted; T010 (E11) is accepted on content and waits only for its
`status: done` line.** Verified on the research server: 55 `meta.json` read, all at MAXN with `bench_commit`,
20 / 100 iterations where §2 applies, `raw_files` hashes re-computed for every file of the E11, E4-cropcache and
E0 directories and a sample of the rest (one mismatch, below); the `model_inputs.csv` checker exits 0 on all nine
E11 files; `results/` is 39 MB, the largest committed file 6.6 MB, nothing over 50 MB; the write partition was
respected except the disclosed `docs/` corrections (accepted — see AGENTS.md, which now states the exception
the owner granted you). The independent-review corrections you made to your own drafts (the +3.13 → +0.83 ms
step-0 artefact, the missed `deepcopy`, the empty `raw_files`, the 0.10 ms "kernel" that was dispatch, the
50–66 % → 84 % idle envelope) are exactly what protocol §11 is for; keep reporting them.

**The numbers change the research.** The ladder is now measured — floor 3.115 (spec) / 3.403 (measured peak),
eager 52.56–52.94, compiled 16.30, crop-not-clone 10.37 ms per step — and the split contradicts the working
hypothesis (expert 31 % at N = 4, not 59 %). Both go to the research side as they stand; nothing here asks you
to re-measure to a different answer.

| # | your question / finding | answer | made binding in |
|---|---|---|---|
| 1 | Off-chip traffic: reads validate via `…op_read_lookup_miss.sum`, writes only via `…op_write.sum`; §3.3 as written would report zero write bytes | **Accepted; §3.3 corrected to the two-counter definition** with your validation ratios cited. `GBps_from_misses` / `pct_of_measured_peak` stay blank until an nsys duration of the same kernel exists (T004 step 3 supplies it). Case (d) under ncu replay is recorded in §3.3 (5) and §12. | protocol §3.3, §12; T004 |
| 2 | Memory rail: tracks traffic (+5.0 W on a copy load), rails do not double-count, only the absolute scaling of `VDDQ_VDD2_1V8AO` is open | **Unchanged: no schematic on this side either.** Report it as `VDDQ_VDD2_1V8AO (regulator input)`, absolute and above idle, tagged `[ASSUMPTION]`; sum the four rails as module power (your T005 check is the evidence). No calibration step. | protocol §4 (as before); T004 step 4 |
| 3 | Power-mode plan | **Superseded by the owner's direct commitment** in your `STATUS.md` standing item: `MODE_30W` and `MODE_15W` are **required** for π0's E7, one human reboot each, after MAXN is complete. Protocol §1 and T004 step 4b say exactly what to run in each capped session (E7 loops + one E3-style latency run). You still never confirm a reboot prompt and never use `--force`. | protocol §1; T004 step 4b |
| 4 | **nsys reports embed a session token**; four `nsys_toy.{nsys-rep,sqlite}` files pushed in T001 (commit `410a535`) contain it; `.gitignore` now excludes `results/**/*.nsys-rep` and `*.sqlite` | **Right call, and thank you for saying so plainly.** The exception is now the rule: protocol §3.5 / §6e, AGENTS.md, T004 — binaries stay in `scratch/`, listed `committed: false`; the full CSV exports are the committed raw record. The token rotation and any history rewrite are the **human's** decisions (the research side never rewrites pushed history either); until told otherwise treat the pushed copies as compromised-and-rotated, not as something to fix in git. `scan_for_secrets()` writing the matched bytes into `meta.json` — fixed by you — is now forbidden explicitly in §3.5. | protocol §3.5, §6e, §3.1, §7; AGENTS.md; T004 |
| 5 | Bandwidth peak varies 1.19× between processes with clocks and temperature constant | **Accepted as an open anomaly.** Rule: quote both floor rows, headline against the measured peak, and re-measure the peak at the start of every profiling session (`bench/micro_peak.py`) so a ratio uses a same-session peak. Re-check after the capped-mode reboots as you proposed. If T004's nsys timelines show the *same* kernel at two speeds in one process, that is worth one paragraph; do not chase it further inside T004. | protocol §9, §12; T004 |
| 6 | `docs/00_context.md` / `01_measurement_plan.md` corrected on device under the owner's instruction | **Accepted; the edits were checked line by line against the reports.** AGENTS.md now states the exception: factual corrections to those two files, marked and disclosed, are yours; rules, schemas and task bodies are not. | AGENTS.md |

**Small things to close, none blocking (do them at the start of the next session, one commit):**
1. `tasks/T010_pi0_hybrid_inputs.md` has no `status: done` line although the report, the consolidated file and the
   checker are all in; append it with the deviations (E11d ran 5 warm-up / 30 timed per point, below §2's 20 / 100 —
   state it there and in the report, or re-run the 18 points at 20 / 100 if it costs under an hour; the two-stream
   concurrency harness runs the prefix loop from a second Python thread — recorded, T011 fixes it).
2. `results/20260922_E4_pi0_reduce-overhead+cropcache_N10/meta.json` lists `equivalence.md` at 1,228 B / a stale sha256;
   the file is 1,407 B (edited after hashing). `bench/refresh_raw_files.py` on that directory.
3. `reports/E4_pi0_compile.md` §9 "Independent pass — recorded below once complete" is still empty although the
   review's three catches are described in `STATUS.md`; paste them there so the report is self-contained.
4. `results/20260921_E3_pi0_eager_probe/` and `results/evidence_E4_pi0_reduce-overhead_N4/` have `raw_files: []`;
   list what is there (the probe's CSVs, the evidence log) or say why nothing is raw.
5. `results/20260922_E5_pi0_eager_N{4,10}/` were pushed without `meta.json` / `summary.md` (T004 in progress) —
   fine as work in progress; they are complete when T004 step 1 is.
6. `STATUS.md` still carries the answered 2026-09-21 questions and the resolved "Blocked" / "Carried into the MAXN
   re-run" sections; prune when convenient.

**What the research side takes from E11 (for your information; no action):** `T_host_layer` is modelled as one
CUDA graph per layer (0.177 ms) with the compiled-sequence 0.085 ms as the lower bound; launches are priced at
~29 µs (single kernel) / ~62 µs (`F.linear`), 35 per layer eager; transfers at 40 µs + bytes / 36 GB/s; stream
handoff 109 µs; the prefix is a constant 816 tokens for this checkpoint; the two-process sharing result (1.001×)
is the one used for `host_layer_slowdown`, the two-stream one is not used. T011/T012 will carry the same segment
list and the corrected concurrency harness.

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
