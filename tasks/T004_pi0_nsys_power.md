# T004 — π0 kernel timeline, SM utilisation, roofline and power (E5, E6, E7)  · priority P1 · depends on: T003

Model: π0 — **LeRobot's π0** (`lerobot @ f0d2b37beb`, checkpoint `lerobot/pi0_libero_finetuned_v044` @ `45dcc8f`),
the same environment, fixture and harness as T002/T003 (this line was "openpi PyTorch" before T002's accepted
route change). Modes: eager and `reduce-overhead` (T003's best stock mode). N = 4 and N = 10. MAXN + `jetson_clocks`.
T010 (E11) has run before this task: reuse its `bench/phase_loops.py` (prefix-only / action-expert-only / full loops)
for step 4, and its nsys captures — `results/20260922_E11f_pi0_pipelining/nsys/{eager,reduce-overhead}_N{4,10}_*`
already cover every configuration of step 1 (2 inferences per capture): analyse those exports here instead of taking
new captures (reference them in `meta.json.notes`; the `results/20260922_E5_pi0_eager_N{4,10}/` directories you
started from them need their `meta.json`, `summary.md` and `raw/` like any other run). Take a new capture only if a
configuration is missing or the window turns out too short.

**Rules revised 2026-09-22 (protocol updated the same day; the protocol wins where this file is shorter):**
- **No `.nsys-rep` / `.sqlite` in git** (protocol §3.5 exception — they embed a session token). Commit the full
  `nsys stats` CSV exports under `nsys/` (compressed above 1 MB), list the binaries with `committed: false`.
- **Off-chip bytes use two counters** (protocol §3.3): reads from `…op_read_lookup_miss.sum`, writes from
  `…op_write.sum`, both × 32 B. `bench/parse_ncu.py` must implement exactly that.
- **Floors and gap ratios against the measured peak** (protocol §9): run `bench/micro_peak.py` once at the start
  of each profiling session, cite that session's peak next to the `hw_constants.json` value, and give the
  bandwidth spread (T005: up to 1.19× between processes) with every `× floor` figure.
- **Temperature log for every run** (protocol §1): E4 had none. A tegrastats log per run, stopped properly
  (`bench/common.py:stop_tegrastats()`; `pgrep tegrastats` empty before each timed run), is part of acceptance.
- **Per-region latency under `reduce-overhead` does not exist yet** (E4's instrumentation graph-breaks the model):
  step 1's nsys NVTX projection is the first and only per-region split of the compiled model. Report
  `vision` / `vlm_prefix` / `ae_total` / per-`ae_step` from the NVTX ranges for both modes, next to E3's eager
  CUDA-event numbers and T003 §4.1's `step_only` numbers, and say how the three agree.

0. Physical floor per region → `floor.csv` (protocol §9), both `peak_source` rows, before any profiling.
1. E5 / nsys (protocol §3.1): capture window = 1–3 steady-state inferences (or the E11f exports); compute kernels
   per step, kernel time vs region time, GPU idle fraction **per inference** (T010's per-inference grouping, not
   the whole-capture envelope), gap histogram, top-20 kernels by time per region with launch configuration →
   `kernels_per_region.csv`, `gaps.csv`, `timeline_summary.csv`. For `reduce-overhead` use `--cuda-graph-trace=node`
   so the graph-replayed kernels are visible, and state the kernel count per step both ways (graph nodes vs
   launches). T010 already reports `gpu_idle_ae_total` = 84 % eager at both N; E5 must give the same quantity for
   the compiled mode and the per-kernel time split behind it (which kernels fill the 16 %).
2. E5 / ncu (protocol §3.2–3.3; on this device: `sudo $(which ncu)`, the per-run clock check of §3.2 — ncu's own
   SM clock came out −0.9 to −1.9 % against 1300.5 MHz in T005, within the 2 % rule; stall reasons from the `raw`
   page; no DRAM counters — off-chip bytes from the two sysmem counters above): write `bench/ncu_target.py` and
   `bench/parse_ncu.py` (`docs/03_instruction_level_analysis.md` §11 items 1–2). One filtered run per region, in
   this order: `ae_layer[0]/`, `ae_step[0]/`, `vlm_prefix/`, `vision/`; eager at N = 4 first, then `reduce-overhead`
   (record whether ncu sees kernels inside CUDA-graph replays; if it does not, profile the `step_only` variant
   with graphs off and say so). One `ae_step` is the same work at N = 4 and N = 10: capture it once per mode.
   Export both pages; produce `ncu/columns.txt`, `ncu/kernels.csv`, `ncu/region_utilisation.csv` with SM
   occupancy / activity, per-pipe utilisation and warp-stall reasons (do they sum to ≈ 1?). Never unfiltered; if a
   region is too large, follow the fallback in §3.2 and report the kernel-time coverage. Commit every `.ncu-rep`
   (compressed; ≈ 7 MB per kernel uncompressed, ~76× with zstd — budget `-c` so no file passes 50 MB compressed)
   and both full page exports (protocol §3.5 / §6). Run the profiled Python without token variables (§3.5).
3. E6: roofline CSV per the plan for `vision`, `vlm_prefix`, `ae_step` (analytic rows, and `source = ncu` rows built
   from SM-throughput FLOPs and the two-counter sysmem bytes — never from `Memory Throughput`, which is L2 here),
   against the Orin's own peaks from `results/20260922_E1_peak/hw_constants.json` (187.47 GB/s, 33.85 TFLOP/s,
   ridge 180.6 FLOP/B) **and** the spec (204.8 / 42). Achieved GB/s uses the nsys duration of the same kernel.
4. E7 at MAXN: power per rail for idle / prefix-only / action-expert-only / full loops (`bench/phase_loops.py`),
   eager and `reduce-overhead`, N = 4 and N = 10; energy per inference and per denoising step (J = mean W above
   idle × region time, and absolute). All four rails from sysfs (protocol §4; T005 showed they do not double-count
   and may be summed), the memory rail reported as absolute and above-idle `VDDQ_VDD2_1V8AO (regulator input)`.
   Then `MODE_50W` per protocol §1 if it switches without a reboot (last item of the session).
   4b. **Capped modes are now required** (owner's commitment, `STATUS.md` standing item; protocol §1 updated):
   after MAXN is complete, write "next: resume E7 at mode 2" in `STATUS.md` and stop; the human reboots into
   `MODE_30W`, then later `MODE_15W`. In each capped session: §1 gate in that mode (record SM count, GPU and EMC
   clocks, `jetson_clocks` applied), then only (i) the four E7 loops above for eager and `reduce-overhead` at N = 10,
   and (ii) one E3-style latency run at N = 4 and N = 10 in both modes (20 warm-up / 100 timed, regions on for
   eager only) so that latency and energy per step exist per power mode, and **(iii) in the `MODE_30W` session
   only: T014 Step 4** — E12 Step 1 (the π0 prefix ∥ expert co-run, `bf16_both`, N = 10) at 8 SMs, per
   `docs/E12_corun/README.md` §5 and protocol §14 (added 2026-09-24; no extra reboot). Result directories carry
   the mode in their name; nothing from a capped mode is ever averaged with MAXN.
Result directories: `results/<date>_E5_pi0_<mode>_N<N>/`, `results/<date>_E6_pi0/`, `results/<date>_E7_pi0_<mode>_<powermode>/`,
`results/<date>_E3_pi0_<mode>_N<N>_<powermode>/` for 4b(ii).
Acceptance: `reports/E5_pi0_timeline.md` answering (a) kernels executing vs GPU waiting, per mode, and the
per-region split of the compiled model; (b) region-weighted and kernel-weighted SM occupancy, pipe utilisation and
stall reasons per region, next to `GR3D_FREQ`, set against the paper's 22 %, ending with the root-cause chain of
protocol §10 (T005 and T010 already point at CPU dispatch: 969 launches per step, ~62 µs each, the GPU idle 84 % of
`ae_total`; E5 must confirm or refute that from the timeline and say what the remaining 16 % of kernel time is
bound by); `reports/E6_pi0_roofline.md` (short; the CSV is the deliverable); `reports/E7_pi0_power.md` with W per
rail per phase and mJ per step and per inference, per mode and per power mode. Every result directory has its raw
files committed under `raw/` and listed in `meta.json.raw_files` (E5: nsys CSV exports, `.ncu-rep`, `torch.profiler`
trace if taken; E7: raw tegrastats logs and sysfs samples; all: command stdout; `.nsys-rep`/`.sqlite` listed
`committed: false`), a tegrastats temperature log per run, no committed file over 50 MB, repository-size lines in
`STATUS.md` (protocol §6f). Self-review (§11).

status: claimed 2026-09-22
