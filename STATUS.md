# STATUS — on-device agent (Jetson AGX Thor)

*Maintained by the on-device agent. The research side reads it and never edits it. Newest first inside each
section; the log at the bottom is one line per event.*

## Now
T002 done 2026-10-22 (commit 3a5417b). T003 done 2026-10-22. Next: T004 (baseline latency, all six models). Owner's priorities (2026-10-22): the task order stays T002 → T015; the
latency breakdown for `eager_fp32` and `best_bf16` of all six models (T004, T005) comes first, then the ncu analysis,
**`best_bf16` before `eager_fp32` inside T006–T008** (the mode order inside a task is not fixed by the plan).

**How root works here:** the agent has no sudo. The owner chose docker instead of a sudoers file: `env/rootdo.sh`
runs allow-listed commands (`jetson_clocks`, `nvpmodel -q`, `tegrastats`, `cat /sys/kernel/debug/*`,
`chown-repo`, `ncu`) as host root through `docker run --privileged --pid=host … nsenter -t 1`. It is never used to
install anything.

## Needs the human
*(none open. H2 — Hugging Face login + Cosmos-Reason2 licence — resolved 2026-10-22, see the log in `STATUS.md`.)*

## Questions for the research side
- 2026-10-22 — **A hidden performance state changes the expert's timings by up to 1.26×** (T003,
  `results/20261022_E1_state_probe/`). After ≈ 55 s of sustained heavy GPU load (≈ 75 °C), cold weight-streaming
  GEMMs (the π0 expert's pattern) run at ≈ 245 instead of ≈ 166 GB/s for a few seconds; int8 GEMMs jump 258 → 330 TOP/s.
  No readable clock changes. Model inference loops may or may not enter this state. Conservative reading until
  answered: every timing run classifies the state (`bench/gate.py: memory_state_probe()`) before and after each block;
  baselines are reported in the **normal** state; blocks that end in the fast state are flagged and, where they
  change a phase share, repeated after a cool-down. Is the normal state the right baseline for the PIM comparison?
- 2026-10-22 — **Roofline ceiling: burst or sustained?** (T002) VDD_GPU is current-limited at 5,060 mA (≈ 100 W).
  A bf16 4096³ GEMM loop delivers 99.5 TFLOP/s for the first 0.5 s, then a sustained median of 93.97 TFLOP/s (−5.6 %),
  falling further as the die warms (91.9 at 60 s, 80 °C), with
  soctherm `oc3` throttle events that devfreq does not show (`results/20261022_E0_platform_MAXN/`). Conservative
  reading until answered: T003 records both; `hw_constants.json` takes the **highest measured rate** as the
  ceiling (the roof is what the hardware can deliver; short model kernels see the burst rate) and lists the
  sustained rate beside it, labelled. Every run records OC deltas and is marked `throttled` on any event.
- 2026-10-22 — **Pin a revision for `nvidia/Cosmos-Reason2-2B`?** `docs/00_context.md` pins the GR00T N1.7
  checkpoint but not the backbone repo its config names (`model_name: nvidia/Cosmos-Reason2-2B`), which
  `Gr00tPolicy` loads at run time. Done meanwhile (conservative reading): `main` downloaded once on 2026-10-22 =
  **9ce19a195e423419c349abfc86fd07178b230561**, recorded in `env/README.md` and every N1.7 `meta.json`; only that
  snapshot is used. Please confirm or name another revision.
- 2026-10-22 — **The GPU is shared.** Other users run long sessions on this Thor (one is another Claude session,
  "[thor] 2bit kernel"). VDD_GPU swung 6.7–30.8 W (10 s sysfs sample; 36.7 W instantaneous in tegrastats) with nothing of ours running. On Thor `nvidia-smi` lists no
  processes, so gate item 5 ("no other GPU job") cannot be checked per process. Proposed check before every
  measured run: no other user's GPU process in `ps`, plus tegrastats `GR3D_FREQ` load ≈ 0 % and VDD_GPU at its idle
  level for 10 s. Runs that fail it wait and retry. Can the owner reserve the device during measurement windows?
- 2026-10-22 — **For information: the Orin's LeRobot loader.** lerobot @ f0d2b37b `from_pretrained` ignores
  `revision` and loads `main` (Thor: it returned a random model offline; caught). `main` has since moved for
  `lerobot/pi05_libero_base` and `lerobot/smolvla_base`. Thor loads the pinned snapshots explicitly. Whether the
  Orin's π0.5 / SmolVLA runs used the pinned bytes depends on what `main` was when the Orin downloaded them. The
  pins in `docs/00_context.md` look like the Orin's resolved snapshots, which would make them consistent.

## Repository size (protocol §6f)
After T003 (2026-10-22): `du -sh results/` 11 M · `du -sh .git` 9.5 M · `size-pack` 2.35 MiB (newer objects still
loose) · largest committed files: `env/tool_help/ncu_query_metrics_chip_gb10b.txt` 919,077 B;
`results/20261022_E2_ae_kernels/raw/samples.csv` 528,212 B; `results/20261022_E2_ae_kernels_v3/raw/samples.csv`
528,174 B; `results/20261022_E2_ae_kernels_v2/raw/samples.csv` 527,989 B; `results/20261022_E1_peak/raw/latency_samples_bw_p1.csv` 515,156 B.

## Log
- 2026-10-22 T001 claimed; fixtures copied to `bench/fixtures/`.
- 2026-10-22 No sudo; owner approved the docker route (`env/rootdo.sh`). MAXN (id 0) is current.
- 2026-10-22 torch 2.10.0+cu130 passes A8 once Triton uses CUDA 13 `ptxas`; Nsight tools unpacked without root.
- 2026-10-22 ncu needs root (works via rootdo); no `dram__` counters, L2 sysmem counters ≈ 1.00× analytic on the toy.
- 2026-10-22 nsys secrets re-test: NOT CLEAN; nsys binaries never committed.
- 2026-10-22 Smoke: π0, π0.5, SmolVLA, GR00T N1.6, N1.5 pass; N1.7 blocked (gated backbone, H2); LIBERO renders.
- 2026-10-22 Independent review of T001: 14 findings fixed (addenda directory, columns.txt, roofline-section metrics, flags, hashes, GR00T P/S, rail ranges, meta fields).
- 2026-10-22 T001 closed done-partial (N1.7 smoke waits for H2); docs/00 L2 and docs/05 A8(b) hints corrected in place.
- 2026-10-22 H2 resolved by the owner (HF login, licence accepted); Cosmos-Reason2-2B @ 9ce19a19 downloaded; N1.7 smoke passes (P 273, S 41); T001 done.
- 2026-10-22 T002 done: locked 1575 MHz; idle module power 33.27 W; VDD_GPU over-current limit ≈ 100 W found (sustained GEMM −5.6 %, temperature-dependent, invisible to devfreq); gate.py now records OC deltas. Independent review fixed.
- 2026-10-22 T003 done: BW 258.2 GB/s; bf16 129.8 / int8 330.6 / tf32 62.4 / fp32 6.43 T(FL)OP/s; counters validated (0.1 %); expert projections 3.29 ms/step (1.37× floor); DRAM latency ≈ 520 ns; hidden fast memory-path state found. Independent review: 3 conclusions corrected (ceiling modes, bandwidth claim retracted, latency v2).
