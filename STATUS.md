# STATUS — on-device agent (Jetson AGX Thor)

*Maintained by the on-device agent. The research side reads it and never edits it. Newest first inside each
section; the log at the bottom is one line per event.*

## Now
**Running: T007 C1 (`best_bf16`).** Deliverable 1 is done: `best_bf16` chosen and confirmed for all six models
(`results/20261024_E4i_summary`, `reports/E4i_best_combinations.md`). T006 `best_bf16` (roofline per part and per
kernel) is done for all six with 100 % coverage (`reports/E6_roofline_all.md`). Next: T007 C1 for all six `best_bf16`,
then the rest of T005, then `eager_fp32` for T006/T007, then T008 onwards. Resume point: `PROGRESS.md`.

**How root works here:** the agent has no sudo. The owner chose docker instead of a sudoers file: `env/rootdo.sh`
runs allow-listed commands (`jetson_clocks`, `nvpmodel -q`, `tegrastats`, `cat /sys/kernel/debug/*`,
`chown-repo`, `ncu`) as host root through `docker run --privileged --pid=host … nsenter -t 1`. It is never used to
install anything.

## Needs the human
*(none open. H2 — Hugging Face login + Cosmos-Reason2 licence — resolved 2026-10-22, see the log in `STATUS.md`.)*

## Questions for the research side
- 2026-10-25 — **The owner put an exact code-level variant into `best_bf16` for π0 / π0.5 (`kv_shallow`).** LeRobot's
  `denoise_step` deep-copies the whole prefix KV cache at every denoising step (`modeling_pi0.py:919`,
  `modeling_pi05.py:890`): a graph break, a ~15 MB device copy and ~2.7 ms of host time per step during which the GPU
  waits (T007 timeline: π0 / π0.5 expert steps 25–27 % idle, 99 % of it CPU-starved). transformers' cache update
  rebinds (`torch.cat`) and never writes the prefix tensors, so a shallow copy is exact: bitwise-equal actions, π0 expert
  64.7 → 50.1 ms, e2e 143.8 → 128.7 ms in one process (`results/20261025_E4_cpu_overhead_diag/`). Protocol §16 leaves
  such variants to the research side; the owner decided on 2026-10-25 to include it. It enters through the normal
  path (E4i stage 4 search, equivalence, `e4i_summary.py`, confirmation). The pre-change `best_bf16` rows stay
  committed (`results/20261024_E4i_summary/`) for comparison. Please say if the PIM study needs the stock-code numbers
  as its baseline instead.
- 2026-10-22 — **For every device agent: cuBLAS picks slow GEMM kernels at some token counts; zero-padding the rows
  fixes it, bit-identically** (`reports/NOTE_gemm_shape_cliff_m_padding.md`). π0's prefix gate/up GEMM at M = 816 runs
  at 31 instead of ≈ 110 TFLOP/s on Thor; padding to M = 1024 cut π0's optimised frame 211.7 → 170.9 ms with
  identical actions. The Orin's own E4h data shows the same class of problem in π0's down projection (5.55 ms, SM 26 %).
  Please pass the note to the Orin agent.
- 2026-10-22 — **GR00T's shipped code computes logits it discards** (T004, `results/20261022_E3_groot_region_split_check/`).
  The backbone's vocabulary `lm_head` runs on every prefix token and costs 3.5 / 3.9 / 19.9 ms per inference
  (N1.7 / N1.6 / N1.5; 14 % of N1.5's e2e). The Orin folded it into the prefix (backbone − vision). Conservative reading
  until answered: Thor reports it as its own region and keeps it in "other"; the prefix is the decoder stack; the
  Orin-definition prefix is reported beside it. Removing it is a model change and belongs to T014 (E14), not to the
  `best_bf16` search. Should the PIM study count it as prefix work?
- 2026-10-22 — **Orin vs Thor GR00T e2e are different quantities.** The Orin's GR00T e2e is `model.get_action` on
  pre-collated inputs (`reference/orin/bench/groot_phases.py:631`); Thor's E3 e2e is `Gr00tPolicy.get_action`,
  which includes 14.7–19.3 ms of CPU processing. Thor records both, and every Thor-vs-Orin GR00T comparison uses the
  model call.
- 2026-10-22 — **GR00T N1.7's expert time varies between processes** (72.2–85.7 ms at N = 4, nine runs;
  `results/20261022_E3_groot_n17_state_vs_latency/`). Not the memory-path state. The cause is open; T006 compares a
  fast and a slow process. Until then N1.7 numbers carry that range.
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
After T004 (2026-10-22): `du -sh results/` 24 M · `du -sh .git` 16 M · `size-pack` 2.35 MiB (newer objects still
loose) · 780 files under `results/` · largest committed files: `env/tool_help/ncu_query_metrics_chip_gb10b.txt`
919,077 B; `results/20261022_E3_groot_n15_eager_Nsweep_v2/latency_samples.csv` 541,887 B;
`results/20261022_E3_groot_n16_eager_Nsweep_v2/latency_samples.csv` 540,151 B;
`results/20261022_E3_groot_n17_eager_Nsweep_v2/latency_samples.csv` 540,056 B;
`results/20261022_E2_ae_kernels/raw/samples.csv` 528,212 B.

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
- 2026-10-22 T004 done: E3 for six models (v2 GR00T after review), E4 π0 stock modes (best whole max-autotune 360.7 ms, ≤ 15 %); unused GR00T lm_head found; N1.7 process-to-process expert variation (72–86 ms) open. Two independent review passes fixed.
- 2026-10-22 T005: pi0 prefix slower than pi0.5's traced to a cuBLASLt kernel-choice cliff at M = 816 (gate/up GEMM 3.8x slower); fixed by linear_mpad (bit-identical); note for other agents in reports/NOTE_gemm_shape_cliff_m_padding.md.
- 2026-10-22 T005: six ladders done; E4i stage 1 done for all six models (N1.5 9/10); attention-backend switches (SDPA / flash-attn) all slower than Inductor's own fusion of eager attention; forced autotuning helps experts, hurts vision/prefix; paused by the owner, resume point in PROGRESS.md.
- 2026-10-23 T005 resumed: N1.5 divergence diagnosed (vision compile changes its output 4.6 %; no graph bug); bf16_islands (π0.5 466 MB of fp32 adaRMS weights) and expert-only autotuning take π0.5 183.1 -> 163.2 ms and π0 157.8 -> 148.7 ms; a candidate disturbed by another user's GPU job led to the stability guard; paused again by the owner (shared GPU in use).
- 2026-10-24 Resumed by the owner; order: best_bf16 + breakdown, then ncu for best_bf16 (T006/T007), then the rest.
- 2026-10-24 E4i stage 2 done for all six; best_bf16 chosen; confirmations within 2 % for π0, π0.5, SmolVLA, N1.7; paused by the owner before the N1.6 / N1.5 confirmations.
- 2026-10-24 T006 best_bf16 done for all six (E5 nsys + E6 roofline, 100 % coverage, 0 join mismatches); E6 reports generated; T007 C1 smoke test started (SmolVLA).
- 2026-10-25 T007 timeline found host stalls in π0/π0.5 expert steps (per-step KV deepcopy) and N1.7's eager vision/prefix; N1.7 vision made compilable (vis_fa_static): stage 3 best vis_fa_at_ae_at 120.8 vs 131.9 ms (confirmation queued); owner put kv_shallow into best_bf16 for π0/π0.5 (stage 4 queued). T006 GR00T expert floor bug fixed (weights per step): gap 1.43/1.47/1.94x.
