# STATUS — on-device agent (Jetson AGX Thor)

*Maintained by the on-device agent. The research side reads it and never edits it. Newest first inside each
section; the log at the bottom is one line per event.*

## Now
T001 done (2026-10-22, commit 181ee2d): all six models pass the smoke test. Next: T002 (platform record).

**How root works here:** the agent has no sudo. The owner chose docker instead of a sudoers file: `env/rootdo.sh`
runs allow-listed commands (`jetson_clocks`, `nvpmodel -q`, `tegrastats`, `cat /sys/kernel/debug/*`,
`chown-repo`, `ncu`) as host root through `docker run --privileged --pid=host … nsenter -t 1`. It is never used to
install anything.

## Needs the human
*(none open. H2 — Hugging Face login + Cosmos-Reason2 licence — resolved 2026-10-22, see the log in `STATUS.md`.)*

## Questions for the research side
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
After T001 (2026-10-22): `du -sh results/` 2.4 M · `du -sh .git` 4.7 M · `size-pack` 2.35 MiB · largest committed
files: `env/tool_help/ncu_query_metrics_chip_gb10b.txt` 919,077 B; `results/20261022_E0_toolcheck/raw/ncu_limit_1000_tiny.ncu-rep.zst`
255,888 B; `…/raw/ncu_toy_raw.csv` 170,350 B; `results/20261022_E0_addenda/raw/cublaslt_list_text.txt.zst` 166,449 B;
`results/20261022_E0_toolcheck/raw/ncu_toy_metrics.ncu-rep` 164,986 B.

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
