# E7 / E3 / E12 at the capped power modes — MODE_30W and MODE_15W (T004 step 4b, T014 step 4)

Jetson AGX Orin 64 GB. Four power modes, **four different perturbations, not one power sweep** (E0,
`/etc/nvpmodel.conf`, each read back by the §1 gate of every run):

| mode | SMs (TPC mask) | GPU | EMC | CPUs |
|---|---|---:|---:|---|
| MAXN | 16 (0) | 1300.5 MHz | 3199 MHz | 12 @ 2.20 GHz |
| MODE_50W | 16 (0) | 816 MHz | 3199 MHz | 12 @ 1.50 GHz |
| MODE_30W | 8 (240) | 612 MHz | 3199 MHz | 8 @ 1.73 GHz |
| MODE_15W | 6 (248) | 408 MHz | **2133 MHz** | 4 @ 1.11 GHz |

Sessions: MODE_30W 2026-09-26 11:15–14:09, MODE_15W 2026-09-27 18:25–22:57, each after an owner reboot,
`jetson_clocks` applied, clocks locked (min = max) in every run's gate, never throttled. Nothing from a capped mode is
averaged with MAXN. Result directories carry the mode in their name.

## 1. Energy and latency per inference, full frame (E7; `results/20260927_E7_power_modes_compare`)

Above-idle module energy per inference (J), each run against its own 60 s idle baseline in the same mode; median
latency (ms) of the timed loop. E·D = energy × delay.

| model | rung | N | MAXN ms / J | 50W ms / J | 30W ms / J | 15W ms / J | 15W vs MAXN latency × / energy × / E·D × | 15W vs 30W latency × / energy × |
|---|---|---:|---|---|---|---|---|---|
| π0 | `bf16_both` | 10 | 425 / 13.70 | 626 / 10.85 | 1258 / 12.36 | 2188 / 12.74 | 5.151 / 0.93 / 4.789 | 1.739 / 1.03 |
| π0 | `eager` | 10 | 1083 / 25.00 | 1583 / 19.17 | 2533 / 19.07 | 4475 / 22.09 | 4.132 / 0.883 / 3.65 | 1.767 / 1.159 |
| π0.5 | `bf16_both_graphs` | 10 | 482 / 16.57 | 689 / 12.88 | 1345 / 15.77 | 2386 / 15.93 | 4.95 / 0.961 / 4.759 | 1.775 / 1.01 |
| π0.5 | `eager` | 10 | 3042 / 100.44 | 4700 / 74.09 | 10931 / 80.44 | 18990 / 83.97 | 6.243 / 0.836 / 5.22 | 1.737 / 1.044 |
| SmolVLA | `vlm_graphs` | 10 | 88 / 2.84 | 128 / 2.20 | 236 / 2.27 | 449 / 2.55 | 5.089 / 0.896 / 4.562 | 1.903 / 1.123 |
| SmolVLA | `eager` | 10 | 741 / 5.86 | 1042 / 4.83 | 987 / 4.58 | 1566 / 3.99 | 2.115 / 0.68 / 1.439 | 1.588 / 0.87 |
| GR00T N1.7 | `best` | 4 | — | 535 / 7.71 | 598 / 8.25 | 1026 / 7.69 | no MAXN run at this precision | 1.717 / 0.932 |
| GR00T N1.7 | `eager` | 4 | — | 536 / 7.60 | 602 / 8.30 | 1039 / 7.73 | no MAXN run at this precision | 1.726 / 0.932 |
| GR00T N1.6 | `best` | 4 | — | 486 / 7.24 | 563 / 7.42 | 942 / 7.57 | no MAXN run at this precision | 1.671 / 1.02 |
| GR00T N1.6 | `eager` | 4 | — | 478 / 7.20 | 561 / 7.41 | 955 / 7.59 | no MAXN run at this precision | 1.704 / 1.024 |
| GR00T N1.5 | `best` | 4 | — | 353 / 8.57 | 658 / 9.65 | 1191 / 8.76 | no MAXN run at this precision | 1.812 / 0.908 |
| GR00T N1.5 | `eager` | 4 | — | 352 / 8.52 | 658 / 9.64 | 1191 / 8.80 | no MAXN run at this precision | 1.812 / 0.913 |

Prefix and action-expert rows per mode, with the source directories, are in `e7_power_modes.csv` / `.md` there.

**Reading (derived).**
- **Energy per inference barely moves; latency does.** From MAXN to MODE_15W the frame is 4.1–6.2× slower for π0 /
  π0.5 / SmolVLA's graph rung while the above-idle energy falls only 4–16 % (SmolVLA eager: 32 %). Energy × delay is
  1.4–5.2× worse. **No capped mode is an efficiency route for these models on this device** — the 50W finding
  (E·D worse in all 12 pairable points) extends to 30W and 15W.
- **The lowest energy per inference is MODE_50W for 4 of the 6 configurations with a MAXN point** (clocks cut, all
  SMs and full EMC kept); π0 `eager` is lowest at 30W (19.07 vs 19.17 J at 50W) and SmolVLA `eager` at 15W (3.99 J).
  Across all four modes a configuration's idle-subtracted frame energy spans at most 1.47× (SmolVLA `eager`; the others 1.26–1.36×) — the work
  is the same, the rate changes 2–6×.
- **15W vs 30W:** 1.59–1.90× slower (SMs 8 → 6 and GPU 612 → 408 MHz give 2.0× of compute; EMC 3199 → 2133 gives
  1.5× of bandwidth), energy 0.87–1.16×.

**Energy basis, disclosed.** The MAXN, 50W and 30W runs were made before the sampler fix (T016 step 4): their rail
sampler started before warm-up / compile, understating compiled phases. Their energies here are the **timed-window
re-analysis of their own committed raw samples** (`results/20260926_E7r_pi0_energy_reconcile_MAXN/E7_all_dirs_timed_window.csv`);
the 15W runs used the fixed sampler and their `power.csv` directly (source per cell in the CSV: `w` / `d`). The re-run
at MAXN (E7r) agreed with the timed-window re-analysis within 0.7 %. GR00T's capped queue uses the `eager` / `best`
rungs at matmul "high"; MAXN has only `eager_fp32` / `vlm_compile`, so GR00T has no MAXN column (as in the 50W section
of `E7_power_energy.md`).

## 2. π0 eager latency per stage (E3 harness, 20 warm-up / 100 timed, regions on)

| stage (median ms) | MAXN | MODE_30W | MODE_15W | 30W / 15W vs MAXN | SM × GPU-clock ratio 30W / 15W | CPU-clock ratio 30W / 15W |
|---|---:|---:|---:|---|---|---|
| vision | 239.6 | 940.7 | 1836.7 | 3.93 / 7.67 | 4.25 / 8.50 | 1.27 / 1.98 |
| vlm_prefix | 232.9 | 847.6 | 1466.2 | 3.64 / 6.29 | 4.25 / 8.50 | 1.27 / 1.98 |
| per denoising step (N=10 − N=4, / 6) | 52.9 | 65.2 | 108.4 | 1.23 / 2.05 | 4.25 / 8.50 | 1.27 / 1.98 |
| e2e, N = 10 | 1011.6 | 2457.3 | 4377.2 | 2.43 / 4.33 | | |
| e2e, N = 4 | 694.0 | 2066.1 | 3726.6 | 2.98 / 5.37 | | |

Directories `results/20260921_E3_pi0_eager_N{4,10}`, `results/20260926_E3_pi0_eager_N{4,10}_30W`,
`results/20260927_E3_pi0_eager_N{4,10}_15W`. **Reading:** vision tracks the GPU (90–92 % of the SM × clock ratio,
i.e. near compute-bound); the Gemma prefill less so (86 % / 74 %); **the eager denoising step tracks the CPU clock**
(1.23 vs 1.27, 2.05 vs 1.98) — the host-issue-bound expert E5 found at MAXN stays host-bound at every mode.

## 3. E12 at 8 and 6 hardware SMs (T014 step 4 at 30W; the same queue at 15W, beyond T014)

| quantity | MAXN reference | MODE_30W | MODE_15W | 15W / 30W |
|---|---|---:|---:|---:|
| 2048² bf16 GEMM, whole GPU (ms) | 0.8915 (8-SM green) | 1.984 (8 SMs) | 3.978 (6 SMs) | **2.005** (SM × clock 2.000) |
| same GEMM, 4-SM green partition (ms) | — | 3.949 | 5.913 | **1.497** (clock 1.500) |
| compiled prefix alone, `graphs_both` (ms) | 246.28 (whole) / 326.94 (green 8) | 876.84 | 1497.22 | 1.708 |
| compiled expert alone, `owngraph_expert` (ms) | 101.36 / 149.05 | 286.13 | 542.95 | 1.898 |
| eager co-run: T_pipe vs A_iso + B_iso (ms) | — | 2439.1 vs 2441.6 | 4306.5 vs 4311.1 | — |
| eager co-run s | — | 0.995 | 0.995 | |
| equivalence (fresh KV) | PASS | PASS 6/6, stale 5/5 | PASS, stale 5/5 | |
| energy pipelined / serial | — | 0.9996 | 1.0060 (time 1.0029) | |
| green SM curve, eager: prefix / expert, smaller partition vs whole GPU | | measured (8 SMs = whole) | 4 → 6 SMs: prefix **1.42×** (SM ratio 1.50), expert **1.02×**; a 2-SM request was granted 4 | |

**Reading.** (1) The compute-bound GEMM scales exactly with SM count × GPU clock across 30W and 15W (within 0.3 %),
and the 15W memory cap does not touch it. (2) The compiled stages scale by 1.71 (prefix) and 1.90 (expert) from 30W
to 15W — between the EMC ratio (1.50) and the compute ratio (2.00), the expert closer to compute. (3) The eager co-run
does not overlap at either capped mode (T_pipe = A_iso + B_iso within 0.1 %, pipelining saves no energy): eager π0 is
CPU-issue-bound, so there is nothing for two streams to overlap, as at MAXN. (4) Inside MODE_15W the eager prefix
scales with SMs (1.42× from 6 to 4 SMs) while the eager expert barely moves (1.02×) — the same split as the MAXN
green curve (prefix SM-hungry, expert SM-insensitive).

## 4. Anomalies and deviations

- `corun_green.py` control 2 asked for an 8-SM green partition; MODE_15W has 6, and the driver refused
  (`CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION`, 21:58). Fixed to 4 SMs when the device has fewer than 8 (MAXN / 30W
  unchanged), re-run the same night; the failed partial output is kept in `scratch/failed_runs/`. With 6 SMs the
  script's "half the GPU" verdict needs a 3-SM point that green contexts do not grant, so that verdict is absent at 15W.
- Metadata nulls found and fixed during the session (no measured value changed): eager `rung` / `compile_mode`
  (`power_phases.py` and 13 other writers), GR00T `gr00t_root`, `corun.py`'s `cuda` / `jetpack` / `l4t` (read from keys
  `software_versions()` never set). Committed metas back-filled with a `backfill_note`.
- Latency at a capped mode includes the slower CPU; a stage's slowdown is therefore compared with both the GPU
  (SM × clock) and the CPU clock ratio above, never attributed to one without that comparison.

## Reproduce

```
bash bench/queues/queue_e7_capped.sh      # in MODE_30W or MODE_15W, after the owner's reboot
bash bench/queues/queue_e12_capped.sh
python3 bench/e7_capped_compare.py --out results/<date>_E7_power_modes_compare
```
