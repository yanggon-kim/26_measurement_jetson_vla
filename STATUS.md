# STATUS — on-device agent

*Maintained by the on-device agent. The research side reads it and never edits it.*

## ✅ 2026-09-28 17:45 — T004 4b(ii) complete in both capped modes; device back in MAXN

* Compiled π0 latency, N = 4 / 10: 30W `results/20260928_E3_pi0_reduce-overhead_N{4,10}_30W` 1844.9 / 2061.5 ms;
  15W `results/20260929_E3_pi0_reduce-overhead_N{4,10}_15W` 3369.4 / 3765.9 ms (MAXN 497.8 / 597.3). Report
  `reports/E7_capped_modes.md` §2b: compilation buys only 1.11–1.19× when capped; the fixed part scales with the GPU.
* Every task in `tasks/` now has its acceptance items delivered or explicitly marked not measured with the reason.
* Device back in **MAXN** after the owner's reboot (17:45): 16 SMs, 12 CPUs, GPU locked 1300.5 MHz (`jetson_clocks`). Nothing running; no task open.

---

## ✅ 2026-09-28 (afternoon) — completeness audit: every task re-checked, gaps measured, one wrong number corrected

The owner asked for a completeness check against the task list. Findings and what was done (all MAXN, one GPU job at a
time, 10-minute watchdog; ~50 commits, all pushed):

* **Wrong number corrected:** SmolVLA E11a `attention_whole` timed a Python return (0.0156 ms, 0 kernels). Real: 3.56 ms
  eager (layer 0, self-attention), 2.64 ms (layer 9, cross). Kernel counts are now keyed by layer.
* **T007:** π0 S sweep (never run) — `results/20260928_E5b_pi0_ssweep`. **T008:** E5 at N = 4 and the ae_layer ncu
  captures (SmolVLA best not measurable: layer ranges break its CUDA-graph replay).
* **E5b (T007/T008/T009):** docs/03 §0 file set for all 8 directories; reports in §10's seven parts. Two tool bugs fixed
  (backward branches never detected; π0 best's dominant `Kernel2` variants never dumped).
* **T011:** π0.5 / SmolVLA E11a without kernel counts → complete re-runs; xfer_exact measured; consolidated files and
  reports rebuilt (T010 structure, cross-model table).
* **T012:** the "E11a only" scope is withdrawn — GR00T N1.7 E11a–f + consolidated + report delivered. GR00T's DiT has
  32 blocks (16 cross-attention); it hands over backbone features and re-projects cross K/V every step (3.18 ms/step).
* **Hygiene:** six E11d raw/ stdouts recovered from queue logs; null meta writers fixed in 9 more scripts; watchdog
  no longer counts CPU-only analysis or `ncu --import` as GPU jobs.
* **Still open — needs the owner:** T004 4b(ii), compiled E3-style latency at N = 4 / 10 in MODE_30W and MODE_15W, needs
  two reboots. Not started.
* Repository: committed results 713 MB; size-pack 158.10 MiB. bench commit e568440.

---

## ✅ 2026-09-28 — owner-requested analyses and the vision energy split (MAXN); no task open

* **E7v — vision / language / action energy measured separately** (12 runs, 6 models × eager / best, MAXN, fixed
  sampler): `results/20260928_E7v_*`, summary `results/20260928_E7v_summary`, `reports/SYNTHESIS_all_measurements.md`
  §2.1. Two new power-loop phases (`bench/phase_loops.py`, `bench/power_phases.py`, `bench/groot_phases.py --phases`):
  `vision` replays the image-encoder calls of one prefix pass; `vlm` (the language part) runs the unchanged prefix with
  the encoder answered from recorded outputs. Closures: vision + language vs prefix within 0.4–1.8 %, prefix + action vs
  full within 0.3–4.9 %. Best-rung energy shares vision / language / action: π0 17 / 59 / 24 %, π0.5 15 / 60 / 26 %,
  SmolVLA 50 / 11 / 39 %, GR00T 20–31 / 28–45 / 35–41 %.
* **E5c — the action expert by component** (derived from the committed E5 nsys runs):
  `results/20260928_E5c_expert_components`, synthesis §3.4. π0's compiled attention products (36 per step, `cutlass_75_wmma`
  kernels) take 4.19 ms/step ≈ all 126 weight projections (4.54 ms); the whole prefix K/V cache (15.05 MB) is re-copied
  into the CUDA graph's inputs every denoising step; GR00T's compiled head is 91–94 % weight GEMMs.
* **Reports:** `reports/SYNTHESIS_all_measurements.md` (all measurements in one place, component shares of latency and
  energy), `reports/FEASIBILITY_realtime_control.md` (per-frame latency vs robot control requirements from the π0,
  GR00T N1 and VLA-Perf papers; only SmolVLA meets 10 Hz on Orin).
* Device left at MAXN, clocks locked, nothing running. New work waits for the research side.

---

## ✅ MODE_15W SESSION DONE (2026-09-27 18:25–22:57) — the task queue is empty; next, owner: back to MAXN

Device after the owner's reboot: `MODE_15W`, TPC mask 248 → **6 SMs**, GPU locked **408 MHz**, EMC **2133 MHz**, 4 CPUs
@ 1.11 GHz — as `/etc/nvpmodel.conf` defines it. Both power-measurement paths smoke-tested at 15W before the queue.

* **E7 (T004 4b)**: all 12 power loops (π0, π0.5, SmolVLA N = 10; GR00T N1.5/1.6/1.7 N = 4; eager + best) and E3 π0
  latency at N = 4 / 10. Report **`reports/E7_capped_modes.md`** (MAXN / 50W / 30W / 15W side by side). Headline:
  **energy per inference barely moves across power modes (1.26–1.47× spread) while latency grows 2–6×; E·D at 15W is
  1.4–5.2× worse than MAXN — no capped mode is an efficiency route.** The eager denoising step tracks the CPU clock
  (1.23× / 2.05× at 30W / 15W vs CPU 1.27× / 1.98×); vision tracks SM × GPU clock (90–92 %).
* **E12 at 6 SMs** (beyond T014, same queue): GEMM control scales exactly with SM × clock (15W/30W 2.005 vs 2.000;
  same 4-SM partition 1.497 vs clock 1.500); compiled prefix / expert 1497 / 543 ms (15W/30W 1.71 / 1.90); eager co-run
  serial (s 0.995), equivalence PASS, pipelining saves no energy (1.006).
* **Defects found and fixed:** `corun_green.py` control 2 requested 8 SMs on a 6-SM device (fixed → 4 when < 8; step
  re-run); null `rung` / `compile_mode` / `gr00t_root` / `cuda` / `jetpack` / `l4t` in 14 meta writers (fixed; ~100
  committed metas back-filled, no measured value changed). Remaining pre-existing nulls not touched: `model` / `N` in
  microbenchmark and rails-check metas (not applicable there), `shipped_config.*` (the checkpoint's own config values).
* Task files: T004 and T014 got their missing closing status blocks (T004's E6 correction and step 4b; T014's Step 4).

**Next, owner:** done — the device was returned to MAXN on 2026-09-28. **No task is open**; new work waits for the research side
(and its answers to the T016 questions below).

---

## ✅ T017 DONE (2026-09-27 ~17:10, MAXN) — next: the MODE_15W session (needs the owner's reboot)

π0 padding trim, decomposed (`reports/E14b_pi0_padtrim.md`); all four variants in one bracketed session of the E12b
`graphs_both` harness, drift 0.03 %. Verdicts: `nocam`, `trimprompt`, `nopad` all **PASS** (not bitwise, inside the
rung's own floor), fp32 gate passed (REL ≤ 1.5e-6), K/V handover bytes exact.

| variant | prefix ms (ρ) | vision ρ | expert ms (ρ) | serial ms (ρ) | prefix bytes ρ (r+w) |
|---|---|---:|---|---|---:|
| `pad` | 245.4 | 1 | 101.6 | 347.6 | 1 (21.60 GB) |
| `nocam` | 177.8 (0.724) | 0.665 | 88.3 (0.870) | 269.4 (0.775) | 0.801 |
| `trimprompt` | 252.3 (**1.028**) | 0.999 | 69.8 (**0.687**) | 323.4 (0.930) | 0.999 |
| `nopad` | 180.1 (0.734) | 0.668 | 66.3 (0.653) | 248.5 (0.715) | 0.792 |

New and unexplained: trimming the 27 prompt tokens slows the Gemma prefill (odd lengths 789 / 533 vs aligned 816 / 560)
and cuts the expert by 31 % without moving its bytes. Repository after T017: committed results 534 MB, size-pack
158.10 MiB + loose 88.64 MiB.

**Next:** done — see the MODE_15W block above.

---

## ✅ T016 DONE (2026-09-27 ~14:50, MAXN) — T017 claimed and running next; MODE_15W session still pending (owner reboot)

All five steps and rows 1–8 answered; reports `E13_pi0_bandwidth_steal.md`, `E8s_libero_seeded.md`,
`E14_prefix_savings.md`, and the dated addenda in `E7_power_energy.md` (step 4) and `E12b_pi0_compiled_corun.md`
(step 5). Status block appended to `tasks/T016_prefix_bandwidth_and_followups.md`.

| row | answer |
|---|---|
| 1 | σ = 1.247 at 77 GB/s stolen, 1.28 at the SM-free maximum (84.9 GB/s; ~94 not reachable) → **> 1.247** |
| 2 | 37 % duty: σ = 1.125 → **1.04–1.247** |
| 3 | within ±0.02 of 1 + f(σ_full − 1) → **linear in bytes exposed** |
| 4 | channel confinement: **no** (boot-time BCT) |
| 5 | seeded N = 2 vs N = 10: **replicated** (4 / 4 suites; pooled 300 vs 259 / 400, p = 1.3e-5) |
| 6 | repeat agreement **95 / 100** (on the line) |
| 7 | GR00T `keep1` / `kvonce` / both: **bitwise** (50 / 50, compiled and eager, controls bitwise) |
| 8 | H1 share of prefix excess reads **0.67** (on the 2/3 line) |

Also: π0 `nopad` prefix 253.0 → 189.6 ms, frame 420.5 → 319.5 ms, prefix DRAM reads 17.65 → 14.64 GB; E7 energy
residual cause found (sampler before warm-up; 0.49–0.71 % after the fix; 28 committed compiled rows move ≥ 5 %);
E12b corruption = the single KV buffer.

**Findings that change earlier numbers:**
* **E8 / E8s LIBERO fed the empty camera as a real image.** `bench/libero_eval.py` supplied
  `observation.images.empty_camera_0` as an all-zero image, so LeRobot treated it as present (mask 1, 256 tokens
  attended); in training (and in every fixture / timing run) it is a −1 image with mask 0. Paired comparisons are
  unaffected; absolute success rates are for that input. Same arm, attended vs masked: 71 vs 66 / 100 (p = 0.36).
  `--empty-camera masked` added; correction notes in `E8_libero.md` and `E8s_libero_seeded.md`.
* GR00T's first E14 action sets were invalid (unseeded random observation per process) — re-run, kept as
  `invalid_unseeded_inputs/`.

**Questions for the research side (new):**
1. E8 / E8s ran with the empty camera attended (above). Re-run any LIBERO arms with `--empty-camera masked`, or read
   them as they are? (Most conservative default taken: nothing re-run beyond the T016 nopad pair; all numbers kept
   with the note.)
2. π0 `nopad` and GR00T `keep1` / `kvonce` are labelled variants; whether any enters the GPU reference is OD-1 (yours).

Repository (protocol §6f), at close of T016: committed `results/` 529 MB (1.4 GB on disk incl. uncommitted
scratch-grade files), `git count-objects -vH`: size-pack 158.10 MiB + loose 84.34 MiB; `.git` 245 MB.

**Where things sit:** T017 done (block above). The MODE_15W session still needs the owner's reboot.

---

## ▶ ORDER (owner, 2026-09-26): T016 BEFORE MODE_15W

The owner chose **order B**: reboot MODE_30W → MAXN now, run **T016** (step 1 first, pushed before step 2),
and do the MODE_15W session afterwards (MAXN → 15W → MAXN, two more owner reboots). T016's own text says to
finish capped queues first; the owner's direct order wins (T016 "When"). The 15W queues are ready and fixed.

---

## ✅ MODE_30W SESSION DONE (2026-09-26 11:15–14:09) — MODE_15W next (needs the owner's reboot)

Device after the owner's reboot: `MODE_30W`, TPC mask 240 → **8 SMs**, GPU locked **612 MHz**, CPU 8 cores
@ 1.728 GHz, EMC 3199 MHz, fan max — as `/etc/nvpmodel.conf` defines it. All results `results/20260926_*`.

* **E7 (T004 4b)**: power loops eager + best for π0, π0.5, SmolVLA and GR00T N1.5/1.6/1.7 at N = 10, and π0
  E3 latency at N = 4 / 10 (eager e2e 2066 / 2457 ms; per-step 67.6 / 66.2 ms, 1.25–1.28× MAXN = the CPU
  clock ratio 1.27: the eager expert stays host-bound).
* **T014 Step 4 (E12 at 8 hardware SMs)**, against the MAXN software-partition references:

| GPU-bound quantity | MAXN green 8 SMs @ 1300.5 MHz | MODE_30W 8 SMs @ 612 MHz | ratio | bracket (1.00–2.125×) |
|---|---:|---:|---:|---|
| 2048² bf16 GEMM control | 0.8915 ms | 1.9840 ms | **2.23×** | 5 % above the clock ratio |
| compiled expert (`owngraph_expert`) | 149.41 ms | 286.13 ms | **1.92×** | inside |
| compiled prefix (`graphs_both`) | 327.70 ms | 876.84 ms | **2.68×** | **ABOVE — prediction falsified for this stage** |
| eager prefix, 8 / 4 SMs (green curve) | 779.8 / 1425.1 ms | 1789.7 / 3091.6 ms | 2.30× / 2.17× | ~clock ratio |

  Reading: for compute (the GEMM) and for the eager stages a software 8-SM partition matches 8 hardware SMs
  after clock scaling (≤ 8 % off). **The compiled `graphs_both` prefix does not** — it slows 26 % more than
  the clock ratio allows. Not explained yet (CPU: its host issue equals its wall time at both modes, so
  the 1.27× slower CPU is one candidate; not tested). Eager co-run at 30W is serial again (T_pipe 2439.1 vs
  serial 2437.3 ms) and saves no energy (0.9996); equivalence PASS 6/6 bitwise, stale control 5/5.
* **Defects found and fixed during the session** (the capped queues were written at MAXN and never
  exercised): `corun.py`/`corun_green.py`/`corun_energy.py` refused the MAXN-only gate (now
  `--allow-capped`); `run_e3.py` had no `--outdir` (added — the default name carried no power mode);
  `corun_green.py`'s verdict was hard-coded 8-vs-16 (read 8-vs-8 at 30W; now half vs whole GPU of the
  mode); `corun.py` equivalence wrote no `meta.json` (added; 30W back-filled). The failed steps were
  re-run in the same boot (`scratch/queues/fixup_30w.sh`).

**Next, owner:** `sudo nvpmodel -m 1` (MODE_15W: 6 SMs, GPU 408 MHz, 4 CPU cores @ 1.11 GHz, EMC 2133 MHz),
confirm the reboot, then `claude --continue`; I run both queues (now fixed) under the 10-minute watchdog.

---

## ✅ PRE-REBOOT WORK DONE (2026-09-26 ~00:00) — T006, T008, T009 done; E12b review answered; only the reboot session remains

**Closed today** (every item on its task file's status block, one result directory per commit):

* **T006** — `reports/E3_E4_models.md` (per-step slopes eager → compiled: π0 52.94 → 16.30, π0.5
  61.60 → 18.40, SmolVLA 61.46 → 4.33, GR00T N1.7 51.75 → 18.13 ms; gap to floor compiled 4.8×, 3.2×,
  4.0×, **1.05×**) and **`reports/E8_libero.md`**: π0's full compiled `bf16_both` rung succeeds as often as
  the shipped checkpoint on all four LIBERO suites (**249 vs 252 / 400**, paired p = 0.87) with a 2.45×
  faster forward; `libero_spatial` N = 10 / 4 / 2 → **62 / 74 / 85** / 100 (fewer steps did not hurt —
  measured, not explained). The same arm run twice gave 63 and 56: π0's flow noise is on an unseeded
  torch RNG, so episode outcomes are not seed-determined; arm differences of a few episodes are noise.
* **T009** — GR00T N1.7 N-sweep + `reports/E3_E4_groot.md`. NVIDIA's own `benchmark_inference.py` first
  failed (torchcodec missing; kept), then ran after installing torchcodec `--no-deps` from the repo's own
  Orin wheel (`env/INSTALL_LOG.md`; freeze diff = that one line): its eager action head agrees with ours
  to **0.6 %**.
* **T008** — the last item, step 6's S sweep for π0.5 / SmolVLA (`results/20260925_E5b_{pi05,smolvla}_ssweep`).

**Corrections found and made (all disclosed in their reports / commits):**

1. **S-sweep timings were harness overhead** (`bench/micro_proj_sweep.py` put an event pair + sync
   around every call: +45–75 µs on Orin). Kernel times now come from the sweeps' own ncu captures
   (`s_sweep_ncu.csv`). **`E5b_groot.md` corrected: GR00T's largest DiT projection is at 1.17–1.20× its
   memory floor (was quoted 1.6×), `to_q` at 1.44–1.47× (was 3.5–3.7×).**
2. `bench/opt_ladder.py` wrote π0's checkpoint into every model's `meta.json`; 4 π0.5 / SmolVLA metas
   corrected with originals kept (data unaffected).
3. E8's harness wrote no per-step log (T006 requires one); fixed mid-run, the affected arm re-run.

**E12b review — the device's answers** (addendum at the end of `reports/E12b_pi0_compiled_corun.md`):

* **EMC-vs-bytes gap: measured** (`results/20260925_E12b_pi0_dram_bytes_MAXN`; Orin has no `dram__`
  counters, so L2 fills + writes). **The EMC calibration is bytes** (EMC-implied / measured = 0.96 prefix,
  0.995 expert). **The prefix really moves 21.6 GB/frame — 3.7× E6's analytic bytes** (activations of the
  816-token prefill spill past the 4 MB L2; 76 % of its 74 GB of L2 reads hit) — i.e. ~88 GB/s of real
  demand, which matters for EV-02. The expert: 8.2 GB/frame (1.29× E6, reads alone 1.10×).
* `e12b_diagnose.py` never existed in git — said so in the harness; the M2 diagnosis stays a suspect
  list. `variant_tokens` all defined from the code. `hog_matched` was matched to E6 bytes, so ~22 % light.
* T014 Step 4: STATUS table restated with the confined 8/8 references and a bracketed prediction (block
  below); `queue_e12_capped.sh` step 0c measures the compiled stages at `MODE_30W`; its script's MAXN
  references reproduce M3 within 0.25 %.
* Not done: a one-change-at-a-time M2 isolation; `graphs_both` vs rung-prefix equivalence.

**Questions for the research side (new):**

* The `owngraph_expert` question below is still open.
* Many older `meta.json` files carry `model_commit: null` under that key (the commit is often recorded
  under another key, e.g. `checkpoint_commit` / `revision`). Provenance gap, not a wrong value — want a
  back-fill pass, or is the per-report citation enough?
* E8: should a follow-up seed torch per episode (paired noise) and extend the N sweep to the other
  suites / the stock checkpoint? Not done — outside the owner's Core scope.

**Repository (§6f):** tracked files 413.9 MB (results/ 408.6 MB tracked, 1.2 G on disk incl. untracked
scratch); largest tracked file 13.6 MB (`results/20260922_E4g_pi0_ncu_eager/ncu/ae_step0_details.csv`);
`git count-objects -vH` size 246.36 MiB loose, size-pack 62.49 KiB.

**What remains:** only the owner's capped-mode reboot session — `bench/queues/queue_e7_capped.sh` and
`bench/queues/queue_e12_capped.sh` at MODE_30W, then MODE_15W (sections below). Nothing else is queued.

---

## 🛑 CORRECTION (2026-09-25, from T015 M0/M2/M3) — T014's compiled green-context claims were wrong

**Do not use** T014's "the action expert is SM-independent at `bf16_both` (4 SMs cost 4.9 %)", the
derived 12/4 co-run bound of 271.26 ms, or "the expert can neither block nor be blocked at the CTA
dispatcher" (`reports/E12_pi0_corun.md` §5, and the E6/STATUS text that cited them).

* **Why they were wrong.** E12b M0: a CUDA graph obeys the partition it was *captured* in, not the
  green stream it is replayed on (primary-captured graph on a 4-SM stream: 0.99×; eager launches and
  in-partition captures: 3.24–3.35×). `bf16_both` replays CUDA-graph trees recorded outside the
  partition, so T014's compiled "4-SM" points ran on the whole GPU.
* **What is true instead.** Confined properly, the **compiled expert on 4 SMs is 254.9 ms, 2.50× its
  16-SM time** — SM-sensitive. On the whole GPU the two stages **do** block each other: E12b M2's
  timeline shows kernels of both streams executing together for 3.1 % of a co-run frame, with the
  expert queued 175 ms/frame behind the prefix's 1008-CTA GEMMs (head-of-line blocking). The
  research side's review was right to reject the claim.
* **What stands.** Everything eager: the eager expert is launch-bound and SM-insensitive (eager
  launches are confined, M0), `issue/wall`-free facts such as `T_pipe = A + B` in eager, and E6's
  byte/FLOP analysis.

---

## ✅ T015 / E12b DONE (2026-09-25) — one Orin GPU does not overlap π0's prefix and a fair expert well

`reports/E12b_pi0_compiled_corun.md`. **Best valid co-run `O` = 1.247** (green 12/4; poor, band ≥ 1.15);
whole GPU **1.39**. Why: on the whole GPU the streams time-slice — kernels of both execute together for
3.1 % of a frame, the expert queued 175 ms/frame behind the prefix's 1008-CTA GEMMs (head-of-line
blocking, KerColle's mechanism, on silicon); partitioned, **the stages still slow each other through the
memory system** (`s_part` 1.16–1.26, `p_part` 1.05–1.10; hog control passes) — **row 2 lands:
partitioning SMs does not partition bandwidth**. Fair expert (`bf16_both` + hoist, bitwise equal to
stock): **110.95 ms** — rows 7 and 8 do not land for it. **A labelled exact variant,
`owngraph_expert` (the whole 10-step loop as one CUDA graph, bitwise equal), runs 101.2 ms and DOES
fall in row 8's band (93.1–102.4) — whether it enters the GPU baseline is your call** (question below).
Co-run energy per frame **1.020 ×** fair serial (row 10 does not land). All equivalence checks bitwise,
stale controls detected, 0 non-finite.

**Question for the research side:** should `owngraph_expert` (capturing the N-step expert loop as one
CUDA graph; exact, GPU-agnostic, bitwise identical) join the hoist in the fair GPU baseline? It moves the
GPU expert from 110.95 to 101.2 ms at N = 10.

**Repository (§6f):** results/ 372.7 MB tracked, 1.2G on disk; largest E12b file 9.0 MB.

**Queue then (2026-09-25 evening):** pre-reboot queue resumed — now DONE, see the block above.

---

## (was: NOW) T015 / E12b — where it sat in the queue

**The owner directed it directly**: T015 was taken immediately, and the pre-reboot queue the owner had
started earlier (`bench/queues/queue_prereboot.sh`: E3/E4 N sweep → GR00T N1.5/N1.6 E6 → E8 LIBERO) is
**paused**, to resume after T015. Its four completed N-sweep points are kept in
`scratch/paused_runs/20260925_E3E4_pi05_Nsweep/` (not deleted). The capped-mode session (T004 step 4b,
T014 Step 4) still waits for the owner's reboot and comes after both.

### T014 Step-4 prediction — RESTATED per E12b spec §14 (the earlier one was wrong)

The earlier prediction scaled the **eager** expert (587.56 ms at 8 green SMs) by the GPU-clock ratio. That was
the wrong quantity: the eager expert is **host-issue-bound** (flat SM curve; E5 84 % idle), and `MODE_30W`
slows the **CPU** too (8 cores at ≤ 1.73 GHz vs 12 at 2.2016 GHz, recorded 2026-09-25 10:17 at MAXN: all 12
cores at 2 201 600 kHz, GPU 1 300 500 000 Hz). A deviation from it would not have shown the green-context
result wrong. **The fair cross-check uses GPU-bound quantities, scaled by the GPU clock only:**

> **Revised 2026-09-25 evening after `analysis/E12b_review.md`.** The first version of this table
> (written 24 min before M0) quoted T014's **unconfined** 8-SM compiled values (prefix 320.71, expert
> 113.70 ms), which M0 withdrew, and scaled them by the GPU clock alone; and the capped queue did not
> measure compiled stages at all, so those rows could be neither tested nor falsified. Fixed: the
> references are E12b M3's **confined** 8/8 values; the prediction is a **bracket** (a memory-bound
> stage need not slow with the GPU clock); `queue_e12_capped.sh` step 0c now measures the compiled
> stages through the same harness (`bench/e12_capped_compiled_stages.py`), and the same script is
> queued at MAXN (whole GPU + green 8/8) as its same-script reference.

| quantity (GPU-bound) | MAXN, green 8 SMs @ 1300.5 MHz (confined) | MODE_30W prediction (8 hardware SMs) |
|---|---:|---:|
| 2048² bf16 GEMM control | 0.8909–0.8921 ms/GEMM | ≈ 0.89 × (1300.5 / f_GPU,30W) ms (compute-bound) |
| compiled prefix (`graphs_both`) | A_part **326.94 ms** | between 326.94 and 326.94 × (1300.5 / f_GPU,30W) |
| compiled expert (`owngraph_expert`, graph replay) | B_part **149.05 ms** | between 149.05 and 149.05 × (1300.5 / f_GPU,30W) |

Lower end = memory-bound (no GPU-clock effect), upper end = compute-bound (full clock ratio); the
`MODE_30W` EMC rate is recorded in step 0 / the script's `meta.json`, and a stage **below** the lower end
or **above** the upper end falsifies the green-8 ≈ hardware-8 equivalence for that stage. Sources:
`results/20260925_E12b_pi0_green_corun_N10_MAXN/corun_ratios.csv` (split 8/8).

with the CPU clocks recorded in both modes (`/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`). The
eager-expert comparison stays in the queue as a CPU+GPU observation, **not** as a test of the green-context
method. `bench/queues/queue_e12_capped.sh` now records the CPU clocks and runs the GEMM control first.

---

## ⚠ CORRECTION (2026-09-25, same day) to the GR00T E6 block just below: it compared mismatched rungs

**The GR00T row was a CUDA-graphed action head; the π0/π0.5/SmolVLA rows were eager.** GR00T's
region time came from the `vlm_graphs` ladder run (`--head-mode reduce-overhead`), and I set it
against eager LeRobot numbers without checking the rung. Matched, from GR00T's own ladder rows
(bytes identical across rungs):

| | eager | action head compiled |
|---|---:|---:|
| GR00T N1.7 `ae_step` | 52.13 ms · **33.2 %** peak BW · **3.02×** floor | 20.82 ms · **83.0 %** · **1.20×** (`ae_compile`) |
| π0 `ae_step` | 57.35 ms · **5.9 %** · **16.85×** | 16.87 ms · **20.2 %** · **4.96×** (`bf16_both`) |
| π0.5 `ae_step` | 61.14 ms · 15.1 % · 6.61× | 18.56 ms · 31.4 % · 3.18× (`bf16_both_graphs`) |
| SmolVLA `ae_step` | 61.29 ms · 1.8 % · 56.50× | 4.49 ms · 24.2 % · 4.14× (`vlm_graphs`) |

**Matched-rung E6 landed** (`results/20260925_E6_rungs/`, `reports/E5_E5b_models.md`): every row
above uses Part B's unprofiled ladder time and bytes counted after that rung's casts. Compiling
brings every model 2–14× closer to its floor; **only GR00T becomes bandwidth-bound**. The
compiled-vs-compiled comparison is the fair one, and it supports the GR00T framing: remove launch
overhead and GR00T's DiT sits at the bus, while the LeRobot experts stay 3–5× off theirs.

**What changes:** "GR00T is bandwidth-bound and near-optimal" is true only once its head is
compiled; eager, it is launch-limited too (3.0× floor). **What survives, and is sharper:** at
matched eager rungs GR00T is still ~5× closer to its roof than π0, and **compiling GR00T's head
alone removes almost all of its launch overhead** (3.02× → 1.20×). The fair test of the PIM
framing is therefore compiled-vs-compiled, which the matched-rung E6 now being built will give
for every model — now landed (table above). The block below stays superseded.

---

## 🔬🔬 (SUPERSEDED — see the correction above) GR00T N1.7's action step runs at 86 % of peak bandwidth, 1.16× its floor

**This is the strongest measured support in the whole campaign for a memory-side accelerator,
and it points at a different model than the study started from.**

| model | `ae_step` ms | bytes/step | % peak BW | gap × floor | reading |
|---|---:|---:|---:|---:|---|
| π0 | 52.79 | 638 MB | 6.45 % | 15.51× | launch-bound |
| π0.5 | 62.69 | 1 735 MB | 14.76 % | 6.77× | launch-bound |
| SmolVLA | 60.80 | 203 MB | 1.78 % | 56.04× | launch-bound, extreme |
| **GR00T N1.7** | **20.06** | **3 241 MB** | **86.17 %** | **1.161×** | **bandwidth-bound** |

GR00T N1.7 streams a 1.09-billion-parameter DiT across the memory bus every denoising step and
already sustains **161.5 GB/s of the 187.47 GB/s** this device can deliver, at **1.161× its
physical floor** — protocol §9 reads < 1.5× as *near-optimal*. **There is no software overhead
left to remove.** The only ways to make that step faster are to move less data or to move it
faster, which is precisely what processing-in-memory offers. For π0 and SmolVLA the same
accelerator would target a ceiling they use 6.5 % and 1.8 % of.

**Why the byte count can be trusted**, since the result rests on it: `ae_step` bytes assume
every step reads the whole action head, and that is validated by the gap ratio itself. A ratio
below 1.0 is physically impossible — nothing beats the floor computed from the bytes it moves.
Over-counting by even 20 % would have pushed the floor past the measured 20.06 ms and the ratio
under 1. At 1.161× the count is consistent and cannot be a large overestimate.

**Suggested framing:** make GR00T N1.7 (or a comparably large DiT action head) the subject of
the bandwidth argument, and cite π0/π0.5/SmolVLA as the contrasting case where the same stage is
launch-bound and a bandwidth accelerator would not help. That is a stronger paper than treating
all four as one population, and every number for it is now measured on real silicon.

FLOPs for GR00T are deliberately **not** counted: its DiT blocks alternate self- and
cross-attention, so π0's linear-plus-self-attention formula does not apply and using it would
have produced an authoritative-looking wrong number. Compute columns are empty, not 0.
Outstanding: N1.5 and N1.6 (separate `Isaac-GR00T` branches and venvs), and GR00T's `vision`
byte split (Cosmos-Reason2-2B does not name its tower `vision_model`/`vision_tower`, so the
script recorded nothing rather than guessing).

---

## 🔬 FOR THE PAPER — E6 now exists, and it changes which argument the PIM case can make

**Every action expert measured is on the memory side of Orin's roofline, and none is anywhere
near it.** E6 (roofline) had never been run for any model; `bench/roofline.py` was written for
it and never invoked, and `floor.csv` existed only for π0. Both gaps are closed for π0, π0.5 and
SmolVLA.

| model | `ae_step` ms | intensity | side of ridge (180.57 FLOP/B) | % peak BW | % peak compute | gap × floor |
|---|---:|---:|---|---:|---:|---:|
| π0 | 52.79 | 60.0 | memory | **6.45 %** | **2.14 %** | **15.51×** |
| π0.5 | 62.69 | 28.8 | memory | **14.76 %** | **2.36 %** | **6.77×** |
| SmolVLA | 60.80 | 52.5 | memory | **1.78 %** | **0.52 %** | **56.04×** |

Roof = this Orin's own measured peaks (E1/T005: 187.47 GB/s, 33.85 TFLOP/s bf16). Protocol §9
reads a gap above 10× as *a structural problem — launch-bound, idle GPU, too little
parallelism*.

**The premise holds analytically and fails operationally.** Intensity below the ridge means a
memory-side accelerator aims at the right half of the chart — that part of the thesis survives,
and our analytic intensities (VLM 816.0, expert 60.0 FLOP/B) match arXiv:2604.24447's 840 and
64.5, so both derivations describe the same model. But no expert reaches even 15 % of peak
bandwidth or 3 % of peak compute. **Memory is the nearer of two distant roofs, not a limit any
of these models hits.**

**Five independent measurements now agree on what actually limits the expert**, which is why
this is stated as a conclusion:

| evidence | experiment | figure |
|---|---|---|
| GPU idle during the expert | E5 | 84.2 % over 968 kernels/step (π0 eager) |
| no configuration is memory-latency-bound | E5b | long-scoreboard 19.5–47.6 %, none meets §7's ≥ 60 % |
| the expert does not use the SMs | E12 §5 | 4× fewer SMs costs it **1.3 %** |
| the host issues for the whole wall clock | E12 §1 | `issue/wall` = 1.0000 |
| distance from the physical floor | **E6** | **6.8–56×** |

**What we suggest this means for the paper.** If the argument is *bandwidth* — the expert is
memory-bound, so put compute in memory — these tables contradict it: the expert uses 1.8–14.8 %
of the bandwidth it already has. If the argument is *launch and dispatch overhead* — hundreds to
a thousand small kernels per denoising step, a GPU idle 84 % of the phase, a stage that runs the
same on 4 SMs as on 16 — every measurement here supports it, and the ceiling on the win is what
that overhead costs. **That is a change of reason, not a verdict against the design.** It also
implies the win should be largest for SmolVLA (56× floor, 1183.7 kernels/step) and smallest for
π0.5 (6.8×), which is a prediction the design can be tested against.

**Also worth having:** π0's `vlm_prefix` is the one efficient region anywhere — 1.93× its floor
at 51.92 % of peak compute. There is little to win there by any means.

**One error found and fixed in our own derivation, disclosed because it would have been
invisible:** SmolVLA's config declares `num_hidden_layers = 32` while the loaded model has
**16** (it truncates at load and prints so). Reading the config doubled its KV-cache bytes and
attention FLOPs — a floor wrong by 2× that looked entirely plausible. `bench/floors_models.py`
now counts layers from the loaded module list and refuses to fall back to the config. The
superseded run is in `scratch/superseded/`, not deleted.

**Still missing:** E6 for the three GR00T variants. GR00T is a different family (a DiT action
head, not a Gemma expert behind PaliGemma) with differently named regions
(`vlm_backbone`, `vlm_prefix_excl_vision`), and one function covering both would produce a floor
that looks authoritative and is wrong. T009 needs its own pass.

---

## 🛑 FOR THE RESEARCH SIDE — E5b's raw data reached the repository only on 2026-09-25

**If you pulled before commit `ffccd18` (2026-09-25 06:40) and looked for E5b's per-kernel raw
data, it was not there.** The reports were; the data behind them was not. Re-pull and it is.

**What happened.** 492 result files were sitting **untracked** in `results/` while
`STATUS.md` and my own watchdog both reported E5b as "DONE 2026-09-25, 6/6 configurations, 24
captures". Affected: all six `20260924_E5b_*` directories (SASS inventories, stall
attribution, per-kernel CSV exports, the compressed `.ncu-rep.zst`), several `_E7_*` power
directories' `raw/` (tegrastats logs, `rail_samples.csv`) and the E12 two-process run. They
were never committed by the sessions that produced them. AGENTS.md is explicit — *"Commit the
raw data. We want as much raw data as possible"* — and I did not, while reporting the work
complete.

**How it was found, and the honest part of that.** Not by design. I wrote
`bench/backfill_meta.py` to repair missing `meta.json` files, and its `git log
--diff-filter=A` field reported *my own commit from this session* as the commit that first
added `results/20260924_E5b_pi0_best`. I nearly dismissed that as a bug in my script. It was
not: the directory genuinely entered git for the first time in `ffccd18`, because that commit
used `git add -A`.

**So a `git add -A` I should not have used is what closed the gap.** That commit was audited
afterwards against the rules it could have broken — 0 forbidden extensions
(`.nsys-rep`/`.sqlite`/weights), largest file 8.03 MB against the 50 MB cap, 0 secret-pattern
hits across all 492 files. It was safe, but it was safe by luck plus `.gitignore`, not because
I checked before committing. The lesson is the reverse of the usual one: **the per-commit
check AGENTS.md requires would have caught the accident, and a per-task check that the task's
own raw data is tracked would have caught the original omission.**

**What is verified compliant**, so you know the audit's scope: the 34 `.nsys-rep` + 34
`.sqlite` remain correctly excluded (§3.5, they embed a session token); the four oversized
`.ncu-rep` in `20260922_E4g_pi0_ncu_eager` (88–213 MB) are documented in that directory's
`RAW_FILES.md` with bytes, sha256, a secret scan and a private HF dataset mirror. Repository
now: **345.70 MB tracked under `results/`**, 1.2 GB on disk, `scratch/` 2.2 GB never
committed — well inside the ~2 GB ceiling of protocol §6.

**`meta.json` back-filled for 33 directories** (`bench/backfill_meta.py`). It recovers only
what evidence supports: the exact command where the capture's own `.sh` wrapper survives
(21 directories), the file inventory with sha256 and bytes per AGENTS.md, and the commit that
added the directory. **Power mode, clocks, temperature and throttling are left `null` with a
reason** — they were never recorded, and the campaign-wide "it was at MAXN" is not a per-run
measurement. `20260925_E12_pi0_2proc_N10_MAXN` additionally carries `_run_outcome` stating it
is the OOM-killed run and that none of its numbers may be used; it is still reported BROKEN by
the watchdog, which keys on `corun_ratios.csv`, not on `meta.json`.

Also checked and found sound: the empty `20260923_E8q_pi05_libero_bf16both` is a **deliberate
documented skip** — π0.5's fp32 control scored 0/30 because `pi05_libero_base` is a base
checkpoint, and a bf16 arm compared against zero distinguishes nothing. Git cannot store an
empty directory, so nothing was ever published under that name.

---

## ✅ RESOLVED + 🛠 A FREE 33.6 % OPTIMISATION TO π0 — E7's `ae` time reconciled (2026-09-25)

**Short version for the research side: `r = A/B = 1.50` from E7 is CORRECT and you may quote
it. And π0 leaves 33.6 % of its action-expert time on the table for a one-line change.**

E12's isolated expert measured **111.01 ms** where E7's `ae` reports **167.36 ms** at the same
rung (`bf16_both`), N = 10 and power mode — 50.8 % apart, while the prefix agreed to +0.24 %.
That was chased to the end, because a third of one stage unexplained invalidates any `r`
derived from it.

**Cause: one line.** π0's shipping `sample_actions` (`modeling_pi0.py:858-859`) — and E7's
`phase_builders.make_ae`, which faithfully copies it — builds the timestep tensor *inside* the
denoise loop:

```python
time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
```

`torch.tensor(<python float>, device="cuda")` copies from **pageable** host memory, so the host
blocks until the copy lands — i.e. until the stream drains every kernel queued ahead of it. On a
launch-bound stage that throws away the host's run-ahead once per denoise step.

| expert variant, π0 `bf16_both`, N = 10, MAXN | median | n |
|---|---:|---:|
| timestep tensors hoisted out of the loop | **111.02 ms** | 406 |
| E7's / shipping π0's per-step construction | **167.09 ms** | 270 |
| *E7's reported `ae`* | *167.36 ms* | — |

**167.09 vs 167.36 = 0.16 %.** Reconciled. Two other explanations were measured and rejected
first — E7's Rails/tegrastats samplers (+0.19 %) and in-process residency of the prefix's
graphs and pools (+0.29 %) — so this is not offsetting errors.

**What you can use.**
1. **`r = 1.50` stands.** E7 correctly measured the shipping model.
2. **A second free optimisation to π0, worth 33.6 % of the action expert** (56.07 ms of
   167.09 at N = 10): hoist the N constant timestep tensors out of the denoise loop. Simpler
   than the `att_masks` de-sync — no memoisation, no compile-safety question. If the paper
   quotes an expert-stage latency for π0, it is worth saying that a third of it is a pageable
   host copy rather than expert compute.
3. **π0 hands the GPU back to the host `1 + 2N` times per frame** (21 at N = 10): `1 + N`
   `att_masks` conversions plus `N` timestep copies. This *strengthens* E12's finding that
   nothing overlaps; it does not change it. `T_pipe` is still `A_iso + B_iso`.

**My own error, disclosed.** `reports/E12_pi0_corun.md` §6 said precomputing those tensors was
"not labelled, because §7 names it as the example of a permitted harness fix". I wrote that
believing the change was free without measuring it. It is worth a third of the stage. Corrected
in §3 and §6 of that report. **"Permitted by the protocol" was allowed to stand in for
"measured to be small".**

**Also corrected, narrower:** E12 §3 called the `att_masks` de-sync "worth taking regardless".
It is **eager-only as implemented** — under `bf16_both` it raises `accessing tensor output of
CUDAGraphs that has been overwritten`, because the memoised tensor is allocated inside the
compiled region. E12's MAXN run is eager, so its 19 % figure is sound for what it measured.

**Also fixed, in my own tooling:** `bench/watchdog.sh` §2c globbed only `*_E7_*`, `*_E11*`,
`*_E5*`, so it reported **7** result directories missing `meta.json` when the true number is
**33** (E4, E8q and E12 were never examined), and it printed `missing[:4]` beside the full
count. Both fixed; it now scopes every result directory and prints all of them grouped by
experiment. The `meta.json` back-fill for the genuine runs among those 33 is queued.

Evidence: `scratch/NOTE_e7_ae_time_discrepancy.txt`, `scratch/e12_sampler_test/*.json`,
`bench/corun_2proc.py --e7-timesteps|--preheat-prefix|--sampler`.

---

## 🛑 CORRECTION FOR THE RESEARCH SIDE — two E11 quantities were WRONG in a pushed commit

**If you pulled commit `493c548` (2026-09-24 ~19:00) and used `gpu_idle_ae_total` or
`gpu_idle_frac_ae_total` for π0.5 or SmolVLA, discard those two values and re-pull.**
Everything else in that commit is unaffected.

**What was wrong.** `bench/pipelining.py` launched its nsys sub-capture as
`nsys_target.py --steps N --inferences 2` with no `--model`, and that flag defaults to
`pi0`. So E11f's `gpu_idle` rows for **every** model were π0's numbers recorded under that
model's name. Two of §13.8's 35 quantities were affected, in the consolidated
`model_inputs.csv` that you read directly.

**How it was proven**, so you can check rather than take my word:

| | kernels/step N=4 | N=10 | `ae_total` N=4 | idle N=4 |
|---|---:|---:|---:|---:|
| E11f rows labelled `smolvla` (wrong) | 968.25 | 968.70 | 343.54 ms | 0.8426 |
| π0's own E5 eager | 968.25 | 968.7 | 343.75 ms | 0.8427 |
| SmolVLA's real value (E5) | **1183.7** | — | — | **0.9273** |

**Fixed** in `a49e7e9`; E11f re-run for π0.5 and SmolVLA and both re-consolidated. Each
re-run now verifies its own identity by comparing kernels/step against that model's own E5
capture, and prints `*** STILL THE WRONG MODEL ***` if they disagree:

```
VERIFY pi05    N=4: 1102.25 vs its own E5 1102.70 -> OK    N=10: 1102.70 vs 1102.70 -> OK
VERIFY smolvla N=4: 1183.25 vs its own E5 1183.70 -> OK    N=10: 1183.70 vs 1183.70 -> OK
```

**π0's own E11f rows were never wrong** — the bad default happened to be π0 — so they are
not re-measured. The contaminated directories are kept at
`scratch/wrongmodel_20260924_E11f_{pi05,smolvla}_pipelining` rather than deleted, so the
wrong numbers stay inspectable beside the commit that explains them.

**Why it got through.** It wrote wrong numbers instead of crashing, and every artefact was
internally self-consistent — it simply described the wrong model. Nothing cross-checked
identity. A kernel launch count turns out to be a good fingerprint for exactly this: it is
deterministic, model-specific, and already recorded independently in E5.

**A claim of mine that this invalidated:** I had reported SmolVLA's eager `ae` GPU-idle as
84.0 % and argued it was a counter-example to the GPU-idle hypothesis registered in
`reports/E7_power_energy.md`. That was π0's number. SmolVLA's real eager `ae_step` idle is
**0.9273**, and with the correct values the three LeRobot models are monotonic in idle drop
against their E7 energy-to-time ratios (π0.5 0.198/0.96, π0 0.370/0.80, SmolVLA
0.548/0.45). The hypothesis is **supported**; my falsification of it was an artefact of this
bug. It still needs the GR00T models before anyone writes it up as confirmed.

---

## ✅ MODE_50W is DONE and the device is back at MAXN (2026-09-25 00:50)

18 runs at MODE_50W: 12 LeRobot + 6 GR00T, 0 failures. `sudo nvpmodel -m 3` and the return
`-m 0` both took with **no reboot prompt**, because 50W's `TPC_PG_MASK` is 0, the same as
MAXN's, and TPC gating is what is applied at boot. Both switches were run with stdin closed
so a prompt would have received EOF rather than a confirmation.

**The gate after restoring is byte-identical to the one recorded before switching**: 16 SMs,
mask 0, 1300.5 MHz, min==max locked, CPU 2201.6 MHz. Checked against the stored pre-switch
gate rather than merely checking "is it MAXN", because a device back in MAXN with unlocked
clocks would pass the naive test and silently corrupt everything measured afterwards.

### The one robust result: underclocking loses, on every model

`reports/E7_power_energy.md` has the full table, 12 pairable points, paired on
`(model, rung, matmul_precision)`:

* **energy·delay is WORSE at 50W in all 12** — 1.196× to 1.400×. No model, eager or
  compiled, benefits.
* energy alone falls **0.6 % to 15.1 %**, and costs **37 % to 55 %** more latency.
  SmolVLA eager saves essentially nothing (×0.994).
* So the frequency knob already shows that trading latency for energy loses here. Worth
  knowing for a study whose case is partly about joules.

### Three hypotheses registered, three falsified

1. Energy-to-time ratio tracks GPU-idle fraction (registered in `reports/E7_power_energy.md`)
   — monotonic over π0/π0.5/SmolVLA, broken by GR00T.
2. Clock-insensitive fraction tracks GPU-idle fraction (registered 2026-09-24 23:36) —
   same shape, broken by GR00T, and **inverted**: GR00T's compiled head has 0.0008 idle and
   the *highest* clock-insensitive fraction of all 12 points (37.3 %). It also fails within
   GR00T — N1.6 and N1.7 have near-identical eager idle and fractions 2.2× apart.
3. The action expert is memory-latency bound — **no configuration** meets §7's ≥60 %
   threshold (E5b, six configurations, 19.5–47.6 % time-weighted).

**Two independent relationships both fitted the LeRobot three and both broke on the DiT
head.** Treat any relationship fitted on those three as LeRobot-specific until a DiT model
confirms it — two for two is enough to make that the default rather than a caveat.

---

## ⚠ CORRECTION — E11f's "high priority stream" was the DEFAULT priority

`bench/pipelining.py` read `lo, hi = priority_range()` and created its contender stream with
`priority=lo`, under a comment asserting `lo` was the highest. torch returns
**`(least_priority, greatest_priority)`** and in CUDA a **lower number is a higher
priority**, so on this device `(0, −3)` means `0` is the default and `−3` is the genuinely
high-priority stream. Verified:

```
Stream(priority= 0) -> priority= 0    [least]   == default Stream() -> 0
Stream(priority=-3) -> priority=-3    [greatest]
```

So every committed E11f two-stream number — π0.5 `two_streams` 2.75×, SmolVLA 1.79–1.80× —
was measured with **no priority differentiation between the streams**. The field
`high_priority_stream_accepted: true` is literally true and substantively empty: it accepted
the default. **No high-priority stream has been measured on this device yet.**

The numbers are not wrong for what they are — a two-stream contention measurement at equal
priority — they were mislabelled. `pipelining.py` now records
`contender_stream_priority` and `contender_stream_is_elevated` instead of a boolean that
reads as a claim, and its behaviour is deliberately unchanged so it keeps measuring what its
committed results measured. A genuinely elevated stream is measured in **E12 Step 3**, as
its own labelled mechanism.

Flagged by `docs/E12_corun/README.md` §5, which caught it before I built on it.

---

## ▶ NEXT, needs you: resume E7 at MODE_30W (id 2), then MODE_15W (id 1)

**Everything that can be done without a reboot is done.** MODE_50W ran unattended tonight
because its TPC mask is identical to MAXN's. 30W and 15W gate TPCs, which is applied at
boot, so they need you.

### Why these two genuinely need a reboot and 50W did not

From `/etc/nvpmodel.conf` on this board, plus the clocks measured after switching:

| mode | `TPC_PG_MASK` | SMs | `EMC MAX_FREQ` | GPU clock | reboot? |
|---|---:|---:|---|---:|---|
| MAXN (0) | 0 | 16 | `-1` unrestricted | 1300.5 MHz | — |
| **MODE_50W (3)** | **0** | **16** | `-1` | **816.0 MHz** | **no — already done** |
| MODE_30W (2) | 240 | **8** | `-1` | (unmeasured) | **yes** |
| MODE_15W (1) | 248 | **6** | **2133 MHz capped** | (unmeasured) | **yes** |

**These four are not one sweep.** 50W is frequency-only at constant SMs and constant memory
bandwidth. 30W halves the SM count. 15W cuts SMs further *and* caps memory. Any table that
lines all four up as "power budget vs latency" is conflating three different perturbations —
say which knob moved in each.

### Exact commands

```bash
# 1. switch, confirm the prompt yourself, and reboot
sudo nvpmodel -m 2          # MODE_30W   (use -m 1 for MODE_15W)
sudo reboot

# 2. after it comes back
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
sudo nvpmodel -q            # must say MODE_30W
sudo jetson_clocks && sudo jetson_clocks --fan

# 3. run BOTH queues, in this order. Each refuses to start in the wrong mode.
bash bench/queues/queue_e7_capped.sh  2>&1 | tee scratch/queues/queue_e7_30w.log
bash bench/queues/queue_e12_capped.sh 2>&1 | tee scratch/queues/queue_e12_30w.log

# 4. when both modes are done, return to MAXN (this also needs a reboot)
sudo nvpmodel -m 0 && sudo reboot
```

**`queue_e12_capped.sh` is new (2026-09-25) and is T014 step 4.** It runs the pipeline, the
equivalence gate (re-run, not inherited — fewer SMs change the schedule and nothing downstream
is valid until it passes *at that mode*), the green-context SM curve, and the energy arms. It
records the GPU clock **before** measuring, because without it the comparison below cannot be
made.

**It carries a prediction written before the run**, which is the point of running it: on
2026-09-25 CUDA green contexts gave π0 a *software* 8-SM partition at MAXN, and MODE_30W gates
8 SMs in *hardware*. Those should agree.

| π0 eager N=10, median ms | prefix | expert |
|---|---:|---:|
| MAXN, whole GPU (16 SMs) | 478.08 | 580.79 |
| MAXN, green context, 8 SMs | 779.79 (1.631×) | 587.56 (**1.012×**) |
| **MODE_30W, 8 SMs in hardware** | **predicted ≈ 779.79 × (clock ratio)** | **predicted ≈ 587.56 × (clock ratio)** |

If they agree after accounting for clock, the green-context method is validated against real
silicon and E12 §5's conclusion stands. **If the expert slows far more than the clock ratio,
the green-context result is wrong and E12 §5 needs revisiting** — which is exactly why the
prediction is recorded here in advance rather than fitted afterwards.

`issue/wall` should stay ≈ 1.0000: fewer SMs does not make the host issue kernels faster.

**I never confirm a reboot prompt and never use `nvpmodel --force`** (AGENTS.md), which is
why these steps are yours rather than mine.

### What the capped runs should produce

Per T004 step 4b: the four power loops for **eager + best at N=10 only**, plus one E3-style
latency run at N=4 and N=10. Nothing from a capped mode is ever averaged with MAXN; the
directory name carries the mode. `--allow-capped` is implemented in `power_phases.py`,
`phase_loops.py` and `groot_phases.py`, and `gate()` still records the real mode.

### What 50W already told us, so you know what to expect

π0 N=10, module rail, MAXN → 50W: power ×0.62, latency ×1.46, **energy ×0.90**, above-idle
energy ×0.77, and **energy·delay 1.32× worse**. A 37 % clock cut bought 10 % energy and cost
47 % latency. Expect 30W and 15W to be worse on energy·delay still, and expect the SM
reduction to change the shape rather than just the scale.

---

## ⛳ STANDING ITEM — E7 capped power modes need two human reboots (do not lose this)

The owner has committed to two reboots for the capped-mode half of E7 and asked to be
reminded. **These are required, not optional** (they override the research side's
"optional" in `docs/04_research_answers.md` answer 3).

| mode | how | status |
|---|---|---|
| MAXN | current | done / in progress with every task |
| `MODE_50W` (id 3) | same TPC mask as MAXN → try `sudo nvpmodel -m 3` with **no** reboot, as the last item of a MAXN session | agent can do unattended |
| **`MODE_30W` (id 2)** | **needs a human reboot** — `sudo nvpmodel -m 2`, confirm, then `sudo jetson_clocks` | **PENDING — ask the owner** |
| **`MODE_15W` (id 1)** | **needs a human reboot** — same procedure | **PENDING — ask the owner** |

The agent never confirms a reboot prompt and never uses `nvpmodel --force` (`AGENTS.md`).
After each reboot: `sudo jetson_clocks && sudo jetson_clocks --fan`, then re-run the E7
power points. Write "next: resume E7 at mode <id>" here before any switch (protocol §1).

## 2026-09-24 (night) — E11 complete for six models, E5 complete, and my registered hypothesis is FALSIFIED

### The hypothesis E7 registered is falsified as a general claim

`reports/E7_power_energy.md` registered that the action expert's energy-to-time ratio
tracks its GPU-idle fraction, and named E5 as the test. E5 is now complete and the verdict
is negative.

| model | expert-region idle, eager → best | drop | E7 energy/time ratio |
|---|---|---:|---:|
| π0.5 | 0.6429 → 0.4453 | 0.198 | 0.96 |
| π0 N=10 | 0.8417 → 0.4714 | 0.370 | 0.80 |
| SmolVLA | 0.9273 → 0.3795 | 0.548 | 0.45 |
| GR00T N1.5 | 0.5258 → 0.0008 | 0.525 | **1.05** |
| GR00T N1.7 | 0.7153 → 0.0008 | 0.715 | 0.88 |
| GR00T N1.6 | 0.7224 → 0.0010 | 0.721 | 0.66 |

Within the three LeRobot models it is monotonic, which is why it looked like support
earlier. It does not survive GR00T: SmolVLA and N1.5 have almost the same drop (0.548 vs
0.525) and the two extreme ratios (0.45 vs 1.05). **Do not write it up as confirmed.** The
family-specific version is true and the general one is not.

### The finding that replaces it, and it is stronger

**GR00T's compiled DiT action head has essentially no GPU idle**: 0.0008 / 0.0010 / 0.0008
for N1.5 / N1.6 / N1.7, gap median 0.0 µs, `kernel_union_ms` equal to `wall_ms` within
0.1 ms. The LeRobot experts retain **38–47 %** idle after the same class of optimisation.

So CUDA graphs remove launch starvation *completely* in a DiT head and only *partly* in a
flow-matching expert. That splits the PIM argument by architecture: GR00T's compiled head
is genuinely saturated, so its bound is real work — the regime a memory-side accelerator is
for — while the LeRobot experts are still gap-bound and Amdahl limits what faster memory can
do there until the launches are addressed.

Caveat: `ae_step` (per denoising step, LeRobot) and `ae_total` (per inference, GR00T) are
different regions. The idle *fraction* compares; the wall times do not.

### The 84 % idle figure is π0-specific

π0's 84 % eager `ae_step` idle has been the motivating number for the idle-time argument.
It does not generalise: π0.5 is **64.3 %** while running *more* kernels per step (1102.7 vs
968.7), which also rules out kernel count alone setting the idle fraction. Compiled, the
three converge (0.4714 / 0.4453 / 0.3795).

### E11: complete for six models

| model | run | rows | parts_missing |
|---|---|---:|---|
| π0 | `20260924_E11_pi0_inputs` | 1564 | `[]` |
| π0.5 | `20260924_E11_pi05_inputs` | 1159 | `[]` |
| SmolVLA | `20260924_E11_smolvla_inputs` | 860 | `[]` |
| GR00T ×3 | `20260924_E11a_groot_n1{5,6,7}_hostseg` | E11a only | — |

`reports/E11_models.md`. T011 and T012 both carry `status: done` with every deviation
listed. Measured P per model: π0 816, π0.5 968, SmolVLA 177 — π0's literal carries over to
neither.

### E5: complete, 14/14

`reports/E5_timeline.md`, now with the **corrected** idle fraction computed from Part B's
unprofiled `ae_total_ms` and cross-checked two ways (π0 eager 0.7639 against the documented
0.756; SmolVLA eager 0.8894 against 0.8854 derived from E11f's `ae_total`). The compiled
half exists: `--rung` keeps the NVTX regions, so the abandoned
`20260922_E5_pi0_reduce-overhead_N4` is superseded by `20260924_E5_pi0_best_N4`.

π0 compiled vs eager: `ae_step` idle 0.8417 → 0.4714, kernels/step 968.7 → 372.7.

### Questions for the research side (new)

1. **`check_model_inputs.py`'s `MODELS` set has no `groot_n16` or `groot_n15`**, so their
   `model_inputs.csv` is rejected. The rows are written and each `meta.json` records it.
   Relabelling the `model` field to pass the checker would be falsifying it, so it was not
   done. Please extend the set.
2. **Two E11 quantities are absent under compiled modes and need not be.** See the
   correction notice above and `scratch/NOTE_e11f_compiled_idle_is_obtainable.txt`: E5
   measures the same quantity per region via `--rung`. Closing it is a consolidation change
   — say whether you want it sourced from the E5 run with `from=<E5 dir>` in `variant`.

### In flight

E5b (T007) instruction-level for six configurations. π0.5 eager cannot reach §5's 60 %
coverage: **eight of its kernels are launch-bound** (SM 0.1–2.5 %, a few µs each against a
62 µs gap), carrying ~57 % of its `ae_step` time. The shortfall is reported at 43.4 % rather
than the bar being lowered — and it is itself a finding, since a memory-side accelerator
cannot help a phase waiting on launches.

bench commit `d2797536f055`.

## 2026-09-24 — E7 energy begins, and the expert is the CHEAP phase

The owner's instruction: run the rest of the campaign, with everything needing a reboot
last. Order being executed: **E7 at MAXN → E11 (T011/T012) → E5 → E5b → MODE_50W →
MODE_30W/15W (the two human reboots)**.

### E7 COMPLETE — 14/14 points at MAXN

`reports/E7_power_energy.md` is generated by `bench/make_e7_report.py`. Every run is a
60 s idle baseline in the same power mode, then the prefix / action-expert / full loops
for 60 s each, four INA3221 rails at 20 Hz, tegrastats alongside (protocol §4).

#### Finding 1 — the action expert occupies the frame without drawing the power

Its energy-to-time ratio is **below 1 in all 14 points**. Eager values:

| model | expert % of TIME | expert % of above-idle ENERGY | ratio |
|---|---:|---:|---:|
| SmolVLA | 86.2 | 71.2 | 0.83 |
| GR00T N1.7 | 58.4 | 45.1 | 0.77 |
| GR00T N1.5 | 43.3 | 31.3 | 0.72 |
| GR00T N1.6 | 63.1 | 44.1 | 0.70 |
| π0.5 | 20.5 | 14.1 | 0.69 |
| π0 (N=10) | 55.5 | 22.4 | **0.40** |
| π0 (N=4) | 33.3 | 10.6 | **0.32** |

π0's prefix draws 40.9 W above idle; its expert draws 9.7 W. Six model variants, three
codebases, two action-head architectures (Gemma expert, DiT), two GR00T backbones. Since
the PIM argument is an argument about energy, this is the most direct evidence the study
has produced against its starting premise.

**Every compiled configuration raises its own model's ratio** (π0 0.40 → 0.82, SmolVLA
0.83 → 0.78 is the one exception, N1.6 0.70 → 0.82, N1.5 0.72 → 0.89). Hypothesis, stated
in the report as something E5 can falsify: the ratio tracks the phase's **GPU-idle
fraction**, because idle time fills the frame while drawing nothing above idle. E5 has
π0's eager `ae_step` at 84.3 % idle against vision 1.2 % and prefix 1.6 %.

#### Finding 2 — latency is not a usable proxy for energy

Frame energy, eager → best:

| model | time saved | energy saved | ratio |
|---|---:|---:|---:|
| GR00T N1.5 | 1.57× | 1.65× | **1.05** |
| π0.5 | 6.31× | 6.09× | 0.96 |
| π0 (N=4) | 2.24× | 1.97× | 0.88 |
| GR00T N1.7 | 1.59× | 1.39× | 0.88 |
| π0 (N=10) | 2.55× | 2.04× | 0.80 |
| GR00T N1.6 | 2.04× | 1.34× | 0.66 |
| SmolVLA | 8.40× | 3.74× | **0.45** |

SmolVLA runs 8.4× faster drawing 2.24× the power (19.1 → 42.7 W), so it delivers less
than half the energy saving its speedup implies. **GR00T N1.5 goes the other way at
1.05** — its optimisation removes work. The range 1.05–0.45 is the point: a speedup
predicts the energy saving only to within a factor of two, and this side had no way to
know that from Part B.

Three independent measurements now agree, measuring different things:

* **Part B** — no optimised configuration has an expert that dominates its frame.
* **E7** — the expert is 55.5 % of π0's latency at N=10 but 22.4 % of its energy.
* **E5** — the expert is 84 % GPU-**idle**, so most of that latency is not work at all.

### A correction to a number this side has been quoting

The "84 % GPU idle" figure is the **`ae_step` region**, not the frame. From the committed
`results/20260922_E5_pi0_eager_N4/timeline_summary.csv`: vision 1.2 %, vlm_prefix 1.6 %,
`ae_step` 84.3 %, whole capture 36.5 % (N=4) / 54.7 % (N=10). The dispatch-bound phase is
the action expert specifically; the other two keep the GPU fed. Two generators said
"frame" and now say which. This sharpens the finding rather than weakening it.

### Two plan assumptions that turned out to be wrong, and one that was right

* **"Compiled captures cannot carry NVTX regions."** Withdrawn. True of `--compile-mode`
  (applied to `sample_actions`, which swallows the frame), false of the ladder **rungs**,
  which compile the three wrap points individually with the instrumentation applied
  afterwards — which is exactly why Part B has per-region numbers for compiled rungs. E5's
  compiled half is measurable; `bench/nsys_target.py --rung` does it.
* **"The two-stream harness must be fixed before T011."** It was fixed in `773d2a9` on
  2026-09-22. What is actually true is narrower and is a *measurement*: both committed
  E11f runs carry `bench_commit` `de325f77` / `5eeddd5c`, which predate that commit, so
  π0's two-stream numbers on disk are still GIL-confounded and **π0's E11f must be
  re-run**. Queued.
* **"GR00T's DiT blocks may not be patchable."** They are — `DiT.forward` (`dit.py:310`)
  calls `block(...)`, so `transformer_blocks[i].forward` is reached. T012 is feasible and
  its driver is written (`bench/groot_segments.py`).

### Defects found by our own guards, all corrected

* `assert_real_weights` ran **after** the rung's bf16 cast, so π0's `bf16_both` E7 points
  exited with "325/400 tensors match the checkpoint". The guard was right: 74 layernorm
  modules and the vision tower had deliberately been cast. Verify the load, *then* modify
  the precision. Those two points re-run.
* The GR00T E11 driver emitted invented `quantity` names (`t_seg_<name>`). §13.8's
  `quantity` is a **closed table** — `t_host_segment` etc., with the segment in `key`.
  `check_model_inputs.py` would have rejected every row.
* `_short_name` returned `enable_if` — a **return type** — as a kernel name, because it
  cut at the first `<` before stripping the return type. π0.5's dominant compiled expert
  kernel is in fact cuBLAS's `internal::kernel`, a **GEMV** (`cublasGemvParamsEx`) at
  13.2 % SM throughput taking 96.4 % of the captured `ae_step` time. That is the skinny
  matrix–vector signature this side predicted.
* `pkill -f "TreeLauncherSubreaper"` in a queue's sweep matched the `sh` running it — the
  third self-matching-process incident here. Every pattern now brackets its first
  character.

### Correction to a claim in an earlier commit message (2026-09-24)

Commit `e190d34` said π0.5's optimisation gives "a 6.6× energy reduction against Part B's
6.2× latency reduction — so on this model the optimisation is not trading power for
speed, it is removing work." **That is wrong and the conclusion is backwards.** The 6.6×
was the sum of the `prefix` and `ae` loops' above-idle energy, which is not the whole
frame, compared against the whole frame's latency.

Like for like, on the `full` loop:

| | eager | best | ratio |
|---|---:|---:|---:|
| latency | 3041.59 ms | 482.09 ms | **6.31×** |
| energy (absolute) | 132.99 J | 21.85 J | **6.09×** |
| energy (above idle) | 100.00 J | 16.53 J | **6.05×** |
| mean module power | 43.72 W | 45.33 W | 1.04× |

π0.5's optimised configuration draws **more** power and saves slightly **less** energy
than time. It is buying a little speed with power, not removing work. The result is still
strongly positive — 6× less energy per inference — but the direction of the sub-claim was
inverted. `reports/E7_power_energy.md` now carries the like-for-like table and this
correction.

### Question for the research side — `check_model_inputs.py` MODELS set

`docs/reference_scripts/check_model_inputs.py` has
`MODELS = {"pi0", "pi05", "smolvla", "groot_n17", "none"}`. The owner later added **GR00T
N1.6 and N1.5** to the study, and their E11a runs will therefore produce a
`model_inputs.csv` the checker rejects with *"model 'groot_n16' not in …"*.

That file is yours (`docs/` is not ours to edit), so: **may `groot_n16` and `groot_n15`
be added to `MODELS`?** Nothing else about their rows differs — same quantities, same
units, same header.

Until then this side does **not** relabel the rows to get past the checker. The runs go
ahead, the file is written, and `bench/groot_segments.py` prints the schema mismatch and
records it in `meta.json.schema_note`, so it is visible as a schema question rather than
rediscovered as a checker failure.

### Repository size (protocol §6f)

`du -sh results/` = **920 MB**; `git count-objects -vH` size-pack = **62.49 KiB** (the
pack is small because the E7 raw files were only just added and most of `results/` is
already packed upstream). The 707.6 MB of oversized `.ncu-rep` files remain in the private
HF dataset `yang-gon/vla-orin-ncu-raw`, not in git.

### Tasks claimed today

T007, T008, T009, T011, T012 — `status: claimed 2026-09-24`. T004's E7 half is in
progress under the same claim from 2026-09-22.

### E5b needs no exploratory captures

`bench/pick_e5b_targets.py` applies `docs/03` §1 (skip kernels below 5 % SM throughput
whose duration is comparable to the 62 µs launch gap) and §5 (cover ≥ 60 % of `ae_step`
kernel time) to C1's committed per-kernel data. Every model clears the bar with 1–8
kernels; 22 captures in total. π0.5's compiled expert needs exactly one.

## 2026-09-23 (late) — combination search, accuracy validation, and a defect in the GR00T ladder

New reports: **`reports/E4i_best_combinations.md`** (generated), **`reports/E4h_vlm_prefix_kernels.md`**
(per-kernel prefix analysis), **`reports/E4_PARTS_ABC_GUIDE.md`** (orientation for this side).

### Best configuration per model, measured

| model | best rung | e2e (ms) | vision | prefix | expert | ae % | vs eager |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolVLA | `vlm_graphs` | 92.30 | 31.26 | 7.93 | 44.86 | 48.6 | 7.88× |
| GR00T N1.5 | `all_compile_autotune` | 168.75 | 32.22 | 70.73 | 60.55 | 35.9 | 1.51× |
| GR00T N1.6 | `all_compile_autotune` | 174.59 | 48.80 | 52.61 | 72.83 | 41.7 | 1.87× |
| GR00T N1.7 | `vlm_compile` | 240.49 | 70.82 | 87.59 | 79.57 | 33.1 | 1.55× |
| π0 | `bf16_both` | 424.46 | 56.74 | 192.86 | 168.68 | 39.7 | 2.52× |
| π0.5 | `bf16_both_graphs` | 487.81 | 56.57 | 239.47 | 185.60 | 38.0 | 6.21× |

### The finding that bears on the PIM premise

**The action expert's share peaks at the eager baseline in every one of the six models.**
Optimisation shrinks the expert faster than the rest of the frame in five of six (π0
54.9 → 39.7 %, SmolVLA 85.3 → 48.6 %, N1.7 56.7 → 33.1 %, N1.6 61.5 → 41.7 %, N1.5
38.6 → 35.9 %). The exception is π0.5, whose fp32 prefix was so pathological (2162 ms,
71 % of the frame) that fixing it promoted the expert from 20.4 % to 38.0 %.

Across six models and ~20 configurations, **no optimised deployment has an action expert
that dominates its frame**. In four of six, the VLM prefix is the largest optimised phase.

### Two defects found in our own method, both corrected

* **The GR00T ladder was not cumulative.** `vis_compile` left the LLM eager and
  `vlm_compile` left vision eager, so they were never compiled together — the LeRobot
  ladder did accumulate. The missing combination is worth 6.1 % (N1.6) and 7.3 % (N1.5).
* **Inductor silently disables GEMM autotuning here.** `is_big_gpu()` requires 68 SMs
  (`torch/_inductor/utils.py:1426`); Orin has 16. Forcing it: **+9.4 % on N1.5**, +1.30×
  on π0's fp32 prefix, but only 0.5 % on N1.6 and a net loss on π0 because it slows the
  *eager* expert (allocator pressure on a launch-bound phase). Orin's 48 KB/block shared
  memory also leaves **no valid Triton config** for π0's bf16 shapes.

### Accuracy of the bf16 rungs

* **π0 — not shown to degrade.** fp32 6/30 vs bf16 4/30, z = 0.70, differences in both
  directions. Resolves "broken", not "a few percent".
* **π0.5 — NOT VALIDATED.** fp32 baseline 0/30; `pi05_libero_base` is a base model, not a
  LIBERO finetune. **π0.5's 6.21× carries no accuracy statement.**

### Still open, in priority order

1. **E7 energy — no data at all, for any model.** The PIM argument is about joules and we
   have none. Capped power modes need two human reboots (standing item above).
2. **T011 / T012 hybrid-model inputs** for π0.5, SmolVLA and GR00T; T010 did π0 only.
3. `bench/check_report_numbers.py` extended to the E4* reports.
4. GR00T N1.7 cannot use `all_compile`: its Qwen3-VL vision encoder passes
   `int(grid_thw[:,1:].max().item())` to `flash_attn_varlen_func`, which dynamo cannot
   trace. Its 240.49 ms is a structurally weaker optimum, not a tuning failure.

## 2026-09-23 — Part A + B complete for six models; Part C in progress

`reports/E4d_phase_balance.md` (generated from the row JSONs) and
`reports/E4g_c1_c2_bottlenecks.md` (generated from the ncu CSVs). Both regenerate from
raw data, so neither can drift from the measurements.

### Which models are action-head bottlenecked, identical treatment per model

| model | N | AE eager | AE best | e2e eager (ms) | e2e best (ms) |
|---|---:|---:|---:|---:|---:|
| SmolVLA | 10 | 85.3 % | 48.6 % | 727.46 | 80.97 |
| GR00T N1.6 | 4 | 61.5 % | 39.5 % | 327.22 | 186.12 |
| GR00T N1.7 | 4 | 56.7 % | 33.1 % | 372.48 | 240.49 |
| π0 | 10 | 54.9 % | 39.7 % | 1069.04 | 424.46 |
| GR00T N1.5 | 4 | 38.6 % | 31.0 % | 255.22 | 200.85 |
| π0.5 | 10 | 20.4 % | 38.2 % | 3028.56 | 489.15 |

**The plan's prediction was wrong.** It predicted the share would track N, putting GR00T
lowest at N = 4. N1.6 and N1.7 sit *above* π0 at N = 10. C3 gives the mechanism: N1.6 and
N1.7 carry **32** DiT blocks, N1.5 **16**, all width 1536. Within one family at one N the
share moves with the size of the head — architecture beats step count.

**Correction to earlier numbers circulated for π0.** "54.9 % → 18.9 %" paired the N = 10
eager share with the N = 4 optimised share; π0's directory holds both ladders and the
generator keyed rows by rung alone. At N = 10 it is **39.7 %**. π0's N = 4 ladder is now
reported separately as a within-model control on N (32.3 % → 18.7 %).

### C1 — what each phase is bound by (π0 complete, five models to go)

| phase | mode | SM % | Mem % | Occ % | LongSB | verdict |
|---|---|---:|---:|---:|---:|---|
| vision | eager | 65.3 | 58.3 | 35.6 | 3.38 | mixed |
| vision | best | 43.6 | 59.1 | 22.7 | 1.93 | latency (`long_scoreboard`) |
| vlm_prefix | eager | 40.4 | 48.6 | 70.7 | 34.44 | latency (`long_scoreboard`) |
| vlm_prefix | best | 51.6 | 55.1 | 22.4 | 2.73 | latency (`mio_throttle`) |
| ae_step_0 | eager | 22.7 | 21.6 | 18.5 | 3.59 | latency (`imc_miss`) |
| ae_step_0 | best | 23.3 | 56.5 | 39.2 | 11.44 | latency (`long_scoreboard`) |

**The PIM premise is conditional on optimisation, and this is the most consequential
result so far.** "The action expert is the memory-bound phase" is **false for the eager
baseline** — eager it is issue-limited (constant-cache misses 35 %, instruction-fetch
starvation 18 %, Long Scoreboard only third at 16 %, memory throughput 21.6 %) — and
**true for the compiled deployment**, where memory throughput reaches 56.5 % and Long
Scoreboard triples to become dominant. A PIM argument built on eager numbers is arguing
from launch overhead; the same argument on a compiled deployment is arguing from a real
memory-latency bound.

Our eager SM 22.7 % matches arXiv:2604.24447's reported 22.3 % for the Action Expert on
AGX Orin, an independent check on the capture.

The prefix moves the other way: compilation removes **92 %** of its memory-latency stall
(34.44 → 2.73 cycles/issue) and the limit becomes `mio_throttle`. That is why CUDA graphs
on π0's prefix were a net loss in Part B — the stall they would target is already gone.

### Questions for the research side

1. Does the conditional result change the PIM target? If the expert is only memory-bound
   *after* compilation, a study quoting eager numbers is measuring launch overhead.
2. Should the paper's Finding #4 be re-examined? It calls the VLM compute-bound from
   operational intensity (840 FLOP/B analytic; ours 816), but our counters show the eager
   prefix stalling **74 %** on Long Scoreboard at 40 % SM and 49 % memory throughput. High
   arithmetic intensity does not help when latency-bound at low occupancy.
3. π0.5 ships **fp32** (`config dtype: float32`) while π0 ships bf16. Casting π0.5 to
   bf16 through LeRobot's own path gives **2.16×** on the VLM prefix where compilation
   gave 1.04×. Is fp32 deployment the intended configuration upstream?

### Raw data too large for git

Four `.ncu-rep` from the superseded overnight C1 run (88–203 MB each; GitHub rejects
> 100 MB) are published at
`https://huggingface.co/datasets/yang-gon/vla-orin-ncu-raw` — **private** dataset,
707.6 MB, byte counts matching the sha256 table in
`results/20260922_E4g_pi0_ncu_eager/RAW_FILES.md`. Scanned clean beforehand: zero
credential patterns and zero environment-variable-shaped strings, confirming `.ncu-rep`
does not embed the process environment (unlike `.nsys-rep`, which does on this device and
is never published).

## Headline after T010 (E11): the hybrid model's six [ASSUMPTION] terms, measured

`reports/E11_pi0_hybrid_inputs.md` · nine result directories · 1,564 rows in
`results/20260922b_E11_pi0_inputs/model_inputs.csv`, 33 of the 35 quantities §13.8 names.

| term | assumed | measured | |
|---|---|---|---|
| kernels per host segment | 1 (2 for attention) | **1–10.1**; ~35 launches per layer | 10× too few |
| `T_host_layer` | 0.16 ms | **0.1827 ms** as one CUDA graph per layer | 1.14× — reachable |
| | | 0.4455 ms compiled sequence | 2.78× |
| | | 1.5290 ms eager sequence | 9.56× |
| transfer fixed cost | **0** | **38–40 µs** pinned, either direction | not zero |
| transfer bandwidth | 143 GB/s | **36 GB/s** pinned | 4× optimistic |
| synchronisation | **free** | 0.8 µs query … **68.7 µs** stream hand-off | not free |
| `kv_copied_bytes_per_step` | **0** | **15,040,512 B every step**, 5.5824 ms | not zero |

**Only one implementation reaches the mapping's 0.16 ms: a single CUDA graph over the whole
layer.** Six separate graphs cost 1.6×, a compiled sequence 2.8×, eager 9.6×. Fusion across
segment boundaries is real and worth 2.5×, but it is a graph that closes the gap, not a compiler.

**The GPU is idle 75.6 % of the action-expert region** at both N = 4 and N = 10 — corrected for
nsys inflating the region's wall time ×1.53 while leaving kernel durations alone. The uncorrected
figure is 84 % and charges the profiler's overhead to the model. In CPU terms: the expert loop
pins **one core at 100 %** while eleven of twelve do nothing.

**Overlap belongs in a separate process.** A prefix loop beside the host-segment sequence costs
1.2 % as two processes and **10.4 ×** on the executing part as two CUDA streams in one process.
The contention is for the GIL, not the GPU.

**The prefix is a constant for this checkpoint.** `P_processed = 816` at every point of the sweep
while `P_valid` moves 262 → 807; every region is flat to within 0.7 %. This is a property of the
input format — three always-encoded image slots and a prompt padded to 48 — not of the hardware,
and **it is not expected to transfer to π0.5**, whose `tokenizer_max_length` is 200 and
`empty_cameras` 0.

### ⚠ Three defects in T010 were visible only in the output, never as an error

Recorded because the pattern matters more than the individual bugs. A compiled callable whose
results are discarded is dead-code-eliminated — measured here at 0.0662 ms vs 0.6605 ms for the
same matmul — and the published "compiled sequence = 0.085 ms" measured nothing. The background
hog's duty cycle never applied, so half a table was one condition four times. Fixing that
introduced a closure rebinding that killed every `cpu_spin_control` worker, whose six tracebacks
went to `DEVNULL` with the hog's stdout. Each produced a plausible number and no warning.

Countermeasures now in place: `bench/bw_hog.py --self-test` (every hog kind for 2 s, fails unless
each emits samples with an achieved duty; `mempressure.py` calls it before loading the model), and
`bench/check_report_numbers.py --task=T010`, which re-derives 96 figures from the CSVs and fails if
any is absent from the report.

## Headline after T005 (E1/E2): the expert's GEMMs are already at the speed of memory

**The gap is not weight streaming.** E2 put the action expert's 126 projection GEMMs — 622.85 MB,
97.6 % of the 638 MB that T002's `ae_step` floor was built on — into one CUDA graph, so the
Python interpreter is out of the timed region:

| | ms | × floor |
|---|---|---|
| analytic floor (622.85 MB ÷ **measured** 187.47 GB/s) | 3.322 | 1.00× |
| **the 126 projections, replayed as one CUDA graph** | **4.411** | **1.33×** |
| the same, issued back to back from Python | 8.656 | 2.61× |
| a whole eager denoising step (T002/E3), floor 3.403 ms | 52.56 | 15.4× |

The three MLP projections hold 73 % of the weight bytes and run at **87–90 % of measured peak
bandwidth**. The sensitivity runs the safe way: at the bottom of the measured bandwidth range the
stack multiple falls to 1.13×, so the conclusion strengthens. A memory system that removed *all*
off-chip weight traffic for the projections would save at most 4.4 ms of a 52.6 ms step.

**A denoising step launches 969 GPU kernels, and only 17 % of them are GEMMs** (slope over
N ∈ {2, 10}; the fit predicts N = 4 exactly). 48 % elementwise, 16 % copies, 8 % KV-cache
concatenation, 8 % normalisation. **At S = 51 the marginal cost of a projection is ~62 µs
whatever its shape**, against 8–55 µs of GPU work, so every projection is launch-bound there.
(That column is `max(CPU, GPU)`; above S ≈ 128 it tracks the GPU instead and is no longer a
dispatch cost.) `PiGemmaRMSNorm` costs **275 µs marginal against 17.7 µs of GPU**, 36 times a
step.

**Measured peaks (protocol §8), replacing the spec sheet everywhere:**

| quantity | spec | measured | ratio |
|---|---|---|---|
| memory bandwidth | 204.8 GB/s | **187.47 GB/s** | 0.915 |
| bf16 dense GEMM | 42 TFLOP/s | **33.85 TFLOP/s** | 0.806 |
| ridge point | ≈ 205–208 FLOP/B | **180.6 FLOP/B** | 0.87 |

**E11's analytic host model is 2× pessimistic on compute**: `eff_flops_frac` 0.40 assumed vs
**0.806** measured, `eff_bw_frac` 0.70 vs **0.915**. **`bench/fill_measured_floors.py` filled the
90 `peak_source = measured` rows** every `floor.csv` had left blank since T002.

**Two research-side questions answered.** (1) The L2 sysmem-miss substitute for DRAM traffic is
exact for **reads** (1.000–1.009 on four kernel configurations with analytically known traffic)
but the write path is **identically zero** in every `*_lookup_miss` counter — writes register as
L2 *hits*. Off-chip traffic must be `op_read_lookup_miss.sum × 32 B` **+** `op_write.sum × 32 B`;
using the miss counter for both, as protocol §3.3 has it, reports zero write traffic. Also:
§3.3(3) forbids deriving GB/s from ncu's `Duration`, so those columns are deliberately blank
until an nsys run supplies a duration. (2) The four INA3221 rails do **not** double-count and may
be summed: a load that raises `VDD_GPU_SOC` by 51.8 W raises `VIN_SYS_5V0` by only 8.7 W.

### ⚠ Unexplained: bandwidth varies run to run by up to 1.19×

22 fresh processes running the identical kernel span 159.9–188.6 GB/s, and whole-E1 passes gave
peaks of 161.8, 183.4, 185.6, 187.5 and 189.1 GB/s — with the GPU clock (1300.5 MHz), the EMC
clock (3199 MHz, sampled *during* the timed loop) and the die temperature (47–51 °C) constant
across the range, and identical device pointers. Sustained load does not throttle it either
(12 repeats of an 8192² GEMM: clock stable, no decay, die to 58.4 °C). Five checks are backed by
committed files in `results/20260922_E1_peak/raw/`; six more were run interactively and not
retained, and are reported as reasoning rather than evidence. **Consequence: a bandwidth-derived
floor carries roughly ±8 %, and a measured-vs-floor ratio should use a peak from the same
session.** Re-check after the power-mode reboots.

Report: `reports/E1_E2_microbench.md`.

## Headline after T003 (E4): compilation closes two thirds of the gap, and one line closes half of what is left

**The ladder is complete** (ms per denoising step, π0, MAXN):

| rung | ms/step | × floor |
|---|---|---|
| analytic bandwidth floor | 3.115 | 1.0× |
| research-side "optimistic bound" (2 × floor), *assumed unreachable* | 6.230 | 2.0× |
| measured eager (T002) | 52.56 | 16.9× |
| **measured compiled, best stock mode (`reduce-overhead`)** | **16.30** | **5.2×** |
| *modified: crop instead of cloning the KV cache* | *10.37* | *3.3×* |

The research side's ~90 ms/step estimate is **5.5× too pessimistic** — it applied Jetson
**Thor's** 1.51× compile gain to a derived 136 ms eager step, and both inputs were wrong here.
Measured compile gain is **3.22× per step** (1.39× end-to-end at N = 4, 1.69× at N = 10): a
single end-to-end factor is not transferable across N, only the slope is.

**What each mechanism contributes.** Compilation 2.32×, CUDA graphs a further 1.39×,
autotuning **nothing measurable**. `max-autotune` costs 5× the compile time of `default` for
latency it only gets from CUDA graphs — use `reduce-overhead`.

**Triton GEMM templates win no action-expert shape.** With Inductor's `min_sms = 68` gate
defeated in-process, Triton won **0 of 126** action-expert autotune blocks and 54 of 697
vision/prefix blocks; on four of five expert shapes the best Triton candidate is **~2× slower**
than cuBLAS. The committed `output_code` shows why: the expert graph is **127
`extern_kernels.mm`** — every matmul to cuBLAS — with Triton only for fused RoPE, GeGLU,
RMSNorm and casts. **The SM-count restriction is not why the expert sits above its floor.**

**One line still dominates.** `copy.deepcopy(past_key_values)` at `modeling_pi0.py:919` is
every graph break in the model (33 of 33 break events across all logs). It **cannot be
deleted** — the negative control fails, because the expert appends its suffix K/V regardless
of `use_cache=False` — but `DynamicCache.crop(P)` is an exact substitute: **bitwise identical
actions**, zero graph breaks, 10.37 ms/step. That reaches **1.67× the "optimistic bound"**
that `docs/00_context.md` calls a stand-in for a perfect software stack. **The headroom a PIM
design must beat is much smaller than the original ladder implied.**

Full report: `reports/E4_pi0_compile.md`.

### Errors caught in T003, for the record

Two I caught myself before they reached the report: my own region instrumentation
graph-breaks a compiled model (so E4's primary numbers are uninstrumented, per protocol §2),
and the `forced` variant silently reused cached kernels until both arms were re-run with cold
caches. Three the independent review caught: **all 20 result directories had empty `raw/` and
`raw_files: []`** — the same defect T002's review found, reintroduced verbatim in new code, so
every log-derived number was unverifiable from the repository; CUDA-graph use was *inferred*
from a timing gap rather than measured (now confirmed directly); and several counts were
wrong (autotune blocks double-counted 494→247, 1646→823) including a quoted log line that
existed in no file. Two protocol violations disclosed: a result directory overwritten in
place (§5), and `throttled: false` asserted with no tegrastats log (§1).

## Headline after T002 (E3): the action expert is **not** the dominant phase

**π0, LeRobot implementation, eager, MAXN: one denoising step costs 52.94 ms** (slope fit over
N ∈ {1,2,4,6,8,10}, r² = 0.99999). The research ladder assumes 136 ms — the real device is
**2.6× faster** than assumed, and **17.0× the 3.115 ms bandwidth floor**.

The phase split contradicts the working hypothesis. At the paper's N = 4 the action expert is
the *smallest* of the three phases, and `vision` is the largest:

| phase | paper (Orin, N=4) | measured here, N=4 | measured, N=10 |
|---|---|---|---|
| `vision` | 110 ms (12 %) | **239.4 ms (34.5 %)** | 239.6 ms (23.7 %) |
| `vlm_prefix` | 258 ms (28 %) | 233.1 ms (33.6 %) | 232.9 ms (23.0 %) |
| `ae_total` | 543 ms (59 %) | **215.5 ms (31.1 %)** | 533.0 ms (52.7 %) |
| end-to-end | 920 ms | **694.1 ms** | 1011.6 ms |

The expert *is* memory-bound as predicted (floor 3.115 ms memory vs 0.912 ms compute) — it
just is not the biggest phase. Full report: `reports/E3_pi0_eager.md`.

Three `docs/00_context.md` values are now **confirmed from the loaded model**, not assumed:
**P = 816** (3 × 256 + 48; the doc marked it assumed), **S = 51**, **623.01 MB** of expert
weights per step, and 38.29 GFLOP against the doc's 38.7.

## Three findings

* **80.2 ms per inference encodes a dummy image.** `empty_cameras: 1` makes the policy
  synthesise a third camera filled with −1 and push it through the full SigLIP tower:
  **33.3 % of `vision`, 11.6 % of an N = 4 inference**, and its attention mask is zero so the
  result is discarded. Caching one constant embedding removes it.
* **10.4 % of every denoising step is a Python deep copy of the KV cache.**
  `modeling_pi0.py:919` runs `copy.deepcopy(past_key_values)` inside `denoise_step`, cloning
  all 36 KV tensors (**15.04 MB**) every step: **5.68 ms/step, 56.8 ms and 150 MB per
  inference at N = 10**. At an effective 2.65 GB/s it is Python/allocator cost, not bandwidth
  — and it alone is **1.82× the entire 3.115 ms memory floor of a step**.
  `docs/00_context.md` assumes `kv_copied_bytes_per_step = 0`. The expert branch passes
  `use_cache=False`, so whether the copy is needed at all is one experiment.
* **Denoising step 0 costs +0.83 ms** over the rest — *not* the +3.13 ms first reported. The
  larger figure was ~74 % my own instrumentation: `ae_layer[j]` events land on one step and
  cost it ~2.57 ms. Control: `results/20260921_E3_pi0_eager_probe/step0_control.csv`.

## ⚠ A trap that would have silently produced garbage

PyPI `lerobot==0.4.4` matches the checkpoint's `_v044` name and would have loaded a
**randomly initialised model**: it needs a patched transformers fork, it lacks the
`lm_head → embed_tokens` alias fix this checkpoint requires (the weight exists only as a
safetensors metadata alias, which `load_file` does not reconstruct), and
`PI0Policy.from_pretrained` **catches load failures and returns the model anyway** with only a
printed warning. A random π0 still runs and still yields plausible latencies.
Pinned `lerobot @ f0d2b37beb` instead, and `bench/pi0_common.py:assert_real_weights()` now
refuses to measure unless 400 sampled tensors match the checkpoint file.
`transformers` is pinned to **5.3.0 exactly** — 5.4.0 moved a `× √hidden_size` rescale that
this ref still applies itself, which would scale every prefix embedding by ≈ 45, silently.

## Route change: LeRobot's π0, not openpi's (owner decision)

`tasks/T002_pi0_eager.md` prescribed openpi's PyTorch port. It was researched in full and
rejected on setup cost — 12 GB orbax/OCDBT conversion, a mandatory JAX import, a
`transformers == "4.53.2"` string-equality check with five files copied over site-packages,
and a `requires-python >=3.11` gate — none of which measures anything. LeRobot's port is the
same architecture (the model prints *"a direct port of the OpenPI implementation"*), with a
ready-made PyTorch checkpoint. **The weights are a different fine-tune**, so the paper
comparison is across implementations *and* checkpoints; that is stated in the report.
The openpi route stays open and is the obvious way to test whether the phase-split
disagreement is implementation or measurement. `docs/00_context.md` and the task file were
updated to match.

## Device is at MAXN — measurement can proceed

The human rebooted into **MAXN (id 0)** on 2026-09-21 and it is persisted
(`/var/lib/nvpmodel/status` = `pmode:0000`). `jetson_clocks` + `--fan` are applied and are
now run **before every capture**, per protocol §1. CUDA reports **16 SMs**, `TPC_PG_MASK` 0,
GPU locked at **1 300.5 MHz** (confirmed under load), 12 CPU cores, EMC 3 199 MHz.

The authoritative E0 record is **`results/20260921_E0_platform_maxn/`**.
`results/20260921_E0_platform/` is the earlier MODE_30W run, kept only for comparison; **no
number in it characterises the Orin.**

**Three things do not survive a reboot** and must be re-applied at the start of every session:
```bash
sudo jetson_clocks && sudo jetson_clocks --fan
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'   # ncu counters
```
plus the sudo password itself — **passwordless `sudo` is still not configured**
(`AGENTS.md` lists it as a prerequisite). The user is in group `sudo` but a password is
required; it was supplied interactively and used through a `SUDO_ASKPASS` helper kept
outside the repository, which `/tmp` clearing at reboot removes. A `NOPASSWD` sudoers entry
for `nvpmodel`, `jetson_clocks`, `tegrastats`, `ncu` and reading `/sys/kernel/debug` would
remove this friction.

Note for E7's power-mode sweep: **every mode change that alters TPC gating needs a reboot**,
and declining the prompt does not stage it (`nvpmodel` reports `optMask is 1, no request for
power mode` and leaves the persisted mode untouched). Answer `YES`, or use `--force`.

## Device

| | |
|---|---|
| board | NVIDIA Jetson AGX Orin Developer Kit (`tegra234`), host `cocojet` |
| L4T / JetPack | **R36.5.0** / **6.2.3+b81** — JetPack 6 prerequisite **met**, no reflash needed |
| CUDA / cuDNN / TensorRT | 12.6.68 / 9.3.0.75 / 10.3.0.30 |
| kernel / python | 5.15.185-tegra (`oot`) / 3.10.12 |
| GPU | Orin, cc 8.7 → **`sm_87`** (confirmed by torch *and* `cuobjdump` on the system cuBLAS), 48 max warps/SM |
| **SMs** | **16** at MAXN (`TPC_PG_MASK` 0, 8 of 8 TPCs); was 8 in MODE_30W |
| GPU clock | **1 300.5 MHz** locked and confirmed under load; was 612 MHz in MODE_30W |
| EMC | **3 199 MHz** = LPDDR5-6400, as expected |
| memory / disk | 61 GiB RAM, 8 × 3.8 G zram (no disk swap); NVMe 1.8 T, 1.7 T free |
| idle power (MAXN, clocks locked, 58.6 s @ 10 Hz) | **module power 11 012 mW median** (the four rails of protocol §4, from sysfs); tj 41.9 °C; not throttled. MODE_30W gave 8 522 mW / 39.9 °C. |

## Software stack that works

**torch 2.8.0 + torchvision 0.23.0 + triton 3.4.0**, JetPack CUDA 12.6 wheels (cp310
aarch64) from `https://pypi.jetson-ai-lab.io/jp6/cu126`, in `~/venvs/jetson_base`.
`torch.cuda.is_available()` true, `arch_list = ['sm_87']`, bf16 GEMM correct to 4.2e-3 vs
fp32, int8 `torch._int_mm` works.
**`torch.compile` works in every mode**: inductor `default`, `reduce-overhead`,
`max-autotune`, `max-autotune-no-cudagraphs`, plus `aot_eager` and `eager`
(`results/20260921_E0_platform_maxn/stack_verification.json`).
**But `max-autotune` never uses Triton GEMM templates on an Orin** — see point 5 below.
Trap to remember: `pip install torch==2.8.0` with `--extra-index-url https://pypi.org/simple`
installs PyPI's **CPU-only** `2.8.0+cpu`, because pip prefers a manylinux wheel over the
index's `linux_aarch64` one. Install the three CUDA wheels by full URL with `--no-deps`
(`env/setup.sh`).

Profilers: **`ncu` 2024.3.1.0**, **`nsys` 2024.5.4.34**, `cuobjdump`/`nvdisasm` 12.6.68 —
all shipped with JetPack, nothing to install. `--set` values offered: `basic`, `detailed`,
**`full`**, `nvlink`, `pmsampling`, `roofline` (protocol expected ≥ basic/roofline/full ✓).

## Which ncu metrics this Orin exposes

Full verdicts: `results/20260921_E0_platform_maxn/ncu/metric_presence.csv` — **52 present,
14 present-but-zero, 9 absent**; every exported column: `ncu/columns.txt` (1 117 raw
columns). `present_zero` is the "empty" case T001 asks about, made precise: the column
exists and every value is exactly 0.

**Present with non-zero values:** all protocol §3.3 occupancy metrics (`SM Busy`,
`SM Active Cycles`, `Achieved Occupancy`, `Theoretical Occupancy`,
`Achieved Active Warps Per SM`, `sm__warps_active.avg.pct_of_peak_sustained_active`);
`Duration`, `Compute (SM) Throughput`, `Memory Throughput`, `Mem Busy`, `L2 Hit Rate`,
both `Executed Ipc`; the LSU global/shared counters this kernel exercises; all launch /
occupancy-limit metrics; pipe utilisation for `lsu` (21.6 %), `alu` (13.5 %), `adu`,
`cbu`, `fma`; **15 of the 18 warp-stall reasons**.

**Present but zero on the toy kernel (14):** mostly legitimate for this kernel — the local
and atomic LSU counters, shared-atom wavefronts, the `xu` pipe, `L1/TEX Hit Rate`, and the
`membar` / `sleeping` / `tex_throttle` stalls. **Three are not legitimate: every
`dram__*`.** A zero on one kernel is not evidence a counter works; re-check per kernel.

**Absent (9):** `DRAM Throughput`, `DRAM Utilization`, `dram__bytes{,_read,_write}.sum`,
`sm__inst_executed_pipe_fp16...active`, `sm__inst_executed_pipe_{fma,alu}...elapsed`,
`gpc__cycles_active.avg.per_warp`.

**Five things later tasks must do differently:**
1. **No DRAM counters — established three ways.** `ncu --query-metrics` lists **4 426
   metrics on `Orin (GA10B)` and none is `dram__*`** (`env/tool_help/ncu_query_metrics.txt`);
   an explicit `--metrics dram__bytes.sum,...` returns **`n/a`** (`raw/ncu_dram_probe.log`);
   and the `--set full` placeholder columns all read 0. ncu's `Memory Throughput` here
   equals `lts__throughput` to two decimal places — it is **L2**, not DRAM. Off-chip *read*
   bytes must come from `lts__t_sectors_*_sysmem_*_lookup_miss.sum × 32 B`; on the toy
   kernel at MAXN that gave **8.4931 MB against 8.4931 MB** of analytically compulsory
   reads — **ratio 1.0000** (the MODE_30W run gave 1.0032). The **write** path is still not
   validated: the kernel's 0.42 MB output never appears as a sysmem miss.
2. **Stall reasons are on the `raw` page only**, as
   `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`. The `details` page has
   a `Warp State Statistics` section but no `Stall …` rows.
3. **`ncu` needs root here.** `perf_event_paranoid=1` (protocol §12) is not sufficient;
   use `sudo $(which ncu) ...` (plain `sudo ncu` fails — `secure_path`), then `chown` the
   `.ncu-rep` before exporting.
4. **Check ncu's clock against the real one every time.** `--clock-control base` pins the
   SM clock. At MAXN with `jetson_clocks` applied the two agree (ncu 1.29 GHz vs 1 300.5 MHz
   measured, `Duration` 57.09 µs vs nsys 54.6 µs, +4.5 %). In the MODE_30W run, taken with
   unlocked clocks, they did not (305.70 MHz vs 612 MHz). ncu `Duration` is still never a
   latency (protocol §3.2) — but lock the clocks first and the gap is small.
5. **`max-autotune` cannot use Triton GEMM templates on any Jetson Orin.**
   `torch/_inductor/utils.py` hard-codes `min_sms = 68` (`# 3080`) for CUDA; Orin has 16, so
   the check fails **in every power mode** — this was initially mis-attributed to MODE_30W
   and is disproved by the MAXN run (`raw/verify_stack.stdout.log` line 3). `max-autotune`
   falls back to ATen/cuBLAS; its autotune log compares only `addmm` vs `bias_addmm`.
   **E4 must present "max-autotune" here as "default + CUDA graphs + ATen autotuning".**

Also: **`tegrastats` does not print `VDDQ_VDD2_1V8AO`** on this carrier board — the memory
rail is only in sysfs at `/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon4/in2_*`.
`bench/collect_platform.py` now samples all four rails there at 10 Hz
(`power_idle_sysfs.csv`), so protocol §4 module power is available. That channel reads a
**constant 5 040 mV bus**, not 1.8 V — see question 2. `tegrastats` also **appends** to
`--logfile`; delete the file first. And nsys's `--capture-range=cudaProfilerApi` kills the
target at `cudaProfilerStop` unless `--capture-range-end=stop` is passed.

## Done

* **T005 (E1, E2) — microbenchmarks, substitute validation, rail check.** Result directories
  `results/20260922_E1_peak/`, `results/20260922_E2_ae_kernels/`,
  `results/20260922_E2_kernel_counts{,_N2,_N4,_N10}/`, `results/20260922_E7_rails_check/`
  (the `20260921_*` copies are kept, each with a `SUPERSEDED.md`) ·
  Report: `reports/E1_E2_microbench.md` ·
  Harness: `bench/micro_peak.py`, `micro_ae_kernels.py`, `ncu_validation.py`,
  `run_ncu_validation.sh`, `parse_ncu_validation.py`, `power_rails_check.py`,
  `refresh_raw_files.py`, `sustained_load_probe.py`, `bandwidth_mode_probe.py`,
  `count_step_kernels.py`, `analyze_step_kernels.py`, `fill_measured_floors.py`, plus
  `collect_raw()` / `scan_for_secrets()` / `classify_kind()` in `common.py`.
  `hw_constants.json` is complete (protocol §8). Corrected `docs/01_measurement_plan.md`'s
  E2 table as owner (k and v are two separate 1024→256 projections, not one 1024→512).

* **T002 (E3) — π0 end-to-end latency, eager.** Six result directories
  `results/20260921_E3_pi0_eager_N{1,2,4,6,8,10}/` · Report: `reports/E3_pi0_eager.md` ·
  Environment: `env/setup_pi0.sh`, `env/requirements_pi0.txt`, `env/README.md` §10 ·
  Fixture: `bench/fixtures/` (HuggingFaceVLA/libero ep 0 frame 0) ·
  Harness: `bench/pi0_common.py`, `regions.py`, `run_e3.py`, `analyze_e3.py`,
  `fetch_fixture.py`, `measure_preprocess.py`.

* **T001 (E0) — bring-up and platform record. COMPLETE, all four steps.** Step 1 (MAXN +
  `jetson_clocks`) was satisfied after the human authorised the reboot on 2026-09-21.
  Result: **`results/20260921_E0_platform_maxn/`** (authoritative), with
  `results/20260921_E0_platform/` kept as the MODE_30W comparison ·
  Report: `reports/E0_platform.md` ·
  Environment: `env/README.md`, `env/setup.sh`, `env/requirements_base.txt`,
  `env/tool_help/` (12 files) · Harness: `bench/common.py`, `verify_stack.py`,
  `ncu_toy.py`, `ncu_columns.py`, `hw_constants.py`, `collect_platform.py`,
  `gpu_clock_probe.py`, `finalize_raw.py`, `make_meta_e0.py`.
  An independent sub-agent pass (protocol §11) reviewed the run; it found one fabricated
  process claim in the report and nine smaller defects, all fixed and listed in
  `reports/E0_platform.md` §9.

### Carried into the MAXN re-run (accepted, not yet done)
* Apply `sudo jetson_clocks` **before** every capture, not after, and record
  `devfreq cur_freq` alongside each one (`bench/gpu_clock_probe.py`). In this run the two
  toy profiler captures were taken with `schedutil` free to scale 306–612 MHz, and that
  clock was not recorded.
* ~~Re-validate the L2 sysmem-miss substitute for off-chip **write** traffic, and against the
  measured bandwidth peak from E1.~~ **DONE in T005.** Reads validate to 1.000–1.009 on four
  kernels with analytically known traffic. Writes are **identically zero** in every
  `*_lookup_miss` counter — a write sector is recorded as an L2 *hit* — and are recovered
  exactly by `lts__t_sectors_aperture_sysmem_op_write.sum × 32 B` instead. See
  `results/20260922_E2_ae_kernels/ncu/sysmem_miss_validation.csv`.
* Confirm 16 SMs and re-check `max-autotune` (it is refused at 8 SMs).

## In progress

(none)

## Blocked

* (nothing) — T001's MAXN blocker was resolved by the human's reboot and T002 is done.

Historical, for the record: T002 was blocked until the reboot, because it would have run at half the
  SMs and half the GPU clock, so it should wait for the reboot.

## Proposed task Q1 — which models are action-head bottlenecked? (owner request, 2026-09-22)

**Queued, not started.** Runs after π0's phase characterisation and prefix optimisation.

**Why.** The study assumes the action expert is the memory-bound phase worth moving into
memory. Measured on π0, that holds only before optimisation. The three phases start almost
balanced — **33.9 / 32.9 / 32.3 %** eager fp32 — and optimisation *unbalances* them:

| region | eager fp32 | optimised | speedup | share before | share after |
|---|---:|---:|---:|---:|---:|
| vision | 241.31 | 99.92 | 2.42× | 33.9 % | 27.0 % |
| `vlm_prefix` | 234.06 | 192.78 | **1.21×** | 32.9 % | **52.2 %** |
| `ae_total` | 230.12 | 69.72 | **3.30×** | 32.3 % | **18.9 %** |

Each factor is surgical — it moves one region and leaves the others within ±2 %: TF32 is
**2.32× on vision alone**; compiling `denoise_step` is **2.31× on the expert alone**, and
CUDA graphs a further **1.40×**; compiling `pg.forward` is **1.22× on `vlm_prefix` alone**.
CUDA graphs fail outright on both `embed_image` and `pg.forward` (static-buffer overwrite).

arXiv:2604.24447 reports the expert at 59 % of π0 on an AGX Orin and builds two
optimisations (DP-Cache, V-AEFusion) on that. Their configuration is **uncompiled** —
inferred from their own tables and confirmed against ours phase by phase within 11 % — and
they never re-measure the split after the compilation their own Table 3 applies. So the
premise is a property of the baseline, not of deployed inference.

**The question.** Is there a model for which the premise holds under an *identical*
treatment — identically optimised, or identically un-optimised? The controlling ratio is
`N × cost_per_step / (vision + prefix + N × cost_per_step)`, so the prediction, recorded in
advance so it can be wrong, is that dominance tracks the denoising-step count far more than
architecture.

**Model set, fixed by the owner 2026-09-22.** Diffusion Policy, ACT and OpenVLA are out.
Six models, all confirmed obtainable:

| model | source | checkpoint | N | reachable |
|---|---|---|---|---|
| π0 | lerobot `f0d2b37` | `lerobot/pi0_libero_finetuned_v044` | 10 | **measured** |
| π0.5 | lerobot `pi05` port | `lerobot/pi05_libero_base` | 10 | yes |
| SmolVLA | lerobot | `lerobot/smolvla_base` | 10 (`num_steps`) | yes |
| GR00T **N1.5** | Isaac-GR00T branch `n1d5` @ `4af2b62` | `nvidia/GR00T-N1.5-3B` | — | HTTP 200 |
| GR00T **N1.6** | Isaac-GR00T branch `n1d6` @ `9b37aa1` | `nvidia/GR00T-N1.6-3B` | — | HTTP 200 |
| GR00T **N1.7** | pinned `376ba890`, **installed** | `nvidia/GR00T-N1.7-3B` | 4 | verified |

**The GR00T family is the controlled experiment here, and that is why it is worth all three
versions.** N1.7 replaced N1.6's **Eagle** VLM backbone with **Cosmos-Reason2-2B**
(Qwen3-VL) and "expands the state/action dimensions and increases the model action horizon"
(`README.md:99,105`). So across N1.5 → N1.6 → N1.7 the action head stays in the same DiT
family while the prefix changes substantially. If action-head share moves with the backbone
swap and not with the expert, that settles whether dominance is architectural or is simply
a ratio between two independently-sized halves.

Practical cost: each GR00T version is a **separate branch with a separate model package**
(`gr00t_n1d6` → `gr00t_n1d7`), so each needs its own checkout and environment. N1.7's is
built and verified (`env/setup_groot.sh`); N1.5 and N1.6 are not, and their dependency sets
may differ — N1.7 pins torch 2.10.0 from the jetson jp6-cu126 index.

**Blocked on T006's model registry.** `bench/` is π0-specific in six ways, and the three
wrap points the ladder needs (`embed_image`, `pg.forward`, `denoise_step`) differ per model
— `regions.py`'s `ae_layer[j]` records nothing at all for SmolVLA, whose layer body is
inlined in `SmolVLMWithExpertModel.forward` (verified). Comparison targets for AGX Orin are
saved from the authors' leaderboard in `scratch/T004/vla_leaderboard_20260922.json`:
ACT 23.93, GR00T 256.22, SmolVLA 777.38, π0 920.58, OpenVLA 951.56, π0.5 2994.02,
Diffusion Policy 4379.08 ms.

**Comparison targets** for AGX Orin from the authors' leaderboard, saved in
`scratch/T004/vla_leaderboard_20260922.json`: GR00T 256.22, SmolVLA 777.38, π0 920.58,
π0.5 2994.02 ms. (Their figure is for one GR00T version and does not say which.)

## Questions for the research side

1. **DRAM traffic.** `docs/02_protocol.md` §3.3 and E6 assume ncu's `DRAM Throughput` /
   `DRAM Utilization`. They do not exist on Orin (every `dram__*` counter reads 0). We
   propose `lts__t_sectors_aperture_sysmem_*.sum × 32 B` as the measured off-chip byte count
   (validated to 0.4 % on a kernel with known traffic) and will label it
   *measured (L2 sysmem-miss sectors)*. Please confirm that is acceptable for the roofline
   points of E6, and note that ncu's `Memory Throughput` on this device must be read as
   **L2** throughput wherever it appears.
   **T005 update — the substitute is validated, but not in the form the protocol states it.**
   On four kernels with analytically known traffic, reads recover to **1.000–1.009** via
   `lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum × 32 B`. The **write** path is
   **identically zero** in every `*_lookup_miss` counter, including on a kernel that
   demonstrably writes 256 MiB: a write sector is recorded as an L2 lookup *hit*, because the
   L2 is write-back / write-allocate, so it can never appear in a miss counter. Writes are
   recovered exactly (1.001–1.008) by `lts__t_sectors_aperture_sysmem_op_write.sum × 32 B`.
   **So off-chip traffic = `op_read_lookup_miss.sum × 32 B` + `op_write.sum × 32 B`.** Using
   the miss counter for both, as §3.3 has it, would report every kernel's write traffic as
   zero. Evidence: `results/20260922_E2_ae_kernels/ncu/sysmem_miss_validation.csv`. Note also that protocol §3.3(3) forbids deriving achieved GB/s from ncu's `Duration`; those columns are left blank until an nsys run of the same kernel supplies a duration.
2. **Memory-rail power.** `VDDQ_VDD2_1V8AO` exists only in sysfs, and the bus voltage its
   INA3221 channel reports is ≈ 5 V, not 1.8 V. Before E7 quotes memory power we will have
   to establish the correct scaling; if the research side knows the intended interpretation
   of this rail on the AGX Orin devkit carrier, that would save a calibration step.
   **T005 update.** The rail behaves sensibly whatever its scaling: it rises +5.0 W on a pure
   DRAM-copy load and only +5.5 W on a tensor-core load, i.e. it tracks memory traffic rather
   than compute, which is what the name implies. The four rails were also checked for
   double-counting and are independent — a load that raises `VDD_GPU_SOC` by 49.9 W raises
   `VIN_SYS_5V0` by only 8.7 W — so E7 may sum them. What remains open is only the absolute
   scaling of this one rail. Evidence: `results/20260922_E7_rails_check/power_rails_check.csv`.
3. **Power-mode plan.** Since MAXN needs a reboot on this board, so does every return from a
   capped mode. E7's power-mode sweep therefore cannot be done inside one session: it will
   need one reboot per mode, each authorised by the human. Is that acceptable, or should E7
   be limited to MAXN plus whichever capped modes can be entered without a reboot?

## Repository size (protocol §6f)

After T010 (2026-09-22):

```
du -sh results/   106M
du -sh .git        25M
git count-objects -vH:  count: 1821   size-pack: 62.49 KiB

five largest committed files (protocol §6f):
   6,584,184  results/20260922b_E11f_pi0_pipelining/nsys/eager_N10_cuda_gpu_trace.csv
   6,584,072  results/20260922_E5_pi0_eager_N10/nsys/eager_N10_cuda_gpu_trace.csv
   3,301,941  results/20260922b_E11f_pi0_pipelining/nsys/eager_N4_cuda_gpu_trace.csv
   3,301,929  results/20260922_E5_pi0_eager_N4/nsys/eager_N4_cuda_gpu_trace.csv
   2,466,365  results/20260922_E4_pi0_max-autotune+forced_N4/raw/compile_and_stdout.log
```

`find results -size +50M` is empty; the largest committed file is 6.6 MB, well inside §6's limit.
T010 added ~84 MB: nine result directories plus the superseded passes §5 requires be kept, 216,000
per-iteration segment samples, 61 `torch.profiler` traces (~23× under zstd -19), per-core
`/proc/stat` samples and the nsys CSV exports. 453 raw-file entries were secrets-scanned before
commit with zero matches. At 106 MB of results the ~2 GB ceiling is not close.

After T005 (2026-09-22):

```
du -sh results/   22M
du -sh .git       9.1M
git count-objects -vH:  count: 797   size-pack: 62.49 KiB

five largest committed files (protocol §6f):
   2,466,365  results/20260922_E4_pi0_max-autotune+forced_N4/raw/compile_and_stdout.log
   2,466,365  results/20260922_E4_pi0_max-autotune+forced_N10/raw/compile_and_stdout.log
   1,577,638  results/20260921_E2_ae_kernels/raw/latency_samples.csv
   1,227,335  results/20260921_E2_kernel_counts_N10/raw/step_trace.json.zst
     794,544  env/tool_help/ncu_query_metrics.txt
```

`find results -size +50M` is empty, and no committed file exceeds 2.5 MB. T002's figures were
3.8 M / 3.6 M / 362 objects; T003's were 11 M / 5.0 M / 592.

T005 added ~5 MB: two full E1/E2 passes (the `20260921_*` set is kept with a `SUPERSEDED.md`
because the report cites the E1 runs as evidence of bandwidth variation), five ncu reports
compressed 84×, and four `torch.profiler` traces compressed ~23×. Every `.zst` records its
original's sha256, size and a secrets scan run **before** compression.

### Size budget calibration from the toy captures (T001 step 3c)

| file | capture | uncompressed | `zstd -19 -T0` | ratio |
|---|---|---|---|---|
| `nsys_toy.nsys-rep` | 3 inferences of one `F.linear`, 6 kernels | 108 402 B | 41 708 B | 2.6× |
| `nsys_toy.sqlite` | derived from the above | 536 576 B | 53 989 B | 9.9× |
| `ncu_toy.ncu-rep` | **1 kernel**, `--set full`, 30 replay passes | **7 072 411 B** | **92 396 B** | **76.5×** |
| `ncu_toy.ncu-rep` (MAXN) | same, 16 SMs | 7 107 467 B | 92 232 B | 77.1× |

Only the `.ncu-rep` was committed compressed (protocol §6b: > 1 MB); the other two are
under 1 MB and are committed as-is.

**What this implies for later tasks.** The ncu report is ≈ **7 MB per captured kernel**
uncompressed at `--set full`, and compresses ~76× because it is mostly sparse. A π0
`ae_step[0]/` capture is expected to hold several hundred kernels → **order 1–2 GB
uncompressed, order 20–30 MB as `zstd -19`** — committable, but only just, and the
uncompressed intermediate needs room in `scratch/`. Budget `-c` limits accordingly and
check before committing. `bench/finalize_raw.py` compresses above 1 MB and splits above
50 MB with a verified `REASSEMBLE.txt`.

The nsys side is cheap by comparison (≈ 18 KB per kernel uncompressed here, and dominated
by fixed overhead), so the 1–3 inference capture window of protocol §3.1 is not the
constraint; ncu is.
