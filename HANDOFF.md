# HANDOFF — jetson

*Root: `/home/yanggon/0007_26summer/05_VLA_LPDDR/jetson` · Owner: `jetson-owner` · Last updated: 2026-09-22 by jetson-owner*

## Current status
**2026-09-22: T002 (E3), T003 (E4), T005 (E1/E2) received, verified, accepted; T010 (E11) complete on content
(status line still "claimed"); T004 claimed by the device.** 34 device commits `c1c7fe7..e6a0af7`. Checked here:
55 `meta.json` (all MAXN, `bench_commit`, 20/100 where §2 applies; E11d ran 5/30 — deviation to be stated),
`raw_files` sha256 recomputed for E11*, E4-cropcache, E0 and a sample of the rest (one stale hash:
`…cropcache_N10/equivalence.md`), `check_model_inputs.py` exit 0 on 9 files, `results/` 39 MB, largest file 6.6 MB,
size-pack < 1 MB (device reports 62 KiB), nothing > 50 MB. `analysis/ladder.py` reproduces every ladder number from
the committed CSVs. Write partition: the device edited `docs/00_context.md` and `docs/01_measurement_plan.md`
(factual corrections, disclosed, "owner's standing instruction") — accepted and now codified as a narrow exception
in AGENTS.md; user to confirm.

**SECURITY (user action):** nsys binary reports embed the system-wide process environment; the four
`results/20260921_E0_platform{,_maxn}/raw/nsys_toy.{nsys-rep,sqlite}` pushed in T001 (commit `410a535`) contain a
session token of the device agent's harness. The device untracked them (`5ec4788`), added `.gitignore` rules and
says the owner is rotating the token. History was not rewritten (our rule). Repo must be private; user decides on
rotation confirmation / rewrite.

**Key research facts (π0 LeRobot port `lerobot @ f0d2b37beb`, ckpt `lerobot/pi0_libero_finetuned_v044`, MAXN
16 SMs / 1300.5 MHz / EMC 3199, bf16, batch 1, n = 100 per point, device commits per `meta.json.bench_commit`):**
- Ladder, ms/step (slope over N): floor 3.115 spec / 3.403 measured-peak; **eager 52.56–52.94 (16.9× / 15.4×)**;
  compiled `default` 22.61, `max-autotune` 16.71, **`reduce-overhead` 16.30 (5.2× / 4.8×)**, `step_only` 16.34,
  `is_big_gpu` defeated 16.23 (Triton won 0/126 expert shapes), **crop-not-clone KV 10.37 (3.3× / 3.05×, bitwise
  identical actions in eager)**. Research assumed 136 / ≈ 90: 2.6× / 5.5× too pessimistic. Compile gain per step
  3.22× (2.32× compile, 1.39× CUDA graphs, autotune 0); e2e 1.39× at N = 4, 1.69× at N = 10.
  Files: `results/20260921_E4_pi0_summary.csv`, `results/20260921_E3_pi0_eager_N*/`, `reports/E3_pi0_eager.md`, `reports/E4_pi0_compile.md`.
- Split eager N = 4: vision 239.4 / vlm_prefix 233.1 / ae_total 215.5 / e2e 694.0 ms → AE **31.1 %** (paper 59 %);
  N = 10: 239.6 / 232.9 / 533.0 / 1011.6 → 52.7 %. Compiled (`step_only`, instrumented): AE 12.5 % (N = 4), 26.1 % (N = 10);
  e2e 551 / 652 ms. Vision includes an 80 ms dummy-camera SigLIP pass (`empty_cameras: 1`). P = 816 (533 valid), S = 51,
  623.01 MB expert weights, 38.29 GFLOP/step, peak mem 9.04 GB. Pre/postprocess 2.0 ms total.
- T005: peaks **187.47 GB/s (0.915)**, **33.85 TFLOP/s bf16 (0.806)**, ridge 180.6; bandwidth varies 1.19× between
  processes (unexplained). 126 expert projections = 622.85 MB run at **1.33× floor as one CUDA graph** (MLP ones at
  87–90 % of peak), 2.61× from Python; **969 kernels/step, 17 % GEMM**; dispatch ~62 µs per `F.linear`, ~29 µs per
  single-kernel op; launch-bound crossover S ≈ 128 (MLP), not 16. Write-traffic counter corrected (`op_write.sum`).
  Rails independent, summable. `results/20260922_E1_peak/hw_constants.json`, `results/20260922_E2_ae_kernels/`.
- T010/E11 (`results/20260922_E11_pi0_inputs/model_inputs.csv`, 1,589 rows): host layer 1.502 ms eager / 0.177 as one
  graph / 0.085 compiled sequence (assumed 0.16); ~35 launches per layer; transfers 40 µs + 36 GB/s pinned (assumed
  0 + 143); sync 10.6 µs idle, 109 µs stream handoff; KV 15.04 MB deep-copied every step (5.53 ms); prefix constant
  816 tokens for any input (no P slope); **GPU idle 84 % of `ae_total`** (vision/prefix 1–2 %); prefix slowdown under
  a bandwidth hog 2.22× of which 1.75× is time-slicing (bandwidth part ≈ 1.27×); two-process sharing 1.001×;
  CPU 99.8 % of one core during the expert. `reports/E11_pi0_hybrid_inputs.md`.

**My changes, pushed 2026-09-22:** protocol §1 (capped modes required, per owner), §3.1/§3.5/§6e/§7 (nsys binaries
not committed; CSV exports are the raw record; secrets-scan output rule), §3.3 (two-counter off-chip bytes, ncu
replay caveat), §9 (measured-peak floor, same-session peak), §12 (six pitfalls), §13.7 (two-stream rule);
AGENTS.md (nsys exception; docs/ factual-correction exception); T004 rewritten (LeRobot, E11f captures, two
counters, temperature logs, compiled per-region from nsys, step 4b capped modes); T006/T007/T011 route lines;
`docs/04` 2026-09-22 answers (6 items + 6 small fixes); `docs/00_context.md` E11/T005 table; `analysis/ladder.py`.

### Task queue
| task | content | priority | depends |
|---|---|---|---|
| T001 | bring-up, E0 — **done, accepted** (`results/20260921_E0_platform_maxn/`) | P0 | — |
| T013 | withdrawn (E0 at MAXN was done inside T001); number reserved | — | — |
| T002 | π0 eager (E3) — **done, accepted** (`results/20260921_E3_pi0_eager_N*/`) | P0 | T001 |
| T003 | π0 compiled (E4) — **done, accepted** (`results/2026092{1,2}_E4_pi0_*/`) | P0 | T002 |
| T010 | π0 E11 → `model_inputs.csv` — **complete on content, accepted; `status: done` still to be appended** | P1 | T003 |
| **T004** | **claimed 2026-09-22**; π0 E5 (nsys from E11f exports + ncu), E6, E7 incl. required capped modes (4b) | P1 | T003 |
| T005 | E1/E2 — **done, accepted** (`results/20260922_E1_peak/`, `_E2_ae_kernels/`) | P1 | T001 |
| T007 | π0 E5b instruction level | P1 | T004, T005 |
| T006 | π0.5 + SmolVLA E3/E4, LIBERO E8 | P2 | T003 |
| **T011** | **π0.5 + SmolVLA E11** (right after T006 step 1) | P2 | T006, T010 |
| T008 | π0.5 + SmolVLA E5/E5b/E6/E7 | P2 | T006, T004, T007 |
| T009 | GR00T N1.7 full set | P3 | T008 |
| **T012** | **GR00T N1.7 E11** | P3 | T009, T010 |

Execution order is the table at the top of `docs/01_measurement_plan.md` (rows above are already in that order).

### Key numbers
See "Key research facts" above; `python analysis/ladder.py` regenerates the ladder and split tables. Handed to main
2026-09-22 as "for microarch-owner / evaluation-owner / paper-owner" items (result paths, device commit `e6a0af7`,
MAXN, n = 100). Idle module power at MAXN 11,012 mW (E0). E7 power numbers not yet measured.

## Recent decisions
- **2026-09-22, after the 34-commit pull — my judgement calls, user to confirm:** (1) the device's disclosed
  `docs/` corrections are accepted and the exception codified in AGENTS.md (factual corrections to
  `docs/00`/`01` only, marked and disclosed); (2) capped power modes are now *required* for π0 E7 (the owner told the
  device directly; protocol §1 + T004 4b restate it); (3) nsys binaries are permanently uncommitted (token leak);
  full CSV exports are the raw record; (4) floors are headlined against the measured peak with the spread stated;
  (5) T006/T011 default to LeRobot ports for π0.5 and SmolVLA, openpi only as fallback; (6) no history rewrite
  for the leaked token — user's call; (7) T010's two-stream concurrency rows are not used (GIL-confounded).
- **2026-09-21, after T001 — my judgement calls, user to confirm:** (1) E7 capped modes reduced: MAXN required,
  MODE_50W only if it switches without a reboot (last item of last MAXN session), 30 W / 15 W optional and
  human-scheduled (was: full sweep required for π0); (2) sysmem-miss substitute accepted with conditions (analytic
  bytes stay primary; read-only until T005 validates writes; nsys time, never ncu Duration); (3) memory rail is
  reported as "regulator input", absolute + above idle, no rescaling — interpretation tagged `[ASSUMPTION]` (not from
  a schematic); four- and three-rail sums both reported until T005's double-count check; (4) T003 gets one labelled
  variant that overrides Inductor's SM-count check; the ladder uses unmodified modes only; (5) ncu clock rule: keep
  `base`, lock clocks, discard captures where `SM Frequency` and `cur_freq` differ > 2 %; (6) answers live in a new
  `docs/04_research_answers.md`, binding parts in the protocol; (7) a task overtaken by events is withdrawn in
  place, never deleted or renumbered.
- **E11 design (2026-09-21), my judgement calls — user to confirm:** (1) schema has 18 columns: the brief's 15 plus
  `bench_commit`, `variant` (`name=value;…`) and `timing` (`wall_isolated | gpu_event | wall_launch | wall_loop | slope |
  analytic`); the **primary** host-segment number is `wall_isolated` (sync → launch → sync from an idle GPU, with a
  0 / 2 ms pause), because in the target system every segment is SYNC-bounded; (2) `cudagraph` is measured both per
  segment (realistic) and as one whole-layer replay (lower bound = the simulator's `loop` feature, 1 launch/layer);
  (3) segment `key` names are exactly those of `microarch/03_tools/trace_gen.py` manifests (`rmsnorm1, rope, softmax,
  residual_rmsnorm2, gelu_gate_mul_up, partials_residual, residual, attention_gpu`) so `host_model.py` can join on them;
  (4) memory-pressure levels are hog **duty cycles** (25–100 %) with achieved GB/s recorded, plus two controls
  (`gpu_spin_control`, `cpu_spin_control`) — removes the dependency on E1/T005; (5) added beyond the brief: host
  segments concurrent with the next frame's prefix (two processes / two streams) → `host_layer_slowdown`, because a
  100 µs segment can queue behind ms-long prefix kernels; spin-wait check so `cpu_util` is not misread;
  `kv_copied_bytes_per_step` (does `torch.cat` re-copy the prefix KV each step?); `ae_step` vs P fit; (6) GR00T E11 is
  its own task T012, not folded into T009 (needs T010's scripts; T009 is already large); (7) AGENTS.md task selection
  changed from "lowest number" to the plan's order table so T010 runs after T003 without renumbering; (8) SmolVLA /
  GR00T segment lists are **not** established here (op_trace.py is π0-only): T011/T012 tell the device to derive and
  record them; "4 partials" for those models is tagged `[ASSUMPTION]`; (9) restated numbers (E3/E4 latencies in the
  consolidated file) use a copy rule — tidy CSV copied to `raw/restated/<run>/` — so every `source_file` exists inside
  the result directory.
- Source of the ncu / SASS material: the user's skill `/home/yanggon/99_personal_project/02_SKILL/skills/workload-analysis`
  (server only). Only flags, metric and column names it documents are used; everything unverified on Orin is
  "discover with `--help` / exported header and record" (`env/tool_help/`, `ncu/columns.txt`).
- The skill documents two `cuobjdump` spellings (`-sass -fun` / `-symbols` vs `--dump-sass -arch=sm_XX --function` /
  `--list-text`). Both are given in docs/03 §2, the agent uses whichever its version accepts. Reported to main as
  a judgement call (not treated as a stop-condition contradiction).
- **Data policy (user decision 2026-09-21, via main): commit as much raw data as possible.** `.nsys-rep`, `.sqlite`
  (if < 50 MB compressed), `.ncu-rep`, full SASS/PTX dumps, analysed `.cubin`, tegrastats logs, torch.profiler traces,
  command stdout, compile logs, per-iteration samples and full ncu CSV exports go to the tracked
  `results/<run>/raw/`; `meta.json.raw_files` = path, sha256, bytes (+ compression, original hash, split parts).
  Summaries are in addition to raw, never instead. Never committed: weights/checkpoints, datasets, caches, venvs,
  vendor `.so`, secrets. Size rules are binding in protocol §6 (small captures: nsys 1–3 inferences; `zstd -19`;
  50 MB per file; shorten first, `split -b 45M` + `REASSEMBLE.txt` last resort; no LFS; `.sqlite` rule; size report
  in `STATUS.md`; stop near 2 GB). Inductor/Triton caches and uncompressed originals live in the ignored `scratch/`.
  Added on my own judgement: a secrets scan of raw files (protocol §3.5) because profiler reports can record the
  process environment / command line. Commits `5752f2e`, `62a162f`, `c2d3ed0` (pushed). This supersedes the
  conservative policy of `d42ae1a`.
- Review depth: self-review checklist for every task; independent sub-agent pass only for P0 tasks (T001–T003).
- Region utilisation has two derived definitions: kernel-weighted and region-weighted (idle = 0, region time from
  nsys); the region-weighted one is what is compared with the paper's 22 %.
- Latency-budget cycle costs are from Ada/Blackwell; tagged `[ASSUMPTION]` for Orin.

## Open issues
- (high, user) **Leaked session token in git history** (commit `410a535`, four nsys files, now untracked). Repo
  must be private; rotation to be confirmed; no rewrite by agents.
- (medium) **Bandwidth peak varies 1.19× between processes** with clocks/temperature constant (T005) — floors carry
  ±8 %; rule: same-session peak. Unexplained.
- (medium) **π0 is the LeRobot port, different fine-tune from openpi's** — paper comparison is across
  implementations and checkpoints; the phase-split disagreement (vision 239 vs 110 ms) may be image count (3 SigLIP
  passes incl. a dummy) or implementation. openpi cross-check remains an optional later task.
- (low) T010 `status: done` missing; `…cropcache_N10/equivalence.md` stale hash; E4 report §9 independent pass
  empty; E11d 5/30 iterations; E5 partial dirs without meta (T004 WIP) — all listed in `docs/04` for the device.
- (medium) **E11 unknowns on the device:** no documented PyTorch route to a zero-copy / managed allocation (protocol
  §13.3 lists candidates only); whether `torch.cuda.Stream(priority=…)` has any effect on Tegra; whether the openpi
  input path can drop an image slot or always encodes three and pads the prompt (then processed P is constant and the
  P fit degenerates — the task says to report that as a result). The memory-pressure experiment is a proxy; the docs
  say so and ask for the limits in every report.
- (low) The device may already be past "pick a task" when it next pulls; the order-table rule only matters after T003.
- (high) **Repository size.** Raw files are committed; growth is unbounded in principle. Guards: 50 MB per file, size
  report in `STATUS.md` after each task, stop-and-ask near 2 GB `size-pack`. Watch T001's reported toy `.nsys-rep` /
  `.ncu-rep` sizes and tighten §6 (e.g. `-c`, fewer inferences) before T004/T007 if they extrapolate badly. History
  cannot be shrunk later without a rewrite, which is forbidden — an over-size commit is permanent.
- (medium) Raw files may leak: secrets (mitigated by the scan rule), hostnames / user paths inside `.nsys-rep`, and
  SASS disassembly of NVIDIA's proprietary cuBLAS kernels. User should confirm the GitHub repo is private.
- (high) Which agent framework and credentials the device will use is unknown; the device needs push access.
- (resolved by T001) ncu on Orin exposes occupancy / pipes / stalls / LSU; no DRAM counters.
- (medium) **Passwordless sudo is still not configured on the device**; every session needs the password again (ncu, jetson_clocks, perf_event_paranoid). User action.
- (medium) ncu runs the profiled Python as root: model caches / HOME under sudo may bite in T004 (protocol §3.2 gives a pattern, unverified).
- (medium) ncu report ≈ 7 MB per kernel: an `ae_step[0]/` capture may be 1–2 GB uncompressed (20–30 MB zst) — watch T004.
- (medium) Whether cuBLAS on JetPack ships SASS for `sm_87` and whether ncu sees CUDA-graph replays: unknown; docs tell the agent to record what it finds.
- (medium) openpi π0 checkpoint may need JAX → PyTorch conversion on the device (T002 step 1).
- (resolved by T001) `torch.compile` / Triton work; mode ids, wheel index, nsys report names confirmed.
- (medium) openpi may want a torch other than 2.8.0 / pip may swap in the CPU wheel (T002 header warns).

## Next steps
0. **Next pull:** expect T010 `status: done` + the six small fixes, then T004 results (E5 from the E11f exports,
   ncu per region, E6, E7 at MAXN, then "next: resume E7 at mode 2" and a stop for the human reboot). Check the
   E5 idle split for the compiled mode and that `parse_ncu.py` uses the two counters. Then T007 (E5b) / T006.
0a. On each pull: check `STATUS.md` "Repository size", `find results -size +50M`, and that `raw_files` hashes match
   (`sha256sum`) for a sample of files.
1. (done 2026-09-21) T001; (done 2026-09-22) T002, T003, T005, T010 received and analysed; ladder script in
   `analysis/ladder.py`; "for microarch-owner:" items handed to main 2026-09-22.
2b. (done 2026-09-22) T010 checker run on every `model_inputs.csv`; consolidated file handed to main. Still planned:
   `analysis/model_inputs_to_hostparams.py` (join on `microarch/03_tools/trace_gen.py` segment keys) if
   microarch-owner wants a derived parameter file rather than the CSV.
3. When T004/T007 arrive: `analysis/` scripts for region utilisation and the stall/LSU tables; hand to evaluation.

## Interfaces with other sub-projects
- → microarch-owner: measured latency/power/kernel data for `host_model.py`, `energy_model.py`, roofline points.
  Interface for the hybrid model: `results/<date>_E11_<model>_inputs/model_inputs.csv` (protocol §13.8; mapping table
  in `docs/00_context.md`). `transfer_time` should become `fit_a + fit_b·bytes`; SYNC points get `t_sync`.
- → paper-owner: the measured baseline ladder, SM-occupancy and instruction-level evidence, LIBERO results.
- Results flow through evaluation-owner (collected with provenance) before paper / site.
