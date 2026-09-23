# PROGRESS — phase-balance study, π0 → π0.5 → SmolVLA → GR00T N1.7/N1.6/N1.5

**Started 2026-09-22 18:35 PDT · budget 26 h (owner, revised) · ends ≈ 2026-09-23 20:35 PDT**
**⛔ DON'T STOP. A turn ends only when the next job is already running.**

I keep this file current. Every item is `todo` / `running` / `done` / `blocked`, with the
evidence or the exact failing command.

---

## The question this answers

Which models are action-head bottlenecked, under an *identical* treatment — identically
optimised, or identically un-optimised? π0 says the answer depends entirely on whether you
compile: **62.6 % of the frame un-optimised, 18.9 % optimised.** If that is true of every
model, the PIM premise is a statement about compilation, not about VLAs.

## Scope and budget

The owner raised the budget to **26 h** and asked for it done properly, so **every model
gets the full routine**: Part A (paper configuration, fp32 control, per-phase SM
utilisation, operational intensity) and Part B (the full 9-rung attribution ladder). No
reduced ladders.

**One N per model — its own configured default**, which is the deployment setting and the
one the authors' leaderboard numbers use. Verified from each config rather than assumed,
and the field name differs three ways, which the registry must carry:

| model | field | default N |
|---|---|---:|
| π0 | `num_inference_steps` | **10** |
| π0.5 | `num_inference_steps` | **10** |
| SmolVLA | **`num_steps`** | **10** |
| GR00T N1.7 | **`num_inference_timesteps`** | **4** |

π0's N=4 ladder is already done and is *kept as the control*, not as waste: it is a
within-model test of the hypothesis that N drives the action-head share. Same model, same
machine, only N changes — AE share **32.3 % at N=4 against 54.9 % at N=10**, both eager
fp32. That is the cleanest evidence available that the share is a property of iteration
count. It also explains the paper's stray claim that π0 "requires only 4 denoising steps":
4 is GR00T's default, not π0's.

| model | N | Part A | Part B | new work | est. |
|---|---:|---|---|---|---:|
| π0 | 10 (+4 control) | done / running | running + B2–B4 | — | 3.5 h |
| model registry | — | — | — | the gate for everything below | 2 h |
| π0.5 | 10 | full | full 9 rungs | wrap points differ | 3.5 h |
| SmolVLA | 10 | full | full 9 rungs | `regions.py` needs a new hook — layer body is inlined | 4 h |
| GR00T N1.7 | 4 | full | full 9 rungs | different codebase; DiT instrumentation | 4 h |
| GR00T N1.6 | 4 | full | full 9 rungs | own checkout + env; reuses N1.7's instrumentation | 3 h |
| GR00T N1.5 | 4 | full | full 9 rungs | own checkout + env | 3 h |
| cross-model report | | | | | 1 h |
| | | | | **total** | **≈ 24 h** |

**Ordering rule inside each model, kept for risk not for economy.** The two endpoint
configurations — un-optimised and best-optimised — run *first*, before that model's full
ladder. They are what answers the question, so if one model's instrumentation overruns,
the next model still gets its answer rather than the overrun eating the queue. This is
sequencing, not a reduction in scope.

---

## Now — updated every 2 min by `bench/progress_monitor.py --mode fast`

<!-- FAST:BEGIN -->
**Updated 23:17:11** · elapsed **4.70 h** of 26 (18 %) · 51.1 GiB free · 0.72 GiB swap

> **nothing is running.** If this persists, the chain broke — check the last log below and start the next item.

Latest log lines:

- `groot_probe.log` · File "/home/yanggon/venvs/groot/lib/python3.10/site-packages/torch/__init__.py", line 431, in <module>
- `groot_probe.log` · from torch._C import *  # noqa: F403
- `groot_probe.log` · ImportError: libcudss.so.0: cannot open shared object file: No such file or directory
- `smolvla_nvg.log` · File "/home/yanggon/00_VLA/26_measurement_jetson_vla/bench/opt_ladder.py", line 404, in <lambda>
- `smolvla_nvg.log` · rows.sort(key=lambda r: (int(r["N"]), [w[0] for w in want].index(r["rung"])))
- `smolvla_nvg.log` · ValueError: 'ae_compile' is not in list
<!-- FAST:END -->

## Results — re-derived from every result CSV every 20 min (`--mode slow`)

<!-- SLOW:BEGIN -->
**Re-derived from every result CSV on disk at 23:17** (36 configurations)

| model | N | configuration | vision | vlm_prefix | ae_total | e2e | **AE share** |
|---|---:|---|---:|---:|---:|---:|---:|
| pi05 | 10 | eager_fp32 | 239.45 | 2162.47 | 617.74 | 3028.56 | 20.4 % |
| pi05 | 10 | eager_tf32 | 103.11 | 534.28 | 611.42 | 1254.86 | 48.7 % |
| pi05 | 10 | ae_compile | 103.43 | 549.79 | 258.12 | 917.88 | 28.1 % |
| pi05 | 10 | ae_graphs | 104.28 | 549.50 | 245.58 | 905.68 | 27.1 % |
| pi05 | 10 | vis_compile | 102.38 | 556.54 | 244.73 | 909.88 | 26.9 % |
| pi05 | 10 | vis_graphs | 100.25 | 548.77 | 245.23 | 900.66 | 27.2 % |
| pi05 | 10 | vlm_compile | 99.82 | 516.60 | 240.75 | 863.47 | 27.9 % |
| pi05 | 10 | whole_graphs | — | — | — | 860.09 | — |
| pi05 | 10 | bf16_vision | 88.05 | 558.22 | 244.43 | 897.61 | 27.2 % |
| pi0 | 4 | eager_fp32 | 241.31 | 234.06 | 230.12 | 711.75 | 32.3 % |
| pi0 | 4 | eager_tf32 | 104.18 | 234.06 | 230.75 | 575.28 | 40.1 % |
| pi0 | 4 | ae_compile | 103.26 | 234.25 | 99.94 | 444.35 | 22.5 % |
| pi0 | 4 | ae_graphs | 103.70 | 234.49 | 71.37 | 416.85 | 17.1 % |
| pi0 | 4 | vis_compile | 101.55 | 234.54 | 70.58 | 413.63 | 17.1 % |
| pi0 | 4 | vis_graphs | 100.20 | 234.60 | 71.43 | 413.74 | 17.3 % |
| pi0 | 4 | vlm_compile | 99.92 | 192.78 | 69.72 | 369.52 | 18.9 % |
| pi0 | 4 | whole_graphs | — | — | — | 366.12 | — |
| pi0 | 10 | eager_fp32 | 241.01 | 234.57 | 586.91 | 1069.04 | 54.9 % |
| pi0 | 10 | eager_tf32 | 102.71 | 234.04 | 573.48 | 916.45 | 62.6 % |
| pi0 | 10 | ae_compile | 103.58 | 234.35 | 241.28 | 585.80 | 41.2 % |
| pi0 | 10 | ae_graphs | 103.36 | 234.74 | 172.18 | 516.42 | 33.3 % |
| pi0 | 10 | vis_compile | 101.88 | 234.55 | 173.17 | 516.44 | 33.5 % |
| pi0 | 10 | vis_graphs | 100.04 | 234.95 | 174.45 | 515.99 | 33.8 % |
| pi0 | 10 | vlm_compile | 99.18 | 192.76 | 168.84 | 466.99 | 36.2 % |
| pi0 | 10 | whole_graphs | — | — | — | 463.86 | — |
| smolvla | 10 | eager_fp32 | 33.30 | 65.77 | 620.56 | 727.46 | 85.3 % |
| smolvla | 10 | eager_tf32 | 33.16 | 64.75 | 612.92 | 718.64 | 85.3 % |
| smolvla | 10 | ae_compile | 33.27 | 64.18 | 161.93 | 267.42 | 60.6 % |
| smolvla | 10 | ae_graphs | 33.25 | 65.71 | 43.82 | 150.32 | 29.2 % |
| smolvla | 10 | vis_compile | 35.15 | 65.92 | 43.65 | 152.70 | 28.6 % |
| pi05 | 10 | paper config (eager+TF32) | 103.61 | 550.51 | 626.90 | 1287.32 | 48.7 % |
| pi05 | 10 | paper config (eager+TF32) | 239.39 | 2166.69 | 617.87 | 3030.99 | 20.4 % |
| pi0 | 10 | paper config (eager+TF32) | 103.00 | 234.30 | 571.58 | 915.15 | 62.5 % |
| pi0 | 10 | paper config (eager+TF32) | 241.51 | 234.36 | 580.24 | 1062.42 | 54.6 % |
| smolvla | 10 | paper config (eager+TF32) | 33.31 | 64.60 | 610.92 | 716.81 | 85.2 % |
| smolvla | 10 | paper config (eager+TF32) | 33.26 | 64.49 | 608.01 | 713.69 | 85.2 % |
<!-- SLOW:END -->

## Live status

| # | item | state | evidence |
|---|---|---|---|
| 0 | π0 Part A — operational intensity (A1) | **done** | vision **398.1**, `vlm_prefix` **816.0**, `ae_step` **60.0** FLOP/B vs measured ridge 180.57. Paper: 840 / 64.5. `bench/vision_flops.py` |
| 1 | π0 Part A — paper config reproduced | **done** | N=10 eager+TF32: **916.45 ms vs their 920.58 (0.45 %)**; split 11.2/25.5/**62.6** % vs their 12/28/**59** |
| 2 | π0 Part B ladder, N=4 | **done** | 9 rungs; two independent runs agree to 1.81 % worst case |
| 3 | π0 Part B ladder, N=10 | **running** | 3 of 9 rungs; memory flat 50.6→50.9 GiB per rung |
| 4 | π0 Part A — SM utilisation (A2) | queued | chained on ladder PID; needs sudo for GPU metrics |
| 5 | π0 Part B — B2 bf16 vision, B3 bf16 layernorms, B4 graphs on `pg.forward` | todo | B3 is the lever with a mechanism; B4 should be small at 816 FLOP/B |
| 6 | π0 LIBERO quick check (B2/B3 change numerics) | todo | 100 episodes per variant, ~45 min each |
| 7 | **model registry** — the gate for everything below | **done (LeRobot)** | `bench/model_registry.py`. N field differs 3 ways; wrap points uniform once the container attr is known (`paligemma_with_expert` / `vlm_with_expert`). GR00T descriptor deliberately pending until its instrumentation exists. |
| 8 | π0.5 — Part A + full ladder | todo | `lerobot/pi05_libero_base` |
| 9 | SmolVLA — Part A + full ladder | todo | `lerobot/smolvla_base`; **`regions.py` records nothing** — layer body inlined |
| 10 | GR00T N1.7 — Part A + full ladder | todo | installed, verified; `env/setup_groot.sh` |
| 11 | GR00T N1.6 — Part A + full ladder | todo | branch `n1d6` @ `9b37aa1`, `nvidia/GR00T-N1.6-3B`; own env |
| 12 | GR00T N1.5 — Part A + full ladder | todo | branch `n1d5` @ `4af2b62`, `nvidia/GR00T-N1.5-3B`; own env |
| 14 | `reports/E4d_phase_balance.md` — the cross-model table | todo | one row per model: AE share un-optimised vs optimised |

## Results so far

| model | config | vision | `vlm_prefix` | `ae_total` | e2e | **AE share** |
|---|---|---:|---:|---:|---:|---:|
| π0 | N=10 eager fp32 | 241.01 | 234.57 | 586.91 | 1069.04 | 54.9 % |
| π0 | N=10 eager **+TF32** (= paper config) | 102.71 | 234.04 | 573.48 | **916.45** | **62.6 %** |
| π0 | N=4 eager fp32 | 241.31 | 234.06 | 230.12 | 711.75 | 32.3 % |
| π0 | N=4 fully compiled | 99.92 | 192.78 | 69.72 | 369.52 | **18.9 %** |
| *(their π0 on AGX Orin)* | *N=10 eager* | *110.47* | *257.76* | *543.14* | *920.58* | *59 %* |

## Rules that are not optional

1. **One process per rung.** `del policy` does not free the 8.28 GiB model — `torch.compile`
   retains references and on Orin GPU memory is system memory. Two runs were OOM-killed at
   rung 9 before this.
2. **Memory floor 20 GiB, enforced**, recorded before and after every rung.
3. **Preflight before every long run.** Three defects were caught this way, each of which
   would have produced a complete, plausible, wrong table.
4. **Wait on a PID, never `pgrep -f <pattern>`** — it matches the shell that contains the
   pattern. This has cost two waits already.
5. **Write results incrementally**; a kill must cost one rung, not the run.
6. **Commit `bench/` before the run that uses it**, so `bench_commit` names the code that ran.

## Log

- **18:35** file created. π0 N=10 ladder at 3/9, Part A armed behind it.
- **19:05** model registry written and committed for the LeRobot family; GR00T pending.
- **18:45** one N per model, its configured default: π0/π0.5/SmolVLA 10,
  GR00T 4. Three different field names. π0's N=4 ladder kept as the within-model control.
- **18:40** owner raised the budget to 26 h — full 9-rung ladder for every model, no
  reduced ladders. Endpoint-configs-first kept inside each model as risk sequencing.
