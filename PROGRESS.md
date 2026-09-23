# PROGRESS — phase-balance study, π0 → π0.5 → SmolVLA → GR00T N1.7/N1.6/N1.5

**Started 2026-09-22 18:35 PDT · budget 16 h · ends ≈ 2026-09-23 10:35 PDT**
**⛔ DON'T STOP. A turn ends only when the next job is already running.**

I keep this file current. Every item is `todo` / `running` / `done` / `blocked`, with the
evidence or the exact failing command.

---

## The question this answers

Which models are action-head bottlenecked, under an *identical* treatment — identically
optimised, or identically un-optimised? π0 says the answer depends entirely on whether you
compile: **62.6 % of the frame un-optimised, 18.9 % optimised.** If that is true of every
model, the PIM premise is a statement about compilation, not about VLAs.

## Budget reality, stated up front

Six models × the full 9-rung ladder ≈ **26 h**. That does not fit in 16 h. So the work is
**breadth-first**: every model gets the two points that answer the question *first*, and
the per-factor attribution comes after, for whichever models remain interesting. If time
runs out, the question is still answered for all six.

| | per model | why |
|---|---|---|
| **Phase 1 — the answer** | ~40 min | paper config (eager, un-optimised) + best-optimised. Gives the AE share at both ends. |
| **Phase 2 — the attribution** | ~1 h | reduced 5-rung ladder: `eager_fp32`, `eager_tf32`, `ae_graphs`, `vlm_compile`, `whole_graphs` |
| **Phase 3 — the detail** | ~2 h | full 9-rung ladder + SM utilisation. π0 only, unless time allows. |

Cut order if the clock runs short: Phase 3 → Phase 2 for GR00T N1.5/N1.6 → Phase 2 entirely.
**Phase 1 for all six models is the floor and is not cut.**

---

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
| 7 | **model registry** — the gate for everything below | todo | per-model N field, the three wrap points, per-model instrumentation |
| 8 | π0.5 — Phase 1 | todo | `lerobot/pi05_libero_base` |
| 9 | SmolVLA — Phase 1 | todo | `lerobot/smolvla_base`; **`regions.py` records nothing** — layer body inlined |
| 10 | GR00T N1.7 — Phase 1 | todo | installed, verified; `env/setup_groot.sh` |
| 11 | GR00T N1.6 — Phase 1 | todo | branch `n1d6` @ `9b37aa1`, `nvidia/GR00T-N1.6-3B`; own env |
| 12 | GR00T N1.5 — Phase 1 | todo | branch `n1d5` @ `4af2b62`, `nvidia/GR00T-N1.5-3B`; own env |
| 13 | Phase 2 ladders, models 2–6 | todo | 5 rungs each |
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
