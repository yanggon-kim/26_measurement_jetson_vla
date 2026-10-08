# E13 / E14 / E14b / E8s and the late addenda — research-side review (2026-10-08)

*jetson-owner, on the research server. Reviews what the device returned for T016 (E13, E8s, E14, E7r, E12c) and T017
(E14b), plus the owner-requested analyses that followed (E12d, the analytic co-run report, E7 capped §2b / §3, E7v,
E5c). Each report is checked against its pre-registration as issued: T016 rows 1–8 (rows 1–7 as issued, row 8 from
Amendment 1, both before any measurement) and T017 rows 1–9. Pulled range `1c5cd12..2984671` (173 commits; last
device commit 2026-09-29 01:28 PDT). **Measured** = a committed row. **Derived** = arithmetic on committed rows,
never measured as such; every derived number here is tagged `[derived]`. This file is analysis. It is not an
instruction to the device.*

All runs are MAXN + `jetson_clocks` (16 SMs, GPU 1300.5 MHz, EMC 3199 MHz), π0 LeRobot `f0d2b37` / ckpt `45dcc8fc`,
batch 1, N = 10 unless stated, never throttled, 0 non-finite (every report states it). Bench commits: T016 close
`3b4cea0`, T017 close `fa805b0`, T004 4b(ii) / E12d `e568440` and later.

## 1. Verdict per report

| report | pre-registered rows | honest? | defects (§8) |
|---|---|---|---|
| `E13_pi0_bandwidth_steal.md` (T016 step 1) | 1 > 1.247 (σ 1.247 at 77.1 GB/s, 1.28 at the 85 GB/s max; 94 not reachable SM-free) · 2 1.125 (middle band) · 3 linear (±0.015) · 4 no · 8 0.67, on the 2/3 line | yes. The unreachable 94 GB/s point and the failed in-session calibration are disclosed and are not papered over. Row 8's fit quality is stated, and 0.67 is read as "not distinguishable from mixed" | D1, D2, D3 |
| `E8s_libero_seeded.md` (step 2) | 5 replicated (4/4 suites, pooled p 1.3e-5) · 6 95/100, exactly on the line | yes. The per-suite p values are labelled unadjusted, and the pooled test is the pre-registered one | D5 (camera) |
| `E14_prefix_savings.md` (step 3) | 7 bitwise (GR00T `keep1`, `kvonce`, both: 50/50, compiled and eager) | yes. The first, invalid GR00T action sets are kept and labelled | D4, D6 |
| `E14b_pi0_padtrim.md` (T017) | 1 0.734 · 2 0.665 · 3 1.028 · 4 −0.010 · 5 0.792 · 6 0.653 · 7 248.5 ms · 8 exact · 9 PASS ×3 | yes. The failed bf16 gate attempt is reported, and so is the duplicate launch. Row 3's slowdown and row 6's miss are stated plainly | D7 |
| E7 correction block (step 4, EV-10) | residual target < 5 % | met: 0.49–0.71 % | — (it is the fix) |
| E12b addendum (step 5, EV-13) | 2 × 2 table | complete; the script is in git | — |
| E12 addenda (E12d), `ANALYTIC_corun_feasibility.md` | none (owner requests) | measured vs derived kept apart | D8 |
| `E7_capped_modes.md` §2b, §3 | none | yes | D9 (open, not a defect) |
| SYNTHESIS §2.1 (E7v), §3.4 (E5c) | none | closure checks shown (0.4–1.8 %, 0.3–4.9 %) | — |

The device met T016's and T017's acceptance lists item by item. Two exceptions are both allowed by the specs: ~94 GB/s
was unreachable without SMs (A1 says to state the highest rate reached), and the optional `kernel_bytes.csv` per
variant was not produced. No pre-registered band was moved.

## 2. E13: σ vs stolen bandwidth, and question (a)

**Measured** (`results/20260926_E13a_pi0_bwsteal_ce+cpu_MAXN/sigma.csv`, n = 30 per segment, bracketed):
σ = 1.000 / 1.060 / 1.120 / 1.181 / 1.247 / 1.278 / 1.281 at 0 / 20 / 40 / 59.8 / 77.1 / 82.9 / 84.8 GB/s stolen.
The x-axis is the load's own byte count during the prefix, 2 × the bytes copied (`bench/e13_bwsteal.py:157`), so it
is DRAM read + write traffic. The load was 50 % writes. **The curve steepens at the top**: the local slope is
0.0030 per GB/s below 60 GB/s, 0.0038 from 60 to 77 and 0.0053 from 77 to 83 `[derived]`. "1 + 0.0032·x" is the
low-load secant, not a law.

**Checking main's mapping.** The per-channel share of the 178.5 GB/s achievable bus is 11.16 GB/s.

| PIM channels k | stolen if the PIM's occupancy = channel share | σ linear 0.0032 | σ interpolated between measured points | with the in-session load-byte EMC fit (≈ 196 GB/s full bus) |
|---:|---:|---:|---:|---:|
| 4 | 44.6 GB/s | 1.143 | **1.134** | 1.148 |
| 6 | 66.9 GB/s | 1.214 | **1.208** | 1.233 |
| 8 | 89.2 GB/s | 1.285 | beyond the reachable range (≥ 1.28) | — |

All values `[derived]`. Main's 1.21 / 1.14 hold to ±0.02. The full-bus uncertainty (D1) is the larger error bar.

**Bandwidth steal through shared arbitration vs a hard channel partition.** These are different mechanisms. E13
measured only the first.
1. **What E13 is.** A foreign read + write stream shares all 16 channels with the prefix through the memory
   controller's queues. The prefix keeps its full channel parallelism, but every access waits behind foreign
   requests: row conflicts and read/write turnarounds. The bus was never full (EMC 41 % → 76 % at the maximum,
   against 89 % for a pure stream). So E13's slowdown is **queueing latency**, not a bandwidth cap. This is the
   right model for a memory-side engine that issues ordinary requests through the MC. The copy-engine half also
   shares the GPU's memory hub / SMMU path, which a PIM would not (D2).
2. **A hard, static partition** (GPU data mapped only onto 16 − k channels, PIM weights on k) has no foreign
   traffic on the GPU's channels. It has a hard cap of (16 − k)/16 of the bus **all the time**, whether the PIM is
   busy or not, so the duty cycle (E13b) does not apply. The lower bound on its slowdown is a per-kernel cap model on
   E13d's committed per-kernel bytes. Each kernel's time becomes max(t, bytes / BW_left). The model adds no latency
   for fewer channels (`analysis/e13_cap_model.py`, `[derived]`):

   | channels left (k) | 12 (4) | 11 (5) | 10 (6) | 9 (7) | 8 (8) |
   |---|---:|---:|---:|---:|---:|
   | BW left GB/s | 133.9 | 122.7 | 111.6 | 100.4 | 89.2 |
   | σ_cap, lower bound (frames f0 / f1) | 1.044 / 1.045 | 1.062 / 1.063 | **1.084 / 1.085** | 1.119 / 1.120 | 1.193 / 1.189 |

   The prefix's mean draw is 85 GB/s, but 45–49 % of its kernel time runs above 90 GB/s and 14–15 % above 150 GB/s.
   So the cap bites before the mean says it should. **A hard partition at k = 6 then costs between ~1.08 (cap only)
   and ~1.21 (E13's contention value at the same bandwidth left).** E13's number is not a strict upper bound
   either: fewer channels also mean deeper queues for the GPU's own requests. Neither bound is measured. Stock
   Orin cannot confine the GPU (row 4: the map is in the boot-time BCT). F-a1 below gives a measurable proxy.
3. **A shared address map with a PIM that seizes channels** (AiM all-bank mode on channels that the GPU's hashed
   interleave also uses) is neither case. Every GPU line that falls on a seized channel stalls until the PIM
   releases it. That is **blocking**, which E13 does not model at all: it can approach serialisation. The design
   must state which of the three it assumes.
4. **Duty cycle.** E13b's 37 % result (σ 1.125, linear in the bytes exposed, row 3) applies only to case 1, with
   the PIM busy for 37 % of the frame. In the balanced throughput mode (case 005), the PIM expert on k channels runs
   ≈ 93.1 × 16/k ms `[derived; channel-linear scaling is microarch's to confirm]`, which is 248 ms at k = 6 ≈ the
   prefix. Its duty is then ≈ 1, so the **full-time** σ is the right one at the balance point. Main's use of the
   full-time σ is correct there.
5. **What it gives the design** `[derived]`, at k = 6: period ≈ max(245.4 × σ, 248) = 266–296 ms, against the
   measured best GPU co-run of 307.15 ms (E12b green 12/4) and the GPU serial frame of 347.6 ms. That is 1.04–1.15×
   over the co-run and 1.17–1.31× over serial, for the cap and the contention end respectively. At k = 5 the PIM
   (≈ 298 ms) sets the period. At k = 8 the prefix does (σ ≥ 1.19). k ≈ 6 is the balance point. For
   brainstorm-owner and microarch-owner, via main.

**E13d (row 8).** GEMMs read 15.4–15.5 GB against 8.08 GB of unique inputs. Four shapes carry 98 % of the 7.4 GB
excess: Gemma MLP gate/up 2.74 GB (weights, ~2.2× re-read at M = 816), SigLIP MLP 2.45 GB (mostly weights), Gemma
down 1.72 GB (activations, its 26.7 MB input exceeding the 4 MB L2), and attention projections 0.34 GB. 0.67 is a
non-negative fit with 18–55 % residuals, and cuBLAS changes kernels along the sweeps, so **read it as "mixed, weight-leaning"**.
The research lever follows: the gate/up re-read factor falls to ~1.1× at M ≤ 408 (measured on the sweep), so
splitting the prefill into two M-halves would cut ~1.4 GB per frame of weight re-reads. That is an observation for
microarch's EV-17 estimate, not a measured variant.

## 3. E14b / E14: the padding trim, and questions (b) and (c)

**Measured** (E14b, `graphs_both` harness, one bracketed session, pad drift 0.03 %; prefix / expert / serial ms):
`pad` 245.41 / 101.60 / 347.64 · `nocam` 177.77 / 88.35 / 269.37 · `trimprompt` 252.28 / 69.81 / 323.41 · `nopad`
180.12 / 66.32 / 248.48. Prefix bytes (reads + writes ≤): 21.60 / 17.30 / 21.58 / 17.10 GB. Expert reads 7.03 /
6.92 / 7.01 / 6.91 GB. K/V exact for all four. The fp32 gate passed (worst REL 1.5e-6). Compiled bf16 was 0/64
bitwise and PASS ×3 against the floor F (MAXABS 0.296, REL 0.0245). The pad repeat was bitwise 64/64. E14's rung
path agrees: prefix 0.749, frame 420.5 → 319.5 ms. Its masked-camera LIBERO pair gave 70 vs 66 / 100, p = 0.29.

**(c) The two anomalies have one probable cause: integer alignment, with opposite signs in the two stages.**
- **Prefill.** Gemma GEMMs run with M = prefix length. 816 and 560 are ≡ 0 (mod 8) and fast. 789 and 533 are ≡ 5
  (mod 8): +6.7 and +2.5 ms in `vlm_prefix` while moving the same bytes (0.999).
- **Expert.** The expert's attention runs over prefix + 51 suffix keys (E5c: "867-token cache"). pad 867 ≡ 3 and
  nocam 611 ≡ 3 (mod 8) are slow. trimprompt 840 = 8·105 and nopad 584 = 8·73 are fast. nocam's expert saving
  (13.25 ms) matches token-linear scaling of E5c's attention time: 256/867 × 4.19 ms × 10 = 12.4 ms `[derived]`.
  So nocam keeps the same slow kernel. trimprompt saves 31.8 ms, ≈ 76 % of the whole attention time, while its
  expert bytes drop 0.3 %. That is a **kernel switch**, consistent with E5c's finding that pad's 36 attention
  products per step run in `cutlass_75_wmma` kernels at ~1.6 TFLOP/s and ~30 GB/s, "neither compute- nor
  bandwidth-bound".
- **Why the signs oppose.** With a fixed 51-token suffix, a prefix aligned to 8 makes the key length ≡ 3 (mod 8),
  and the reverse. LeRobot's padded prompt (48) happens to align the prefill and misalign the expert. That is
  exactly the 4-of-4 pattern above `[derived; untested]`.
- **One clue does not fit.** In the LIBERO rung path with the camera attended, `trimprompt` (also 840 keys) gave
  426.5 vs 426.5 ms static (E14 speed diagnosis) and 423.3 vs 427.5 ms in the arm. That is ~0–4 ms, not ~25 ms.
  E14's rung-path `nopad` did save 38 ms in the expert, so the rung path does show the effect at 584 keys. Either
  the attended-camera input or something in the LIBERO loop differs. This is unresolved, and F-c1 settles it.
- Also measured, not a defect: on the checkpoint's own bf16 eager path, `trimprompt` moved one of 64 inputs by
  MAXABS 1.04 (REL 4.8e-2). That is > F_maxabs, but it is not the pre-registered comparison. The flow integrator
  is sensitive on some inputs. The compiled verdict and LIBERO (70 vs 71, p = 1.00, the attended arm) stand.

**(b) "nopad expert 66.3 ms < AiM full design 93.1 ms": not a fair comparison as it stands.** Five issues:
1. **Different K/V lengths.** 66.3 ms attends over 584 keys. The AiM full design keeps attention in the PIM
   (microarch `col5_primary_design_point.md` §4b: "stock with attention on the PIM", prefix KV in the PIM), at
   the trace's padded length. microarch-owner should confirm the length. A fair pair simulates the PIM at 584 keys
   too. The PIM's attention MACs shrink with the trim, and its weight MACs do not.
2. **Most of the GPU gain is a kernel-selection artefact, not less work.** If the alignment reading holds, a GPU
   expert at the *padded* length with 5 masked keys appended (867 → 872, exact) would already get most of the
   ~30 ms. **The GPU's 101.6 ms pad expert would then be a ~70 ms expert carrying a removable attention-kernel
   penalty** `[hypothesis; F-c3 tests it]`. Per step that is ~7.0 ms against AiM's 9.31 ms. The GPU expert would
   beat the AiM full design per step at either length.
3. **Harness tokens.** 66.3 ms carries `hoist + owngraph_expert + dbuf_kv` (OD-1, not the rung). The rung-path
   nopad expert is 129.9 ms (E14: 319.5 − 189.6).
4. **Which mode.** In latency mode `[derived]`: pad serial 347.6 vs 245.4 + 93.1 = 338.5 ms (PIM 1.027×). nopad
   serial 248.5 vs 180.1 + 93.1 = 273.2 ms (**PIM 0.91×, a loss**) unless the PIM is re-simulated at 584 keys. In
   throughput mode the comparison is not expert vs expert. It is the period, max(prefix·σ, PIM expert) vs the GPU
   co-run, so 66 vs 93 matters only through the balance point.
5. **Floor.** The nopad GPU expert moves 6.91 + 0.99 GB, a 44.3 ms byte floor at 178.5 GB/s. 66.3 ms is 1.50× that
   floor. The GPU is closer to its floor than the research side assumed.

A fair statement needs both sides at the same token count, one GPU harness named, and the attention-kernel
artefact either removed on the GPU or quoted beside it.

## 4. E8s: N = 2 is more accurate, question (d)

**Measured**: seeded, paired, all four suites. N = 1 / 2 / 3 / 4 / 6 / 10 pooled 295 / 300 / 286 / 284 / 267 / 259
of 400. N = 1–4 each beat N = 10 at p ≤ 0.003; N = 6 does not (p 0.26). No suite has N = 10 as its best arm. The
repeat control agreed on 95/100. The forward fits 16.9 ms/step + 258.2 ms (rung path).

**What it implies.**
- **The configured N is a deployment choice that cuts the PIM's target by ~5×.** At N = 2 the expert is 2 steps:
  ~34 ms on the rung path, ~13–20 ms in the `owngraph` harness (6.6–10.2 ms/step), against 18.6 ms for AiM at
  9.31 ms/step `[derived]`. The expert falls from ~29 % of the serial frame (101.6/347.6) to ~5–8 % in the harness (~12 % on the rung path). A latency-mode
  PIM gain is then bounded by ~1.08×, and in throughput mode the expert no longer sets anything. A PIM study argued
  at N = 10 must say that π0's own fine-tune is more accurate at N = 1–4. Otherwise a reviewer who runs N = 2 will
  say it.
- **It is a user decision of OD-1's kind.** The checkpoint's default N = 10 stays the reference unless the user
  says otherwise. Present N as an axis (gain vs N) and never pick it silently.
- **Scope limits:** one fine-tune (`pi0_libero_finetuned_v044`), one rung (`bf16_both`), LIBERO only, π0 only, and
  every arm with the empty camera attended (§5). The mechanism is untested. The repeat control sits exactly on the
  95 % line.

## 5. The empty-camera finding, question (e)

`bench/libero_eval.py` fed `empty_camera_0` as an all-zero image with mask 1 (attended) in **every E8 and E8s arm**.
Training, `pi0_common.build_batch` and every timing / bytes / equivalence run use LeRobot's −1 image with mask 0.

- **No latency, byte, energy, K/V or equivalence number is affected.** The shapes are identical (816 tokens either
  way; only mask values differ), the forwards match (pad 427.5 ms with the camera attended or masked), and the
  non-LIBERO harnesses never used `libero_eval.py`.
- **Paired LIBERO comparisons stand for the input they used**: `bf16_both` vs stock (249 vs 252), the seeded N
  sweep, and the trimprompt arm.
- **Absolute LIBERO success rates are for a non-training input.** Never quote them as "π0's success rate". The one
  masked-vs-attended check (`pad` N = 10, spatial) gave 66 vs 71 / 100, p = 0.36. The N × camera interaction is
  untested.
- **A mislabeled directory:** `results/20260926_E8s_pi0_libero_spatial_N10_seeded_nopad` is `trimprompt` with the
  camera attended. Its `meta.json` and `summary.md` still say `--prefix-variant nopad` (D5). The correct nopad
  LIBERO pair is `…_maskedcam` vs `…_maskedcam_nopad`: 66 vs 70 / 100, p = 0.29.
- Nothing the research side quotes is invalidated. Any absolute rate, any "nopad 70 vs 71" and the E8 "85 vs 62"
  (unseeded, superseded by E8s) need the caveat or the replacement. That is for evaluation-owner, via main.

## 6. T016 steps 4–5 and the owner-requested analyses

- **E7r (step 4) changes a quoted number.** The rail sampler started before warm-up / compile, so compiled phase
  energies were understated. π0 `bf16_both`, N = 10, above idle: **prefix 6.40 → 10.36 J (+62 %), expert 3.11 →
  3.25 J, full 13.67 → 13.70 J** (`results/20260926_E7r_pi0_energy_reconcile_MAXN`; re-measured, residual 0.49–0.71 %).
  28 of 132 committed phase rows move by ≥ 5 %, all of them compiled; the corrected values are in
  `E7_all_dirs_timed_window.csv`. The expert's energy share at the best rung falls from ~33 % to **24 %** (E7v
  agrees: π0 17 / 59 / 24 % vision / language / action). For microarch-owner (`energy_model.py`) and
  evaluation-owner.
- **E12c (step 5):** the single KV buffer is the operative cause (own pool + single buffer gave 98/100 wrong; shared
  pool + single buffer hangs). `dbuf_kv` is necessary and sufficient. `owngraph_expert` is not needed for
  correctness. This closes EV-13.
- **E12d (eager co-run alternatives):** with one issuer, kernels of the two streams run concurrently 0.0 % of the
  time, which settles the issue/wall objection by measurement. The graph-captured eager expert runs 4.37× faster
  (frame 599 ms). Two slim processes overlap (O 1.206, φ 0.75, period 686.9 ms; the prefix absorbs the sharing, p
  1.45). None beats the compiled co-run (307 ms). Framing II (O 1.247) is unchanged.
- **ANALYTIC co-run feasibility:** the resource floors leave π0 only 1.15–1.16× overlap headroom (compute-limited
  prefill). At 10 Hz π0 needs 1.52× Orin's bf16 compute and 1.59× its bandwidth, but at 816 padded tokens.
  With nopad the prefill FLOPs fall by ~35 %, so the compute need is ≈ 1.03× `[derived]`, still > 1.
- **E7 capped §2b / §3:** compilation buys only 1.11–1.19× in capped modes; the fixed part (vision + prefix) is
  87–92 % of the compiled N = 4 frame. For brainstorm case 006 (power cap) `[derived]`: at MODE_15W, N = 10 the
  expert is ~17.5 % of the compiled frame (10 × 66.08 ms steps / 3765.9 ms). Expert offload is bounded by ~1.21×. At
  MODE_30W the compiled `graphs_both` prefix takes 876.84 ms. If its bytes do not depend on the mode (not
  measured), it draws ~25 GB/s of a full-rate EMC, so **in capped modes the channel partition would cost the GPU
  little**. That is a point in case 006's favour, conditional on that bytes assumption. The 2.68× slowdown at 30W
  vs green-8 at MAXN (clock ratio 2.125) stays unexplained (D9).
- **E5c:** π0's compiled expert spends 4.19 ms/step on attention (cutlass_75_wmma), as much as on all 126 weight
  projections (4.54 ms, 1.33× floor). It re-copies the 15.05 MB K/V into graph inputs every step (0.36 ms, ~150 MB
  per frame). Both are GPU-side inefficiencies that a fair GPU reference may remove. The attention one is §3's
  alignment hypothesis.
- **E7v:** the action phase is the cheapest per unit time (π0 19.8 W above idle vs 40–42 W for vision and language).
  For π0, the energy case for an expert-only PIM covers at most 24 % of the frame's above-idle energy.

## 7. What this means for the research (one paragraph per finding)

1. **Throughput mode survives only in a narrow band.** A full-time steal of ≥ 77 GB/s breaks O = 1.247. A hard
   partition at k ≈ 6 costs 1.08–1.21 (§2), giving 1.04–1.15× over the GPU co-run `[derived]`. The design must say
   which memory-sharing mechanism it assumes (§2 items 1–3); the three differ by more than the margin.
2. **The GPU expert reference is softer than assumed.** Its pad attention probably carries a ~30 ms alignment
   penalty (§3), and its K/V copy costs 0.36 ms/step. A fixed GPU expert at ~7 ms/step beats AiM's 9.31 ms/step.
   The latency-mode PIM case at π0 (≈ 2.5 %) disappears against it.
3. **The configured N and the trim are both baseline decisions (OD-1)** that shrink the PIM's share of the frame.
   The paper needs them as axes, not footnotes.
4. **Energy:** after the E7r correction the expert is 24 % of π0's best-rung energy. The prefix's language part
   (59 %) is the energy target, not the expert.

## 8. Defects and corrections

- **D1 (E13, disclosed):** the in-session EMC hog calibration was not reproducible (122.6–181.8 GB/s at the same
  EMC%). "Left" uses E12b's 178.5. The load-byte fit implies ≈ 196 GB/s at 89 % EMC, so "left" is uncertain by
  ~10 %. σ vs stolen is unaffected.
- **D2 (E13):** the slope steepens above 60 GB/s (§2), so "σ ≈ 1 + 0.0032·x" understates σ near the top. Half the
  load is copy-engine traffic through the GPU memory hub, a path a PIM does not use. The CE-only and CPU-only σ at
  matched rates were not reported separately.
- **D3 (E13b, disclosed):** the "end" window sits at the unloaded frame's last 93 ms, so it covers ~158–251 of a
  ~269 ms loaded frame. f = 0.315 refers to the unloaded timeline. Row 3 still lands (−0.015).
- **D4 (E14, disclosed):** GR00T time `meta.json` were back-filled. GR00T bytes are one inference per variant.
- **D5 (E8s):** `results/20260926_E8s_pi0_libero_spatial_N10_seeded_nopad` is `trimprompt` with the camera
  attended, but its own `meta.json` / `summary.md` carry no note. Only the E14 report says so. The directory needs a
  `NOTE.md` (device side).
- **D6 (E14 vs E14b):** SYNTHESIS §0 item 10 mixes E14's rung-path frame saving (−101 ms) with E14b's harness bytes
  ratio (0.79). Both are right; they come from different harnesses.
- **D7 (E14b):** none in the measurement. The explanation of both anomalies is "not isolated", and §3 proposes one.
- **D8 (ANALYTIC caveats):** "activations spill to DRAM (E13 / E14)" contradicts E13d. Writes bound spills at
  ≤ 3.86 GB; the excess is re-reads.
- **D9 (open):** the compiled prefix at MODE_30W is 2.68× the MAXN green-8 time against a 2.125× clock ratio.
  Unexplained.

## 9. Proposed follow-ups (not issued; each needs the user's approval, then a task file)

- **F-c1** (~30 min, highest value): nsys kernel names and durations for one expert step and one prefill in the
  four E14b variants, plus the rung path at `trimprompt` with the camera masked and attended. It tells whether the
  attention kernel switches with the key length, and why LIBERO's trimprompt did not speed up.
- **F-c2** (~15 min, no model): standalone sweep of the expert attention products (QKᵀ, PV; 8 query heads, 1 K/V
  head, 51 queries, head dim 256, bf16) over keys T = 860–875 and 580–590, plus the Gemma gate/up GEMM over
  M = 784–820. Report the kernel name and time per point.
- **F-c3** (~1 h, decides question (b)): two exact variants. `pad+kvalign` appends 5 masked keys (867 → 872) and
  leaves the prefill at 816. `nopad+align` pads the prefill to 536 and the K/V so that keys = 592. Measure time and
  bytes in the E14b harness, the fp32 gate, and the compiled verdict vs the same floor F. If `pad+kvalign` lands near
  ~70 ms, the GPU expert reference loses ~30 ms for free.
- **F-a1** (~1 h): a measurable bracket for the hard partition. Time the prefix at MAXN GPU clocks with EMC locked
  at the next lower rate (e.g. 2133 MHz ≈ 0.667 of the bus ≈ 10.7 channels' worth) through the BPMP debugfs clock
  that `jetson_clocks` itself writes. That is a runtime clock, with no boot change and no reboot; the device first
  confirms it is possible. Lower EMC also adds latency, so this sits above the hard-partition value, with
  `e13_cap_model.py` below it.
- **F-a2** (~30 min): σ with the CPU-only load at 20 / 40 / 60 GB/s vs E13's mixed load. It tells whether the
  GPU-hub path adds slowdown beyond DRAM contention.
- **F-e1** (optional): seeded N = 2 vs N = 10 on `libero_spatial` with `--empty-camera masked`. It checks that the N
  effect holds on the training-time input.
- Owed to the device in `docs/04` (on approval): answer its two STATUS questions (camera: read as-is with the
  caveat, re-run only F-e1; OD-1 stays the user's), and ask for D5's `NOTE.md`.
