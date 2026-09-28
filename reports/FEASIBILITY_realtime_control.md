# Why acceleration is needed — VLA latency on Jetson AGX Orin vs robot control requirements (2026-09-28)

A derived note for the paper: measured per-frame latencies from this study set against control-rate
requirements taken from the literature. No new measurement. Device: Jetson AGX Orin 64 GB, **MAXN** (16 SMs,
GPU 1300.5 MHz, EMC 3199 MHz), batch 1. Latencies: `reports/E4i_best_combinations.md` (e2e medians, ladder rungs).

## 1. One frame (camera images + instruction → one action chunk), measured

| model | denoising steps N | actions per chunk | eager, as shipped | **best on Jetson** | speed-up | best, inferences / s | best, ms per action of the chunk (derived) |
|---|---:|---:|---:|---:|---:|---:|---:|
| π0 (3.3 B) | 10 | 50 | 1069 ms | **424 ms** | 2.52× | 2.4 Hz | 8.5 |
| π0.5 | 10 | 50 | 3029 ms | **488 ms** | 6.21× | 2.1 Hz | 9.8 |
| SmolVLA (0.45 B) | 10 | 50 | 727 ms | **92 ms** (81 ms as one graph) | 7.88× (8.98×) | 10.8 Hz (12.3 Hz) | 1.8 |
| GR00T N1.7 | 4 | 40 | 372 ms | **240 ms** | 1.55× | 4.2 Hz | 6.0 |
| GR00T N1.6 | 4 | 50 | 327 ms | **175 ms** | 1.87× | 5.7 Hz | 3.5 |
| GR00T N1.5 | 4 | 16 | 255 ms | **169 ms** | 1.51× | 5.9 Hz | 10.5 |

"Best" = the best committed configuration per model (bf16 casts, TF32, `torch.compile`, CUDA graphs; details in
`reports/SYNTHESIS_all_measurements.md` §1). Further measured variants for π0: prefix padding trim 248.5 ms per frame
(`reports/E14b_pi0_padtrim.md`, compiled co-run harness), N = 2 denoising steps ≈ 292 ms (`reports/E8s_libero_seeded.md`,
where N = 2 is also *more* successful on LIBERO). A π0 frame costs ≈ 258 ms fixed + 16.9 ms per denoising step
(E8s, linear fit). In the LIBERO loop the compiled π0 reaches a 19.8 Hz wall-clock control rate only because it
executes each 50-action chunk open-loop and re-plans every 2.5 s (`results/20260926_E8s_pi0_libero_spatial_N10_seeded/summary.json`).

## 2. What robots need (literature)

| source | requirement / reference point |
|---|---|
| π0 [1] | robots controlled at **20 Hz** (UR5e, Franka) or **50 Hz** (all others); action chunk H = 50; inference every **0.8 s** (after 16 actions) or every **0.5 s** (after 25 actions) — so latency must stay below 500–800 ms for that open-loop schedule. π0 itself: **73 ms on an RTX 4090** (image encoders 14 ms, observation forward 32 ms, 10 flow steps 27 ms; 86 ms off-board). |
| GR00T N1 [2] | System 2 (vision-language) at **10 Hz** on an L40; System 1 (action generation) produces closed-loop motor actions at **120 Hz**; sampling a 16-action chunk takes **63.9 ms on an L40** (bf16). |
| VLA-Perf [3] | "we define a **10 Hz** inference frequency as **acceptable** … and **100 Hz** as **high-performance**" (camera frame rates are 24–60 Hz); real-time means **10–100 ms** latency. π0 on the next-generation Jetson Thor: 19.0 Hz; π0-XL: 2.1 Hz. |
| VLA survey [4] | background: autoregressive VLAs typically reach 3–5 Hz end to end. |

## 3. Feasibility (derived from §1 and §2)

| target | latency budget | meets it on Jetson Orin (best) | misses it |
|---|---:|---|---|
| reactive closed-loop control, "acceptable" [3] | ≤ 100 ms (10 Hz) | SmolVLA only (92 / 81 ms) | π0 4.2×, π0.5 4.9×, GR00T N1.5–N1.7 1.7–2.4× over; as shipped 2.6–30× over |
| high-performance [3] | ≤ 10 ms (100 Hz) | none | all by 8–49× |
| π0's own open-loop chunk schedule, 50 Hz robot [1] | ≤ 500 ms | π0 (424), π0.5 (488), SmolVLA, GR00T — with little margin for π0 / π0.5 | as shipped: π0 (1069), π0.5 (3029) |
| π0's own schedule, 20 Hz robot [1] | ≤ 800 ms | all optimised models | as shipped: π0, π0.5 |
| GR00T N1's System-2 rate [2] | ≤ 100 ms (10 Hz) | none of the GR00T versions (169–240 ms) | 1.7–2.4× |

**Edge vs datacenter / desktop:** Jetson's best π0 is **5.8× slower** than π0 on an RTX 4090 (424 vs 73 ms; eager 14.6×).
GR00T N1.x on Jetson needs 169–240 ms against GR00T N1's 63.9 ms on an L40 (different model generation — indicative only).

**Where the time goes, and why this motivates acceleration** (`reports/SYNTHESIS_all_measurements.md` §1.3, §3):
after optimisation, the action expert is 33–49 % of the frame and vision + language prefill the rest. The prefix is
bandwidth-hungry (π0's moves 21.6 GB per frame, 3.7× its analytic bytes); GR00T's compiled action head already streams
its weights at 75–87 % of peak bandwidth (1.05–1.33× its memory floor) — software alone cannot take it further; the
LeRobot experts remain 3–5× above their memory floor in small, memory-latency-bound kernels. Compilation has already
removed the launch overhead that dominated eager execution, so the remaining gap to a 10 Hz budget is a memory-system
problem, not a compute problem.

## 4. Draft paragraph for the paper

> **Latency on the edge.** On a Jetson AGX Orin 64 GB at its maximum power mode, one inference of the six
> flow-matching VLAs — from camera images and instruction to an action chunk — takes 255 ms to 3,029 ms as shipped
> (eager PyTorch), and still 92 ms to 488 ms after the best combination of bf16 casting, TF32, `torch.compile` and
> CUDA graphs we found (1.5–7.9× faster). Only the smallest model (SmolVLA, 0.45 B parameters) reaches the 10 Hz rate
> commonly treated as the minimum for reactive closed-loop control [3]; π0 and π0.5 need 424 ms and 488 ms (2.4 Hz
> and 2.1 Hz), 5.8× slower than π0's 73 ms on a desktop RTX 4090 [1], and the GR00T family needs 169–240 ms against
> GR00T N1's 63.9 ms on an L40 [2]. These models remain usable only by executing long action chunks open-loop — π0
> re-plans every 0.5 s on a 50 Hz robot [1] — which trades reactivity for throughput. The gap is not a matter of raw
> compute: the optimised models spend 33–49 % of the frame in the action expert and the rest in the vision and
> language prefix, which is limited by memory bandwidth rather than arithmetic.

## References

1. K. Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164 —
   https://arxiv.org/html/2410.24164v1 (Table I inference breakdown on RTX 4090; §V control rates and chunk schedule).
2. NVIDIA, "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots," arXiv:2503.14734 —
   https://arxiv.org/html/2503.14734v1 (System 2 at 10 Hz, System 1 at 120 Hz; 63.9 ms for a 16-action chunk on L40).
3. "How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf," arXiv:2602.18397 —
   https://arxiv.org/html/2602.18397v1 (10 Hz acceptable / 100 Hz high-performance; 10–100 ms real-time; Jetson Thor π0 19.0 Hz).
4. "Vision-Language-Action Models: Concepts, Progress, Applications and Challenges," arXiv:2505.04769 —
   https://arxiv.org/html/2505.04769v2 (background only).

Quotes and numbers from [1]–[3] were read from the papers' arXiv HTML pages on 2026-09-28.
