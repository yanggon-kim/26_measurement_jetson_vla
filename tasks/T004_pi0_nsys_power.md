# T004 — π0 kernel timeline and power (E5, E7)  · priority P1 · depends on: T003

1. E5: Nsight Systems on eager and best-compiled π0 at N = 4 and 10; export the CSVs of protocol §3; compute
   kernels per step, kernel-time vs region-time, GPU idle fraction, gap histogram, top-20 kernels.
2. E7: power per rail for idle / prefix-only / action-expert-only / full loops; energy per inference and per
   denoising step; repeat the full loop at MAXN, 50 W, 30 W, 15 W.
Acceptance: `reports/E5_pi0_timeline.md` answering "kernels executing vs GPU waiting", and `reports/E7_pi0_power.md`
with W per rail per phase and mJ per step.
