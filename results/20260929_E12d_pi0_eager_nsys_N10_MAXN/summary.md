# E12d arm (a) — does T014's eager co-run overlap at all? (nsys, direct measurement)

* **What:** one Nsight Systems capture of T014's eager in-process pipeline (π0, N = 10, `pi0_nosync`, `corun.run_pipeline`,
  MAXN, 1300.5 MHz), NVTX ranges `pipeline` (5 frames), `prefix_alone`, `expert_alone`, `serial`; analysed with
  `bench/e12b_nsys_analyze.py` (E12b M2 step 5, unchanged). Replaces T014's issue/wall argument with the timeline.
* **Answer:** kernels of the two streams execute simultaneously for **0.0 ms of 6364 ms (0.0 %)** of the co-run, while the
  GPU is busy only **47.9 %** of that span (union of both streams). Alone, the prefix keeps the GPU **99.2 %** busy; the
  expert only **15.9 %** (9678 kernels per frame, 135 ms of kernel time per frame against ~524 ms unprofiled stage time).
* **Reading:** the expert's kernels land in the prefix's inter-kernel gaps (378.5 ms/frame of the expert's gaps coincide
  with prefix kernels) — interleaved, never concurrent. Kernel durations barely change (+5.0 ms prefix, +0.3 ms expert per
  frame), so there is no GPU contention; the GPU is idle half the time and still nothing overlaps → the limit is on the
  issue side (one Python thread feeding both streams), which arms (b) and (c) test causally.
* Profiled times are inflated; ratios come from unprofiled runs. `.nsys-rep` / `.sqlite` not committed (protocol §3.5).
