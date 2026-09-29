# E12d arm (c) — the eager co-run as two processes with slim weights (no MPS)

* **What:** π0 eager (N = 10, `pi0_nosync`, MAXN) split into a prefix process that dropped `gemma_expert`
  (574.8 M params) and an expert process that dropped `paligemma` (3450.0 M params); KV + pad (15.04 MB/frame) handed over
  through a POSIX shared-memory double buffer (cudaHostRegister'd, rc 0) with per-slot semaphores. Each process has its
  own Python interpreter, so the two stages are issued by two CPU cores. Bracketed, n = 120 per condition, 71 s loop.
* **Memory:** prefix process 7.24 GiB CUDA-allocated / 13.0 GiB RSS, expert process 1.12 / 6.63 GiB; MemAvailable
  51.0 → 32.4 GiB with both ready (T014's two full-model processes were OOM-killed).
* **Equivalence:** the expert process's actions are bitwise identical to a single-process serial run of the full model on
  6 frames; stale control 5/5 detected; 0 non-finite.

| ms (median, p5–p95) | value |
|---|---:|
| A_iso prefix process (incl. D2H 0.50 ms) | 473.54 (473.23–473.90) |
| B_iso expert process (incl. H2D 0.68 ms) | 568.39 (567.30–569.61) |
| T_serial, lockstep across the processes | 1043.97 (1043.16–1045.15) |
| **T_pipe (71 s)** | **686.93** |
| A_co / B_co | 686.94 (686.19–687.55) / 581.60 (581.09–584.15) |
| O / φ / p / s | **1.206 / 0.753** / 1.450 / 1.021 |

* **Reading:** with two issuing threads (processes), the **same eager kernels overlap**: the frame period falls 1044 → 687 ms
  (1.52× over lockstep; 1.45× over T014's one-process pipeline, 994.9 ms). The expert keeps its isolated speed (s 1.02);
  the prefix absorbs the sharing (p 1.45). *Interpretation, not measured (no timeline of the two-process run):* the prefix's kernels wait behind the expert's many small, host-paced kernels. This is
  the causal test T014 lacked: separate host issue → overlap appears.
