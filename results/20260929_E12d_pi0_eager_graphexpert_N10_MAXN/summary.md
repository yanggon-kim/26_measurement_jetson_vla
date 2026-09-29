# E12d arm (b) — the eager expert as ONE CUDA graph, eager prefix, in-process co-run

* **What:** T014's eager configuration (π0, N = 10, `pi0_nosync`, MAXN) with only the expert's launch mechanism changed:
  the unmodified eager `denoise_step` loop captured once per KV-buffer parity (E12b's `owngraph_expert` + `dbuf_kv`
  harness), the prefix issued eagerly on its own stream; bracketed, n = 120 per condition, 71 s pipeline loop.
* **Equivalence:** pipelined = serial bitwise on every frame; stale control 5/5 detected; the graph expert is **bitwise
  identical** to the plain (non-graph) eager expert; 0 non-finite.

| ms (median, p5–p95) | value |
|---|---:|
| A_iso eager prefix | 471.06 (470.64–471.39) |
| B_iso **graph** expert | **123.53** (123.47–123.87) |
| B_iso plain eager expert, same session | **540.24** |
| T_serial | 599.46 (599.03–599.87) |
| T_pipe (prefix-first, 71 s) | **593.21** |
| O / φ / p / s | 1.259 / 0.049 / 1.034 / 2.302 |

* **Reading:** removing the expert's host issue makes it **4.37× faster with bitwise-identical actions** (the host was 77 %
  of the eager expert), and the whole frame 995 → 599 ms. But the co-run still barely overlaps (φ 0.05): the eager prefix
  keeps the GPU ~99 % busy (arm a), so the graph expert's 124 ms of GPU work can only slot into it — it stretches to 284 ms
  (s 2.30) and T_pipe ≈ A + B. With the expert graphed, the co-run is **GPU-bound**, not host-bound. expert-first order:
  T_pipe 579.75, O 1.231, φ 0.154.
