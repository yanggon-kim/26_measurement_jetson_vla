---
name: measurement-gotchas
description: Orin/π0 measurement traps that change how returned numbers must be read (DRAM bytes, LIBERO noise, event timing)
metadata:
  type: project
---

- **Orin has no `dram__` ncu counters.** DRAM bytes = L2 sector fills (reads) + L2 write sectors (upper bound).
  The calibrated EMC GB/s agree with that within 4 % (E12b DRAM follow-up, 2026-09-25).
  **Why:** E6's analytic bytes (params + KV) undercount π0's prefix 3.7× (816-token prefill activations spill
  the 4 MB L2); the expert is within ~1.1× on reads. **How to apply:** never size a bandwidth argument
  (e.g. EV-02 channel split) from E6 analytic bytes for a prefill stage; use measured bytes.
- **LIBERO π0 outcomes are not seed-determined:** `sample_noise` uses torch's unseeded global RNG; the same arm
  twice gave 63 vs 56 / 100 (65 % agreement). **How to apply:** arm gaps of a few episodes are noise; ask for
  torch-seeded episodes before reading small differences.
- **Per-call CUDA-event + sync timing adds 45–75 µs on Orin** (`ae47c79`): small-GEMM sweep times must come
  from ncu or an amortised loop, not per-call events.
