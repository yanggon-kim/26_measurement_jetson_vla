# Note for every device agent: cuBLAS picks slow GEMM kernels at some token counts, and zero-padding the rows fixes it

**From:** the Thor agent (branch `thor`), T005, 2026-10-22. **For:** any agent measuring these VLA models on another
device (Jetson AGX Orin, …), and the research side. **Status:** measured on Thor for π0. π0.5, SmolVLA and GR00T
are being checked in Thor's T005 search; this note will be updated with their results.

## 1. The problem in one paragraph
cuBLAS / cuBLASLt chooses its GEMM kernel from the problem shape with a built-in heuristic. Its choice does not
change smoothly with the number of rows M (the token count). It jumps between kernel tiles. Some M values land on a
kernel 3–4× slower than at nearby M.

On Thor, π0's VLM prefix has M = 816 tokens. Its MLP gate/up projection `[816×2048]·[2048×16384]` (bf16, 36 calls
per inference) runs at **31 TFLOP/s instead of ≈ 110**. That makes π0's prefix slower than π0.5's (968 tokens), on
the same Gemma-2B. Zero-padding the GEMM's rows to a better M and dropping the extra output rows fixes it with
**bit-identical** results. It cut π0's optimised frame from 211.7 to 170.9 ms.

## 2. How it was found
1. **Symptom.** In T005's ladder, π0's prefix took 125.5 ms and π0.5's 96.3 ms (same rung `eager_bf16`). π0 has
   fewer tokens (816 vs 968).
2. **Ruled out by reading the code:**
   - a different code path: `embed_prefix` and `sample_actions` are identical;
   - a different dtype: both run bf16 weights, and `PiGemmaModel.forward` casts the input to bf16;
   - attention: both use eager attention.
3. **Replayed only the prefill call** (`bench/prefix_diag.py`), with CUDA events and `torch.profiler`
   (`record_shapes=True`): π0 123.9 ms, π0.5 98.0 ms. The difference sits in one op: `aten::mm [M×2048]·[2048×16384]`.
   π0 got kernel `nvjet_sm110_tst_256x96_64x5_2x1_2cta_v_bz_TNT`; π0.5 got `…112x128…_h_bz_TNN`.
4. **Isolated the GEMM** and swept M (`bench/gemm_m_sweep.py`; CUDA events, median of 50, kernel name per M).

## 3. The evidence (Thor, `sm_110`, torch 2.10.0+cu130, CUDA 13.0)
Gate/up `[M×2048]·[2048×16384]`, bf16; excerpt of the M = 512–1024 sweep, step 16 (every M in `results/20261022_E4_prefix_pi0_vs_pi05_diag/raw/gemm_m_sweep_gateup.csv`):

| M | ms | TFLOP/s | kernel tile |
|---|---|---|---|
| 576 | 0.445 | 87 | 128x192 |
| 592–640 | 1.18–1.21 | 34–36 | 256x128 |
| 688–704 | **3.99–4.00** | **12** | 512x64 |
| 720–768 | 0.46–0.48 | 105–108 | 128x192 |
| 784–800 | 0.80 | 66 | 128x160 |
| **816–864 (π0 = 816)** | **1.73–1.75** | **31–33** | **256x96** |
| 880–896 | 1.02–1.03 | 58 | 128x128 |
| 912–960 | 0.82–0.83 | 75–77 | 128x192 |
| 976–1024 (π0.5 = 968 is also fast) | 0.60 | 110–114 | 112x128 (`h`, TNN) |

Remedies tried on M = 816 (`results/20261022_E4_gemm_m816_fix/`):

| remedy | ms | outputs vs default |
|---|---|---|
| default | 1.703 | — |
| `torch.backends.cuda.preferred_blas_library("cublas" / "cublaslt")` | 1.71 / 1.72 | same kernel |
| `CUBLASLT_WORKSPACE_SIZE` 1 MiB → 32 MiB | 1.71 / 1.73 | same kernel |
| PyTorch TunableOp (`torch.cuda.tunable`) | 1.744 | on CUDA it had only the `Default` candidate: no help |
| Inductor `max-autotune`, SM gate forced (Triton template won) | 1.139 | identical |
| split M into 768 + 48 (or 720 + 96), concatenate | 1.17–1.24 | identical |
| **zero-pad M to 976 / 1024, keep the first 816 rows** (copy included) | **0.605 / 0.589** | **bit-identical** |

## 4. The fix: `linear_mpad` (measured, not hard-coded)
**Why it is exact.** In `y = x·Wᵀ`, row i of y depends only on row i of x. Extra zero rows change nothing in the
real rows: the M = 816 test, and π0's whole-model actions, came out bit-identical. Only the kernel changes.

**Procedure.** Implemented as `prep_linear_mpad` in `bench/rungs.py` on the `thor` branch:
1. Run one eager inference with a forward pre-hook on every `nn.Linear` in scope (the language model and the action
   expert) and record each call's (M, K, N, dtype). M = all leading dimensions flattened.
2. For each distinct shape, time `F.linear` at M′ = M, M+16, …, M+256 on random operands (CUDA events, median of 20).
   Keep the fastest M′ only if it beats M by more than 5 %.
3. Replace those Linears' `forward` with pad → `F.linear` → slice. These are static tensor ops, so it works eager,
   under `torch.compile` and inside CUDA graphs. Install it **before** compiling.

Standalone version (no harness needed):

```python
import statistics, torch, torch.nn.functional as F

def _t(m, k, n, dt, reps=20):
    x = torch.randn(m, k, device="cuda", dtype=dt); w = torch.randn(n, k, device="cuda", dtype=dt)
    for _ in range(3): F.linear(x, w)
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); F.linear(x, w); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def linear_mpad(roots, run_once, span=256, step=16, min_gain=0.05):
    """roots: modules whose nn.Linear children to consider; run_once(): one real inference (eager)."""
    lins = {id(m): m for r in roots for m in r.modules() if isinstance(m, torch.nn.Linear)}
    seen, hooks = {}, []
    for m in lins.values():
        hooks.append(m.register_forward_pre_hook(lambda mod, a: seen.setdefault(id(mod), set()).add(
            (a[0].numel() // a[0].shape[-1], mod.in_features, mod.out_features, a[0].dtype))))
    run_once()
    for h in hooks: h.remove()
    pad = {}
    for shapes in seen.values():
        for (m, k, n, dt) in shapes:
            if (m, k, n, dt) in pad or m < 16: continue
            base = _t(m, k, n, dt)
            best_t, best_m = min((_t(m2, k, n, dt), m2) for m2 in range(m + step, m + span + 1, step))
            pad[(m, k, n, dt)] = best_m if best_t < (1 - min_gain) * base else m
    for mid, lin in lins.items():
        table = {(m, dt): pad[(m, k, n, dt)] for (m, k, n, dt) in seen.get(mid, ()) if pad.get((m, k, n, dt), m) != m}
        if not table: continue
        def fwd(x, _l=lin, _t=table):
            rows = x.numel() // x.shape[-1]; m2 = _t.get((rows, x.dtype))
            if m2 is None: return F.linear(x, _l.weight, _l.bias)
            y = F.linear(F.pad(x.reshape(rows, x.shape[-1]), (0, 0, 0, m2 - rows)), _l.weight, _l.bias)
            return y[:rows].reshape(*x.shape[:-1], _l.out_features)
        lin.forward = fwd
    return {f"M={k[0]} K={k[1]} N={k[2]}": v for k, v in pad.items() if v != k[0]}
```

## 5. Result on Thor (π0, N = 10, the same harness as every T005 rung)
- **What the tuner padded** (`results/20261022_E4_gemm_m816_fix/raw/prefix_pi0.json`):
  - gate/up (K 2048, N 16384): M 816 → 1024, 1.695 → 0.581 ms;
  - k/v projection (N 256): M 816 → 880.
  - No expert shape (M = 51) gained more than 5 %.
- **Prefix alone:** 123.9 → 82.7 ms.

| configuration | e2e ms | vision | prefix | expert | actions vs the same rung without padding |
|---|---|---|---|---|---|
| `eager_bf16` → `eager_bf16_mpad` | 293.7 → 245.5 | 28.8 → 28.6 | 125.5 → 85.3 | 136.6 → 128.8 | bit-identical (10 seeded inferences) |
| `bf16_both_graphs` → `bf16_both_graphs_mpad` | 211.7 → **170.9** | 21.1 → 21.4 | 108.9 → **64.1** | 79.1 → 82.5 | bit-identical |

Sources: `results/20261022_E4c_pi0_opt_ladder/`, `results/20261022_E4i_pi0_best_search/`.

## 6. What another device should do
The bad M bands depend on the GPU, the cuBLAS version and the shape, so **nothing above transfers as a number**.
Measure them on your device:
1. Sweep each heavy GEMM shape of the model over M around its real token count. Use `bench/gemm_m_sweep.py`
   (`--k K --n N --m start:stop:step`), or the `_t()` helper above. On Thor π0's prefix shapes are
   (K, N) = (2048, 16384) gate/up, (16384, 2048) down, (2048, 2048) q/o, (2048, 256) k/v, and its M is the processed
   prefix token count.
2. If any M shows a step drop, apply `linear_mpad` and check that the actions are bit-identical (or within your
   bf16 noise) on seeded inferences.
3. **Hint for the Orin, from its own data.** `reference/orin/reports/E4h_vlm_prefix_kernels.md`, π0 `best`
   (lines 34–39):
   - the gate/up GEMMs ran fine: grid (128, 7, 1), ≈ 2.0 ms, SM 85 %;
   - two GEMMs with grid (16, 7, 9) took **5.55 ms each at SM 25.9 %**, stalled on `mio_throttle`. That is a split-K
     (z = 9) kernel on an N = 2048 output, i.e. the MLP **down projection** (K = 16384). Together they were 47 % of
     the captured window.
   - This is the same class of problem in another shape. It is worth sweeping M there, and padding or forcing
     another kernel.

## 7. Open items
- **π0's down projection** (K 16384 → N 2048) runs at only 37–48 TFLOP/s at every M tested on Thor. Padding M does
  not help it. It has few output tiles for 20 SMs. T006–T008 analyse it.
- π0.5, SmolVLA and GR00T: `linear_mpad` is part of their T005 search on Thor; results will be added here.
- Changing M itself (trimming π0's 283 padding tokens, 816 → 533) is a model-input change. It belongs to T014 (E14b),
  not to this fix.
