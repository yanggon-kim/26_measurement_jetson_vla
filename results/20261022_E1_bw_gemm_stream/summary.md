# E1 addendum: DRAM bandwidth of back-to-back skinny GEMMs (T003, follow-up to E2) — 2026-10-22, MAXN, 0 OC events

`F.linear(x[S, K], W_i)` over 1 GiB of distinct random bf16 weights per pass (no reuse possible), 20 passes per
row, 2 processes. **Weight bytes ÷ time** is the DRAM-relevant rate: x and the output stay in L2 back to back.

| W shape | S = 1 | S = 51 | S = 256 |
|---|---|---|---|
| 4096×1024 (gate / up) | 47.6 µs/call → 176 GB/s | 48.8 → 172 | 39.3 → **214** |
| 2048×1024 (q) | 20.5 → 205 | 20.7 → 203 | 19.7 → **213** |
| 1024×4096 (down) | 40.8 → 206 | 38.9 → 215 | 40.1 → **209** |

(Weight-only GB/s computed from p1's µs per call; p2 agrees within 3.2 %. The CSVs' `GBps` columns count activations
too, up to 293 GB/s, which is not DRAM traffic and exceeds the 273 GB/s LPDDR5X spec.) The 258.23 GB/s E1 ceiling
stands; these skinny GEMMs reach 67–85 % of it (p2's best row: 219 GB/s).
