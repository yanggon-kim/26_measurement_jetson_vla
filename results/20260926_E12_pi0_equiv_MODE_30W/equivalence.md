# E12 fresh-KV equivalence (`docs/E12_corun/README.md` §6)

**VERDICT: PASS** — 0 failing row(s); the negative control changed the actions in 5/5 rows.

Non-negotiable 1: each frame's expert must read that frame's own KV cache. A pipeline that shares, reuses or overwrites a cache across frames is a different mechanism (stale KV) and would invalidate every timing.

`stale_control` is the §6.3 negative control: expert(k−1) is handed frame k−2's cache. **PASS for that row means it CHANGED the actions, as it must** — if a deliberately corrupted cache were indistinguishable from a correct one, the check would have no power.

* Consecutive serial frames differ by max|d|=5.513814e+00 in the action chunk. §6.1 asks for this: it is the scale at which a stale-KV mix-up would show, and a near-zero value would mean the observations are not materially different and the check is weak however it is written.

| mechanism | path | frames | max abs diff | bitwise identical | verdict |
|---|---|---:|---:|---|---|
| `pipeline` | eager | 6 | 0.000e+00 | True | PASS |
| `stale_control` | eager | 5 | 1.956e-01 | False | PASS |
