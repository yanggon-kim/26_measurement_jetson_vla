# E11d — prefix latency vs its input (measurement plan, part 4)

Run directory: `20260922b_E11d_pi0_reduce-overhead_prefixsweep`  
Written by `bench/prefix_sweep.py` at 2026-09-22T13:59:14Z.

## Answer — prefix latency vs prefix length

| mode | images | prompt | prompt tok | image tok | P processed | P valid | region | median ms | p5 | p95 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reduce-overhead | 1 | short | 48 | 768 | 816 | 262 | e2e | 464.661 | 464.308 | 465.124 | 100 |
| reduce-overhead | 1 | typical | 48 | 768 | 816 | 277 | e2e | 465.010 | 464.665 | 465.575 | 100 |
| reduce-overhead | 1 | long | 48 | 768 | 816 | 295 | e2e | 464.898 | 463.763 | 465.539 | 100 |
| reduce-overhead | 2 | short | 48 | 768 | 816 | 518 | e2e | 463.449 | 462.870 | 465.102 | 100 |
| reduce-overhead | 2 | typical | 48 | 768 | 816 | 533 | e2e | 463.709 | 462.987 | 464.466 | 100 |
| reduce-overhead | 2 | long | 48 | 768 | 816 | 551 | e2e | 462.782 | 462.268 | 463.563 | 100 |
| reduce-overhead | 3 | short | 48 | 768 | 816 | 774 | e2e | 463.112 | 462.523 | 464.435 | 100 |
| reduce-overhead | 3 | typical | 48 | 768 | 816 | 789 | e2e | 463.265 | 462.559 | 464.148 | 100 |
| reduce-overhead | 3 | long | 48 | 768 | 816 | 807 | e2e | 463.354 | 462.781 | 464.358 | 100 |

## Fits

| fit | region | x | a (ms) | b | unit b | r2 | points | note |
|---|---|---|---|---|---|---|---|---|
| ae_step_vs_P | e2e | P_processed |  |  |  |  | 1 | x is constant at 816 across the sweep; no slope can be fitted, which is itself the result |

## Conditions

π0, bf16, MAXN + jetson_clocks. **P_processed is constant at 816 for every point**: the processor always encodes three image slots and pads the prompt to 48 tokens, so what the sweep varies is how many of those 816 positions carry a valid mask (P_valid), not how many are computed. That is why prefix latency is flat.

## Reproduce

```
python bench/prefix_sweep.py --outdir-prefix results/20260922b_E11d_pi0
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
