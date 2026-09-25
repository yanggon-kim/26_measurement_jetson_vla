# E11d — prefix latency vs its input (measurement plan, part 4)

Run directory: `20260924_E11d_smolvla_reduce-overhead_prefixsweep`  
Written by `bench/prefix_sweep.py` at 2026-09-25T00:33:57Z.

## Answer — prefix latency vs prefix length

| mode | images | prompt | prompt tok | image tok | P processed | P valid | region | median ms | p5 | p95 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reduce-overhead | 1 | short | 48 | 65 | 113 | 71 | e2e | 39.685 | 39.399 | 40.578 | 30 |
| reduce-overhead | 1 | typical | 48 | 65 | 113 | 87 | e2e | 39.818 | 39.417 | 40.529 | 30 |
| reduce-overhead | 1 | long | 48 | 65 | 113 | 105 | e2e | 40.272 | 39.545 | 41.295 | 30 |
| reduce-overhead | 2 | short | 48 | 129 | 177 | 135 | e2e | 58.315 | 57.819 | 58.754 | 30 |
| reduce-overhead | 2 | typical | 48 | 129 | 177 | 151 | e2e | 58.335 | 57.900 | 58.733 | 30 |
| reduce-overhead | 2 | long | 48 | 129 | 177 | 169 | e2e | 58.423 | 57.986 | 58.996 | 30 |
| reduce-overhead | 3 | short | 48 | 193 | 241 | 199 | e2e | 76.017 | 75.903 | 76.727 | 30 |
| reduce-overhead | 3 | typical | 48 | 193 | 241 | 215 | e2e | 76.041 | 75.950 | 77.010 | 30 |
| reduce-overhead | 3 | long | 48 | 193 | 241 | 233 | e2e | 75.887 | 75.791 | 76.966 | 30 |

## Fits

| fit | region | x | a (ms) | b | unit b | r2 | points | note |
|---|---|---|---|---|---|---|---|---|
| ae_step_vs_P | e2e | P_processed | 8.0000 | 0.28280555 | ms_per_token | 0.9998 | 3 |  |

## Conditions

π0, bf16, MAXN + jetson_clocks. **P_processed is constant at 816 for every point**: the processor always encodes three image slots and pads the prompt to 48 tokens, so what the sweep varies is how many of those 816 positions carry a valid mask (P_valid), not how many are computed. That is why prefix latency is flat.

## Reproduce

```
python bench/prefix_sweep.py --outdir-prefix results/20260924_E11d_smolvla
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
