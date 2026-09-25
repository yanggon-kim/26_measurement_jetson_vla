# E11d — prefix latency vs its input (measurement plan, part 4)

Run directory: `20260924_E11d_pi05_reduce-overhead_prefixsweep`  
Written by `bench/prefix_sweep.py` at 2026-09-24T21:46:05Z.

## Answer — prefix latency vs prefix length

| mode | images | prompt | prompt tok | image tok | P processed | P valid | region | median ms | p5 | p95 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reduce-overhead | 1 | short | 200 | 768 | 968 | 303 | e2e | 715.603 | 714.371 | 716.765 | 30 |
| reduce-overhead | 1 | typical | 200 | 768 | 968 | 318 | e2e | 716.479 | 714.131 | 720.007 | 30 |
| reduce-overhead | 1 | long | 200 | 768 | 968 | 336 | e2e | 715.462 | 713.746 | 716.535 | 30 |
| reduce-overhead | 2 | short | 200 | 768 | 968 | 559 | e2e | 715.664 | 714.082 | 717.399 | 30 |
| reduce-overhead | 2 | typical | 200 | 768 | 968 | 574 | e2e | 715.492 | 712.924 | 716.808 | 30 |
| reduce-overhead | 2 | long | 200 | 768 | 968 | 592 | e2e | 714.856 | 713.043 | 717.373 | 30 |
| reduce-overhead | 3 | short | 200 | 768 | 968 | 815 | e2e | 713.965 | 712.393 | 715.389 | 30 |
| reduce-overhead | 3 | typical | 200 | 768 | 968 | 830 | e2e | 714.443 | 712.695 | 716.008 | 30 |
| reduce-overhead | 3 | long | 200 | 768 | 968 | 848 | e2e | 712.976 | 712.272 | 715.584 | 30 |

## Fits

| fit | region | x | a (ms) | b | unit b | r2 | points | note |
|---|---|---|---|---|---|---|---|---|
| ae_step_vs_P | e2e | P_processed |  |  |  |  | 1 | x is constant at 968 across the sweep; no slope can be fitted, which is itself the result |

## Conditions

π0, bf16, MAXN + jetson_clocks. **P_processed is constant at 816 for every point**: the processor always encodes three image slots and pads the prompt to 48 tokens, so what the sweep varies is how many of those 816 positions carry a valid mask (P_valid), not how many are computed. That is why prefix latency is flat.

## Reproduce

```
python bench/prefix_sweep.py --outdir-prefix results/20260924_E11d_pi05
```

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
