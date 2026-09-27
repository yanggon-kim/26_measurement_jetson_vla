# Note — files and compression
* `raw/rail_samples.csv.zst` is `raw/rail_samples.csv` (zstd -19) compressed after `meta.json` was written;
  `meta.json.raw_files` lists the uncompressed original's hash.
* `reanalysis_E7_raw.csv` (derived, no GPU): the COMMITTED E7 raw (`results/20260924_E7_pi0_N10_bf16_both_MAXN`)
  trimmed to each loop's timed window -> residual 0.68 % (vs 30.42 % untrimmed, reproducing E7's 6395.8 /
  3112.5 / 13 666.2 mJ exactly before trimming).
