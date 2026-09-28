# 20260924_E11a_groot_n17_hostseg

* **What:** E11a (GR00T DiT block) — task T012
* **Conditions:** date 2026-09-25T00:24:41Z, power mode NV Power Mode: MAXN, model groot_n17
* **Command:** `bench/groot_segments.py --outdir results/20260924_E11a_groot_n17_hostseg --version n17 --checkpoint nvidia/GR00T-N1.7-3B`

Headline — first rows of `model_inputs.csv` (168 rows):

| model | model_commit | bench_commit | quantity | key | variant | value | unit |
|---|---|---|---|---|---|---|---|
| groot_n17 | nvidia/GR00T-N1.7-3B | ec935ef61b92a823cbeb032… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.198986032 | ms |
| groot_n17 | nvidia/GR00T-N1.7-3B | ec935ef61b92a823cbeb032… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.192200998 | ms |
| groot_n17 | nvidia/GR00T-N1.7-3B | ec935ef61b92a823cbeb032… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.212505928 | ms |
| groot_n17 | nvidia/GR00T-N1.7-3B | ec935ef61b92a823cbeb032… | n_kernels_host_segment | adaln_cond | block=0;kind=cross_attn | 0 | count |
| groot_n17 | nvidia/GR00T-N1.7-3B | ec935ef61b92a823cbeb032… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.391969486 | ms |
| groot_n17 | nvidia/GR00T-N1.7-3B | ec935ef61b92a823cbeb032… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.376847427 | ms |

**Anything odd:** notes: ["Both kinds of DiT block are measured. DiT.forward (dit.py:310-326) sends encoder_hidden_states only on even indices when interleave_self_attention is set, so odd blocks are pure self-attention and e

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
