# 20260924_E11a_groot_n16_hostseg

* **What:** E11a (GR00T DiT block) — task T012
* **Conditions:** date 2026-09-25T00:25:33Z, power mode NV Power Mode: MAXN, model groot_n16
* **Command:** `bench/groot_segments.py --outdir results/20260924_E11a_groot_n16_hostseg --version n16 --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d6 --checkpoint nvidia/GR00T-N1.6-3B --embodiment-tag robocasa_panda_omron`

Headline — first rows of `model_inputs.csv` (168 rows):

| model | model_commit | bench_commit | quantity | key | variant | value | unit |
|---|---|---|---|---|---|---|---|
| groot_n16 | nvidia/GR00T-N1.6-3B | 16ab28fe0fb6d14a1fa9e9f… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.19680601 | ms |
| groot_n16 | nvidia/GR00T-N1.6-3B | 16ab28fe0fb6d14a1fa9e9f… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.191556328 | ms |
| groot_n16 | nvidia/GR00T-N1.6-3B | 16ab28fe0fb6d14a1fa9e9f… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.209958991 | ms |
| groot_n16 | nvidia/GR00T-N1.6-3B | 16ab28fe0fb6d14a1fa9e9f… | n_kernels_host_segment | adaln_cond | block=0;kind=cross_attn | 0 | count |
| groot_n16 | nvidia/GR00T-N1.6-3B | 16ab28fe0fb6d14a1fa9e9f… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.382379018 | ms |
| groot_n16 | nvidia/GR00T-N1.6-3B | 16ab28fe0fb6d14a1fa9e9f… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.375211006 | ms |

**Anything odd:** not_measured: ["model_inputs.csv names model='groot_n16', which is NOT in check_model_inputs.py's MODELS set ['groot_n17', 'none', 'pi0', 'pi05', 'smolvla']. The checker will reject this file. The rows are correct;; schema_note: model_inputs.csv names model='groot_n16', which is NOT in check_model_inputs.py's MODELS set ['groot_n17', 'none', 'pi0', 'pi05', 'smolvla']. The checker will reject this file. The rows are correct; t; notes: ["Both kinds of DiT block are measured. DiT.forward (dit.py:310-326) sends encoder_hidden_states only on even indices when interleave_self_attention is set, so odd blocks are pure self-attention and e

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
