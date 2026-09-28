# 20260924_E11a_groot_n15_hostseg

* **What:** E11a (GR00T DiT block) — task T012
* **Conditions:** date 2026-09-25T01:55:59Z, power mode NV Power Mode: MAXN, model groot_n15
* **Command:** `bench/groot_segments.py --outdir results/20260924_E11a_groot_n15_hostseg --version n15 --gr00t-root /home/yanggon/00_VLA/Isaac-GR00T-n1d5 --checkpoint nvidia/GR00T-N1.5-3B --api old --embodiment-tag oxe_droid --data-conf`

Headline — first rows of `model_inputs.csv` (162 rows):

| model | model_commit | bench_commit | quantity | key | variant | value | unit |
|---|---|---|---|---|---|---|---|
| groot_n15 | nvidia/GR00T-N1.5-3B | 8d6eb167c7249d195d0957e… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.238709035 | ms |
| groot_n15 | nvidia/GR00T-N1.5-3B | 8d6eb167c7249d195d0957e… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.235091452 | ms |
| groot_n15 | nvidia/GR00T-N1.5-3B | 8d6eb167c7249d195d0957e… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.256696853 | ms |
| groot_n15 | nvidia/GR00T-N1.5-3B | 8d6eb167c7249d195d0957e… | n_kernels_host_segment | adaln_cond | block=0;kind=cross_attn | 0 | count |
| groot_n15 | nvidia/GR00T-N1.5-3B | 8d6eb167c7249d195d0957e… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.390510482 | ms |
| groot_n15 | nvidia/GR00T-N1.5-3B | 8d6eb167c7249d195d0957e… | t_host_segment | adaln_cond | block=0;kind=cross_attn | 0.382926909 | ms |

**Anything odd:** not_measured: ['adalayernorm1 block0 compiled: RuntimeError: Inference tensors cannot be saved for backward. Please do not use Tensors created in inference mode in computation tracked by autograd. To work around th; schema_note: model_inputs.csv names model='groot_n15', which is NOT in check_model_inputs.py's MODELS set ['groot_n17', 'none', 'pi0', 'pi05', 'smolvla']. The checker will reject this file. The rows are correct; t; notes: ["Segments were reconstructed under torch.autocast(cuda, torch.bfloat16), because this model's own gr00t/model/policy.py:189-191 wraps inference in it (--api=old). The reported bytes are the traffic o

*Generated 2026-09-28 by `bench/make_summaries.py` from this directory's own meta.json and CSV (protocol §5 completeness audit); no number was recomputed. The report that analyses this run is the authority.*
