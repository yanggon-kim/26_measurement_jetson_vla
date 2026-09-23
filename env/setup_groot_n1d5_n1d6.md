# GR00T N1.6 and N1.5 on this device (2026-09-23)

Both need a checkout at a commit that still contains their model module, because the N1.7
refactor removed them. Created as **git worktrees** of the existing clone, so no second
copy of the repository is downloaded:

| version | worktree | commit | module |
|---|---|---|---|
| N1.6 | `/home/yanggon/00_VLA/Isaac-GR00T-n1d6` | `9b37aa1` (2026-06-30, newest commit still containing `gr00t/model/gr00t_n1d6/`) | `gr00t_n1d6` |
| N1.5 | `/home/yanggon/00_VLA/Isaac-GR00T-n1d5` | `4af2b62` (2025-11-05, parent of `4e62473` "GR00T N1.6 Release", which deleted the file) | `gr00t/model/gr00t_n1.py`, class `GR00T_N1_5` |

`4af2b62` was confirmed by reading the file out of that commit rather than by assuming:
`GR00T_N1_5_Config.model_type == "gr00t_n1_5"`, which is exactly what the checkpoint's
`config.json` asks for (`model_type: gr00t_n1_5`, `architectures: ['GR00T_N1_5']`).

## Checkpoints

Both are ungated on the hub and small enough to keep: N1.6 6.6 GB, N1.5 5.4 GB
(1.6 TB free on NVMe). Downloaded with `huggingface_hub.snapshot_download`.
The `models--nvidia--GR00T-N1.6-3B` cache directory already existed but held **only
`config.json` (28 KB)** — the weights had never been fetched.

## Packages added to `~/venvs/groot` (uv, --no-deps so the N1.7 environment is unchanged)

    av==17.1.0          # N1.6 and N1.5 import it through gr00t.data
    numpydantic==1.10.0 # N1.5 only

With these, **N1.6 imports cleanly in the existing environment** — no second venv needed.

## N1.5 is BLOCKED on one dependency, and it is not stubbed

    gr00t/data/transform/state_action.py:21: import pytorch3d.transforms as pt

`pytorch3d` has no aarch64 wheel and builds from source against CUDA, which on this device
is a long CPU-saturating build. Running it while measurements are in flight would corrupt
their CPU-side timings (`data_processing_cpu` is a reported phase), so it is deferred to an
idle window rather than run alongside. It is **not** stubbed out: the module provides the
rotation conversions the state/action transforms use, and faking them would silently change
what the input pipeline does.

Order therefore: **N1.6 first** (ready now), N1.5 after, with the build run only when no
measurement is running.

## One local fix to the N1.6 worktree (2026-09-23), disclosed

`gr00t/model/modules/eagle_backbone.py` computes

    if use_flash_attention:
        extra_kwargs["attn_implementation"] = "flash_attention_2"

and then, in the `nvidia/Eagle-Block2A-2B-v2` branch, calls
`AutoModel.from_config(config, trust_remote_code=True)` **without passing
`extra_kwargs`**. The Eagle remote code then asserts

    AssertionError: Qwen3 must use flash_attention_2 but got None

and N1.6 cannot be loaded at all — every rung failed in 8–9 s. The worktree copy now
forwards `**extra_kwargs` to `from_config`. This is a load-time bug in the vendored
checkout, not a change to what is measured: the branch's own asserts two lines above
require flash attention and bf16, so forwarding the argument gives the configuration that
code already insists on.
