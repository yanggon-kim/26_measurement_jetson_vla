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

## A second environment, `~/venvs/groot16` (2026-09-23) — this replaces the shim attempt

The three versions pin different transformers, and it is a hard conflict:

    Isaac-GR00T @ 376ba89 (N1.7)  transformers==4.57.3   # Qwen3-VL needs >= 4.57
    Isaac-GR00T @ 9b37aa1 (N1.6)  transformers==4.51.3
    Isaac-GR00T @ 4af2b62 (N1.5)  transformers==4.51.3

Under 4.57.6 the Eagle remote code failed on symbols that MOVED or were REMOVED:

    ImportError: cannot import name 'VideoInput' from 'transformers.image_utils'
    ImportError: cannot import name 'BASE_IMAGE_PROCESSOR_FAST_DOCSTRING'
                 from 'transformers.image_processing_utils_fast'

The first was re-exportable (it moved to `transformers.video_utils`), so I shimmed it — and
then hit the second, which is a removed constant. Shimming a chain of deleted internals is
how a measurement ends up running something other than what it claims, so the shim was
abandoned in favour of the environment the checkout actually asks for.

`~/venvs/groot16` is a **copy** of `~/venvs/groot` with `transformers==4.51.3` installed
over it. Copied rather than created fresh because torch here is the Jetson CUDA build; a
new venv would resolve a PyPI torch with no aarch64 CUDA wheel. Verified:
`transformers 4.51.3 | torch 2.10.0 | cuda True`, and both missing symbols present
natively, no shim. It serves **both** N1.6 and N1.5.

`--compat-shims` stays in `bench/groot_phases.py` but is NOT used for these runs; when it
is used it records itself in `meta.json` under `compat_shims`.

## One more version difference, handled in the harness

N1.7's `Gr00tPolicy` resolves a string embodiment tag to the enum; N1.6's does not and
dies with `'str' object has no attribute 'value'`. `groot_phases.py` now resolves the tag
before constructing the policy, so both versions take the same argument.

**N1.6 verified end to end** (eager, robocasa_panda_omron, 3 warmup / 3 timed):
e2e 385.97 ms, vision 62.41, vlm_prefix 72.13, action head 242.93 (62.9 %), CPU 50.86.

## N1.5 unblocked (2026-09-23 10:20)

`pytorch3d` built from source in **627 s** once `activate_orin_venv.sh` was sourced first
(its `setup.py` imports torch, which needs the Jetson `LD_LIBRARY_PATH`). Built at
`TORCH_CUDA_ARCH_LIST=8.7`, `FORCE_CUDA=1`, commit `75ebeeae`.

Further packages into `~/venvs/groot16` (all `--no-deps`):

    setuptools==84.0.0   # inductor needs it to compile; NEITHER groot venv had it, and
                         # its absence is what failed N1.6's vis_compile rung
    timm==1.0.30         # Eagle's RADIOConfig imports it
    decord               # STUB, see below

### The `decord` stub, and why it raises

transformers' `check_imports()` statically scans the remote modeling file and refuses to
load the Eagle processor unless `decord` is importable. decord has **no aarch64 wheel**
(nor does the `eva-decord` fork). In
`gr00t/model/backbone/eagle2_hg_model/processing_eagle2_5_vl.py` decord appears in exactly
one place — `_read_video_decord(video_path)`, which opens a video FILE, with its
`import decord` inside the function body (line 191). This benchmark passes numpy image
arrays built from the checkpoint's own modality config and never a video path, so that
function is unreachable here.

"Unreachable" is a claim, so the stub **raises on every attribute access**. If any code
path does reach decord the run fails loudly and this note is proven wrong, rather than a
number being quietly produced by a different video backend.

### One more silent-failure class, fixed

`_vision_module` looked under `backbone.model`; N1.5's EagleBackbone calls the container
`backbone.eagle_model` (eagle_backbone.py:59). The hook simply never fired and N1.5's first
run reported **vision 0.00 ms** with no error at all. The resolver now searches both
containers, and a vision module that cannot be found is a hard failure rather than a zero.

**N1.5 verified** (eager, oxe_droid, 3/3): e2e 259.99 ms — vision 50.45, prefix 100.22,
action head 99.23 (38.2 %), CPU 48.79.
