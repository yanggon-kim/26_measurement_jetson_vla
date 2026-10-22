# Install log — GR00T N1.7 / N1.6 / N1.5 and LIBERO environments (T001 step 7, 2026-10-22)

All installs are user-level (no sudo exists; none attempted). Venvs created with `uv` 0.13.5 from `/usr/bin/python3.12`
(3.12.3). Full build / install logs: `scratch/t001/setup_{groot17,groot16,groot15,libero}.log`,
`scratch/t001/build_mujoco_2.3.7.log`, load checks `scratch/t001/load_*.log`, `scratch/t001/attn_*.log`
(scratch/ is git-ignored). Re-run any env with `bash env/setup_<name>.sh`; frozen pins in `env/requirements_<name>.txt`.

Index shorthand: **PT** = `https://download.pytorch.org/whl/cu130`; **JAI** = `https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple`;
**PyPI** = `https://pypi.org/simple`. Every step after torch runs with a constraints file holding the exact torch build
and is followed by a `torch.cuda.is_available()` / `torch.version.cuda == 13.*` check (all passed).

| # | env | command (abridged; exact form in the setup script) | version | why | outcome |
|---|---|---|---|---|---|
| 1 | groot17 | `uv venv --python /usr/bin/python3.12 ~/venvs/groot17` | Py 3.12.3 | no ensurepip / python3-venv on this image | ok |
| 2 | groot17 | `uv pip install --index-url PT torch==2.10.0 torchvision==0.25.0` | 2.10.0+cu130 / 0.25.0+cu130, triton 3.6.0 | Thor pyproject of Isaac-GR00T@376ba89 pins torch 2.10.0; A8 route (a) build, same as ~/venvs/lerobot | ok, CUDA 13.0, 20 SMs |
| 3 | groot17 | `uv pip install -c constraints ... <Thor pyproject deps>` (PyPI + PT, unsafe-best-match) | transformers 4.57.6, numpy 1.26.4, peft 0.17.1, ... | `scripts/deployment/thor/pyproject.toml` pins, minus tensorrt/onnx/torchcodec/cudss (not on the inference path) | ok |
| 4 | groot17 | `uv pip install --no-deps --index-url JAI flash-attn==2.8.4 diffusers==0.36.0.dev0` | flash-attn 2.8.4, diffusers 0.36.0.dev0 | Thor pins; no build needed (prebuilt cp312 aarch64 cu130 wheel). Checked vs SDPA in bf16: max abs diff 0 | ok; flash_fwd_kernel runs on sm_110 |
| 5 | groot17 | `uv pip install --no-deps -e ~/src/Isaac-GR00T` + `zz_groot_repo.pth` | gr00t 0.1.0 @ 376ba89 | the model code; .pth as on the Orin | ok, `Gr00tPolicy` imports |
| 6 | groot17 | `hf download Qwen/Qwen3-VL-2B-Instruct <config/tokenizer/processor files only>` (rev 89644892) | no weights | `nvidia/Cosmos-Reason2-2B` (N1.7's backbone config + processor) is gated (HTTP 401, "requires approval"); public base model used ONLY for an env check (scratch/t001/groot_checks/standin_n17.py) | env check ok; pinned load BLOCKED (needs the human) |
| 7 | groot16 | same as 1–2 into `~/venvs/groot16` | torch 2.10.0+cu130 | n1d6 Thor pyproject pins torch 2.10.0 | ok |
| 8 | groot16 | `uv pip install -c constraints ... <n1d6 Thor pyproject deps>` | transformers 4.51.3, av 16.1.0, gitpython 3.1.46 | n1d6 `scripts/deployment/thor/pyproject.toml`; Eagle remote code needs 4.51.x (Orin finding) | ok |
| 9 | groot16 | `uv pip install --no-deps --index-url JAI flash-attn==2.8.4 diffusers==0.36.0.dev0` | 2.8.4 / 0.36.0.dev0 | Thor pins; Eagle-Block2A asserts flash_attention_2 | ok |
| 10 | groot16 | `uv pip install --no-deps -e ~/src/Isaac-GR00T-n1d6` + .pth | @ 9b37aa1 | model code. The Orin's eagle_backbone.py `**extra_kwargs` fix was tested and is NOT needed: checkout left pristine | ok, full load + get_action |
| 11 | groot15 | same as 1–2 into `~/venvs/groot15` | torch 2.10.0+cu130 | N1.5 pins torch 2.5.1 (base) / 2.8.0 (orin extra); neither exists for CUDA 13; `thor` extra pins none | ok |
| 12 | groot15 | `uv pip install -c constraints ... <n1d5 pyproject deps>` | transformers 4.51.3, timm 1.0.14, numpydantic 1.6.7, av 12.3.0, gymnasium 1.0.0, peft 0.17.0, pydantic 2.10.6, accelerate 1.2.1, kornia 0.7.4 | n1d5 `[project.dependencies]` minus ray / tianshou / onnx / pytest; own venv because its pins differ from N1.6's | ok |
| 13 | groot15 | `uv pip install --no-deps --index-url JAI flash-attn==2.8.4 diffusers==0.36.0.dev0` | 2.8.4 / 0.36.0.dev0 | radio_model.py imports flash_attn at module level; diffusers = `thor` extra pin | ok |
| 14 | groot15 | `git clone pytorch3d; checkout 75ebeeae` + `~/opt/pytorch3d_py/pytorch3d -> ~/src/pytorch3d/pytorch3d` + `zz_pytorch3d_py.pth` | pytorch3d 0.7.8 source, **no `_C` extension** | N1.5 imports only `pytorch3d.transforms`, which (with `pytorch3d.common`) is pure Python; avoids the Orin's 627 s CUDA build. Same commit as the Orin. `_C` use would raise | ok (`matrix_to_quaternion(I) == [1,0,0,0]`) |
| 15 | groot15 | write `site-packages/decord/__init__.py` (stub, raises on any attribute) | stub | decord has no aarch64 wheel; transformers `check_imports()` refuses to load the Eagle processor without it — verified: removing the stub gives `ImportError: ... requires ... decord`. Orin precedent | ok |
| 16 | groot15 | `uv pip install --no-deps -e ~/src/Isaac-GR00T-n1d5` + .pth | @ 4af2b62 | model code | ok, full load + get_action |
| 17 | libero | `uv venv` + `uv pip install --index-url PT torch==2.10.0` + numpy 1.26.4 | torch 2.10.0+cu130 | LIBERO's benchmark module imports torch (init states); own torch instead of the Orin's .pth bridge (lerobot venv is numpy 2.x) | ok |
| 18 | libero | `curl` mujoco-2.3.7-linux-aarch64.tar.gz → `~/opt/mujoco-2.3.7` | libmujoco 2.3.7, sha256 422d52e9…8c71 | prebuilt C library for the bindings build | ok |
| 19 | libero | `apt-get download libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxrender-dev libxfixes-dev libxext-dev` + `dpkg -x` → `~/opt/x11dev` | Ubuntu noble arm64 | the sdist build compiles GLFW (`simulate`), which failed with "RandR headers not found"; headers only, build-time only | ok |
| 20 | libero | `nice -n 19 taskset -c 10-13 uv build --wheel mujoco-2.3.7.tar.gz` (MUJOCO_PATH, MUJOCO_PLUGIN_PATH, CMAKE_INCLUDE_PATH set) | mujoco 2.3.7 cp312 wheel (sdist sha256 422041f1…a7d0; wheel 79c7ae83…3f81, cached `~/opt/wheels/`) | robosuite 1.4.1 needs mujoco 2.3.7 (Orin finding); no cp312 aarch64 wheel exists below 3.0.0 | ok, ~2 min |
| 21 | libero | `uv pip install -c constraints robosuite==1.4.1 bddl==1.0.1 gym==0.25.2 hydra-core==1.2.0 easydict==1.9 thop cloudpickle future einops matplotlib opencv-python PyOpenGL pyyaml imageio h5py` | robosuite 1.4.1, bddl 1.0.1, gym 0.25.2, PyOpenGL 3.1.10 | LIBERO's pins as on the Orin | ok |
| 22 | libero | `zz_libero_src.pth` → `~/src/LIBERO`; write `~/.libero/config.yaml` | LIBERO @ 8f1084e | `libero/` has no `__init__.py`; config avoids the interactive first-import prompt | ok |
| 23 | libero | check: `MUJOCO_GL=egl` imports + `libero_spatial` task 0 reset (+5 zero steps), 256×256 | — | A15 | ok: agentview (256,256,3) uint8, all 196608 values nonzero, mean 119.8; EGL renderer "NVIDIA Tegra NVIDIA Thor/PCIe", GL 4.6.0 NVIDIA 580.00. Init states need numpy globals allow-listed for `torch.load` (torch ≥ 2.6), as the Orin's libero_eval.py does |
