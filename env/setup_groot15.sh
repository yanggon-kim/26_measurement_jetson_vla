#!/usr/bin/env bash
# env/setup_groot15.sh — the Isaac-GR00T N1.5 environment on the Thor (T001 step 7, docs/05_setup.md A8/A14).
#
#   bash env/setup_groot15.sh            # re-runnable; rebuilds ~/venvs/groot15
#
# Ported from reference/orin/env/setup_groot_n1d5_n1d6.md. On the Orin N1.5 and N1.6 shared ~/venvs/groot16
# (a copy of the N1.7 venv with transformers 4.51.3 + av, numpydantic, timm, setuptools, a decord stub and a
# source-built pytorch3d). Here N1.5 gets its OWN venv built from the n1d5 checkout's pyproject.toml @ 4af2b62,
# because N1.5 pins a different set (gymnasium 1.0.0, peft 0.17.0, pydantic 2.10.6, timm 1.0.14, ...) than
# N1.6's Thor project; sharing would silently move one of them off its pins. Deviations:
#   * torch 2.10.0+cu130 / torchvision 0.25.0+cu130 (PyTorch SBSA index, A8 route (a), same build as every env
#     here). N1.5's base pins torch 2.5.1 and its Orin extra 2.8.0; neither exists for CUDA 13 / sm_110; its
#     `thor` extra pins no torch. 2.10.0 is the version every other GR00T env here uses.
#   * flash-attn 2.8.4 (Jetson AI Lab SBSA cu130 index, --no-deps; the Orin extra pinned 2.8.2, which has no
#     cu130 aarch64 wheel). REQUIRED: radio_model.py imports flash_attn at module level and the Eagle config
#     asserts flash_attention_2. diffusers 0.36.0.dev0 (the n1d5 `thor` extra's pin) from the same index.
#   * pytorch3d: only `pytorch3d.transforms` is imported (gr00t/data/transform/state_action.py), and that
#     subpackage (plus pytorch3d.common) is pure Python with no use of the compiled `_C` extension. Instead of
#     the Orin's 627 s CUDA source build (which would also load the CPU while measurements run), the pure-Python
#     source tree of pytorch3d @ 75ebeeae (v0.7.8, the commit the Orin built) is put on sys.path WITHOUT the
#     extension. Same code for every function N1.5 calls; any use of pytorch3d._C fails loudly (ImportError).
#   * decord: no aarch64 wheel (nor eva-decord), and the FFmpeg dev headers to build it are not installed (no
#     sudo). transformers' check_imports() refuses to load the Eagle processor unless `decord` is importable;
#     decord is used only in _read_video_decord(video_path) (video FILES), which inference never reaches.
#     As on the Orin, a stub is installed that RAISES on every attribute access, so any real use fails loudly.
#   * not installed: ray, tianshou, tensorflow, onnx / onnxruntime / tensorrt, torchcodec, pytest (training /
#     sim-eval / export only; not on the inference path).
#   * no sudo apt-get (none needed); uv creates the venv (no ensurepip); Python 3.12; repo on sys.path by .pth.
#   * Triton needs CUDA 13.0's ptxas for sm_110a: source env/thor_env.sh before running anything.
set -euo pipefail

VENV="${VENV:-$HOME/venvs/groot15}"
SRC="${SRC:-$HOME/src/Isaac-GR00T-n1d5}"
SHA="4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"
P3D_SRC="${P3D_SRC:-$HOME/src/pytorch3d}"
P3D_SHA="75ebeeaea0908c5527e7b1e305fbc7681382db47"   # v0.7.8, the commit the Orin built
P3D_PY="$HOME/opt/pytorch3d_py"                       # holds only a symlink to the python package
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
TORCH_IDX="https://download.pytorch.org/whl/cu130"
JAI_IDX="https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple"
PY="$VENV/bin/python"
SP="$VENV/lib/python3.12/site-packages"

check_torch() {  # protocol §12: a pip / uv step must never replace the CUDA wheel
  "$PY" - <<'PY'
import torch, numpy
assert torch.cuda.is_available(), "CUDA LOST: a step replaced the CUDA torch wheel"
assert torch.version.cuda.startswith("13"), torch.version.cuda
torch.randn(4, 4, device="cuda").cpu().numpy()
print(f"  ok: torch {torch.__version__} cuda {torch.version.cuda} "
      f"SMs {torch.cuda.get_device_properties(0).multi_processor_count} numpy {numpy.__version__}")
PY
}

[ -d "$SRC/.git" ] || git clone https://github.com/NVIDIA/Isaac-GR00T.git "$SRC"
git -C "$SRC" fetch -q origin "$SHA" 2>/dev/null || true
git -C "$SRC" checkout -q "$SHA"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$SHA" ]
[ -d "$P3D_SRC/.git" ] || git clone -q https://github.com/facebookresearch/pytorch3d.git "$P3D_SRC"
git -C "$P3D_SRC" checkout -q "$P3D_SHA"
[ "$(git -C "$P3D_SRC" rev-parse HEAD)" = "$P3D_SHA" ]

uv venv -q --clear --python /usr/bin/python3.12 "$VENV"
CONSTRAINTS="$(mktemp)"
printf 'torch==2.10.0+cu130\ntorchvision==0.25.0+cu130\ntransformers==4.51.3\nnumpy<2\n' > "$CONSTRAINTS"
UVI=(uv pip install -q --python "$PY" -c "$CONSTRAINTS"
     --index-url https://pypi.org/simple --extra-index-url "$TORCH_IDX" --index-strategy unsafe-best-match)

# 1. the CUDA wheels first
uv pip install -q --python "$PY" --index-url "$TORCH_IDX" torch==2.10.0 torchvision==0.25.0
echo "after torch:"; check_torch

# 2. n1d5 pyproject.toml [project.dependencies] (minus ray, tianshou, onnx, pytest) + the `thor` extra's
#    inference-relevant pins (iopath, pyzmq, nvtx; its opencv_python 4.11.0.86 is left out: the base
#    already pins opencv_python_headless 4.11.0.86 and the two wheels overwrite each other's cv2)
"${UVI[@]}" \
  "albumentations==1.4.18" "av==12.3.0" "blessings==1.7" "dm_tree==0.1.8" "einops==0.8.1" \
  "gymnasium==1.0.0" "h5py==3.12.1" "hydra-core==1.3.2" "imageio==2.34.2" "kornia==0.7.4" \
  "matplotlib==3.10.0" "numpy>=1.23.5,<2.0.0" "numpydantic==1.6.7" "omegaconf==2.3.0" \
  "opencv_python_headless==4.11.0.86" "pandas==2.2.3" "pydantic==2.10.6" "PyYAML==6.0.2" \
  "Requests==2.32.3" "timm==1.0.14" "tqdm==4.67.1" "transformers==4.51.3" "typing_extensions>=4.12.2" \
  "pyarrow==14.0.1" "wandb==0.18.0" "fastparquet==2024.11.0" "accelerate==1.2.1" "peft==0.17.0" \
  "protobuf==4.25.1" "tyro" "iopath==0.1.9" "pyzmq" "nvtx" "setuptools" \
  "huggingface-hub[cli]"
echo "after deps:"; check_torch

# 3. Thor-index wheels, no deps
uv pip install -q --python "$PY" --no-deps --index-url "$JAI_IDX" "flash-attn==2.8.4" "diffusers==0.36.0.dev0"
"${UVI[@]}" "importlib-metadata" "regex" "requests" "safetensors" "Pillow" "filelock"  # diffusers' own deps
echo "after flash-attn/diffusers:"; check_torch

# 4. GR00T itself, no deps, repo on the path; pytorch3d's pure-Python package on the path (no _C)
uv pip install -q --python "$PY" --no-deps -e "$SRC"
echo "$SRC" > "$SP/zz_groot_repo.pth"
mkdir -p "$P3D_PY"; ln -sfn "$P3D_SRC/pytorch3d" "$P3D_PY/pytorch3d"
echo "$P3D_PY" > "$SP/zz_pytorch3d_py.pth"

# 5. the decord stub (see header): importable, raises on any attribute access
mkdir -p "$SP/decord"
cat > "$SP/decord/__init__.py" <<'STUB'
"""STUB installed by env/setup_groot15.sh — NOT the real decord (no aarch64 wheel).
Exists only so transformers' check_imports() lets the Eagle processor load; decord is used only to read video
FILES (_read_video_decord), which inference with numpy frames never reaches. Any real use raises."""
__version__ = "0.0.0+stub"
def __getattr__(name):
    raise ImportError(f"decord is a stub on this machine (env/setup_groot15.sh); decord.{name} is unavailable")
STUB
echo "after gr00t:"; check_torch

# 6. imports and a flash-attn numerical check (flash vs SDPA, bf16) — tiny, not a measurement
"$PY" - <<'PY'
import torch, transformers, flash_attn, diffusers
from flash_attn import flash_attn_func
import torch.nn.functional as F
q, k, v = (torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
o = flash_attn_func(q, k, v)
r = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
err = (o.float() - r.float()).abs().max().item()
assert err < 2e-2, f"flash-attn disagrees with SDPA: {err}"
import pytorch3d.transforms as pt
assert torch.allclose(pt.matrix_to_quaternion(torch.eye(3)), torch.tensor([1., 0., 0., 0.]))
from gr00t.model.policy import Gr00tPolicy  # noqa: F401
from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: F401
print(f"  gr00t ok; transformers {transformers.__version__} flash_attn {flash_attn.__version__} "
      f"(max |flash-sdpa| {err:.2e}) diffusers {diffusers.__version__}; pytorch3d.transforms (pure Python) ok")
PY

uv pip freeze --python "$PY" > "$REPO/env/requirements_groot15.txt"
rm -f "$CONSTRAINTS"
echo "GR00T N1.5 environment ready: $VENV  (pins: env/requirements_groot15.txt)"
echo "NOTE: decord in this venv is a raising stub; pytorch3d is pure-Python @ $P3D_SHA without _C."
