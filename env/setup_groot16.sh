#!/usr/bin/env bash
# env/setup_groot16.sh — the Isaac-GR00T N1.6 environment on the Thor (T001 step 7, docs/05_setup.md A8/A14).
#
#   bash env/setup_groot16.sh            # re-runnable; rebuilds ~/venvs/groot16
#
# Ported from reference/orin/env/setup_groot_n1d5_n1d6.md (the Orin copied its N1.7 venv and put
# transformers 4.51.3 over it). Here the env is built fresh from the n1d6 checkout's own Thor project
# (scripts/deployment/thor/pyproject.toml @ 9b37aa1), with these deviations:
#   * torch 2.10.0+cu130 / torchvision 0.25.0+cu130 from PyTorch's SBSA index (A8 route (a), the same build as
#     every other env here) instead of the Jetson AI Lab build of the same version that the Thor uv.lock names.
#   * triton 3.6.0 (what torch 2.10.0+cu130 requires) instead of the Thor pin 3.5.0.
#   * flash-attn 2.8.4 and diffusers 0.36.0.dev0 (the Thor pins) from pypi.jetson-ai-lab.io/sbsa/cu130,
#     --no-deps. flash-attn is REQUIRED here: the Eagle-Block2A remote code asserts flash_attention_2.
#   * not installed: tensorrt / onnx / onnxscript (TensorRT not scheduled), torchcodec (dataset video decoding;
#     the benchmark feeds numpy frames), extra nvidia-cudnn/cudss (the PyTorch.org wheel brings its own).
#   * added: setuptools (inductor needs it; its absence failed N1.6's compile rung on the Orin), nvtx.
#   * The Orin's one-line eagle_backbone.py fix (forward **extra_kwargs to AutoModel.from_config) is NOT
#     applied and NOT needed here: the checkout is left pristine, and the load check showed
#     config / text_config / vision_config._attn_implementation == flash_attention_2 and flash-attn's
#     flash_fwd_kernel running in one get_action (scratch/t001/attn_n16_nofix.log, T001).
#   * no sudo apt-get (none needed); uv creates the venv (no ensurepip); Python 3.12; repo on sys.path by .pth.
#   * Triton needs CUDA 13.0's ptxas for sm_110a: source env/thor_env.sh before running anything.
set -euo pipefail

VENV="${VENV:-$HOME/venvs/groot16}"
SRC="${SRC:-$HOME/src/Isaac-GR00T-n1d6}"
SHA="9b37aa1ce69c73c6d165233fa88128283bba4508"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
TORCH_IDX="https://download.pytorch.org/whl/cu130"
JAI_IDX="https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple"
PY="$VENV/bin/python"

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

uv venv -q --clear --python /usr/bin/python3.12 "$VENV"
CONSTRAINTS="$(mktemp)"
printf 'torch==2.10.0+cu130\ntorchvision==0.25.0+cu130\ntransformers==4.51.3\nnumpy==1.26.4\n' > "$CONSTRAINTS"
UVI=(uv pip install -q --python "$PY" -c "$CONSTRAINTS"
     --index-url https://pypi.org/simple --extra-index-url "$TORCH_IDX" --index-strategy unsafe-best-match)

# 1. the CUDA wheels first
uv pip install -q --python "$PY" --index-url "$TORCH_IDX" torch==2.10.0 torchvision==0.25.0
echo "after torch:"; check_torch

# 2. the n1d6 Thor pyproject's dependencies, minus the deviations above
"${UVI[@]}" \
  "albumentations==1.4.18" "av==16.1.0" "dm-tree==0.1.8" "huggingface-hub[cli]" "lmdb==1.7.5" \
  "msgpack==1.1.0" "msgpack-numpy==0.4.8" "opencv-python-headless>=4.5,<4.13" "pandas==2.2.3" \
  "peft==0.17.1" "termcolor==3.2.0" "transformers==4.51.3" "tyro==0.9.17" "click==8.1.8" \
  "datasets==3.6.0" "cryptography>=44.0.0" "einops==0.8.1" "gitpython==3.1.46" "gymnasium==1.2.2" \
  "matplotlib==3.10.1" "numpy==1.26.4" "omegaconf==2.3.0" "scipy==1.15.3" "wandb==0.23.0" \
  "pyzmq==27.0.1" "accelerate" "setuptools" "nvtx"
echo "after deps:"; check_torch

# 3. Thor-index wheels, no deps
uv pip install -q --python "$PY" --no-deps --index-url "$JAI_IDX" "flash-attn==2.8.4" "diffusers==0.36.0.dev0"
"${UVI[@]}" "importlib-metadata" "regex" "requests" "safetensors" "Pillow" "filelock"  # diffusers' own deps
echo "after flash-attn/diffusers:"; check_torch

# 4. GR00T itself, no deps, repo on the path
uv pip install -q --python "$PY" --no-deps -e "$SRC"
echo "$SRC" > "$VENV/lib/python3.12/site-packages/zz_groot_repo.pth"
echo "after gr00t:"; check_torch

[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || echo "  WARNING: $SRC has local modifications"

# 5. imports and a flash-attn numerical check (flash vs SDPA, bf16) — tiny, not a measurement
"$PY" - <<'PY'
import torch, transformers, flash_attn, diffusers
from flash_attn import flash_attn_func
import torch.nn.functional as F
q, k, v = (torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
o = flash_attn_func(q, k, v)
r = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
err = (o.float() - r.float()).abs().max().item()
assert err < 2e-2, f"flash-attn disagrees with SDPA: {err}"
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: F401
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: F401
print(f"  gr00t ok; transformers {transformers.__version__} flash_attn {flash_attn.__version__} "
      f"(max |flash-sdpa| {err:.2e}) diffusers {diffusers.__version__}")
PY

uv pip freeze --python "$PY" > "$REPO/env/requirements_groot16.txt"
rm -f "$CONSTRAINTS"
echo "GR00T N1.6 environment ready: $VENV  (pins: env/requirements_groot16.txt)"
