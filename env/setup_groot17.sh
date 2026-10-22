#!/usr/bin/env bash
# env/setup_groot17.sh — the Isaac-GR00T N1.7 environment on the Thor (T001 step 7, docs/05_setup.md A8/A14).
#
#   bash env/setup_groot17.sh            # re-runnable; rebuilds ~/venvs/groot17
#
# Ported from reference/orin/env/setup_groot.sh. The Orin used the repo's scripts/deployment/orin uv project
# (Python 3.10, JetPack 6 torch 2.10.0 cu126). This checkout ships a Thor project too
# (scripts/deployment/thor/{pyproject.toml,uv.lock}); its pins are followed here, with these deviations:
#   * torch 2.10.0+cu130 / torchvision 0.25.0+cu130 come from PyTorch's own SBSA index (download.pytorch.org/
#     whl/cu130, A8 route (a), the same build as ~/venvs/lerobot), not the Jetson AI Lab build of the same
#     version the Thor uv.lock names. Same torch version as the Thor pin; one torch build across all envs.
#   * triton 3.6.0 (what torch 2.10.0+cu130 itself requires) instead of the Thor pin 3.5.0.
#   * flash-attn 2.8.4 and diffusers 0.36.0.dev0 are the Thor pins and come from the Jetson AI Lab SBSA cu130
#     index (pypi.jetson-ai-lab.io/sbsa/cu130), installed --no-deps so that index can never supply torch.
#     flash-attn is checked numerically against SDPA at the end (ABI with the PyTorch.org torch build).
#   * not installed: tensorrt / onnx / onnxscript (TensorRT export is not scheduled, docs/01 §scope),
#     torchcodec (video-file decoding for datasets; the benchmark passes numpy frames; the repo's own LFS
#     wheel is only a pointer file in this checkout), nvidia-cudss-cu13 / extra cudnn (the PyTorch.org wheel
#     brings its own CUDA libraries), deepspeed (x86 only), dev extras.
#   * no `sudo apt-get` (install_deps.sh needs it for NVPL, nvcc, ffmpeg): no sudo on this machine, and none
#     of those is needed by this torch build (NVPL is not linked by it; nvcc comes from env/thor_env.sh).
#   * python3-venv / ensurepip is not installed: `uv` (~/.local/bin) creates the venv. Python 3.12 (system).
#   * The repo goes on sys.path with a .pth file, as on the Orin (editable import hooks were unreliable there).
#   * Triton needs CUDA 13.0's ptxas for sm_110a: source env/thor_env.sh before running anything.
# LOADING N1.7 NEEDS A GATED REPO: the checkpoint's config names the VLM backbone nvidia/Cosmos-Reason2-2B, whose
# config + processor/tokenizer (and weights, overwritten by the GR00T checkpoint) Gr00tPolicy fetches; it is gated
# (HTTP 401 without a token). After the licence is accepted on huggingface.co and `hf auth login` is done:
#   HF_HUB_OFFLINE=0 ~/venvs/tools/bin/hf download nvidia/Cosmos-Reason2-2B
set -euo pipefail

VENV="${VENV:-$HOME/venvs/groot17}"
SRC="${SRC:-$HOME/src/Isaac-GR00T}"
SHA="376ba890cff8c9de64d71d982772a9c36185fdd7"
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
printf 'torch==2.10.0+cu130\ntorchvision==0.25.0+cu130\ntransformers==4.57.6\nnumpy==1.26.4\n' > "$CONSTRAINTS"
UVI=(uv pip install -q --python "$PY" -c "$CONSTRAINTS"
     --index-url https://pypi.org/simple --extra-index-url "$TORCH_IDX" --index-strategy unsafe-best-match)

# 1. the CUDA wheels first
uv pip install -q --python "$PY" --index-url "$TORCH_IDX" torch==2.10.0 torchvision==0.25.0
echo "after torch:"; check_torch

# 2. the Thor pyproject's dependencies (scripts/deployment/thor/pyproject.toml), minus the deviations above
"${UVI[@]}" \
  "albumentations==1.4.18" "dm-tree==0.1.8" "huggingface-hub[cli]" "jsonlines==4.0.0" "lmdb==1.7.5" \
  "msgpack==1.1.0" "msgpack-numpy==0.4.8" "opencv-python-headless>=4.5,<4.13" "pandas==2.2.3" \
  "peft==0.17.1" "termcolor==3.2.0" "transformers==4.57.6" "tyro==0.9.17" "click==8.1.8" \
  "datasets==3.6.0" "cryptography>=46.0.7" "einops==0.8.1" "gitpython==3.1.50" "gymnasium==1.2.2" \
  "matplotlib==3.10.1" "numpy==1.26.4" "omegaconf==2.3.0" "scipy==1.15.3" "wandb==0.23.0" \
  "pyzmq==27.0.1" "accelerate" "setuptools" "nvtx"
echo "after deps:"; check_torch

# 3. Thor-index wheels, no deps (their deps are already present)
uv pip install -q --python "$PY" --no-deps --index-url "$JAI_IDX" "flash-attn==2.8.4" "diffusers==0.36.0.dev0"
"${UVI[@]}" "importlib-metadata" "regex" "requests" "safetensors" "Pillow" "filelock"  # diffusers' own deps
echo "after flash-attn/diffusers:"; check_torch

# 4. GR00T itself, no deps, repo on the path
uv pip install -q --python "$PY" --no-deps -e "$SRC"
echo "$SRC" > "$VENV/lib/python3.12/site-packages/zz_groot_repo.pth"
echo "after gr00t:"; check_torch

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

uv pip freeze --python "$PY" > "$REPO/env/requirements_groot17.txt"
rm -f "$CONSTRAINTS"
echo "GR00T N1.7 environment ready: $VENV  (pins: env/requirements_groot17.txt)"
