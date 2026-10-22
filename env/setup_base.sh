#!/usr/bin/env bash
# env/setup_base.sh — the base tool chain and PyTorch environment of this Thor (T001 steps 3-4). Re-runnable, no root.
#
#   bash env/setup_base.sh && source env/thor_env.sh && ~/venvs/base/bin/python bench/stack_check.py --out <json>
#
# What it does and why (env/README.md has the findings):
#   1. uv into ~/.local/bin: python3-venv / ensurepip is not installed and there is no sudo to install it.
#   2. CUDA 13.0 tools from the JetPack r38.2 apt repo (repo.download.nvidia.com/jetson/common r38.2/main), fetched with
#      `apt-get download` and unpacked with `dpkg -x` into ~/opt/cuda-root (no root): nvcc + ptxas (Triton needs CUDA
#      13's ptxas for sm_110a), cuobjdump, nvdisasm, CUPTI, NVTX, Nsight Compute 2025.3.0, Nsight Systems 2025.3.2
#      (the versions cuda-nsight-*-13-0 depend on). Symlinks in ~/opt/bin.
#   3. ~/venvs/base: torch 2.10.0+cu130 / torchvision 0.25.0+cu130 (triton 3.6.0 comes with it) from PyTorch's SBSA
#      index -- route (a) of docs/05_setup.md A8, the first that passed every check.
#   4. ~/venvs/tools: huggingface_hub[cli] for downloads.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || \
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
export PATH="$HOME/.local/bin:$PATH"

DEBS="nsight-compute-2025.3.0 nsight-systems-2025.3.2 cuda-cuobjdump-13-0 cuda-nvdisasm-13-0 cuda-nvcc-13-0
      cuda-crt-13-0 libnvvm-13-0 libnvptxcompiler-13-0 cuda-cudart-dev-13-0 cuda-cudart-13-0 cuda-cupti-13-0
      cuda-nvtx-13-0 cuda-cccl-13-0 cuda-driver-dev-13-0"
mkdir -p "$HOME/opt/debs" "$HOME/opt/bin"
( cd "$HOME/opt/debs" && apt-get download $DEBS )
for f in "$HOME"/opt/debs/*.deb; do dpkg -x "$f" "$HOME/opt/cuda-root"; done
R="$HOME/opt/cuda-root"
ln -sf "$R/opt/nvidia/nsight-compute/2025.3.0/ncu" "$HOME/opt/bin/ncu"
ln -sf "$R/opt/nvidia/nsight-systems/2025.3.2/target-linux-sbsa-armv8/nsys" "$HOME/opt/bin/nsys"
for t in cuobjdump nvdisasm nvcc ptxas; do ln -sf "$R/usr/local/cuda-13.0/bin/$t" "$HOME/opt/bin/$t"; done

uv venv -q --clear --python /usr/bin/python3.12 "$HOME/venvs/base"
uv pip install -q --python "$HOME/venvs/base/bin/python" --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.10.0 torchvision==0.25.0
uv pip install -q --python "$HOME/venvs/base/bin/python" ninja   # torch load_inline (T003 pointer chase)
uv pip freeze --python "$HOME/venvs/base/bin/python" > "$REPO/env/requirements_base.txt"

uv venv -q --clear --python /usr/bin/python3.12 "$HOME/venvs/tools"
uv pip install -q --python "$HOME/venvs/tools/bin/python" "huggingface_hub[cli]"
uv pip freeze --python "$HOME/venvs/tools/bin/python" > "$REPO/env/requirements_tools.txt"

source "$REPO/env/thor_env.sh"
"$HOME/venvs/base/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
ncu --version | tail -1; nsys --version; cuobjdump --version | tail -1
