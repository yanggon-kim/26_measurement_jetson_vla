#!/usr/bin/env bash
# T009 / T012 — Isaac-GR00T N1.7 on Jetson AGX Orin, JetPack 6.2.3.
#
# VERIFIED WORKING on 2026-09-22: torch 2.10.0, CUDA available, device "Orin",
# compute capability (8, 7), `import gr00t` OK.
#
# Repo: github.com/NVIDIA/Isaac-GR00T at the pinned commit from docs/00_context.md:151,
#   376ba890cff8c9de64d71d982772a9c36185fdd7
# cloned to ~/00_VLA/Isaac-GR00T. Its own Orin path is used rather than a hand-rolled
# environment: scripts/deployment/orin/{pyproject.toml,uv.lock,wheels/}.
#
# NOTE ON TORCH VERSIONS. This venv has torch 2.10.0 from the jetson jp6-cu126 index;
# ~/venvs/pi0 has 2.8.0. They are deliberately separate — GR00T's pyproject pins
# torch==2.10.0 and requires-python >=3.10,<3.11. Every GR00T measurement records its own
# torch version in meta.json, and no cross-model latency comparison may ignore that they
# differ.
set -uo pipefail
R=$HOME/00_VLA/Isaac-GR00T

# ffmpeg first, BY HAND, not through install_deps.sh. That script calls `sudo apt-get`,
# and this machine's sudoers grants NOPASSWD only for jetson_clocks, nvpmodel, tegrastats,
# ncu, chown and /bin/sh — so a bare `sudo apt-get` prompts for a password and hangs an
# unattended run. Simulated first (AGENTS.md: decline anything touching the kernel,
# bootloader or L4T/NVIDIA core packages); it proposed exactly 2 packages, removed none,
# and touched nothing NVIDIA.
sudo -n /bin/sh -c 'apt-get update -qq && apt-get install -y --no-install-recommends ffmpeg'

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

export UV_PROJECT_ENVIRONMENT="$HOME/venvs/groot"
uv sync --project "$R/scripts/deployment/orin" --no-install-project --extra dev
VENV_PY="$HOME/venvs/groot/bin/python"
uv pip install --python "$VENV_PY" --no-deps -e "$R"
uv pip install --python "$VENV_PY" --no-deps nvidia-cudss-cu12
W=$(find "$R/scripts/deployment/orin/wheels" -name 'torchcodec-*.whl' -print -quit)
[ -n "$W" ] && uv pip install --python "$VENV_PY" --force-reinstall --no-deps "$W"

# `uv pip install --no-deps -e` registered no import hook here, exactly as LIBERO's
# editable install did not: `import gr00t` worked only with the repo as cwd. Put the repo
# on the path so an unattended run from any directory finds it.
echo "$R" > "$HOME/venvs/groot/lib/python3.10/site-packages/zz_groot_repo.pth"

# ---------------------------------------------------------------------------------------
# The one fix Isaac-GR00T's own activation script needs on this machine.
#
# scripts/activate_orin.sh builds LD_LIBRARY_PATH from
#     python3 -c 'import site; print(site.getsitepackages()[0])'
# which resolves the SYSTEM python's site-packages, not this venv's. The nvidia pip
# packages that carry libcudss.so.0 live in the VENV, so sourcing it as written leaves
# torch failing with:
#     ImportError: libcudss.so.0: cannot open shared object file
# Source it, then override with the venv's own paths:
# ---------------------------------------------------------------------------------------
cat > "$HOME/venvs/groot/activate_orin_venv.sh" <<'ACT'
# source this INSTEAD of Isaac-GR00T/scripts/activate_orin.sh
source "$HOME/00_VLA/Isaac-GR00T/scripts/activate_orin.sh" >/dev/null
SP=$("$HOME/venvs/groot/bin/python" -c 'import site;print(site.getsitepackages()[0])')
NVLIBS=$(find "$SP/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$SP/torch/lib:${NVLIBS}${LD_LIBRARY_PATH:-}"
export TRITON_PTXAS_PATH=/usr/local/cuda-12.6/bin/ptxas
ACT

source "$HOME/venvs/groot/activate_orin_venv.sh"
"$VENV_PY" -c "
import torch, gr00t
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
print('gr00t', gr00t.__file__)"
