#!/usr/bin/env bash
# env/setup.sh — rebuild the base measurement environment on this Jetson AGX Orin 64GB.
#
# Produced by T001 (E0). Base environment = the one used for E0/E1/E2 (microbenchmarks,
# profiler verification). Model environments (openpi, lerobot, GR00T) are separate and
# are created by T002 onwards; see env/README.md.
#
#   bash env/setup.sh              # create the venv and install everything
#   bash env/setup.sh --verify     # only re-run the verification
#
# Assumes JetPack 6.2.x (L4T R36.5.0, CUDA 12.6, system Python 3.10). It installs
# nothing that touches the kernel, the bootloader or the NVIDIA L4T core packages.
set -euo pipefail

VENV="${VENV:-$HOME/venvs/jetson_base}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------- pinned wheels
# The Jetson wheel index (JetPack 6 + CUDA 12.6) serves non-manylinux
# `linux_aarch64` wheels. pip ranks a manylinux wheel of the same version HIGHER,
# so `pip install torch==2.8.0 --extra-index-url https://pypi.org/simple` silently
# installs PyPI's **CPU-only** 2.8.0+cpu. The three CUDA wheels are therefore pinned
# by full URL and installed with --no-deps. Verified on 2026-09-21.
JP_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"
WHL_TORCH="$JP_INDEX/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl"
WHL_TORCHVISION="$JP_INDEX/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl"
WHL_TRITON="$JP_INDEX/+f/9da/4bcb8e8f0eba0/triton-3.4.0-cp310-cp310-linux_aarch64.whl"

if [[ "${1:-}" != "--verify" ]]; then
  # ---------------------------------------------------------------- apt packages
  # python3-venv, python3-dev and libopenblas-dev were already present on this image;
  # only pip (and its setuptools/wheel deps) had to be installed.
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      python3-pip python3-venv python3-dev libopenblas-dev zstd

  # ---------------------------------------------------------------- virtualenv
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

  # numpy < 2: torch 2.8.0 for JetPack is built against the 1.x ABI.
  "$VENV/bin/python" -m pip install "numpy<2"
  "$VENV/bin/python" -m pip install --no-deps \
      "$WHL_TORCH" "$WHL_TORCHVISION" "$WHL_TRITON"
  # runtime dependencies of torch, from PyPI (pure python / generic aarch64)
  "$VENV/bin/python" -m pip install \
      filelock fsspec jinja2 networkx sympy typing_extensions packaging pillow
fi

# ---------------------------------------------------------------- profiler counters
# ncu needs the perf counters unlocked (protocol §12). This is a runtime sysctl, not a
# boot change, and does not survive a reboot — re-run it each session.
# NOTE: on this device that alone is NOT sufficient; ncu must additionally be run under
# sudo (see env/README.md, "Nsight Compute needs root").
sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'

# ---------------------------------------------------------------- verification
"$VENV/bin/python" "$REPO/bench/verify_stack.py" \
    --out "${VERIFY_OUT:-/tmp/stack_verification.json}"

echo
echo "base environment ready: $VENV"
echo "verification report:   ${VERIFY_OUT:-/tmp/stack_verification.json}"
