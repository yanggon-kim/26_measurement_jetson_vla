#!/usr/bin/env bash
# env/setup_lerobot.sh — the LeRobot environment for π0, π0.5 and SmolVLA on the Thor (T001 step 7).
#
#   bash env/setup_lerobot.sh            # re-runnable; rebuilds ~/venvs/lerobot
#
# Ported from reference/orin/env/setup_pi0.sh. What carries over and what does not:
#   * lerobot @ f0d2b37beb (0.4.5) is kept: it has the lm_head -> embed_tokens alias fix and needs stock
#     transformers (the Orin README's "0.4.4 trap"; asserted at the end of this script).
#   * transformers == 5.3.0 exactly is kept: 5.4 moved a sqrt(hidden) rescale into PaliGemma and lerobot dropped
#     its copy at the same time, which silently breaks the prefix embeddings (Orin env/README.md).
#   * Python 3.12 (system) instead of the Orin's forced 3.10; torch 2.10.0+cu130 from PyTorch's SBSA index
#     (the newest torch lerobot's `torch<2.11` allows) instead of the JetPack 6 wheel; numpy 2 is fine with it.
#   * torch / torchvision are pinned by a constraints file so no later step can swap in a CPU or other wheel.
#   * python3-venv / ensurepip is not installed on this image: `uv` (in ~/.local/bin) creates the venv.
#   * Triton needs CUDA 13.0's ptxas for sm_110a: source env/thor_env.sh before running anything.
set -euo pipefail

VENV="${VENV:-$HOME/venvs/lerobot}"
SRC="${SRC:-$HOME/src/lerobot}"
LEROBOT_SHA="f0d2b37bebddf7e8852cb624712da0ad564601b6"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
TORCH_IDX="https://download.pytorch.org/whl/cu130"

check_torch() {  # protocol §12: a pip / uv step must never replace the CUDA wheel
  "$VENV/bin/python" - <<'PY'
import torch, numpy
assert torch.cuda.is_available(), "CUDA LOST: a step replaced the CUDA torch wheel"
assert torch.version.cuda.startswith("13"), torch.version.cuda
torch.randn(4, 4, device="cuda").cpu().numpy()
print(f"  ok: torch {torch.__version__} cuda {torch.version.cuda} "
      f"SMs {torch.cuda.get_device_properties(0).multi_processor_count} numpy {numpy.__version__}")
PY
}

[ -d "$SRC/.git" ] || git clone https://github.com/huggingface/lerobot.git "$SRC"
git -C "$SRC" fetch -q origin "$LEROBOT_SHA" 2>/dev/null || true
git -C "$SRC" checkout -q "$LEROBOT_SHA"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$LEROBOT_SHA" ]

uv venv -q --clear --python /usr/bin/python3.12 "$VENV"
CONSTRAINTS="$(mktemp)"
printf 'torch==2.10.0+cu130\ntorchvision==0.25.0+cu130\ntransformers==5.3.0\n' > "$CONSTRAINTS"

# 1. the CUDA wheels first
uv pip install -q --python "$VENV/bin/python" --index-url "$TORCH_IDX" torch==2.10.0 torchvision==0.25.0
echo "after torch:"; check_torch

# 2. lerobot (editable, pinned commit) with the pi and smolvla extras; torch held by the constraints file
uv pip install -q --python "$VENV/bin/python" -c "$CONSTRAINTS" \
    --index-url https://pypi.org/simple --extra-index-url "$TORCH_IDX" --index-strategy unsafe-best-match \
    -e "$SRC[pi,smolvla]"
echo "after lerobot:"; check_torch

# 2b. sentencepiece: the ungated PaliGemma tokenizer is a bare tokenizer.model (no tokenizer.json, which the gated
#     hub repo ships and the Orin used), so transformers must convert it; without sentencepiece it falls back to a
#     TikToken extractor and fails (T001 smoke, attempt 2).
uv pip install -q --python "$VENV/bin/python" -c "$CONSTRAINTS" \
    --index-url https://pypi.org/simple --extra-index-url "$TORCH_IDX" --index-strategy unsafe-best-match \
    sentencepiece
echo "after sentencepiece:"; check_torch

# 3. the imports the benchmark needs, and the correctness properties of this ref
"$VENV/bin/python" - <<'PY'
import inspect, transformers
from lerobot.policies.pi0 import PI0Policy            # noqa: F401
from lerobot.policies.pi05 import PI05Policy          # noqa: F401
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
from lerobot.policies.factory import make_pre_post_processors        # noqa: F401
import lerobot.policies.pi0.modeling_pi0 as m
src = inspect.getsource(m)
assert "siglip import check" not in src, "this ref needs a patched transformers fork: wrong commit"
assert "lm_head.weight" in src and "embed_tokens.weight" in src, "alias fix missing: weights would load RANDOM"
assert transformers.__version__ == "5.3.0", transformers.__version__
print(f"  lerobot ok (transformers {transformers.__version__}); alias fix present")
PY

uv pip freeze --python "$VENV/bin/python" > "$REPO/env/requirements_lerobot.txt"
rm -f "$CONSTRAINTS"
echo "LeRobot environment ready: $VENV  (pins: env/requirements_lerobot.txt)"
echo "PaliGemma tokenizer: ungated copy ~/models/paligemma_tokenizer/tokenizer.model"
echo "  sha256 8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 (env/README.md)"
