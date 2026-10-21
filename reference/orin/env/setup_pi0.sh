#!/usr/bin/env bash
# env/setup_pi0.sh — the π0 measurement environment (T002 / E3 onwards).
#
#   bash env/setup_pi0.sh
#
# Model route: **LeRobot's π0**, not openpi's PyTorch port. See env/README.md "π0 environment"
# for the full reasoning; short version: openpi costs a 12 GB orbax/OCDBT checkpoint
# conversion, a mandatory JAX install, a `transformers == "4.53.2"` string-equality check with
# five files copied over site-packages, and a `requires-python >=3.11` gate — none of which
# measures anything. LeRobot ships the same architecture ("a direct port of the OpenPI
# implementation … follows the original OpenPI structure for compatibility", printed by the
# model itself on load) with a ready-made PyTorch checkpoint.
#
# ---------------------------------------------------------------------------------------
# WHY THIS EXACT COMMIT — do not "upgrade" without re-reading this
# ---------------------------------------------------------------------------------------
# Pinned: lerobot @ f0d2b37bebddf7e8852cb624712da0ad564601b6 (version string 0.4.5,
# "chore(dependencies): bump transformers v5 (#2964)", 2026-03-05).
#
#   * Only cp310 torch wheels exist for JetPack 6 / CUDA 12.6, so Python 3.10 is forced.
#   * lerobot >= 0.5.0 requires Python >= 3.12  -> unusable here.
#   * lerobot == 0.4.4 (the PyPI release whose name matches the checkpoint's "_v044") is a
#     TRAP. Verified in the installed source before it was discarded:
#       - it requires a **patched transformers fork** (`from transformers.models.siglip
#         import check`), which stock transformers does not satisfy;
#       - it LACKS the `lm_head -> embed_tokens` shared-tensor fix. This checkpoint stores
#         `...language_model.embed_tokens.weight` only as a safetensors metadata ALIAS of
#         `paligemma.lm_head.weight`, and `safetensors.torch.load_file` does not reconstruct
#         aliases, so the key is simply absent;
#       - `from_pretrained` catches load failures, prints a warning, and **returns a model
#         with randomly initialised weights**. The three together produce a silently random
#         model that still runs and still produces plausible-looking latencies.
#   * f0d2b37beb is the one window that is 3.10-clean AND correct: `requires-python >=3.10`,
#     `typing_extensions.Unpack` (not `typing.Unpack`, which is 3.11+), stock transformers,
#     and it HAS the alias fix. All four verified against the installed files.
#
# transformers is pinned to **5.3.0 exactly**, not the `>=5.3.0,<6.0.0` the extra allows:
# transformers 5.4.0 moved a `* sqrt(hidden_size)` rescale inside
# `PaliGemmaModel.get_image_features`, and lerobot removed its own copy of that rescale at the
# same time. Running this ref against 5.4+ scales every prefix embedding by sqrt(2048) ~= 45
# and produces garbage actions **with no error**.
#
# torch is NOT pinned by lerobot (`>=2.2.1,<2.11.0`), so our 2.8.0 CUDA wheel survives;
# `torchcodec` carries an environment marker that excludes aarch64, so it is never installed.
set -euo pipefail

VENV="${VENV:-$HOME/venvs/pi0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_SHA="f0d2b37bebddf7e8852cb624712da0ad564601b6"

JP="https://pypi.jetson-ai-lab.io/jp6/cu126/+f"
WHL_TORCH="$JP/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl"
WHL_TORCHVISION="$JP/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl"
WHL_TRITON="$JP/9da/4bcb8e8f0eba0/triton-3.4.0-cp310-cp310-linux_aarch64.whl"

check_torch() {  # protocol §12: pip silently swaps in PyPI's CPU-only wheel
  "$VENV/bin/python" - <<'PY'
import torch, numpy
assert torch.cuda.is_available(), "CUDA LOST — a pip step replaced the Jetson wheel"
torch.randn(4, 4, device="cuda").cpu().numpy()   # fails loudly on a numpy ABI mismatch
torch.from_numpy(numpy.zeros((2, 2), dtype="float32"))
print(f"  ok: torch {torch.__version__} cuda {torch.version.cuda} "
      f"SMs {torch.cuda.get_device_properties(0).multi_processor_count} numpy {numpy.__version__}")
PY
}

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

# 1. Jetson CUDA wheels FIRST, by full URL, --no-deps (env/README.md has the pip trap)
"$VENV/bin/python" -m pip install --no-deps "$WHL_TORCH" "$WHL_TORCHVISION" "$WHL_TRITON"
"$VENV/bin/python" -m pip install "numpy<2" filelock fsspec jinja2 networkx sympy \
    typing_extensions packaging pillow
echo "after Jetson wheels:"; check_torch

# 2. LeRobot at the pinned commit, with the [pi] extra (transformers + scipy).
#    Do NOT pass `-c constraints.txt`: pip's resolver then fails with
#    "Cannot install imageio[ffmpeg] … ResolutionImpossible" because lerobot requires an
#    extra of a package the constraint file also names.
"$VENV/bin/python" -m pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_SHA}"

# 3. Put transformers and numpy back where they belong (see the header for why 5.3.0).
#    lerobot pulls numpy 2.x via accelerate; the JetPack torch 2.8.0 wheel is built against
#    the numpy 1.x C ABI and every torch<->numpy conversion fails on numpy 2
#    ("Failed to initialize NumPy", tensor_numpy.cpp:82), breaking the policy preprocessing.
#    pip then warns that opencv-python-headless and rerun-sdk "require numpy>=2" — that is a
#    conservative declaration, not a real ABI requirement; both import and work on 1.26.4.
"$VENV/bin/python" -m pip install "transformers==5.3.0" "numpy<2"
echo "after lerobot:"; check_torch

# 4. The imports the benchmark actually needs, and the three correctness properties.
"$VENV/bin/python" - <<'PY'
import inspect, transformers
from lerobot.policies.pi0 import PI0Policy, PI0Config        # noqa: F401
from lerobot.policies.factory import make_pre_post_processors  # noqa: F401
import lerobot.policies.pi0.modeling_pi0 as m
src = inspect.getsource(m)
assert "siglip import check" not in src, "this ref needs a patched transformers fork — wrong commit"
assert "lm_head.weight" in src and "embed_tokens.weight" in src, \
    "missing the lm_head->embed_tokens alias fix — the checkpoint would load as RANDOM weights"
assert transformers.__version__ == "5.3.0", f"transformers {transformers.__version__} != 5.3.0"
print(f"  lerobot π0 ok (transformers {transformers.__version__}); alias fix present; no fork needed")
PY

echo
echo "π0 environment ready: $VENV"
echo "exact pins: $REPO/env/requirements_pi0.txt"
echo
echo "Hugging Face: 'google/paligemma-3b-pt-224' (the tokenizer) is a GATED repo — accept the"
echo "Gemma license once at https://huggingface.co/google/paligemma-3b-pt-224 and 'hf auth login'."
echo "Its tokenizer.model sha256 is 8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6,"
echo "byte-identical to Google's ungated copy at https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
echo "(verified), which is the fallback if the gate is ever a problem."
