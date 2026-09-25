
## 2026-09-25 — git-lfs 3.5.1 (user-space) + Isaac-GR00T demo sample
- `curl -sSL https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-arm64-v3.5.1.tar.gz`,
  binary copied to `~/bin/git-lfs` (sha256 52d57943ec6ff3f0…). No apt, no sudo.
- Why: every `Isaac-GR00T/demo_data/*` parquet was an un-pulled Git LFS pointer (131 B), so T009's
  `scripts/deployment/benchmark_inference.py` failed with "Parquet magic bytes not found".
- `git lfs install --local` (hooks in that one checkout only, no global config) and
  `git lfs pull --include "demo_data/droid_sample/**"` — 211 230 B, PAR1 header. Not committed anywhere.

## 2026-09-25 — torchcodec 0.10.0a0 into ~/venvs/groot (T009 step 1)
- Why: Isaac-GR00T's `scripts/deployment/benchmark_inference.py` decodes the DROID demo video
  through torchcodec and failed without it (`results/20260925_E3E4_groot_n17_benchmark_inference`).
- Source: the repo's own Orin wheel, `Isaac-GR00T/scripts/deployment/orin/wheels/torchcodec-0.10.0a0-cp310-cp310-linux_aarch64.whl`
  (was a Git LFS pointer; fetched with `git lfs pull --include "scripts/deployment/orin/wheels/*"`),
  sha256 `74f85bfa1e30610a3c314fe39bca71bb2f0285e8e7308c60176219a6a3e3bf1a`, 621 818 bytes.
- Command: `uv pip install --python ~/venvs/groot/bin/python --no-deps <wheel>` — `--no-deps` as
  the repo's `orin/install_deps.sh` does, so torch is not touched.
- Check: `uv pip freeze` before/after differs by exactly the torchcodec line; `torch 2.10.0` imports,
  `from torchcodec.decoders import VideoDecoder` imports. FFmpeg 4 runtime (`libavcodec.so.58`) was
  already on the system; no apt change.
