
## 2026-09-25 — git-lfs 3.5.1 (user-space) + Isaac-GR00T demo sample
- `curl -sSL https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-arm64-v3.5.1.tar.gz`,
  binary copied to `~/bin/git-lfs` (sha256 52d57943ec6ff3f0…). No apt, no sudo.
- Why: every `Isaac-GR00T/demo_data/*` parquet was an un-pulled Git LFS pointer (131 B), so T009's
  `scripts/deployment/benchmark_inference.py` failed with "Parquet magic bytes not found".
- `git lfs install --local` (hooks in that one checkout only, no global config) and
  `git lfs pull --include "demo_data/droid_sample/**"` — 211 230 B, PAR1 header. Not committed anywhere.
