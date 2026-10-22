# INSTALL_LOG — every installation on this Thor (on-device agent)

No root was used for any installation: there is no sudo for the agent (T001, A6). Root-only *reads and profiler runs*
go through `env/rootdo.sh` (docker, owner-approved 2026-10-22); it is never used to install anything.
GR00T and LIBERO environments: `env/INSTALL_LOG_groot_libero.md`.

| date (UTC) | command | version | why | outcome |
|---|---|---|---|---|
| 2026-10-22 | `curl -LsSf https://astral.sh/uv/install.sh \| env UV_NO_MODIFY_PATH=1 sh` | uv 0.13.5 → `~/.local/bin` | `python3 -m venv` fails: `ensurepip` missing (python3.12-venv not installed, no sudo) | ok |
| 2026-10-22 | `python3 -m venv ~/venvs/base` | — | first attempt | **failed**: "ensurepip is not available" |
| 2026-10-22 | `uv venv --python /usr/bin/python3.12 ~/venvs/base`; `uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.10.0 torchvision==0.25.0` | torch 2.10.0+cu130, torchvision 0.25.0+cu130, triton 3.6.0, nvidia-cublas 13.1.0.3, cudnn-cu13 9.15.1.9 | A8 route (a); 2.10.0 is the newest torch lerobot's `torch<2.11` allows | ok; `torch.compile` failed until Triton was pointed at CUDA 13 `ptxas` (`env/thor_env.sh`) |
| 2026-10-22 | `apt-get download` + `dpkg -x` into `~/opt/cuda-root`: nsight-compute-2025.3.0, nsight-systems-2025.3.2, cuda-cuobjdump-13-0, cuda-nvdisasm-13-0, cuda-nvcc-13-0, cuda-crt-13-0, libnvvm-13-0, libnvptxcompiler-13-0, cuda-cudart(-dev)-13-0, cuda-cupti-13-0, cuda-nvtx-13-0, cuda-cccl-13-0, cuda-driver-dev-13-0 | ncu 2025.3.0.0, nsys 2025.3.2.367, nvcc 13.0.48, cuobjdump/nvdisasm 13.0.39 | A9: no CUDA toolkit on the image, apt install needs root; Triton needs a CUDA-13 `ptxas` for `sm_110a` | ok; symlinks in `~/opt/bin` |
| 2026-10-22 | `uv venv ~/venvs/tools`; `uv pip install "huggingface_hub[cli]"` | see `requirements_tools.txt` | checkpoint downloads | ok |
| 2026-10-22 | `hf download <repo> --revision <rev>` × 6 (`docs/00_context.md` pins), no token | 39 GB in `~/.cache/huggingface` | A13 | ok, all public |
| 2026-10-22 | `curl -o ~/models/paligemma_tokenizer/tokenizer.model https://storage.googleapis.com/big_vision/paligemma_tokenizer.model` | sha256 8986bb4f…8fc6 (matches) | A13 ungated route for the gated `google/paligemma-3b-pt-224` tokenizer | ok; H2 not needed |
| 2026-10-22 | `hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | snapshot 7b375e1b73b1 | SmolVLA's load path reads its config / processor | ok |
| 2026-10-22 | `git clone` + `checkout` into `~/src/`: lerobot f0d2b37b, Isaac-GR00T 376ba890 / 9b37aa1c (n1d6) / 4af2b622 (n1d5), LIBERO 8f1084e3 | pinned | A14 | ok |
| 2026-10-22 | `bash env/setup_lerobot.sh` (→ `~/venvs/lerobot`) | torch 2.10.0+cu130, transformers 5.3.0, lerobot 0.4.5 editable | π0, π0.5, SmolVLA | ok |
| 2026-10-22 | `uv pip install sentencepiece` in `~/venvs/lerobot` (added to `setup_lerobot.sh`) | 0.2.2 | the ungated tokenizer is a bare `tokenizer.model`; converting it needs sentencepiece (without it transformers fell back to a TikToken extractor and failed) | ok |
| 2026-10-22 | `uv pip install ninja` in `~/venvs/base` | 1.13.2 | torch's load_inline needs ninja to build the T003 pointer-chase kernel | ok |
| 2026-10-22 | `uv pip install --no-deps --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple flash-attn==2.8.4` in `~/venvs/lerobot` (added to `setup_lerobot.sh`) | 2.8.4 (prebuilt cp312 aarch64 cu130 wheel, the build the GR00T venvs use); torch stays 2.10.0+cu130 | T005 E4i: flash-attn candidates for the π0 / π0.5 prefix and the π0.5 expert (owner request: find the true best_bf16 including attention backends) | ok |
