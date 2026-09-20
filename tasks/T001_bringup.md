# T001 — Bring-up and platform record (E0)  · priority P0 · depends on: —

Goal: a working, recorded software stack on the Jetson and the E0 platform record.
0. Check the prerequisites in `AGENTS.md` (JetPack 6.x, passwordless sudo, push access, disk). If JetPack is 5.x, record it and stop: the human must reflash.
1. Set MAXN + `jetson_clocks`; record everything in protocol §1 to `results/<date>_E0_platform/`.
2. Install PyTorch with CUDA for this JetPack (NVIDIA's wheel or container), torchvision, Triton if available.
   Verify: `torch.cuda.is_available()`, a bf16 GEMM runs, **`torch.compile` compiles and runs a small MLP**
   (say which backend/mode worked). Write `env/README.md` + `env/setup.sh` so the stack can be rebuilt.
3. Install Nsight Systems CLI (`nsys`) and Nsight Compute CLI (`nv-nsight-cu-cli`) (JetPack ships them or `apt`); verify `nsys profile` on a
   toy script and `nsys stats` CSV export; note which Nsight Compute metrics the device exposes.
4. Verify `tegrastats` rails listed in protocol §4 exist on this carrier board; record the exact rail names.
Acceptance: E0 result directory, `env/` reproducible, a line in `STATUS.md` with the versions that work.
