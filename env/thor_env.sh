# Source before every bench/ command on this Thor:  source env/thor_env.sh
# Puts the CUDA 13.0 tools (unpacked from the JetPack apt repo without root, env/INSTALL_LOG.md) on PATH and points
# Triton at CUDA 13.0's ptxas: Triton 3.6's bundled ptxas-blackwell (CUDA 12.9) rejects sm_110a (T001 step 3).
CUDA_ROOT_USER="$HOME/opt/cuda-root/usr/local/cuda-13.0"
export PATH="$HOME/opt/bin:$HOME/.local/bin:$PATH"
export CUDA_HOME="$CUDA_ROOT_USER"
export TRITON_PTXAS_PATH="$CUDA_ROOT_USER/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="$CUDA_ROOT_USER/bin/ptxas"
export TRITON_CUOBJDUMP_PATH="$CUDA_ROOT_USER/bin/cuobjdump"
export TRITON_NVDISASM_PATH="$CUDA_ROOT_USER/bin/nvdisasm"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
