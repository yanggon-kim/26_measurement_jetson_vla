#!/usr/bin/env bash
# pytorch3d for N1.5. The first attempt failed in 5 s before compiling anything:
#   ImportError: libcudss.so.0
# pytorch3d's setup.py imports torch to find CUDA, and torch here needs the Jetson
# LD_LIBRARY_PATH that activate_orin_venv.sh sets. Sourcing it first is the fix.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
echo "N1D5B_START $(date -Is)"
while ! grep -q VISC_DONE scratch/T004/queue_viscompile.log 2>/dev/null; do sleep 60; done
echo "vision rungs clear $(date -Is)"

echo "########## $(date +%H:%M) pytorch3d build (sourced env, sm_87)"
t0=$SECONDS
bash -c 'source $HOME/venvs/groot16/activate_orin_venv.sh >/dev/null 2>&1
         export TORCH_CUDA_ARCH_LIST="8.7" MAX_JOBS=6 FORCE_CUDA=1
         VIRTUAL_ENV=$HOME/venvs/groot16 $HOME/.local/bin/uv pip install \
           --python "$HOME/venvs/groot16/bin/python" --no-build-isolation \
           "git+https://github.com/facebookresearch/pytorch3d.git@stable" 2>&1 | tail -20'
echo "EXIT=$? $((SECONDS-t0))s"

echo "########## $(date +%H:%M) N1.5 import smoke test"
bash -c "source \$HOME/venvs/groot16/activate_orin_venv.sh >/dev/null 2>&1; \
         exec \$HOME/venvs/groot16/bin/python -c \"
import sys; sys.path.insert(0,'/home/yanggon/00_VLA/Isaac-GR00T-n1d5')
import gr00t.model; print('  n1d5 gr00t.model OK')
from gr00t.model.gr00t_n1 import GR00T_N1_5; print('  GR00T_N1_5 OK')
\""
echo "EXIT=$?"
echo "N1D5B_DONE $(date -Is)"
