#!/usr/bin/env bash
# N1.5: build pytorch3d ONLY while nothing is measuring (it saturates the CPU and
# data_processing_cpu is a reported phase), then smoke-test the load.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
echo "N1D5_START $(date -Is)"
while ! grep -q N1D6_DONE scratch/T004/queue_fix7.log 2>/dev/null; do sleep 45; done
echo "n1.6 clear $(date -Is)"

echo "########## $(date +%H:%M) pytorch3d build (aarch64, sm_87, no wheel exists)"
t0=$SECONDS
TORCH_CUDA_ARCH_LIST="8.7" MAX_JOBS=6 \
  VIRTUAL_ENV=$HOME/venvs/groot16 $HOME/.local/bin/uv pip install \
  --python "$HOME/venvs/groot16/bin/python" --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@stable" 2>&1 | tail -25
echo "EXIT=$? $((SECONDS-t0))s"

echo "########## $(date +%H:%M) N1.5 import + load smoke test"
bash -c "source \$HOME/venvs/groot16/activate_orin_venv.sh >/dev/null 2>&1; \
         cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
         exec \$HOME/venvs/groot16/bin/python -c \"
import sys; sys.path.insert(0,'/home/yanggon/00_VLA/Isaac-GR00T-n1d5')
import gr00t.model as M; print('  n1d5 gr00t.model OK')
from gr00t.model.gr00t_n1 import GR00T_N1_5; print('  GR00T_N1_5 OK')
\""
echo "EXIT=$?"
echo "N1D5_DONE $(date -Is)"
