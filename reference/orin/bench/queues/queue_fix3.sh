#!/usr/bin/env bash
# 1) find pi0.5's REMAINING fp32 producer with the probe (the conditioning-path cast was
#    necessary but not sufficient), 2) re-run GR00T's vlm_graphs now that its CUDA-graph
#    outputs are cloned.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
PY=$HOME/venvs/pi0/bin/python
export TRITON_CACHE_DIR="$HOME/.cache/triton_groot"; mkdir -p "$TRITON_CACHE_DIR"

echo "FIX3_START $(date -Is)"
while ! grep -q GROOT2_DONE scratch/T004/queue_groot2.log 2>/dev/null; do sleep 45; done
echo "groot2 clear $(date -Is)"
run() { echo "########## $(date +%H:%M) $*"; local t0=$SECONDS; "$@"; echo "EXIT=$? $((SECONDS-t0))s"; }

run $PY bench/pi05_dtype_probe.py --model pi05 --prep base,norms

rm -f results/20260923_E4d_groot_n17_ladder/raw/rows/vlm_graphs.json
echo "########## $(date +%H:%M) groot vlm_graphs (outputs cloned)"
bash -c "export TRITON_CACHE_DIR=$TRITON_CACHE_DIR; \
         source \$HOME/venvs/groot/activate_orin_venv.sh >/dev/null 2>&1; \
         cd /home/yanggon/00_VLA/26_measurement_jetson_vla; \
         exec \$HOME/venvs/groot/bin/python bench/groot_phases.py \
           --outdir results/20260923_E4d_groot_n17_ladder \
           --checkpoint nvidia/GR00T-N1.7-3B --version n17 --warmup 20 --timed 50 \
           --rung vlm_graphs --tf32 on --head-mode reduce-overhead --llm-mode reduce-overhead"
echo "EXIT=$?"
echo "FIX3_DONE $(date -Is)"
