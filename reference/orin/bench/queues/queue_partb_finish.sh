#!/usr/bin/env bash
# Finish Part B for all three LeRobot models, then hand over to GR00T.
# Waits for whatever opt_ladder is running now, so two ladders never share the GPU.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
PY=$HOME/venvs/pi0/bin/python

running() {
  $PY - <<'P'
import sys
from pathlib import Path
for d in Path('/proc').iterdir():
    if not d.name.isdigit():
        continue
    try:
        c = (d/'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
    except OSError:
        continue
    # Match an exact argv ELEMENT, and only of a python process. Substring-matching the
    # joined cmdline is the pgrep -f self-match bug wearing a different hat: the shell
    # that CREATED this script has the whole heredoc in its cmdline, so it contains
    # 'opt_ladder.py' four times and the waiter waited on itself for 26 minutes.
    argv = (d/'cmdline').read_bytes().split(b'\0')
    if not argv or b'python' not in argv[0].rsplit(b'/', 1)[-1]:
        continue
    if any(a.endswith(b'opt_ladder.py') for a in argv[1:]):
        sys.exit(0)
sys.exit(1)
P
}

echo "PARTB_FINISH_START $(date -Is)"
while running; do sleep 60; done
echo "prior ladder clear $(date -Is)"

run() {
  echo "########## $(date +%H:%M) $*"
  local t0=$SECONDS
  "$@"; echo "EXIT=$? $((SECONDS-t0))s"
}

# pi0: only bf16_base is new (it will record 'not applicable' -- the checkpoint is
# already bf16 -- which is the row that makes pi0's column comparable to pi0.5's).
run $PY bench/opt_ladder.py --model pi0 --outdir results/20260922_E4c_pi0_opt_ladder \
    --steps 10 --warmup 20 --timed 10 --rungs bf16_base

# pi0.5: the real Part B. bf16_vision is re-measured because its existing row predates
# the guard commit and so was produced by different code.
rm -f results/20260922_E4c_pi05_opt_ladder/raw/rows/bf16_vision_N10.json
rm -f results/20260922_E4c_pi05_opt_ladder/raw/rows/bf16_norms_N10.json
rm -f results/20260922_E4c_pi05_opt_ladder/raw/rows/bf16_both_N10.json
run $PY bench/opt_ladder.py --model pi05 --outdir results/20260922_E4c_pi05_opt_ladder \
    --steps 10 --warmup 20 --timed 10 --rungs bf16_base,bf16_vision,bf16_norms,bf16_both

# SmolVLA: expected to record three 'not applicable' rows with reasons. Cheap, and it is
# what lets the cross-model table state that LeRobot already ships SmolVLA in the dtype
# state Part B is trying to reach for pi0.
run $PY bench/opt_ladder.py --model smolvla --outdir results/20260922_E4c_smolvla_opt_ladder \
    --steps 10 --warmup 20 --timed 10 --rungs bf16_base,bf16_vision,bf16_norms,bf16_both

echo "PARTB_FINISH_DONE $(date -Is)"
