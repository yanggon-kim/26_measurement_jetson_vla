#!/usr/bin/env bash
# Heartbeat watchdog. Exits after --minutes OR as soon as the queue dies, whichever first,
# printing a status summary. Launched through the Bash tool with run_in_background so that
# its exit WAKES the agent: a file-writing monitor gives the owner visibility but cannot
# start a tool call, which is why the GPU sat idle three times in this session.
set -uo pipefail
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
MINUTES="${1:-20}"; QUEUE="${2:-run_queue3.sh}"
alive() { python3 - "$QUEUE" <<'PY'
import re,sys
from pathlib import Path
pat=sys.argv[1]
for d in Path('/proc').iterdir():
    if not d.name.isdigit(): continue
    try: c=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf-8','replace')
    except OSError: continue
    if pat in c and 'bash ' in c: sys.exit(0)
sys.exit(1)
PY
}
END=$(( $(date +%s) + MINUTES*60 ))
REASON="heartbeat"
while [ "$(date +%s)" -lt "$END" ]; do
  if ! alive; then REASON="queue ended"; break; fi
  sleep 60
done
echo "WAKE: $REASON at $(date +%H:%M)"
echo "--- queue steps"
grep -E "^####|EXIT=" "scratch/T004/${QUEUE%.sh}.log" 2>/dev/null | tail -6
echo "--- running now"
python3 - <<'PY'
import re
from pathlib import Path
n=0
for d in Path('/proc').iterdir():
    if not d.name.isdigit(): continue
    try: c=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode('utf-8','replace')
    except OSError: continue
    m=re.search(r'(\S*(?:opt_ladder|paper_repro|phase_utilisation|run_queue|ncu_target|run_ncu_regions|overnight|expert_dims|measure_S|groot_phases)\S*)', c)
    if m: print('   ', d.name, m.group(1)[:60]); n+=1
print('    (nothing)' if not n else '')
PY
echo "--- newest rows"
ls -t results/*/raw/rows/*.json 2>/dev/null | head -3 | sed 's/^/    /'
