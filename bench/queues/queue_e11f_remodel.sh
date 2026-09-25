#!/usr/bin/env bash
# Re-run E11f for pi0.5 and SmolVLA, whose gpu_idle rows were pi0's numbers.
#
# bench/pipelining.py invoked nsys_target.py with no --model, and that flag defaults to
# "pi0", so the nsys sub-capture profiled pi0 for every model. Fixed in a49e7e9. It wrote
# WRONG NUMBERS rather than crashing, into two of §13.8's quantities
# (gpu_idle_ae_total, gpu_idle_frac_ae_total), and they reached the committed consolidated
# model_inputs.csv for both models.
#
# pi0's own E11f is NOT re-run: the wrong default happened to be pi0, so its rows are right.
# Re-measuring it would produce a second set of numbers for a correct point.
set -u
cd /home/yanggon/00_VLA/26_measurement_jetson_vla
DATE=20260924
PY=$HOME/venvs/pi0/bin/python
export TRITON_CACHE_DIR=/tmp/triton_yanggon
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_yanggon

sweep() {
  sudo -n /bin/sh -c 'pkill -f "[n]cu --profile-from-start"; pkill -f "[T]reeLauncherSubreaper"; pkill -f "[n]sys profile"; tegrastats --stop; pkill -9 -x tegrastats; sleep 2; true' 2>/dev/null
}
run() { local l="$1"; shift; echo "########## $(date +%H:%M) $l"; sweep; "$@"; echo "EXIT=$? $l"; }

for q in "[q]ueue_e5.sh" "[q]ueue_e11.sh" "[q]ueue_e11_fixup.sh" "[q]ueue_e11_recover.sh"; do
  pgrep -u "$USER" -f "$q" >/dev/null 2>&1 && { echo "REFUSING: another queue is running"; exit 2; }
done

echo "E11FREM_START $(date -Is)"
grep -q '"--model", spec.name' bench/pipelining.py || {
  echo "REFUSING: the --model fix (a49e7e9) is absent; the re-run would repeat the bug"; exit 3; }
$PY -m pyflakes bench/pipelining.py | grep -q "undefined name" && { echo "REFUSING: pyflakes"; exit 4; }

for M in pi05 smolvla; do
  D="results/${DATE}_E11f_${M}_pipelining"
  # The existing directory holds a VALID throughput/concurrency measurement and INVALID
  # gpu_idle rows. It cannot be kept as a result (its model_inputs.csv carries the wrong
  # rows) and must not be silently overwritten either (protocol §5), so it moves to scratch
  # where the wrong numbers stay inspectable next to the commit that explains them.
  [ -d "$D" ] && mv "$D" "scratch/wrongmodel_${DATE}_E11f_${M}_pipelining" \
    && echo "  moved the pi0-contaminated run to scratch/wrongmodel_${DATE}_E11f_${M}_pipelining"
  run "$M E11f pipelining (--model fixed)" \
    $PY bench/pipelining.py --model "$M" --outdir "$D" --steps 4 --seconds 60

  # Verify the capture actually profiled THIS model, by kernels/step against its own E5.
  $PY - "$M" "$D" <<'PYE'
import csv, sys, os
m, d = sys.argv[1], sys.argv[2]
f = os.path.join(d, "gpu_idle.csv")
e5 = f"results/20260924_E5_{m}_eager_N10/timeline_summary.csv"
if not (os.path.exists(f) and os.path.exists(e5)):
    print(f"  VERIFY_SKIPPED {m}: missing {f if not os.path.exists(f) else e5}"); raise SystemExit
ref = next(float(x["kernels_per_unit"]) for x in csv.DictReader(open(e5))
           if x["region"] == "ae_step")
for r in csv.DictReader(open(f)):
    if r["mode"] != "eager" or not r.get("n_kernels_in_region"):
        continue
    got = float(r["n_kernels_in_region"]) / float(r["n_ae_step_spans"]) * 2
    ok = abs(got - ref) / ref < 0.05
    print(f"  VERIFY {m} N={r['N']}: kernels/step={got:.2f} vs its own E5 {ref:.2f} "
          f"-> {'OK' if ok else '*** STILL THE WRONG MODEL ***'}")
PYE

  D2="results/${DATE}_E11_${M}_inputs"
  run "$M re-consolidate" $PY bench/consolidate_e11.py --model "$M" --outdir "$D2"
  $PY docs/reference_scripts/check_model_inputs.py "$D2/model_inputs.csv" \
    && echo "CHECK_OK $M" || echo "CHECK_FAILED $M"
done

$PY - <<'PYE'
import glob, json, os
for d in sorted(glob.glob("results/20260924_E11_*_inputs")):
    m = json.load(open(os.path.join(d, "meta.json")))
    miss = m.get("parts_missing")
    print(f"  {os.path.basename(d):32s} rows={m.get('total_rows')}  "
          + (f"*** parts_missing={miss} ***" if miss else "complete"))
PYE
echo "E11FREM_DONE $(date -Is)"
