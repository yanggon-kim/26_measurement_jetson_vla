# SUPERSEDED — kept as evidence, not as a result

**Superseded by the `20260922d_…` directory of the same part.** Its numbers must not be
quoted. Kept because protocol §5 forbids overwriting a result directory, and because the
defect below is worth being able to point at.

This pass was itself a re-measurement: it fixed the duty cycle that never applied in the
`20260922_…` pass. It corrected the GPU hog — the dose-response curve in `mempressure.csv`
is real — but two defects remained, both introduced by that same fix:

1. **`cpu_spin_control` produced nothing.** In the chunking refactor `a *= 1.0000001` moved
   into a closure, and augmented assignment rebinds the name, so `a` became local to
   `do_chunk` and every worker died on its first iteration with `UnboundLocalError`. The
   row is blank where protocol §13.5 requires one of its two controls.
2. **Nothing surfaced that.** `start_hog` sent the hog's stderr to `DEVNULL` along with its
   stdout, so six worker tracebacks went nowhere and the run finished, apparently
   successfully, 47 minutes later.

Both are fixed in `bench/` (commit `eb43ed4`), and `bench/bw_hog.py --self-test` now runs
every hog kind for two seconds and fails unless each produced samples carrying an achieved
duty cycle. `bench/mempressure.py` calls it before loading the model, so this class of
defect costs two seconds instead of a finished result directory.

The `20260922d_…` pass also adds `duty_pct_achieved_alone` and `duty_pct_achieved_during`:
the requested duty is an input, the achieved duty is a measurement, and the analysis must
use the latter.
