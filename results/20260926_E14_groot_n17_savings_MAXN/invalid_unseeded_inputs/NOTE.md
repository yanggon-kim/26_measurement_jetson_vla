# INVALID for equivalence — kept because every run is reported (T016 non-negotiable 1)

These ten action sets (`bench/groot_phases.py --save-actions`, 2026-09-26/27) were each written by a separate
process. `build_observation` fills the observation's pixels and state from **unseeded `np.random`**, so each
process ran on a **different observation**: every cross-process comparison here (variant vs reference, the
compiled and eager repeat controls, eager vs compiled) compares different inputs. `equiv_summary.json` shows it:
the eager reference repeated in a new process differs by REL up to 0.18, and eager vs compiled by REL 1.27.

Fixed in `bench/groot_phases.py` (np.random seeded to 1234 before the observation is built when --save-actions
is given; the observation's sha256 stored in the .npz) and `bench/e14_groot_equiv.py` (refuses pairs without an
equal input hash). All action sets were re-run with the fix; the valid ones are `../actions_*`.
Timing (`../time_*`) is unaffected: latency does not depend on the input values.
