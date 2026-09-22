# SUPERSEDED — kept as evidence, not as a result

This directory is **superseded by the `20260922b_…` directory of the same part.** Its
numbers must not be quoted. It is kept because protocol §5 forbids overwriting a result
directory and because the report cites two of its measurements as evidence of the defects
listed below.

The third adversarial review of T010 found 40 issues in this measurement pass. Three of
them invalidate numbers this directory carries:

1. **The timed sequence returned `None`.** A compiled callable whose outputs are discarded
   is dead-code-eliminated. Measured on this device: a compiled 2048³ bf16 matmul costs
   0.066 ms when its result is discarded and 0.661 ms when it is returned. Every
   `sequence`/`compiled` figure here — and the "18×", "0.53×" and "1.9× better than
   assumed" derived from it — measured nothing.
2. **`bw_hog`'s duty cycle never applied.** One copy is ~3.4 ms of GPU time, so queueing
   copies for `on` seconds and synchronising afterwards drained a queue far longer than the
   20 ms period; the sleep never ran and every duty cycle collapsed to 100 %. The
   E11e table is therefore one condition repeated four times.
3. **The GPU-idle fraction charged nsys's own overhead to the model.** nsys instruments
   every CUDA API call and this workload is launch-bound, so the region wall time it
   reports is inflated while kernel durations are not; the published 84 % is the
   uncorrected figure.

The re-measurement runs the corrected code, which is committed *before* the run so that
the `bench_commit` column of `model_inputs.csv` names the code that actually ran.
