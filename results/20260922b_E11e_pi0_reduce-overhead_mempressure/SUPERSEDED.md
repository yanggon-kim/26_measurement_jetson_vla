# SUPERSEDED — kept as evidence, not as a result

**Superseded by the `20260922d_…` directory of the same part.** Its numbers must not be
quoted. Kept because protocol §5 forbids overwriting a result directory.

This pass ran the corrected harness for every other E11 part, but its hogs still had the
duty-cycle defect: the unit of work (a 256 MiB copy, ~3.4 ms on the GPU and ~10 ms on the
CPU) was larger than the duty window itself, so a hog asked for 25 % overshot and one asked
for 75 % ran past the end of the 20 ms period and never slept. `gpu_hog` reported
53.5 / 107.1 / 133.8 / 172.9 GB/s alone and the prefix slowed by 1.24 / 1.62 / 1.76 / 1.95 —
those are the numbers of a coarse time-slice competitor, not of a bandwidth competitor.
`cpu_hog` was worse: 49.6 GB/s and slowdown 1.076 at *every* duty, with the rate it
reported alone *falling* as its duty rose, because the parent summed a variable number of
its six workers per sample.

Both hogs now move a slice per iteration (32 MiB on the GPU, 16 MiB on the CPU) and emit
the duty cycle they actually achieved. The `20260922d_…` pass measures 36.8 / 72.8 / 107.1 /
145.7 GB/s alone at achieved duties of 26.6 / 51.5 / 75.2 / 100.0 %, and a prefix slowdown
of 1.103 / 1.208 / 1.293 / 1.352.

See also `results/20260922c_E11e_*/SUPERSEDED.md`: the pass between these two fixed the
duty cycle but lost `cpu_spin_control` to a closure bug introduced by that same fix.
