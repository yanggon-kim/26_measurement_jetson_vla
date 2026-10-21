# T013 — E0 again at MAXN  · **WITHDRAWN by the research side 2026-09-21 — nothing to do, do not claim**

This task was written while the device was still in `MODE_30W`. The human rebooted into MAXN and the on-device
agent completed the MAXN record inside T001 (`results/20260921_E0_platform_maxn/`, device commits `cf340fa`,
`30eedfc`, `198965d`) before this file reached it. That record is **accepted** as the authoritative E0: 16 SMs
observed, GPU locked and measured at 1,300.5 MHz, `jetson_clocks` before every capture, ncu's base clock
(1.29 GHz) matching the locked clock, the Inductor SM threshold (68) read from the installed source, 12 of 12
`raw_files` hashes verified on the research server. The number stays reserved (tasks are never renumbered).

What remained of it lives elsewhere: the per-run clock check → protocol §3.2; the substitute's write-path
validation and the rail double-count check → T005; the one stale string in the `MODE_30W` run's `meta.json`
(`jetson_clocks` says the idle log was taken with clocks unlocked; the report, `STATUS.md` and `platform.json` say
locked) → correct it, or say which is true, in your next `STATUS.md` update — a metadata correction in its own
commit is not an overwrite of a result.
