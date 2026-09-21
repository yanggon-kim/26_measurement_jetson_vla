# T013 — E0 again at MAXN, and the ncu clock-control check  · priority P0 · depends on: T001 · blocks: every timed or profiled run

Why: T001's platform record was taken in `MODE_30W` (8 of 16 SMs, GPU ≤ 612 MHz) because MAXN needs a reboot only
the human may make. Its findings are accepted (answers: `docs/04_research_answers.md`); this task produces the
authoritative record and settles what T001 had to leave open. It starts **after the human has rebooted into MAXN**
(protocol §1 gate). If `nvpmodel -q` still reports another mode, leave this task open, keep the note at the top of
`STATUS.md`, and do only preparation work (protocol §1 lists what is allowed; T002 steps 1–3 are).

1. Gate: `sudo nvpmodel -q` → `MAXN`; `sudo jetson_clocks` (and `--fan`) **before anything else**;
   `bench/gpu_clock_probe.py` → GPU `cur_freq` idle and under load. Record `sudo -n true` (is passwordless sudo
   configured now?).
2. Re-run the E0 collection into a **new** directory `results/<date>_E0_platform_maxn/` (never overwrite the
   `MODE_30W` one; reference it in `meta.json.notes`): `bench/collect_platform.py` (60 s idle, four rails from
   sysfs + tegrastats), `bench/verify_stack.py` with a **cold Inductor cache** (a fresh `TORCHINDUCTOR_CACHE_DIR`
   under `scratch/`) so that the `max-autotune` log lines are captured again, `bench/hw_constants.py`.
   Observe and record: `multi_processor_count` at MAXN (the 16 in T001 is derived), the GPU maximum and locked
   frequency, CPU cores online, EMC rate.
3. `max-autotune`: record whether "Not enough SMs to use max_autotune_gemm mode" still appears, and read the
   threshold from the installed source (the function that logs it, in `torch/_inductor/utils.py`; quote file, line,
   and the minimum it compares `multi_processor_count` with) into `stack_verification.json` / the report. The
   research side expects the minimum to be far above 16, i.e. the message to persist at MAXN (protocol §12).
4. Toy profiler captures again, with `jetson_clocks` applied **before** each and `cur_freq` recorded next to each:
   a. nsys toy (add `--capture-range-end=stop`);
   b. ncu toy with `--clock-control base` (the default) **and** with `--clock-control none`, otherwise identical
      (`--set full`, the `toy/` range, `--profile-from-start off`). For each: ncu's `SM Frequency`, run time, report
      size, both page exports, `ncu/metric_presence_<base|none>.csv` (same verdict scheme as T001).
      Write `ncu/clock_control_compare.csv`: `metric, value_base, value_none, ratio` for `SM Frequency`, `Duration`,
      the five occupancy metrics, `Compute (SM) Throughput`, `Memory Throughput`, `Mem Busy`, `L2 Hit Rate`, both
      IPCs, the pipe `_active` columns, all 18 stall ratios, and the sysmem-miss sector counts. State which value
      protocol §3.2's rule selects for later tasks (`none` if `SM Frequency` then equals the locked GPU frequency and
      the metrics are populated; else `base`), and what the 18 stall ratios sum to under each.
   c. the DRAM-counter probe once more at MAXN (`--query-metrics` count of `dram__*`, explicit `--metrics` request),
      so that "no DRAM counters" is established for the mode we measure in, not only for `MODE_30W`.
5. One correction to carry into the new directory: in `results/20260921_E0_platform/meta.json` the field
   `jetson_clocks` says `power_idle.csv` was taken with clocks **unlocked**, while `reports/E0_platform.md` §1.3 /
   §2, `STATUS.md` and `platform.json.device_state.jetson_clocks_show` (Min = Max = 1,728,000) say the committed idle
   log was taken with clocks **locked**. Say in the T013 report which is true for the committed `power_idle*.csv`
   (you may also correct that one string in the old `meta.json` in a separate commit that says so — a metadata
   correction is not an overwrite of a result), and make `bench/make_meta_e0.py` derive the field from what was
   recorded instead of a fixed sentence.

Not in this task (they belong to T005, where kernels with known traffic exist): validating the sysmem-miss
substitute on the write path and against the measured bandwidth peak; the rail double-count check.

Acceptance: `results/<date>_E0_platform_maxn/` complete per protocol §5 with `raw/` (toy `.nsys-rep`, both
`.ncu-rep`, logs, stdout) listed with hashes in `meta.json.raw_files`; `hw_constants.json` with the **observed** MAXN
SM count; `ncu/clock_control_compare.csv`; `reports/E0_platform_maxn.md` (short: what differs from the `MODE_30W`
record, the clock-control decision, the `max-autotune` finding, the idle power at MAXN per rail from sysfs);
`STATUS.md` updated (the "needs the human" block removed or rewritten to what is still needed); repository-size
lines (protocol §6f). Self-review + independent pass (protocol §11, P0).
