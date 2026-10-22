# E0 platform record — T002 (Jetson AGX Thor, 2026-10-22)

## Answer
| quantity | value | n | source |
|---|---|---|---|
| power mode | MAXN, id 0 (the maximum-performance mode; boot default is 120W, id 1) | — | `results/20261022_E0_platform_MAXN/raw/cmd_nvpmodel_q_verbose.txt` |
| SM count | 20 | — | `gpu_properties.json` |
| locked GPU clock under load | 1,575 MHz (min = median = max) | 200 samples, 20 Hz, 10 s | `clock_probe.csv` |
| EMC clock | 4,266 MHz | — | `raw/cmd_emc_rate.txt` |
| idle module power (VIN) | mean 33.27 W, median 33.06, p5–p95 32.47–34.72 | 600 (60 s, 10 Hz) | `idle_power_summary.csv` |
| idle VDD_GPU / VDD_CPU_SOC_MSS / VIN_SYS_5V0 | 7.10 / 8.72 / 5.85 W (means) | 600 each | same |
| GPU rail limit | 5,060 mA warning (≈ 100 W at 19.76 V), 6,080 mA critical | — | `raw/power_limits.txt` |
| bf16 4096³ GEMM under the limit, 20 s | burst 99.5 TFLOP/s (first slice) → sustained median 93.97 TFLOP/s (−5.6 %), still falling at 20 s (93.3) | 39 slices of 0.5 s | `oc_throttle_probe_bf16.csv` |
| same, 60 s with temperatures | burst 98.6 → sustained median 93.06 (−5.6 %) → 91.9 at 60 s; gpu 57.5 → 80.5 °C; r(temperature, TFLOP/s) = −0.96 | 114 slices | `results/20261022_E0_oc_throttle/` |

Measured: everything in the table except the 100 W, which is derived (5.060 A × the 19.76 V idle bus voltage).
"Sustained" means the median of the slices from the first OC event onward, in every file (`bench/t002_finalise.py`).
Not measured here: peaks, ridge and counter validation (T003).

## The finding that matters for every later task: an over-current throttle devfreq does not show
- VDD_GPU's ina3221 channel has a warning limit of 5,060 mA. Under a sustained bf16 GEMM loop:
  - the rail plateaus at 99.7 W;
  - the soctherm `oc3` counter (throttle enabled) fires about 260 times per second (5,245 events in 20 s);
  - the GEMM loses 5.6 % of its throughput (sustained median vs the first slice), and keeps losing as the die warms:
    91.9 TFLOP/s at 60 s and 80 °C, with OC events per slice rising in step with temperature (r = +0.95).
- devfreq's `cur_freq` stays at 1,575 MHz throughout. So "clock at its locked value" (protocol §1 gate item 4) does
  **not** prove the GPU is unthrottled on this device.
- Consequence, applied from now on: every measured run records the soctherm OC counters before and after
  (`bench/gate.py: oc_counts()`, `throttle_verdict()`). Any event marks the run `throttled: true`.
- Model inference is much less power-dense than a GEMM loop, so it may never reach the limit. T004 measures that
  rather than assuming it.
- For T003's peaks this raises a definitional question, written to `STATUS.md`: is the roofline ceiling the burst
  rate or the sustained, current-limited rate? T003 records both either way.

## Conditions
- MAXN, `jetson_clocks` and `jetson_clocks --fan` (applied in this boot at 2026-10-22T03:26Z, verified by the gate).
- The protocol §1 gate passed before every step, including GPU utilisation 0 % from `nvidia-smi`, which is
  system-wide on Thor.
- The device is shared with other users. During the 60 s idle window utilisation was 0 % in 60 of 62 one-second
  samples and 1 % in 2. The idle numbers are clean.
- Temperatures before the idle window: tj 41.6 °C, gpu 41.6 °C, cpu 40.3 °C; fan PWM 255.

## Anomalies, confidence
- The OC counters were already nonzero before T002 (oc2 527, oc3 ≈ 49,000; the board has been up since 2026-09-18),
  likely from other users' work.
- oc3 rose by 5,386 between the idle window and the power-limit read; 5,385 of them were my own trial run of the OC
  probe (into `scratch/`, not a result). That leaves at most 1 event in or around the idle window. The idle
  `throttled: false` rests on that and on VDD_GPU staying ≤ 7.11 W; the counters were not read right at the
  window's end.
- Temperature was not logged during the clock probe or the 20 s OC probe (gate snapshots only); the 60 s re-run
  logs it per slice.
- tegrastats `--interval 100` delivered ~8.3 lines/s, not 10; the sysfs sampler ran at 10 Hz (600 rows).
- ncu's `device__attribute_max_gpu_frequency_khz` and `clock_rate` read 1,048,576 (2²⁰), a placeholder. Memory clock
  and bus width read 0. The clock comes from devfreq, ncu's measured SM frequency (1.5735 GHz,
  `results/20261022_E0_toolcheck`) and tegrastats (1,574 MHz).
- The ina3221 current resolution of 20 mA (≈ 0.40 W per VDD_GPU sample) limits single-sample power precision. Means
  over 600 samples are fine.
- High confidence in every value above. Each is read from this device, and the command is in `meta.json`.

## Reproduce
```bash
source env/thor_env.sh
~/venvs/base/bin/python bench/platform_record.py --out results/<date>_E0_platform_MAXN
~/venvs/base/bin/python bench/gpu_clock_probe.py --seconds 10 --hz 20 --out <dir>/clock_probe.csv
env/rootdo.sh tegrastats --interval 100 --logfile <dir>/raw/tegrastats_idle_60s.log &
~/venvs/base/bin/python bench/power_sample.py --seconds 60 --hz 10 --out <dir>/idle_power.csv; env/rootdo.sh tegrastats --stop
~/venvs/base/bin/python bench/oc_throttle_probe.py --seconds 20 --n 4096 --dtype bfloat16 --out <dir>/oc_throttle_probe_bf16.csv
```

## Deviations
- The task's "jetson_clocks --show before and after applying": clocks were already applied in this boot (T001). The
  record has `--show` before the gate (already locked) and after.
- Added beyond the task: `oc_throttle_probe_bf16.csv` and `raw/power_limits.txt`, because the clock probe revealed
  the OC events.

## Self-review (protocol §11)
| check | result |
|---|---|
| E0 directory in the maximum-performance mode with `jetson_clocks: true` | yes, `results/20261022_E0_platform_MAXN/` |
| locked frequency measured under load | yes, 1,575 MHz, constant (with the OC caveat above) |
| idle power per rail and as module power | yes, 60 s, 10 Hz, + tegrastats |
| static `hw_constants.json` | yes (measured fields left as `n/a: T003`) |
| `meta.json` complete, no null, raw files hashed | yes (32 raw files) |
| secrets scan | clean (42 files, incl. the HF token file contents) |
| sizes in `STATUS.md` | yes (updated at the close) |
| independent review (P0) | done; findings fixed: gate.py logic (locked-clock lookup, OC verdict when unreadable, full jetson_clocks check, idle threshold, unreadable sysfs), one sustained definition, per-field sources in `hw_constants.json`, per-sample CSVs hashed as raw, the finaliser in `bench/`, the 60 s temperature re-run, and the disclosures above |
