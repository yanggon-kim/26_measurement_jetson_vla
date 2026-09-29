# E12d arm (c) — equivalence, two processes

The expert process's actions for 6 frames against a single-process serial run of the full model (corun.run_serial), same frames and seeds.

* pipeline vs single-process serial: **bitwise identical on every frame**
* stale control (expert reuses frame k-1's cache): **5 of 5 detected**
* non-finite action elements over every condition: 0

Rows: `equivalence.csv`.
