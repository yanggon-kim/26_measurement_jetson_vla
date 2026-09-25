# E12b M3 — pipeline equivalence per split (graphs_both, captured in-partition)

Pipelined vs serial per frame through the timing harness; stale control = the expert reads the other KV buffer (another frame's cache).

* **full_gpu**: PASS (bitwise, stale control detected)
* **12/4**: PASS (bitwise, stale control detected)
* **10/6**: PASS (bitwise, stale control detected)
* **8/8**: PASS (bitwise, stale control detected)
* **6/10**: PASS (bitwise, stale control detected)
* **14/2**: PASS (bitwise, stale control detected)

Rows: `equivalence.csv`.
