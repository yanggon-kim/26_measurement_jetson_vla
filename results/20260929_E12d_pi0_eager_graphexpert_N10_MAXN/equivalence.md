# E12d arm (b) — equivalence, eager expert as one CUDA graph

Variant `eager+pi0_nosync+hoist+owngraph_expert_eager+dbuf_kv+prefix_eager+prefix_first`.

* pipelined vs serial through the harness: **bitwise identical on every frame**
* stale control (expert reads the other KV buffer): **5 of 5 detected**
* graph expert vs the plain eager serial path (corun.run_serial, no graph), 6 frames: **bitwise identical**
* power: consecutive serial frames differ by up to 5.4133
* non-finite elements in compared actions: 0

Rows: `equivalence.csv`.
