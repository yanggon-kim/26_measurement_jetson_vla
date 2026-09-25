# E12b M2 — pipeline equivalence, compiled path, whole GPU

Variant `bf16_both+hoist+owngraph_expert+dbuf_kv+prefix_first`. Through the timing harness itself: the same captured graphs run serially (frame k complete before k+1) and pipelined (prefix(k) with expert(k-1)); 6 frames cycling the 5 LIBERO observations, each with its own noise seed.

* pipelined vs serial: **bitwise identical on every frame**
* stale control (expert reads the other KV buffer = another frame's cache): **5 of 5 detected**
* power: consecutive serial frames' actions differ by up to 5.4077 (max abs)
* non-finite elements in any compared action: 0

Rows: `equivalence.csv`.
