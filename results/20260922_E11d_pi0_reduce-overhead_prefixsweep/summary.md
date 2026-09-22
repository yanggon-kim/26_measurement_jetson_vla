# E11d — prefix vs input, reduce-overhead (T010)

The same 9 points compiled. `t_e2e` 466.47-467.36 ms, spread **1.0019x** — flatter still than
eager, and 2.27x faster (eager 1057-1066 ms).

P_processed is constant at 816 for the same reason as the eager run (three encoded image slots,
prompt padded to 48), so there is no slope to fit against P.

Only `t_e2e` is reported: `bench/regions.py`'s context managers force a graph break under
`torch.compile`, so an instrumented compiled run would not measure the graph the model actually
executes. Compile time per input shape is in `meta.json.first_call_seconds_per_shape`.
