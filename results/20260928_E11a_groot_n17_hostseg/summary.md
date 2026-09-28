# E11a - groot_n17 DiT host segments (T012 step 1)

Run directory: `20260928_E11a_groot_n17_hostseg`  
Written by `bench/groot_segments.py` at 2026-09-28T22:44:05Z.

## What

4 DiT blocks (0=cross_attn, 1=self_attn, 16=cross_attn, 17=self_attn) x 3 exec styles, every segment timed four ways (wall_isolated gap 0 / 2 ms, gpu_event, wall_launch; n = 1000); host sequence + whole-layer graph per block; 8 per-step extras. P = 273, S = 41, N = 4.

## Host layer

* `layer=0;kind=cross_attn`: eager sequence 0.723025 ms; compiled sequence 0.575507 ms; cudagraph whole_layer_graph 0.189244 ms; eager sum_of_segments 0.822078 ms; compiled sum_of_segments 1.156119 ms; cudagraph sum_of_segments 0.267931 ms
* `layer=1;kind=self_attn`: eager sequence 0.722225 ms; compiled sequence 0.575972 ms; cudagraph whole_layer_graph 0.188636 ms; eager sum_of_segments 0.827472 ms; compiled sum_of_segments 1.157528 ms; cudagraph sum_of_segments 0.267946 ms
* `layer=16;kind=cross_attn`: eager sequence 0.719857 ms; compiled sequence 0.570772 ms; cudagraph whole_layer_graph 0.188173 ms; eager sum_of_segments 0.823472 ms; compiled sum_of_segments 1.150602 ms; cudagraph sum_of_segments 0.270364 ms
* `layer=17;kind=self_attn`: eager sequence 0.71837 ms; compiled sequence 0.569269 ms; cudagraph whole_layer_graph 0.189676 ms; eager sum_of_segments 0.820511 ms; compiled sum_of_segments 1.150377 ms; cudagraph sum_of_segments 0.268379 ms

## Files

segments.csv, segment_summary.csv, host_layer_summary.csv, kernels_per_call.csv (count pass), nsys/ (host sequences), raw/segment_samples.csv.zst, raw/profiler/, model_inputs.csv

## Not measured

Every quantity this part is responsible for is present in `model_inputs.csv`.
