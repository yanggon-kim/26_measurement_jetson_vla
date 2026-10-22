Superseded by `results/20261022_E2_ae_kernels_v3/` (T003 review): the cold replay for q / o (24 weight copies, 100 calls)
wrapped after 4 calls and reused ws[0..3] within ~16 MiB, under the L2 size (≈ 3 % bias for those two), and
`kernels.csv`'s `floor_ms` counted activations as DRAM bytes (`kernels_floors.csv` corrected that). Two statements in
this directory's summary are wrong and corrected in v3: the graph-replay saving for projections is −0.04 to 1.41 µs
(not 0.5–1.2), and cuBLAS's tile is 256×8 at S ≤ 8, 448×64 at 128 (not "256×56 at S ≤ 51"). The run itself was in the
normal memory-path state and agrees with v3 within 9 % (q 19.7 vs 20.1, gate 48.5 vs 49.6, down 38.0 vs 36.4 µs).
