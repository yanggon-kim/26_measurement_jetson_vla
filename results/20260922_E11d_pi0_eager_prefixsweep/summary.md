# E11d — prefix vs input, eager (T010)

9 points: images {1,2,3} x prompt {short,typical,long}, N=10, MAXN.

**The prefix cost does not depend on the input.** vision 241.58-241.71 ms (1.0005x spread),
vlm_prefix 234.44-234.79 ms (1.0015x), t_e2e 1057.29-1065.79 ms (1.0080x) — while P_valid moves
262 -> 807, a factor of 3.1.

The input format explains it: `config.input_features` declares THREE image slots
(image, image2, empty_camera_0) with `empty_cameras=1`, and a camera that is not supplied is
synthesised as a -1-filled tensor with a zero mask and still encoded into 256 tokens;
`tokenizer_max_length=48` pads every prompt. So **P_processed = 3*256 + 48 = 816 always**.

`prefix_latency_fit` therefore has no slope against P, and `prefix_fit.csv` records that as the
result rather than fitting a line through a constant (protocol §13 asks for exactly this). The
one fit that exists, `vision_vs_images`, is a = 241.57 ms, b = 0.047 ms/image.

**Odd:** the third image slot of the 3-image points carries a copy of the second, because the
dataset supplies two cameras; what varies is how many slots arrive with a valid mask. Regions
are instrumented in eager only — `regions.py` graph-breaks a compiled model.
