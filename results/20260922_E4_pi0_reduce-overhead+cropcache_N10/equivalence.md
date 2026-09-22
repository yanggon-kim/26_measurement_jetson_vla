# cropcache — action equivalence against the unmodified model

A faster model that computes something different is not a result. Checked before the
`cropcache` numbers were reported.

Method: eager (no compilation, so nothing else differs), `num_steps = 10`, **fixed noise**
(`torch.Generator(device="cuda").manual_seed(1234)`) so the two runs are deterministic and
comparable, `sample_actions` called directly. One run with the stock
`copy.deepcopy(past_key_values)`, one inside `crop_instead_of_deepcopy()`.

| check | result |
|---|---|
| output shape / dtype | `(1, 50, 32)` float32, both |
| **bitwise identical** | **True** |
| max absolute difference | **0.000e+00** |
| max relative difference | 0.000e+00 |
| both finite | True |
| control: reference vs itself | identical, 0.000e+00 (confirms the harness is deterministic) |

So cropping the cache back to P after the expert's forward is **exactly** equivalent to giving
each denoising step a private deep copy — not approximately, bitwise. The 15.04 MB clone buys
isolation that a slice provides for free.

Command: see `reports/E4_pi0_compile.md` §3.3; the script is inline in that section's
reproduce block and uses `bench/run_e4.py:crop_instead_of_deepcopy`.
