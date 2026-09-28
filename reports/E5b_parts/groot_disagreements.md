## Two disagreements with the task text, recorded rather than assumed away

* **Block count.** T009 says "action head = 16 DiT blocks". The loaded N1.7 has **32**
  `transformer_blocks` (`shapes.json` → `n_blocks`). Block 0 is a cross-attention block
  (`attn1.to_k` takes 2048-wide backbone features), consistent with alternating blocks.
* **S.** T009 says S = 41; the loaded config's `action_horizon` is **40**. The sweep uses
  T009's 41. The 40-vs-41 difference falls inside the flat region of every curve above, so it
  changes no conclusion. The extra token is plausibly a state token; that is not verified.

Both are raised in `STATUS.md` under questions for the research side.
