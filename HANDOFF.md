# HANDOFF — jetson, branch `thor`

*Research-side state for the Thor campaign. Rewritten, never appended.*

## State now
- Branch created 2026-10-21 from nothing (orphan): instructions, task list T001–T015, empty device tree, Orin
  reference snapshot. No device has pulled it yet; `PROGRESS.md` is all `todo`.
- The move back to the VLA line is **not decided**; this branch only prepares the measurements.

## Open issues
- Every Thor stack fact is a hint (JetPack 7 / CUDA 13, `sm_110`, ~20 SMs, Python 3.12, SBSA wheels, rail names,
  ncu permission, `dram__` counters); T001 records them.
- Expected human items (`docs/05_setup.md` §3): H1 sudoers, H2 Hugging Face + Gemma licence (only if the ungated
  tokenizer route fails), H3 deploy key, H5 maximum-performance mode; H6 / H7 optional.

## Next steps
1. When the device pushes: read `PROGRESS.md`, the head of `STATUS.md` ("Needs the human"), and T001's `env/README.md`.
2. Correct any task text T001's facts contradict (new commit on `thor`), answer questions in `docs/04`.
3. Review T005's `best_bf16.json` and T006's roofline CSVs first: they are the deliverables everything else rests on.
