# Reference scripts (read-only; research side)

Verbatim copies of three scripts from the research side's profiling toolkit, provided as **reference
implementations** for `docs/03_instruction_level_analysis.md` §11. They were written for desktop GPUs and have
**not been validated on the Orin**. The on-device agent does not run them from here: copy them into `bench/`,
adapt them (paths, the "Tensor core" opcode family, Orin column names, the §2 regions and warm-up rule of
`docs/02_protocol.md`), and commit the adapted versions there.

| file | becomes | note |
|---|---|---|
| `ncu_profile_workload.py` | `bench/ncu_target.py` | template: placeholder model, 2 warm-ups — replace with the real model and ≥ 20 warm-ups; the ncu commands in its docstring write to `profiles/`, here use `results/<run>/raw/` (committed, protocol §3.5) |
| `build_inst_inventory.py` | `bench/build_inst_inventory.py` | SASS → per-opcode / per-category / LSU-split inventory; `--inner-loop-range 0xLO-0xHI` |
| `crosscheck_sass.py` | `bench/crosscheck_sass.py` | expects the `parsed.json` schema of `docs/03` §11 item 2 (`pipe_utilization`, `pipe_lsu_split`) |

Comments inside these files mention other documents (`references/…`) that are not in this repository;
everything needed from them is in `docs/02_protocol.md` and `docs/03_instruction_level_analysis.md`. Hardware
numbers in comments refer to other GPUs (protocol §8).

**Exception — `check_model_inputs.py` is run in place, not copied:** it validates an E11 `model_inputs.csv`
against `docs/02_protocol.md` §13.8 (standard library only, no GPU):
`python docs/reference_scripts/check_model_inputs.py results/<run>/model_inputs.csv`. Exit status 0 is an acceptance
criterion of every E11 task. A quantity it does not know is proposed in `STATUS.md`, not added locally.
