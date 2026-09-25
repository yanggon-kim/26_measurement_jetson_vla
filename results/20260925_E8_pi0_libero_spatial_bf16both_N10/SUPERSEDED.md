# SUPERSEDED — do not use these success rates

This run was produced by `bench/libero_eval.py` **before** it settled the scene after each reset.
LeRobot's own LIBERO environment steps the simulator with a no-op action 10 times after reset "so
everything settles" (`lerobot/envs/libero.py:305–308`, `num_steps_wait=10`); this harness acted from
the first frame, while objects were still dropping into place. The defect showed most clearly on
`libero_object` (0 of 88 episodes, run stopped and not committed), and it applies here too.

Replaced by `results/20260925_E8_pi0_libero_spatial_*_N10_settle10/`, same arms, same seeds, harness
fixed. The data here is kept as measured, for comparison, not as a result.
