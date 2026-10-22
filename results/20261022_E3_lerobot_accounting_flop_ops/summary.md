# LeRobot accounting pass with per-op FLOPs (T004 review follow-up) — 2026-10-22, eager_fp32, N = 10

Totals compared with the E3 runs' `accounting.json` (bench a32aa34, before `flop_ops` existed):

| model | region | FLOPs here | FLOPs in E3 | equal | ops counted |
|---|---|---|---|---|---|
| pi0 | vision | 660529741824 | 660529741824 | True | aten._scaled_dot_product_efficient_attention, aten.addmm, aten.convolution |
| pi0 | vlm_prefix | 3332496371712 | 3332496371712 | True | aten.bmm, aten.mm |
| pi0 | ae_step[0] | 38606828288 | 38606828288 | True | aten.addmm, aten.bmm, aten.mm |
| pi05 | vision | 660529741824 | 660529741824 | True | aten._scaled_dot_product_efficient_attention, aten.addmm, aten.convolution |
| pi05 | vlm_prefix | 3974951585792 | 3974951585792 | True | aten.bmm, aten.mm |
| pi05 | ae_step[0] | 38891762176 | 38891762176 | True | aten.addmm, aten.bmm, aten.mm |
| smolvla | vision | 430637580288 | 430637580288 | True | aten._scaled_dot_product_flash_attention, aten.addmm, aten.convolution, aten.mm |
| smolvla | vlm_prefix | 57604239360 | 57604239360 | True | aten.bmm, aten.mm |
| smolvla | ae_step[0] | 11019033600 | 11019033600 | True | aten.addmm, aten.bmm, aten.mm |
