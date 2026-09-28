<!-- inner loop: backward BRA 0x4f553180 -> 0x4f552c40: of the 6 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn_sm87.sass` (PC range 0x4f552c40-0x4f553180)

Total instructions in PC range: **83**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 21 | 25.3% |
| Tensor core | 32 | 38.6% |
| ALU | 22 | 26.5% |
| Uniform | 5 | 6.0% |
| Branch | 3 | 3.6% |
| **Total** | **83** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 21 | 100.0% | 25.3% |
| **LSU total** | **21** | **100.0%** | **25.3%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDSM.16.M88.4` | 12 |
| LSU | Load | `LDGSTS.E.BYPASS.LTC128B.128.CONSTANT` | 8 |
| LSU | Load | `LDGDEPBAR` | 1 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 32 |
| ALU |  | `IADD3` | 9 |
| ALU |  | `IADD3.X` | 8 |
| ALU |  | `ISETP.GT.AND` | 2 |
| ALU |  | `R2P` | 2 |
| ALU |  | `MOV` | 1 |
| Uniform |  | `UIADD3` | 2 |
| Uniform |  | `UISETP.LT.AND` | 2 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 1 |
| Branch |  | `BRA.U` | 1 |
| Branch |  | `DEPBAR.LE` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 32 ÷ 8 = 4.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 21 : 0
