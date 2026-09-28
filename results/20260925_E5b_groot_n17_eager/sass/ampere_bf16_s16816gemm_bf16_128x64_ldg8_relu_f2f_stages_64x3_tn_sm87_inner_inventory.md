<!-- inner loop: backward BRA 0x85bbd920 -> 0x85bbd050: of the 6 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x3_tn_sm87.sass` (PC range 0x85bbd050-0x85bbd920)

Total instructions in PC range: **140**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 37 | 26.4% |
| Tensor core | 64 | 45.7% |
| ALU | 31 | 22.1% |
| Uniform | 5 | 3.6% |
| Branch | 3 | 2.1% |
| **Total** | **140** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 37 | 100.0% | 26.4% |
| **LSU total** | **37** | **100.0%** | **26.4%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDSM.16.M88.4` | 24 |
| LSU | Load | `LDGSTS.E.BYPASS.LTC128B.128.CONSTANT` | 12 |
| LSU | Load | `LDGDEPBAR` | 1 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 64 |
| ALU |  | `IADD3` | 13 |
| ALU |  | `IADD3.X` | 12 |
| ALU |  | `R2P` | 3 |
| ALU |  | `ISETP.GT.AND` | 2 |
| ALU |  | `MOV` | 1 |
| Uniform |  | `UIADD3` | 2 |
| Uniform |  | `UISETP.LT.AND` | 2 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 1 |
| Branch |  | `BRA.U` | 1 |
| Branch |  | `DEPBAR.LE` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 64 ÷ 12 = 5.333. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 37 : 0
