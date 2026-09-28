<!-- inner loop: backward BRA 0x4f68dd70 -> 0x4f68b2d0: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `ampere_sgemm_128x64_tn_sm87.sass` (PC range 0x4f68b2d0-0x4f68dd70)

Total instructions in PC range: **683**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 84 | 12.3% |
| FMA | 512 | 75.0% |
| ALU | 80 | 11.7% |
| Branch | 7 | 1.0% |
| **Total** | **683** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 60 | 71.4% | 8.8% |
| Store | 24 | 28.6% | 3.5% |
| **LSU total** | **84** | **100.0%** | **12.3%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDS.128` | 36 |
| LSU | Load | `LDG.E.LTC128B.CONSTANT` | 24 |
| LSU | Store | `STS` | 24 |
| FMA |  | `FFMA` | 512 |
| ALU |  | `IADD3` | 30 |
| ALU |  | `IADD3.X` | 22 |
| ALU |  | `MOV` | 12 |
| ALU |  | `R2P` | 9 |
| ALU |  | `LOP3.LUT` | 4 |
| ALU |  | `ISETP.GT.AND` | 2 |
| ALU |  | `ISETP.GE.AND` | 1 |
| Branch |  | `BRA.U` | 3 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 2 |
| Branch |  | `NOP` | 1 |
| Branch |  | `S2R` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 512 ÷ 24 = 21.333. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 60 : 24
