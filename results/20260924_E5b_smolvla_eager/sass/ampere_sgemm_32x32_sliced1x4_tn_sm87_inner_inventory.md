<!-- inner loop: backward BRA 0x4cb97ce0 -> 0x4cb96620: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `ampere_sgemm_32x32_sliced1x4_tn_sm87.sass` (PC range 0x4cb96620-0x4cb97ce0)

Total instructions in PC range: **365**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 56 | 15.3% |
| FMA | 256 | 70.1% |
| ALU | 48 | 13.2% |
| Branch | 5 | 1.4% |
| **Total** | **365** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 40 | 71.4% | 11.0% |
| Store | 16 | 28.6% | 4.4% |
| **LSU total** | **56** | **100.0%** | **15.3%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDS.128` | 24 |
| LSU | Load | `LDG.E.LTC128B.CONSTANT` | 16 |
| LSU | Store | `STS` | 16 |
| FMA |  | `FFMA` | 256 |
| ALU |  | `LEA` | 15 |
| ALU |  | `LEA.HI.X` | 15 |
| ALU |  | `LOP3.LUT` | 8 |
| ALU |  | `R2P` | 4 |
| ALU |  | `IADD3` | 2 |
| ALU |  | `IADD3.X` | 1 |
| ALU |  | `ISETP.GE.AND` | 1 |
| ALU |  | `ISETP.GT.AND` | 1 |
| ALU |  | `ISETP.LT.AND` | 1 |
| Branch |  | `BRA.U` | 3 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 1 |
| Branch |  | `S2R` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 256 ÷ 16 = 16.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 40 : 16
