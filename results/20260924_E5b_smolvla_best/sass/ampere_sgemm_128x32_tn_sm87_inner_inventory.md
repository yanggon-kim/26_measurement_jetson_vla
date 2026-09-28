<!-- inner loop: backward BRA 0x4f646410 -> 0x4f6456b0: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `ampere_sgemm_128x32_tn_sm87.sass` (PC range 0x4f6456b0-0x4f646410)

Total instructions in PC range: **215**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 38 | 17.7% |
| FMA | 128 | 59.5% |
| ALU | 42 | 19.5% |
| Branch | 7 | 3.3% |
| **Total** | **215** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 28 | 73.7% | 13.0% |
| Store | 10 | 26.3% | 4.7% |
| **LSU total** | **38** | **100.0%** | **17.7%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDS.128` | 18 |
| LSU | Load | `LDG.E.LTC128B.CONSTANT` | 10 |
| LSU | Store | `STS` | 10 |
| FMA |  | `FFMA` | 128 |
| ALU |  | `LOP3.LUT` | 10 |
| ALU |  | `IADD3` | 9 |
| ALU |  | `IADD3.X` | 8 |
| ALU |  | `R2P` | 7 |
| ALU |  | `MOV` | 5 |
| ALU |  | `ISETP.GT.AND` | 2 |
| ALU |  | `ISETP.GE.AND` | 1 |
| Branch |  | `BRA.U` | 3 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 2 |
| Branch |  | `NOP` | 1 |
| Branch |  | `S2R` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 128 ÷ 10 = 12.8. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 28 : 10
