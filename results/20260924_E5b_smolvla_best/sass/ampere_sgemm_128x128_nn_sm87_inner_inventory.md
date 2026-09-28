<!-- inner loop: backward BRA 0x4f6d25a0 -> 0x4f6cfe00: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `ampere_sgemm_128x128_nn_sm87.sass` (PC range 0x4f6cfe00-0x4f6d25a0)

Total instructions in PC range: **635**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 68 | 10.7% |
| FMA | 512 | 80.6% |
| ALU | 48 | 7.6% |
| Branch | 7 | 1.1% |
| **Total** | **635** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 52 | 76.5% | 8.2% |
| Store | 16 | 23.5% | 2.5% |
| **LSU total** | **68** | **100.0%** | **10.7%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDS.128` | 36 |
| LSU | Load | `LDG.E.LTC128B.CONSTANT` | 16 |
| LSU | Store | `STS` | 16 |
| FMA |  | `FFMA` | 512 |
| ALU |  | `IADD3` | 17 |
| ALU |  | `IADD3.X` | 8 |
| ALU |  | `MOV` | 8 |
| ALU |  | `R2P` | 7 |
| ALU |  | `LOP3.LUT` | 4 |
| ALU |  | `ISETP.GE.AND` | 1 |
| ALU |  | `ISETP.GT.AND` | 1 |
| ALU |  | `ISETP.LT.AND` | 1 |
| ALU |  | `SHF.R.U32.HI` | 1 |
| Branch |  | `BRA.U` | 3 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 2 |
| Branch |  | `NOP` | 1 |
| Branch |  | `S2R` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 512 ÷ 16 = 32.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 52 : 16
