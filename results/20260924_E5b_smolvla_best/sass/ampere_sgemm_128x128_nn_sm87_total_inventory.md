# SASS instruction inventory — `ampere_sgemm_128x128_nn_sm87.sass`

Total instructions: **1232** (file has 1232 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 136 | 11.0% |
| FMA | 611 | 49.6% |
| ALU | 380 | 30.8% |
| Uniform | 1 | 0.1% |
| Branch | 79 | 6.4% |
| Other | 25 | 2.0% |
| **Total** | **1232** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 108 | 79.4% | 8.8% |
| Store | 28 | 20.6% | 2.3% |
| **LSU total** | **136** | **100.0%** | **11.0%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDS.128` | 52 |
| LSU | Load | `LDG.E.LTC128B.CONSTANT` | 26 |
| LSU | Store | `STS` | 17 |
| LSU | Load | `LDG.E.LTC128B.STRONG.GPU` | 16 |
| LSU | Load | `LDS` | 8 |
| LSU | Store | `STG.E.EF` | 8 |
| LSU | Load | `LDG.E.LTC128B.64.CONSTANT` | 4 |
| LSU | Store | `STS.128` | 2 |
| LSU | Load | `LDG.E.EL.LTC128B.STRONG.GPU` | 1 |
| LSU | Load | `MEMBAR.GPU` | 1 |
| LSU | Store | `STG.E.EF.STRONG.GPU` | 1 |
| FMA |  | `FFMA` | 520 |
| FMA |  | `FMUL` | 64 |
| FMA |  | `IMAD.WIDE.U32` | 14 |
| FMA |  | `IMAD.U32` | 12 |
| FMA |  | `FSETP.NE.AND` | 1 |
| ALU |  | `MOV` | 56 |
| ALU |  | `LEA` | 54 |
| ALU |  | `LOP3.LUT` | 46 |
| ALU |  | `ISETP.LT.AND` | 44 |
| ALU |  | `LEA.HI.X` | 42 |
| ALU |  | `IADD3` | 39 |
| ALU |  | `R2P` | 18 |
| ALU |  | `IADD3.X` | 17 |
| ALU |  | `SHF.R.U32.HI` | 12 |
| ALU |  | `ISETP.EQ.AND` | 11 |
| ALU |  | `SHF.L.U32` | 9 |
| ALU |  | `PLOP3.LUT` | 6 |
| ALU |  | `SHF.R.S32.HI` | 5 |
| ALU |  | `BMSK` | 4 |
| ALU |  | `ISETP.NE.AND` | 4 |
| ALU |  | `P2R` | 3 |
| ALU |  | `P2R.B1` | 3 |
| ALU |  | `ISETP.EQ.XOR` | 2 |
| ALU |  | `ISETP.GE.AND` | 2 |
| ALU |  | `ISETP.GT.AND` | 2 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 23 |
| Branch |  | `BRA.U` | 20 |
| Branch |  | `BRA` | 10 |
| Branch |  | `S2R` | 9 |
| Branch |  | `CALL.REL` | 8 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 6 |
| Branch |  | `BAR.SYNC` | 1 |
| Branch |  | `EXIT` | 1 |
| Branch |  | `RET.ABS` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `LEPC` | 8 |
| Other |  | `BMOV.32` | 2 |
