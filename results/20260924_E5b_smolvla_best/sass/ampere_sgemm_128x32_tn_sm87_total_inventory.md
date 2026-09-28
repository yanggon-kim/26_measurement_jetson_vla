# SASS instruction inventory — `ampere_sgemm_128x32_tn_sm87.sass`

Total instructions: **664** (file has 664 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 74 | 11.1% |
| FMA | 175 | 26.4% |
| ALU | 321 | 48.3% |
| Uniform | 1 | 0.2% |
| Branch | 72 | 10.8% |
| Other | 21 | 3.2% |
| **Total** | **664** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 57 | 77.0% | 8.6% |
| Store | 17 | 23.0% | 2.6% |
| **LSU total** | **74** | **100.0%** | **11.1%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDS.128` | 22 |
| LSU | Load | `LDG.E.LTC128B.CONSTANT` | 17 |
| LSU | Store | `STS` | 11 |
| LSU | Load | `LDG.E.LTC128B.STRONG.GPU` | 8 |
| LSU | Load | `LDG.E.LTC128B.64.CONSTANT` | 4 |
| LSU | Load | `LDS` | 4 |
| LSU | Store | `STG.E.EF` | 4 |
| LSU | Load | `LDG.E.EL.LTC128B.STRONG.GPU` | 1 |
| LSU | Load | `MEMBAR.GPU` | 1 |
| LSU | Store | `STG.E.EF.STRONG.GPU` | 1 |
| LSU | Store | `STS.128` | 1 |
| FMA |  | `FFMA` | 132 |
| FMA |  | `FMUL` | 16 |
| FMA |  | `IMAD.WIDE.U32` | 14 |
| FMA |  | `IMAD.U32` | 12 |
| FMA |  | `FSETP.NE.AND` | 1 |
| ALU |  | `LOP3.LUT` | 52 |
| ALU |  | `LEA` | 47 |
| ALU |  | `MOV` | 45 |
| ALU |  | `LEA.HI.X` | 34 |
| ALU |  | `IADD3` | 27 |
| ALU |  | `ISETP.LT.AND` | 24 |
| ALU |  | `IADD3.X` | 17 |
| ALU |  | `R2P` | 13 |
| ALU |  | `ISETP.EQ.AND` | 11 |
| ALU |  | `SHF.R.U32.HI` | 11 |
| ALU |  | `SHF.L.U32` | 9 |
| ALU |  | `PLOP3.LUT` | 6 |
| ALU |  | `SHF.R.S32.HI` | 5 |
| ALU |  | `BMSK` | 4 |
| ALU |  | `ISETP.NE.AND` | 4 |
| ALU |  | `ISETP.GT.AND` | 3 |
| ALU |  | `P2R` | 3 |
| ALU |  | `ISETP.EQ.XOR` | 2 |
| ALU |  | `ISETP.GE.AND` | 2 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `P2R.B1` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA.U` | 20 |
| Branch |  | `NOP` | 20 |
| Branch |  | `BRA` | 10 |
| Branch |  | `S2R` | 9 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 6 |
| Branch |  | `CALL.REL` | 4 |
| Branch |  | `BAR.SYNC` | 1 |
| Branch |  | `EXIT` | 1 |
| Branch |  | `RET.ABS` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `LEPC` | 4 |
| Other |  | `BMOV.32` | 2 |
