# SASS instruction inventory — `ampere_sgemm_128x128_tn_sm87.sass`

Total instructions: **1232** (file has 1232 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 135 | 11.0% |
| FMA | 608 | 49.4% |
| ALU | 392 | 31.8% |
| Uniform | 1 | 0.1% |
| Branch | 71 | 5.8% |
| Other | 25 | 2.0% |
| **Total** | **1232** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 107 | 79.3% | 8.7% |
| Store | 28 | 20.7% | 2.3% |
| **LSU total** | **135** | **100.0%** | **11.0%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDS.128` | 52 |
| LSU | Load | `LDG.E.LTC128B.CONSTANT` | 26 |
| LSU | Store | `STS` | 17 |
| LSU | Load | `LDG.E.LTC128B.STRONG.GPU` | 16 |
| LSU | Load | `LDS` | 8 |
| LSU | Store | `STG.E.EF` | 8 |
| LSU | Load | `LDG.E.LTC128B.64.CONSTANT` | 3 |
| LSU | Store | `STS.128` | 2 |
| LSU | Load | `LDG.E.EL.LTC128B.STRONG.GPU` | 1 |
| LSU | Load | `MEMBAR.GPU` | 1 |
| LSU | Store | `STG.E.EF.STRONG.GPU` | 1 |
| FMA |  | `FFMA` | 520 |
| FMA |  | `FMUL` | 64 |
| FMA |  | `IMAD.WIDE.U32` | 12 |
| FMA |  | `IMAD.U32` | 11 |
| FMA |  | `FSETP.NE.AND` | 1 |
| ALU |  | `LEA` | 70 |
| ALU |  | `MOV` | 63 |
| ALU |  | `LEA.HI.X` | 62 |
| ALU |  | `ISETP.LT.AND` | 44 |
| ALU |  | `LOP3.LUT` | 42 |
| ALU |  | `IADD3` | 25 |
| ALU |  | `R2P` | 17 |
| ALU |  | `ISETP.EQ.AND` | 10 |
| ALU |  | `SHF.L.U32` | 10 |
| ALU |  | `SHF.R.U32.HI` | 10 |
| ALU |  | `IADD3.X` | 6 |
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
| ALU |  | `SEL` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 18 |
| Branch |  | `BRA.U` | 17 |
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
