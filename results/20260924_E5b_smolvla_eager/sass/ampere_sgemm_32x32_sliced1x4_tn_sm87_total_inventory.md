# SASS instruction inventory — `ampere_sgemm_32x32_sliced1x4_tn_sm87.sass`

Total instructions: **896** (file has 896 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 112 | 12.5% |
| FMA | 318 | 35.5% |
| ALU | 362 | 40.4% |
| Uniform | 1 | 0.1% |
| Branch | 78 | 8.7% |
| Other | 25 | 2.8% |
| **Total** | **896** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 76 | 67.9% | 8.5% |
| Store | 36 | 32.1% | 4.0% |
| **LSU total** | **112** | **100.0%** | **12.5%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.LTC128B.CONSTANT` | 34 |
| LSU | Store | `STS` | 33 |
| LSU | Load | `LDS.128` | 31 |
| LSU | Load | `LDS` | 4 |
| LSU | Load | `LDG.E.LTC128B.64.CONSTANT` | 3 |
| LSU | Load | `LDG.E.LTC128B.STRONG.GPU` | 2 |
| LSU | Load | `LDG.E.EL.LTC128B.STRONG.GPU` | 1 |
| LSU | Load | `MEMBAR.GPU` | 1 |
| LSU | Store | `STG.E.EF` | 1 |
| LSU | Store | `STG.E.EF.STRONG.GPU` | 1 |
| LSU | Store | `STS.128` | 1 |
| FMA |  | `FFMA` | 257 |
| FMA |  | `FMUL` | 32 |
| FMA |  | `IMAD.U32` | 13 |
| FMA |  | `IMAD.WIDE.U32` | 12 |
| FMA |  | `FADD` | 3 |
| FMA |  | `FSETP.NE.AND` | 1 |
| ALU |  | `LEA` | 69 |
| ALU |  | `LEA.HI.X` | 60 |
| ALU |  | `LOP3.LUT` | 49 |
| ALU |  | `MOV` | 46 |
| ALU |  | `ISETP.LT.AND` | 38 |
| ALU |  | `IADD3` | 21 |
| ALU |  | `R2P` | 11 |
| ALU |  | `ISETP.EQ.AND` | 10 |
| ALU |  | `SHF.R.U32.HI` | 10 |
| ALU |  | `SHF.L.U32` | 8 |
| ALU |  | `IADD3.X` | 6 |
| ALU |  | `PLOP3.LUT` | 6 |
| ALU |  | `SHF.R.S32.HI` | 6 |
| ALU |  | `BMSK` | 4 |
| ALU |  | `ISETP.NE.AND` | 4 |
| ALU |  | `P2R` | 3 |
| ALU |  | `ISETP.EQ.XOR` | 2 |
| ALU |  | `ISETP.GE.AND` | 2 |
| ALU |  | `ISETP.GT.AND` | 2 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `P2R.B1` | 1 |
| ALU |  | `P2R.B2` | 1 |
| ALU |  | `P2R.B3` | 1 |
| ALU |  | `SEL` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA.U` | 17 |
| Branch |  | `NOP` | 17 |
| Branch |  | `BRA` | 10 |
| Branch |  | `S2R` | 9 |
| Branch |  | `CALL.REL` | 8 |
| Branch |  | `CS2R` | 8 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 6 |
| Branch |  | `BAR.SYNC` | 1 |
| Branch |  | `EXIT` | 1 |
| Branch |  | `RET.ABS` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `LEPC` | 8 |
| Other |  | `BMOV.32` | 2 |
