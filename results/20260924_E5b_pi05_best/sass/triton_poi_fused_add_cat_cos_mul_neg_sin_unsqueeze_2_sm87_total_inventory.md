# SASS instruction inventory — `triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_2_sm87.sass`

Total instructions: **1608** (file has 1608 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 146 | 9.1% |
| FMA | 660 | 41.0% |
| ALU | 603 | 37.5% |
| XU | 16 | 1.0% |
| Uniform | 2 | 0.1% |
| Branch | 156 | 9.7% |
| Other | 25 | 1.6% |
| **Total** | **1608** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 88 | 60.3% | 5.5% |
| Store | 58 | 39.7% | 3.6% |
| **LSU total** | **146** | **100.0%** | **9.1%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDL` | 48 |
| LSU | Store | `STL` | 32 |
| LSU | Store | `STS` | 24 |
| LSU | Load | `LDG.E.CONSTANT` | 16 |
| LSU | Load | `LDG.E.EL.128` | 8 |
| LSU | Load | `LDS` | 8 |
| LSU | Load | `LDG.E.EL.64` | 4 |
| LSU | Load | `LDS.128` | 4 |
| LSU | Store | `STG.E.128` | 2 |
| FMA |  | `IMAD.MOV.U32` | 279 |
| FMA |  | `FFMA.FTZ` | 120 |
| FMA |  | `FSEL` | 40 |
| FMA |  | `IMAD.SHL.U32` | 39 |
| FMA |  | `FMUL.FTZ` | 32 |
| FMA |  | `IMAD.IADD` | 24 |
| FMA |  | `FMUL` | 16 |
| FMA |  | `IMAD.MOV` | 16 |
| FMA |  | `IMAD.WIDE.U32` | 16 |
| FMA |  | `IMAD.X` | 16 |
| FMA |  | `IMAD` | 14 |
| FMA |  | `FADD` | 8 |
| FMA |  | `FADD.FTZ` | 8 |
| FMA |  | `FFMA` | 8 |
| FMA |  | `FSETP.GE.AND` | 8 |
| FMA |  | `FSETP.NEU.AND` | 8 |
| FMA |  | `IMAD.WIDE` | 8 |
| ALU |  | `LOP3.LUT` | 151 |
| ALU |  | `IADD3` | 83 |
| ALU |  | `SHF.R.U32.HI` | 60 |
| ALU |  | `SHF.L.W.U32.HI` | 48 |
| ALU |  | `ISETP.GE.AND` | 33 |
| ALU |  | `LEA.HI` | 24 |
| ALU |  | `P2R` | 20 |
| ALU |  | `SHF.R.S32.HI` | 19 |
| ALU |  | `ISETP.NE.U32.AND` | 17 |
| ALU |  | `F2F.F32.F64` | 16 |
| ALU |  | `I2F.F64.S64` | 16 |
| ALU |  | `IADD3.X` | 16 |
| ALU |  | `ISETP.NE.AND.EX` | 16 |
| ALU |  | `ISETP.GE.U32.AND` | 13 |
| ALU |  | `ISETP.NE.AND` | 12 |
| ALU |  | `ISETP.LT.AND` | 10 |
| ALU |  | `F2I.FTZ.NTZ` | 8 |
| ALU |  | `LEA` | 8 |
| ALU |  | `R2P` | 8 |
| ALU |  | `LEA.HI.X` | 6 |
| ALU |  | `ISETP.GT.U32.AND` | 4 |
| ALU |  | `LEA.HI.X.SX32` | 4 |
| ALU |  | `SEL` | 4 |
| ALU |  | `ISETP.EQ.AND` | 2 |
| ALU |  | `ISETP.LT.U32.AND` | 2 |
| ALU |  | `SGXT.U32` | 2 |
| ALU |  | `SGXT` | 1 |
| XU |  | `DMUL` | 16 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 50 |
| Branch |  | `CS2R` | 26 |
| Branch |  | `R2UR` | 24 |
| Branch |  | `BSSY` | 16 |
| Branch |  | `BSYNC` | 16 |
| Branch |  | `NOP` | 14 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 5 |
| Branch |  | `S2R` | 3 |
| Branch |  | `EXIT` | 2 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `I2FP.F32.S32` | 8 |
| Other |  | `BMOV.32` | 2 |
