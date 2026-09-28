# SASS instruction inventory — `triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_3_sm87.sass`

Total instructions: **1608** (file has 1608 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 142 | 8.8% |
| FMA | 661 | 41.1% |
| ALU | 600 | 37.3% |
| XU | 16 | 1.0% |
| Uniform | 2 | 0.1% |
| Branch | 162 | 10.1% |
| Other | 25 | 1.6% |
| **Total** | **1608** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 90 | 63.4% | 5.6% |
| Store | 52 | 36.6% | 3.2% |
| **LSU total** | **142** | **100.0%** | **8.8%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDL` | 48 |
| LSU | Store | `STL` | 32 |
| LSU | Load | `LDG.E.CONSTANT` | 16 |
| LSU | Store | `STS` | 16 |
| LSU | Load | `LDG.E.EL` | 8 |
| LSU | Load | `LDS` | 8 |
| LSU | Load | `LDG.E.EL.128` | 6 |
| LSU | Load | `LDS.128` | 4 |
| LSU | Store | `STG.E.128` | 2 |
| LSU | Store | `STS.128` | 2 |
| FMA |  | `IMAD.MOV.U32` | 276 |
| FMA |  | `FFMA.FTZ` | 120 |
| FMA |  | `IMAD.IADD` | 33 |
| FMA |  | `FMUL.FTZ` | 32 |
| FMA |  | `FSEL` | 32 |
| FMA |  | `IMAD.SHL.U32` | 32 |
| FMA |  | `IMAD.MOV` | 22 |
| FMA |  | `FMUL` | 16 |
| FMA |  | `IMAD.WIDE.U32` | 16 |
| FMA |  | `IMAD` | 15 |
| FMA |  | `IMAD.WIDE` | 14 |
| FMA |  | `IMAD.X` | 13 |
| FMA |  | `FADD` | 8 |
| FMA |  | `FADD.FTZ` | 8 |
| FMA |  | `FFMA` | 8 |
| FMA |  | `FSETP.GE.AND` | 8 |
| FMA |  | `FSETP.NEU.AND` | 8 |
| ALU |  | `LOP3.LUT` | 144 |
| ALU |  | `IADD3` | 78 |
| ALU |  | `SHF.L.W.U32.HI` | 48 |
| ALU |  | `SHF.R.U32.HI` | 48 |
| ALU |  | `ISETP.GE.AND` | 35 |
| ALU |  | `LEA.HI` | 30 |
| ALU |  | `SHF.R.S32.HI` | 25 |
| ALU |  | `PRMT` | 24 |
| ALU |  | `IADD3.X` | 19 |
| ALU |  | `ISETP.NE.U32.AND` | 17 |
| ALU |  | `F2F.F32.F64` | 16 |
| ALU |  | `I2F.F64.S64` | 16 |
| ALU |  | `ISETP.NE.AND.EX` | 16 |
| ALU |  | `MOV` | 16 |
| ALU |  | `ISETP.NE.AND` | 13 |
| ALU |  | `P2R` | 13 |
| ALU |  | `ISETP.LT.AND` | 12 |
| ALU |  | `F2I.FTZ.NTZ` | 8 |
| ALU |  | `R2P` | 8 |
| ALU |  | `ISETP.LT.U32.AND` | 6 |
| ALU |  | `SHF.L.U32` | 3 |
| ALU |  | `ISETP.GT.AND` | 2 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `LEA` | 1 |
| ALU |  | `SGXT` | 1 |
| XU |  | `DMUL` | 16 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 50 |
| Branch |  | `R2UR` | 28 |
| Branch |  | `CS2R` | 20 |
| Branch |  | `BSSY` | 16 |
| Branch |  | `BSYNC` | 16 |
| Branch |  | `NOP` | 13 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 11 |
| Branch |  | `S2R` | 6 |
| Branch |  | `EXIT` | 2 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `I2FP.F32.S32` | 8 |
| Other |  | `BMOV.32` | 2 |
