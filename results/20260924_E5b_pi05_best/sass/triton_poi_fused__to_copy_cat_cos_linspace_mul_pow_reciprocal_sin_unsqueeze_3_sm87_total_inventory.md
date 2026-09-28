# SASS instruction inventory — `triton_poi_fused__to_copy_cat_cos_linspace_mul_pow_reciprocal_sin_unsqueeze_3_sm87.sass`

Total instructions: **952** (file has 952 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 41 | 4.3% |
| FMA | 272 | 28.6% |
| ALU | 252 | 26.5% |
| XU | 236 | 24.8% |
| Uniform | 8 | 0.8% |
| Branch | 126 | 13.2% |
| Other | 17 | 1.8% |
| **Total** | **952** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 33 | 80.5% | 3.5% |
| Store | 8 | 19.5% | 0.8% |
| **LSU total** | **41** | **100.0%** | **4.3%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.64.CONSTANT` | 25 |
| LSU | Load | `LDL` | 4 |
| LSU | Store | `STL` | 4 |
| LSU | Load | `LDL.64` | 3 |
| LSU | Store | `STL.64` | 2 |
| LSU | Load | `LDG.E` | 1 |
| LSU | Store | `ST.E` | 1 |
| LSU | Store | `STG.E.64` | 1 |
| FMA |  | `IMAD.MOV.U32` | 147 |
| FMA |  | `FSEL` | 54 |
| FMA |  | `IMAD.SHL.U32` | 10 |
| FMA |  | `IMAD.X` | 10 |
| FMA |  | `IMAD` | 9 |
| FMA |  | `IMAD.WIDE.U32` | 8 |
| FMA |  | `IMAD.IADD` | 5 |
| FMA |  | `FFMA` | 4 |
| FMA |  | `FSETP.GEU.AND` | 4 |
| FMA |  | `FSETP.GT.AND` | 4 |
| FMA |  | `IMAD.HI.U32` | 4 |
| FMA |  | `IMAD.MOV` | 4 |
| FMA |  | `IMAD.U32` | 4 |
| FMA |  | `FSETP.NEU.AND` | 2 |
| FMA |  | `FADD.FTZ` | 1 |
| FMA |  | `FSETP.GTU.AND` | 1 |
| FMA |  | `IMAD.WIDE` | 1 |
| ALU |  | `LOP3.LUT` | 61 |
| ALU |  | `IADD3` | 41 |
| ALU |  | `ISETP.NE.AND` | 18 |
| ALU |  | `I2F.F64` | 12 |
| ALU |  | `MOV` | 12 |
| ALU |  | `SHF.R.U32.HI` | 12 |
| ALU |  | `ISETP.GE.AND` | 11 |
| ALU |  | `IADD3.X` | 10 |
| ALU |  | `SEL` | 10 |
| ALU |  | `SHF.R.U64` | 9 |
| ALU |  | `ISETP.GE.U32.AND` | 6 |
| ALU |  | `SHF.L.U64.HI` | 6 |
| ALU |  | `F2I.F64` | 4 |
| ALU |  | `ISETP.EQ.AND` | 4 |
| ALU |  | `ISETP.GT.U32.AND` | 4 |
| ALU |  | `ISETP.NE.OR` | 4 |
| ALU |  | `ISETP.NE.U32.AND` | 3 |
| ALU |  | `LEA.HI` | 3 |
| ALU |  | `SHF.L.U32` | 3 |
| ALU |  | `F2F.F32.F64` | 2 |
| ALU |  | `IMNMX` | 2 |
| ALU |  | `SHF.L.W.U32.HI` | 2 |
| ALU |  | `SHF.R.S32.HI` | 2 |
| ALU |  | `F2F.F64.F32` | 1 |
| ALU |  | `FLO.U32` | 1 |
| ALU |  | `ISETP.EQ.OR` | 1 |
| ALU |  | `ISETP.GE.AND.EX` | 1 |
| ALU |  | `ISETP.GT.AND` | 1 |
| ALU |  | `ISETP.GT.AND.EX` | 1 |
| ALU |  | `ISETP.GT.U32.OR` | 1 |
| ALU |  | `ISETP.NE.AND.EX` | 1 |
| ALU |  | `LEA` | 1 |
| ALU |  | `LEA.HI.X` | 1 |
| ALU |  | `SGXT.U32` | 1 |
| XU |  | `DFMA` | 136 |
| XU |  | `DMUL` | 40 |
| XU |  | `DADD` | 38 |
| XU |  | `DSETP.NAN.AND` | 6 |
| XU |  | `MUFU.RCP64H` | 6 |
| XU |  | `DSETP.GE.AND` | 4 |
| XU |  | `DSETP.NEU.AND` | 4 |
| XU |  | `DMUL.RP` | 1 |
| XU |  | `DSETP.GEU.AND` | 1 |
| Uniform |  | `UMOV` | 5 |
| Uniform |  | `ULDC.64` | 3 |
| Branch |  | `BRA` | 41 |
| Branch |  | `BSSY` | 25 |
| Branch |  | `BSYNC` | 25 |
| Branch |  | `NOP` | 14 |
| Branch |  | `CALL.REL.NOINC` | 12 |
| Branch |  | `RET.REL.NODEC` | 3 |
| Branch |  | `CS2R` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `S2R` | 2 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
