# SASS instruction inventory — `elementwise_kernel.sass`

Total instructions: **864** (file has 864 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 6 | 0.7% |
| FMA | 333 | 38.5% |
| ALU | 291 | 33.7% |
| XU | 108 | 12.5% |
| Uniform | 2 | 0.2% |
| Branch | 105 | 12.2% |
| Other | 19 | 2.2% |
| **Total** | **864** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 4 | 66.7% | 0.5% |
| Store | 2 | 33.3% | 0.2% |
| **LSU total** | **6** | **100.0%** | **0.7%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.64` | 4 |
| LSU | Store | `STG.E.64` | 2 |
| FMA |  | `IMAD` | 201 |
| FMA |  | `IMAD.HI.U32` | 50 |
| FMA |  | `IMAD.MOV.U32` | 45 |
| FMA |  | `HFMA2.MMA` | 14 |
| FMA |  | `IMAD.MOV` | 13 |
| FMA |  | `FSEL` | 6 |
| FMA |  | `FSETP.GEU.AND` | 2 |
| FMA |  | `IMAD.X` | 2 |
| ALU |  | `ISETP.NE.AND` | 60 |
| ALU |  | `IADD3` | 51 |
| ALU |  | `SHF.R.U32.HI` | 51 |
| ALU |  | `MOV` | 47 |
| ALU |  | `LOP3.LUT` | 19 |
| ALU |  | `SEL` | 9 |
| ALU |  | `ISETP.GE.AND` | 6 |
| ALU |  | `ISETP.NE.U32.AND` | 5 |
| ALU |  | `IADD3.X` | 4 |
| ALU |  | `ISETP.GT.AND` | 4 |
| ALU |  | `ISETP.NE.AND.EX` | 4 |
| ALU |  | `LEA` | 4 |
| ALU |  | `LEA.HI` | 4 |
| ALU |  | `ISETP.EQ.AND` | 3 |
| ALU |  | `SHF.L.U32` | 3 |
| ALU |  | `ISETP.EQ.AND.EX` | 2 |
| ALU |  | `ISETP.EQ.U32.AND` | 2 |
| ALU |  | `ISETP.GT.OR` | 2 |
| ALU |  | `ISETP.LT.AND` | 2 |
| ALU |  | `ISETP.NE.OR` | 2 |
| ALU |  | `PLOP3.LUT` | 2 |
| ALU |  | `SHF.L.U64.HI` | 2 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.GT.U32.AND` | 1 |
| ALU |  | `SHF.R.S32.HI` | 1 |
| XU |  | `DFMA` | 41 |
| XU |  | `DADD` | 38 |
| XU |  | `DMUL` | 9 |
| XU |  | `DSETP.NEU.AND` | 8 |
| XU |  | `DSETP.EQ.OR` | 4 |
| XU |  | `DSETP.GT.AND` | 2 |
| XU |  | `DSETP.GTU.AND` | 2 |
| XU |  | `DSETP.GTU.OR` | 2 |
| XU |  | `DSETP.GEU.AND` | 1 |
| XU |  | `MUFU.RCP64H` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 70 |
| Branch |  | `NOP` | 14 |
| Branch |  | `BSSY` | 7 |
| Branch |  | `BSYNC` | 7 |
| Branch |  | `CALL.REL.NOINC` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `S2R` | 2 |
| Branch |  | `RET.REL.NODEC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
| Other |  | `FRND.F64.TRUNC` | 2 |
