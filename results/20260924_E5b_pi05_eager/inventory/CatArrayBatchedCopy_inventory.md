# SASS instruction inventory — `CatArrayBatchedCopy.sass`

Total instructions: **224** (file has 224 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 13 | 5.8% |
| FMA | 74 | 33.0% |
| ALU | 86 | 38.4% |
| XU | 6 | 2.7% |
| Uniform | 2 | 0.9% |
| Branch | 26 | 11.6% |
| Other | 17 | 7.6% |
| **Total** | **224** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 11 | 84.6% | 4.9% |
| Store | 2 | 15.4% | 0.9% |
| **LSU total** | **13** | **100.0%** | **5.8%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDC` | 6 |
| LSU | Load | `LDC.64` | 2 |
| LSU | Load | `LDG.E.U8` | 2 |
| LSU | Store | `STG.E.U8` | 2 |
| LSU | Load | `LDC.U8` | 1 |
| FMA |  | `IMAD` | 33 |
| FMA |  | `IMAD.HI.U32` | 12 |
| FMA |  | `IMAD.MOV` | 12 |
| FMA |  | `IMAD.MOV.U32` | 7 |
| FMA |  | `IMAD.IADD` | 4 |
| FMA |  | `IMAD.X` | 3 |
| FMA |  | `HFMA2.MMA` | 2 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `IADD3` | 30 |
| ALU |  | `ISETP.GE.U32.AND` | 15 |
| ALU |  | `SEL` | 9 |
| ALU |  | `ISETP.NE.U32.AND` | 7 |
| ALU |  | `F2I.FTZ.U32.TRUNC.NTZ` | 6 |
| ALU |  | `I2F.U32.RP` | 6 |
| ALU |  | `LOP3.LUT` | 6 |
| ALU |  | `ISETP.NE.AND` | 3 |
| ALU |  | `MOV` | 2 |
| ALU |  | `IADD3.X` | 1 |
| ALU |  | `SHF.L.U32` | 1 |
| XU |  | `MUFU.RCP` | 6 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 15 |
| Branch |  | `BRA` | 5 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
