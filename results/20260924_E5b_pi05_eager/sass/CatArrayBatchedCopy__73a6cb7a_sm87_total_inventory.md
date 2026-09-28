# SASS instruction inventory — `CatArrayBatchedCopy__73a6cb7a_sm87.sass`

Total instructions: **288** (file has 288 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 13 | 4.5% |
| FMA | 103 | 35.8% |
| ALU | 121 | 42.0% |
| XU | 9 | 3.1% |
| Uniform | 2 | 0.7% |
| Branch | 23 | 8.0% |
| Other | 17 | 5.9% |
| **Total** | **288** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 11 | 84.6% | 3.8% |
| Store | 2 | 15.4% | 0.7% |
| **LSU total** | **13** | **100.0%** | **4.5%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDC` | 4 |
| LSU | Load | `LDC.64` | 4 |
| LSU | Load | `LDG.E` | 2 |
| LSU | Store | `STG.E` | 2 |
| LSU | Load | `LDC.U8` | 1 |
| FMA |  | `IMAD` | 45 |
| FMA |  | `IMAD.HI.U32` | 18 |
| FMA |  | `IMAD.MOV` | 16 |
| FMA |  | `IMAD.MOV.U32` | 12 |
| FMA |  | `IMAD.IADD` | 5 |
| FMA |  | `IMAD.WIDE.U32` | 4 |
| FMA |  | `HFMA2.MMA` | 2 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `IADD3` | 42 |
| ALU |  | `ISETP.GE.U32.AND` | 21 |
| ALU |  | `SEL` | 14 |
| ALU |  | `ISETP.NE.U32.AND` | 10 |
| ALU |  | `F2I.FTZ.U32.TRUNC.NTZ` | 9 |
| ALU |  | `I2F.U32.RP` | 9 |
| ALU |  | `LOP3.LUT` | 9 |
| ALU |  | `ISETP.NE.AND` | 4 |
| ALU |  | `MOV` | 2 |
| ALU |  | `SHF.L.U32` | 1 |
| XU |  | `MUFU.RCP` | 9 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 12 |
| Branch |  | `BRA` | 5 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
