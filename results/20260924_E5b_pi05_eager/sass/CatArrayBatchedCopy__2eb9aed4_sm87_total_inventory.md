# SASS instruction inventory — `CatArrayBatchedCopy__2eb9aed4_sm87.sass`

Total instructions: **216** (file has 216 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 13 | 6.0% |
| FMA | 72 | 33.3% |
| ALU | 84 | 38.9% |
| XU | 6 | 2.8% |
| Uniform | 2 | 0.9% |
| Branch | 22 | 10.2% |
| Other | 17 | 7.9% |
| **Total** | **216** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 11 | 84.6% | 5.1% |
| Store | 2 | 15.4% | 0.9% |
| **LSU total** | **13** | **100.0%** | **6.0%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDC` | 6 |
| LSU | Load | `LDC.64` | 2 |
| LSU | Load | `LDG.E` | 2 |
| LSU | Store | `STG.E` | 2 |
| LSU | Load | `LDC.U8` | 1 |
| FMA |  | `IMAD` | 33 |
| FMA |  | `IMAD.HI.U32` | 12 |
| FMA |  | `IMAD.MOV.U32` | 8 |
| FMA |  | `IMAD.IADD` | 6 |
| FMA |  | `IMAD.MOV` | 6 |
| FMA |  | `IMAD.WIDE.U32` | 4 |
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
| ALU |  | `MOV` | 1 |
| ALU |  | `SHF.L.U32` | 1 |
| XU |  | `MUFU.RCP` | 6 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 11 |
| Branch |  | `BRA` | 5 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
