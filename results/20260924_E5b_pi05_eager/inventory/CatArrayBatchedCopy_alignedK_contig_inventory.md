# SASS instruction inventory — `CatArrayBatchedCopy_alignedK_contig.sass`

Total instructions: **256** (file has 256 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 17 | 6.6% |
| FMA | 93 | 36.3% |
| ALU | 98 | 38.3% |
| XU | 3 | 1.2% |
| Uniform | 2 | 0.8% |
| Branch | 26 | 10.2% |
| Other | 17 | 6.6% |
| **Total** | **256** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 10 | 58.8% | 3.9% |
| Store | 7 | 41.2% | 2.7% |
| **LSU total** | **17** | **100.0%** | **6.6%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Store | `STG.E.64` | 7 |
| LSU | Load | `LDG.E.64` | 5 |
| LSU | Load | `LDC` | 3 |
| LSU | Load | `LDC.64` | 1 |
| LSU | Load | `LDG.E.128` | 1 |
| FMA |  | `IMAD` | 41 |
| FMA |  | `IMAD.MOV.U32` | 16 |
| FMA |  | `IMAD.WIDE.U32` | 13 |
| FMA |  | `IMAD.HI.U32` | 10 |
| FMA |  | `IMAD.IADD` | 6 |
| FMA |  | `IMAD.MOV` | 4 |
| FMA |  | `HFMA2.MMA` | 2 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `IADD3` | 42 |
| ALU |  | `ISETP.GE.U32.AND` | 17 |
| ALU |  | `SEL` | 10 |
| ALU |  | `LOP3.LUT` | 5 |
| ALU |  | `MOV` | 5 |
| ALU |  | `ISETP.NE.AND` | 4 |
| ALU |  | `ISETP.NE.U32.AND` | 4 |
| ALU |  | `F2I.FTZ.U32.TRUNC.NTZ` | 3 |
| ALU |  | `I2F.U32.RP` | 3 |
| ALU |  | `ISETP.GT.U32.AND` | 3 |
| ALU |  | `LEA` | 1 |
| ALU |  | `SHF.L.U32` | 1 |
| XU |  | `MUFU.RCP` | 3 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 8 |
| Branch |  | `BRA` | 7 |
| Branch |  | `EXIT` | 4 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
