# SASS instruction inventory — `elementwise_kernel_with_index_sm87.sass`

Total instructions: **56** (file has 56 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 1 | 1.8% |
| FMA | 8 | 14.3% |
| ALU | 11 | 19.6% |
| XU | 2 | 3.6% |
| Uniform | 2 | 3.6% |
| Branch | 15 | 26.8% |
| Other | 17 | 30.4% |
| **Total** | **56** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Store | 1 | 100.0% | 1.8% |
| **LSU total** | **1** | **100.0%** | **1.8%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Store | `STG.E.64` | 1 |
| FMA |  | `IMAD.MOV.U32` | 4 |
| FMA |  | `FSEL` | 2 |
| FMA |  | `IMAD` | 1 |
| FMA |  | `IMAD.WIDE` | 1 |
| ALU |  | `LOP3.LUT` | 2 |
| ALU |  | `I2F.F64` | 1 |
| ALU |  | `I2F.F64.S64` | 1 |
| ALU |  | `IADD3` | 1 |
| ALU |  | `IADD3.X` | 1 |
| ALU |  | `ISETP.GE.AND` | 1 |
| ALU |  | `ISETP.GE.AND.EX` | 1 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `SHF.R.S32.HI` | 1 |
| XU |  | `DFMA` | 2 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 9 |
| Branch |  | `BRA` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `S2R` | 2 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
