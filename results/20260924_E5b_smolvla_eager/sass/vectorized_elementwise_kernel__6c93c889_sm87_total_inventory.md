# SASS instruction inventory — `vectorized_elementwise_kernel__6c93c889_sm87.sass`

Total instructions: **320** (file has 320 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 30 | 9.4% |
| FMA | 54 | 16.9% |
| ALU | 189 | 59.1% |
| Uniform | 2 | 0.6% |
| Branch | 28 | 8.8% |
| Other | 17 | 5.3% |
| **Total** | **320** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 20 | 66.7% | 6.2% |
| Store | 10 | 33.3% | 3.1% |
| **LSU total** | **30** | **100.0%** | **9.4%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.U8` | 16 |
| LSU | Store | `STG.E.U8` | 8 |
| LSU | Load | `LDG.E` | 4 |
| LSU | Store | `STG.E` | 2 |
| FMA |  | `IMAD.X` | 25 |
| FMA |  | `IMAD.IADD` | 16 |
| FMA |  | `IMAD.MOV.U32` | 9 |
| FMA |  | `IMAD.WIDE.U32` | 2 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| FMA |  | `IMAD.WIDE` | 1 |
| ALU |  | `PRMT` | 50 |
| ALU |  | `ISETP.NE.AND` | 46 |
| ALU |  | `IADD3` | 42 |
| ALU |  | `SEL` | 30 |
| ALU |  | `ISETP.GE.AND` | 15 |
| ALU |  | `IADD3.X` | 2 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `PLOP3.LUT` | 1 |
| ALU |  | `SHF.R.S32.HI` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 8 |
| Branch |  | `NOP` | 8 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
