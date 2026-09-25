# SASS instruction inventory — `vectorized_elementwise_kernel.sass`

Total instructions: **168** (file has 168 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 21 | 12.5% |
| FMA | 36 | 21.4% |
| ALU | 54 | 32.1% |
| Uniform | 2 | 1.2% |
| Branch | 38 | 22.6% |
| Other | 17 | 10.1% |
| **Total** | **168** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 4.8% | 0.6% |
| Store | 20 | 95.2% | 11.9% |
| **LSU total** | **21** | **100.0%** | **12.5%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Store | `STG.E.U8` | 16 |
| LSU | Store | `STG.E` | 4 |
| LSU | Load | `LDC.U8` | 1 |
| FMA |  | `IMAD.X` | 17 |
| FMA |  | `IMAD.IADD` | 16 |
| FMA |  | `IMAD.MOV.U32` | 1 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| FMA |  | `IMAD.WIDE` | 1 |
| ALU |  | `IADD3` | 33 |
| ALU |  | `ISETP.GE.AND` | 16 |
| ALU |  | `PRMT` | 3 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 16 |
| Branch |  | `NOP` | 10 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
