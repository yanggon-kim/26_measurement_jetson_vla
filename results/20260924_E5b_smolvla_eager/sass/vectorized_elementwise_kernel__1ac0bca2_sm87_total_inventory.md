# SASS instruction inventory — `vectorized_elementwise_kernel__1ac0bca2_sm87.sass`

Total instructions: **184** (file has 184 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 24 | 13.0% |
| FMA | 46 | 25.0% |
| ALU | 43 | 23.4% |
| XU | 16 | 8.7% |
| Uniform | 2 | 1.1% |
| Branch | 36 | 19.6% |
| Other | 17 | 9.2% |
| **Total** | **184** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 12 | 50.0% | 6.5% |
| Store | 12 | 50.0% | 6.5% |
| **LSU total** | **24** | **100.0%** | **13.0%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.64` | 8 |
| LSU | Store | `STG.E.64` | 8 |
| LSU | Load | `LDG.E.128` | 4 |
| LSU | Store | `STG.E.128` | 4 |
| FMA |  | `IMAD.WIDE.U32` | 18 |
| FMA |  | `IMAD.IADD` | 12 |
| FMA |  | `IMAD.MOV.U32` | 10 |
| FMA |  | `HFMA2.MMA` | 3 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `IADD3` | 18 |
| ALU |  | `ISETP.GE.AND` | 15 |
| ALU |  | `MOV` | 8 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| XU |  | `DMUL` | 16 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 8 |
| Branch |  | `CS2R` | 8 |
| Branch |  | `NOP` | 8 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
