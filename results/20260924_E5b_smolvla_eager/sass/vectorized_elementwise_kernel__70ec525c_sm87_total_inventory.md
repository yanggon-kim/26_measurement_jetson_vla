# SASS instruction inventory — `vectorized_elementwise_kernel__70ec525c_sm87.sass`

Total instructions: **176** (file has 176 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 20 | 11.4% |
| FMA | 54 | 30.7% |
| ALU | 50 | 28.4% |
| Uniform | 2 | 1.1% |
| Branch | 33 | 18.8% |
| Other | 17 | 9.7% |
| **Total** | **176** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 10 | 50.0% | 5.7% |
| Store | 10 | 50.0% | 5.7% |
| **LSU total** | **20** | **100.0%** | **11.4%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E` | 8 |
| LSU | Store | `STG.E` | 8 |
| LSU | Load | `LDG.E.128` | 2 |
| LSU | Store | `STG.E.128` | 2 |
| FMA |  | `IMAD.WIDE.U32` | 18 |
| FMA |  | `FMUL` | 16 |
| FMA |  | `IMAD.MOV.U32` | 8 |
| FMA |  | `IMAD.IADD` | 6 |
| FMA |  | `HFMA2.MMA` | 3 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `IADD3` | 25 |
| ALU |  | `ISETP.GE.AND` | 15 |
| ALU |  | `MOV` | 8 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 9 |
| Branch |  | `BRA` | 8 |
| Branch |  | `CS2R` | 4 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
