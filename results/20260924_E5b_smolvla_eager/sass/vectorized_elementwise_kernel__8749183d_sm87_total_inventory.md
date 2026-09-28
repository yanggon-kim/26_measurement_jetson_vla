# SASS instruction inventory — `vectorized_elementwise_kernel__8749183d_sm87.sass`

Total instructions: **264** (file has 264 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 20 | 7.6% |
| FMA | 103 | 39.0% |
| ALU | 73 | 27.7% |
| XU | 16 | 6.1% |
| Uniform | 2 | 0.8% |
| Branch | 33 | 12.5% |
| Other | 17 | 6.4% |
| **Total** | **264** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 10 | 50.0% | 3.8% |
| Store | 10 | 50.0% | 3.8% |
| **LSU total** | **20** | **100.0%** | **7.6%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E` | 8 |
| LSU | Store | `STG.E` | 8 |
| LSU | Load | `LDG.E.128` | 2 |
| LSU | Store | `STG.E.128` | 2 |
| FMA |  | `FMUL` | 32 |
| FMA |  | `IMAD.MOV.U32` | 23 |
| FMA |  | `IMAD.WIDE.U32` | 18 |
| FMA |  | `FSETP.GEU.AND` | 16 |
| FMA |  | `IMAD.IADD` | 10 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `HFMA2.MMA` | 1 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `IADD3` | 27 |
| ALU |  | `ISETP.GE.AND` | 22 |
| ALU |  | `PLOP3.LUT` | 16 |
| ALU |  | `ISETP.NE.AND` | 3 |
| ALU |  | `P2R` | 3 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| XU |  | `MUFU.RSQ` | 16 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 10 |
| Branch |  | `BRA` | 8 |
| Branch |  | `CS2R` | 3 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
