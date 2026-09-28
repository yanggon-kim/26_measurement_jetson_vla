# SASS instruction inventory — `triton_per_fused_sum_0_sm87.sass`

Total instructions: **112** (file has 112 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 6 | 5.4% |
| FMA | 15 | 13.4% |
| ALU | 34 | 30.4% |
| Shuffle | 15 | 13.4% |
| Uniform | 2 | 1.8% |
| Branch | 23 | 20.5% |
| Other | 17 | 15.2% |
| **Total** | **112** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 3 | 50.0% | 2.7% |
| Store | 3 | 50.0% | 2.7% |
| **LSU total** | **6** | **100.0%** | **5.4%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDS.64` | 2 |
| LSU | Store | `STS.64` | 2 |
| LSU | Load | `LDG.E` | 1 |
| LSU | Store | `STG.E.64` | 1 |
| FMA |  | `IMAD.X` | 9 |
| FMA |  | `IMAD.MOV.U32` | 4 |
| FMA |  | `IMAD.IADD` | 1 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `IADD3` | 10 |
| ALU |  | `LOP3.LUT` | 6 |
| ALU |  | `ISETP.NE.AND` | 5 |
| ALU |  | `PRMT` | 4 |
| ALU |  | `SEL` | 4 |
| ALU |  | `ISETP.GE.U32.AND` | 2 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `POPC` | 1 |
| ALU |  | `SHF.R.U32.HI` | 1 |
| Shuffle |  | `SHFL.BFLY` | 15 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 13 |
| Branch |  | `BRA` | 3 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `BSSY` | 1 |
| Branch |  | `BSYNC` | 1 |
| Branch |  | `S2R` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
