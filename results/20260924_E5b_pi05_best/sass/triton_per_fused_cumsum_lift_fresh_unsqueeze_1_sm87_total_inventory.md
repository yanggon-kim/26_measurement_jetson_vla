# SASS instruction inventory — `triton_per_fused_cumsum_lift_fresh_unsqueeze_1_sm87.sass`

Total instructions: **72** (file has 72 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 4 | 5.6% |
| FMA | 12 | 16.7% |
| ALU | 13 | 18.1% |
| Shuffle | 5 | 6.9% |
| Uniform | 2 | 2.8% |
| Branch | 19 | 26.4% |
| Other | 17 | 23.6% |
| **Total** | **72** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 2 | 50.0% | 2.8% |
| Store | 2 | 50.0% | 2.8% |
| **LSU total** | **4** | **100.0%** | **5.6%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E` | 1 |
| LSU | Load | `LDS` | 1 |
| LSU | Store | `STG.E` | 1 |
| LSU | Store | `STS` | 1 |
| FMA |  | `FADD` | 6 |
| FMA |  | `IMAD.MOV.U32` | 4 |
| FMA |  | `IMAD.WIDE.U32` | 2 |
| ALU |  | `ISETP.GT.U32.AND` | 4 |
| ALU |  | `ISETP.NE.AND` | 3 |
| ALU |  | `LOP3.LUT` | 2 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `SGXT.U32` | 1 |
| ALU |  | `SHF.R.U32.HI` | 1 |
| Shuffle |  | `SHFL.UP` | 5 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 13 |
| Branch |  | `BRA` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 1 |
| Branch |  | `S2R` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
