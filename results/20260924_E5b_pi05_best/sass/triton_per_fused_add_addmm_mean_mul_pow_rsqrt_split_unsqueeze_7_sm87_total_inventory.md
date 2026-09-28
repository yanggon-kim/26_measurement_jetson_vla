# SASS instruction inventory — `triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_7_sm87.sass`

Total instructions: **120** (file has 120 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 13 | 10.8% |
| FMA | 51 | 42.5% |
| ALU | 10 | 8.3% |
| Shuffle | 8 | 6.7% |
| XU | 1 | 0.8% |
| Uniform | 4 | 3.3% |
| Branch | 16 | 13.3% |
| Other | 17 | 14.2% |
| **Total** | **120** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 10 | 76.9% | 8.3% |
| Store | 3 | 23.1% | 2.5% |
| **LSU total** | **13** | **100.0%** | **10.8%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.EL.128` | 6 |
| LSU | Load | `LDG.E.128` | 2 |
| LSU | Load | `LDS` | 2 |
| LSU | Store | `STS` | 2 |
| LSU | Store | `STG.E.128` | 1 |
| FMA |  | `FADD` | 25 |
| FMA |  | `FFMA` | 12 |
| FMA |  | `FMUL` | 5 |
| FMA |  | `IMAD.WIDE.U32` | 4 |
| FMA |  | `IMAD.WIDE` | 3 |
| FMA |  | `HFMA2.MMA` | 2 |
| ALU |  | `LOP3.LUT` | 3 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LEA` | 1 |
| ALU |  | `MOV` | 1 |
| ALU |  | `SHF.L.U32` | 1 |
| ALU |  | `SHF.R.U32.HI` | 1 |
| Shuffle |  | `SHFL.BFLY` | 8 |
| XU |  | `MUFU.RSQ` | 1 |
| Uniform |  | `UMOV` | 3 |
| Uniform |  | `ULDC.64` | 1 |
| Branch |  | `NOP` | 9 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 2 |
| Branch |  | `BRA` | 2 |
| Branch |  | `S2R` | 2 |
| Branch |  | `EXIT` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
