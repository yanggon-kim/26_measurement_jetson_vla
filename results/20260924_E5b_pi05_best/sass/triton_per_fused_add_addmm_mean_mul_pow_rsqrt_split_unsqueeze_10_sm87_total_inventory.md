# SASS instruction inventory — `triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_10_sm87.sass`

Total instructions: **144** (file has 144 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 17 | 11.8% |
| FMA | 63 | 43.8% |
| ALU | 11 | 7.6% |
| Shuffle | 8 | 5.6% |
| XU | 1 | 0.7% |
| Uniform | 4 | 2.8% |
| Branch | 23 | 16.0% |
| Other | 17 | 11.8% |
| **Total** | **144** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 13 | 76.5% | 9.0% |
| Store | 4 | 23.5% | 2.8% |
| **LSU total** | **17** | **100.0%** | **11.8%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.EL.128` | 8 |
| LSU | Load | `LDG.E.128` | 3 |
| LSU | Load | `LDS` | 2 |
| LSU | Store | `STG.E.128` | 2 |
| LSU | Store | `STS` | 2 |
| FMA |  | `FADD` | 29 |
| FMA |  | `FFMA` | 16 |
| FMA |  | `IMAD.WIDE.U32` | 6 |
| FMA |  | `FMUL` | 5 |
| FMA |  | `IMAD.WIDE` | 5 |
| FMA |  | `HFMA2.MMA` | 2 |
| ALU |  | `LOP3.LUT` | 3 |
| ALU |  | `MOV` | 2 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LEA` | 1 |
| ALU |  | `SHF.L.U32` | 1 |
| ALU |  | `SHF.R.U32.HI` | 1 |
| Shuffle |  | `SHFL.BFLY` | 8 |
| XU |  | `MUFU.RSQ` | 1 |
| Uniform |  | `UMOV` | 3 |
| Uniform |  | `ULDC.64` | 1 |
| Branch |  | `NOP` | 15 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 2 |
| Branch |  | `BRA` | 2 |
| Branch |  | `EXIT` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
