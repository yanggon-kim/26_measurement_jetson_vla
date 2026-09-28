# SASS instruction inventory — `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn_sm87.sass`

Total instructions: **1060** (file has 1064 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 106 | 10.0% |
| Tensor core | 64 | 6.0% |
| FMA | 226 | 21.3% |
| ALU | 477 | 45.0% |
| Uniform | 21 | 2.0% |
| Branch | 141 | 13.3% |
| Other | 25 | 2.4% |
| **Total** | **1060** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 77 | 72.6% | 7.3% |
| Store | 29 | 27.4% | 2.7% |
| **LSU total** | **106** | **100.0%** | **10.0%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDGSTS.E.BYPASS.LTC128B.128.CONSTANT` | 28 |
| LSU | Load | `LDSM.16.M88.4` | 24 |
| LSU | Store | `STS` | 16 |
| LSU | Load | `LDG.E.LTC128B.64.STRONG.GPU` | 8 |
| LSU | Store | `STS.128` | 8 |
| LSU | Load | `LDG.E.LTC128B.64.CONSTANT` | 4 |
| LSU | Load | `LDGDEPBAR` | 4 |
| LSU | Load | `LDS.128` | 4 |
| LSU | Store | `STG.E.EF.64` | 4 |
| LSU | Load | `LDG.E.LTC128B.CONSTANT` | 3 |
| LSU | Load | `LDG.E.EL.LTC128B.STRONG.GPU` | 1 |
| LSU | Load | `MEMBAR.GPU` | 1 |
| LSU | Store | `STG.E.EF.STRONG.GPU` | 1 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 64 |
| FMA |  | `FMUL` | 128 |
| FMA |  | `FADD` | 16 |
| FMA |  | `FFMA` | 16 |
| FMA |  | `FSEL` | 16 |
| FMA |  | `FSETP.GEU.AND` | 16 |
| FMA |  | `IMAD.WIDE.U32` | 16 |
| FMA |  | `IMAD.U32` | 12 |
| FMA |  | `IMAD` | 3 |
| FMA |  | `IMAD.SHL` | 2 |
| FMA |  | `FSETP.NE.AND` | 1 |
| ALU |  | `LOP3.LUT` | 66 |
| ALU |  | `LEA` | 59 |
| ALU |  | `IADD3` | 57 |
| ALU |  | `MOV` | 44 |
| ALU |  | `LEA.HI.X` | 43 |
| ALU |  | `ISETP.LT.AND` | 38 |
| ALU |  | `IADD3.X` | 32 |
| ALU |  | `SHF.R.S32.HI` | 23 |
| ALU |  | `PRMT` | 20 |
| ALU |  | `SHF.L.U32` | 15 |
| ALU |  | `R2P` | 14 |
| ALU |  | `ISETP.EQ.AND` | 11 |
| ALU |  | `SHF.R.U32.HI` | 10 |
| ALU |  | `F2FP.BF16.PACK_AB` | 8 |
| ALU |  | `ISETP.GT.AND` | 7 |
| ALU |  | `PLOP3.LUT` | 6 |
| ALU |  | `P2R` | 5 |
| ALU |  | `BMSK` | 4 |
| ALU |  | `ISETP.NE.AND` | 4 |
| ALU |  | `P2R.B1` | 4 |
| ALU |  | `ISETP.LE.AND` | 3 |
| ALU |  | `ISETP.EQ.XOR` | 2 |
| ALU |  | `ISETP.LE.OR` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `UMOV` | 7 |
| Uniform |  | `UISETP.LT.AND` | 6 |
| Uniform |  | `UIADD3` | 4 |
| Uniform |  | `ULDC` | 4 |
| Branch |  | `CS2R` | 64 |
| Branch |  | `BRA.U` | 21 |
| Branch |  | `NOP` | 17 |
| Branch |  | `BRA` | 13 |
| Branch |  | `CALL.REL` | 8 |
| Branch |  | `S2R` | 8 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 5 |
| Branch |  | `DEPBAR.LE` | 2 |
| Branch |  | `BAR.SYNC` | 1 |
| Branch |  | `EXIT` | 1 |
| Branch |  | `RET.ABS` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `LEPC` | 8 |
| Other |  | `BMOV.32` | 2 |
