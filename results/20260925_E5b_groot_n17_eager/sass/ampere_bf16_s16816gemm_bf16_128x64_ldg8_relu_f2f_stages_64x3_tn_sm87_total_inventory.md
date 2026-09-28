# SASS instruction inventory — `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x3_tn_sm87.sass`

Total instructions: **996** (file has 1000 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 127 | 12.8% |
| Tensor core | 64 | 6.4% |
| FMA | 162 | 16.3% |
| ALU | 495 | 49.7% |
| Uniform | 21 | 2.1% |
| Branch | 106 | 10.6% |
| Other | 21 | 2.1% |
| **Total** | **996** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 94 | 74.0% | 9.4% |
| Store | 33 | 26.0% | 3.3% |
| **LSU total** | **127** | **100.0%** | **12.8%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDGSTS.E.BYPASS.LTC128B.128.CONSTANT` | 39 |
| LSU | Load | `LDSM.16.M88.4` | 30 |
| LSU | Store | `STS` | 16 |
| LSU | Store | `STS.128` | 12 |
| LSU | Load | `LDG.E.LTC128B.64.STRONG.GPU` | 8 |
| LSU | Load | `LDG.E.LTC128B.64.CONSTANT` | 4 |
| LSU | Load | `LDGDEPBAR` | 4 |
| LSU | Load | `LDS.128` | 4 |
| LSU | Store | `STG.E.EF.64` | 4 |
| LSU | Load | `LDG.E.LTC128B.CONSTANT` | 3 |
| LSU | Load | `LDG.E.EL.LTC128B.STRONG.GPU` | 1 |
| LSU | Load | `MEMBAR.GPU` | 1 |
| LSU | Store | `STG.E.EF.STRONG.GPU` | 1 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 64 |
| FMA |  | `FMUL` | 64 |
| FMA |  | `FADD` | 16 |
| FMA |  | `FFMA` | 16 |
| FMA |  | `FSEL` | 16 |
| FMA |  | `FSETP.GEU.AND` | 16 |
| FMA |  | `IMAD.WIDE.U32` | 16 |
| FMA |  | `IMAD.U32` | 12 |
| FMA |  | `IMAD.SHL` | 3 |
| FMA |  | `IMAD` | 2 |
| FMA |  | `FSETP.NE.AND` | 1 |
| ALU |  | `IADD3` | 69 |
| ALU |  | `LOP3.LUT` | 66 |
| ALU |  | `LEA` | 52 |
| ALU |  | `IADD3.X` | 48 |
| ALU |  | `ISETP.LT.AND` | 46 |
| ALU |  | `MOV` | 44 |
| ALU |  | `LEA.HI.X` | 35 |
| ALU |  | `PRMT` | 20 |
| ALU |  | `SHF.R.S32.HI` | 19 |
| ALU |  | `R2P` | 17 |
| ALU |  | `ISETP.EQ.AND` | 11 |
| ALU |  | `SHF.L.U32` | 11 |
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
| ALU |  | `P2R.B2` | 2 |
| ALU |  | `ISETP.LE.OR` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `UMOV` | 7 |
| Uniform |  | `UISETP.LT.AND` | 6 |
| Uniform |  | `UIADD3` | 4 |
| Uniform |  | `ULDC` | 4 |
| Branch |  | `CS2R` | 32 |
| Branch |  | `BRA.U` | 21 |
| Branch |  | `NOP` | 18 |
| Branch |  | `BRA` | 13 |
| Branch |  | `S2R` | 8 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 5 |
| Branch |  | `CALL.REL` | 4 |
| Branch |  | `DEPBAR.LE` | 2 |
| Branch |  | `BAR.SYNC` | 1 |
| Branch |  | `EXIT` | 1 |
| Branch |  | `RET.ABS` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `LEPC` | 4 |
| Other |  | `BMOV.32` | 2 |
