# SASS instruction inventory — `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn.sass`

Total instructions: **788** (file has 792 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 93 | 11.8% |
| Tensor core | 32 | 4.1% |
| FMA | 114 | 14.5% |
| ALU | 399 | 50.6% |
| Uniform | 21 | 2.7% |
| Branch | 108 | 13.7% |
| Other | 21 | 2.7% |
| **Total** | **788** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 66 | 71.0% | 8.4% |
| Store | 27 | 29.0% | 3.4% |
| **LSU total** | **93** | **100.0%** | **11.8%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDGSTS.E.BYPASS.LTC128B.128.CONSTANT` | 28 |
| LSU | Load | `LDSM.16.M88.4` | 18 |
| LSU | Store | `STS` | 16 |
| LSU | Store | `STS.128` | 8 |
| LSU | Load | `LDG.E.LTC128B.64.CONSTANT` | 4 |
| LSU | Load | `LDG.E.LTC128B.64.STRONG.GPU` | 4 |
| LSU | Load | `LDGDEPBAR` | 4 |
| LSU | Load | `LDS.128` | 4 |
| LSU | Load | `LDG.E.LTC128B.CONSTANT` | 2 |
| LSU | Store | `STG.E.EF.64` | 2 |
| LSU | Load | `LDG.E.EL.LTC128B.STRONG.GPU` | 1 |
| LSU | Load | `MEMBAR.GPU` | 1 |
| LSU | Store | `STG.E.EF.STRONG.GPU` | 1 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 32 |
| FMA |  | `FMUL` | 64 |
| FMA |  | `IMAD.WIDE.U32` | 16 |
| FMA |  | `IMAD.U32` | 12 |
| FMA |  | `FADD` | 8 |
| FMA |  | `FFMA` | 8 |
| FMA |  | `IMAD.SHL` | 3 |
| FMA |  | `IMAD` | 2 |
| FMA |  | `FSETP.NE.AND` | 1 |
| ALU |  | `LOP3.LUT` | 61 |
| ALU |  | `IADD3` | 51 |
| ALU |  | `LEA` | 46 |
| ALU |  | `MOV` | 42 |
| ALU |  | `IADD3.X` | 32 |
| ALU |  | `ISETP.LT.AND` | 32 |
| ALU |  | `LEA.HI.X` | 28 |
| ALU |  | `SHF.R.S32.HI` | 22 |
| ALU |  | `ISETP.EQ.AND` | 11 |
| ALU |  | `SHF.L.U32` | 11 |
| ALU |  | `R2P` | 10 |
| ALU |  | `SHF.R.U32.HI` | 10 |
| ALU |  | `PRMT` | 8 |
| ALU |  | `ISETP.GT.AND` | 7 |
| ALU |  | `PLOP3.LUT` | 6 |
| ALU |  | `BMSK` | 4 |
| ALU |  | `F2FP.BF16.PACK_AB` | 4 |
| ALU |  | `ISETP.NE.AND` | 4 |
| ALU |  | `P2R` | 4 |
| ALU |  | `ISETP.EQ.XOR` | 2 |
| ALU |  | `P2R.B1` | 2 |
| ALU |  | `ISETP.LE.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `UMOV` | 7 |
| Uniform |  | `UISETP.LT.AND` | 6 |
| Uniform |  | `UIADD3` | 4 |
| Uniform |  | `ULDC` | 4 |
| Branch |  | `CS2R` | 32 |
| Branch |  | `BRA.U` | 21 |
| Branch |  | `NOP` | 20 |
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
