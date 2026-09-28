# SASS instruction inventory — `Kernel2__cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_nn_align1_sm87.sass`

Total instructions: **749** (file has 752 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 108 | 14.4% |
| Tensor core | 4 | 0.5% |
| FMA | 175 | 23.4% |
| ALU | 359 | 47.9% |
| Shuffle | 1 | 0.1% |
| Uniform | 5 | 0.7% |
| Branch | 70 | 9.3% |
| Other | 27 | 3.6% |
| **Total** | **749** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 81 | 75.0% | 10.8% |
| Store | 27 | 25.0% | 3.6% |
| **LSU total** | **108** | **100.0%** | **14.4%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.LTC128B.U16` | 40 |
| LSU | Load | `LD.E` | 16 |
| LSU | Load | `LDS` | 16 |
| LSU | Store | `STG.E.U16` | 16 |
| LSU | Store | `STS.64` | 8 |
| LSU | Load | `LDG.E.64` | 4 |
| LSU | Load | `LDG.E` | 2 |
| LSU | Load | `LDG.E.STRONG.GPU` | 2 |
| LSU | Store | `STS.128` | 2 |
| LSU | Load | `MEMBAR.ALL.GPU` | 1 |
| LSU | Store | `STG.E.STRONG.GPU` | 1 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 4 |
| FMA |  | `IMAD` | 38 |
| FMA |  | `IMAD.MOV.U32` | 32 |
| FMA |  | `FMNMX` | 16 |
| FMA |  | `FMUL` | 16 |
| FMA |  | `IMAD.IADD` | 15 |
| FMA |  | `IMAD.WIDE.U32` | 12 |
| FMA |  | `IMAD.SHL.U32` | 11 |
| FMA |  | `IMAD.U32` | 10 |
| FMA |  | `FFMA` | 8 |
| FMA |  | `IMAD.WIDE` | 7 |
| FMA |  | `IMAD.X` | 7 |
| FMA |  | `FSEL` | 2 |
| FMA |  | `FSETP.NEU.AND` | 1 |
| ALU |  | `IADD3` | 68 |
| ALU |  | `ISETP.LT.AND` | 48 |
| ALU |  | `PRMT` | 34 |
| ALU |  | `IADD3.X` | 21 |
| ALU |  | `SHF.R.S32.HI` | 21 |
| ALU |  | `F2FP.BF16.PACK_AB` | 16 |
| ALU |  | `ISETP.NE.AND` | 14 |
| ALU |  | `PLOP3.LUT` | 14 |
| ALU |  | `LOP3.LUT` | 13 |
| ALU |  | `MOV` | 13 |
| ALU |  | `LEA` | 11 |
| ALU |  | `LEA.HI.X` | 11 |
| ALU |  | `SEL` | 11 |
| ALU |  | `ISETP.GE.AND` | 9 |
| ALU |  | `LEA.HI` | 9 |
| ALU |  | `R2P` | 8 |
| ALU |  | `ISETP.GT.AND` | 7 |
| ALU |  | `ISETP.NE.U32.AND` | 5 |
| ALU |  | `ISETP.NE.AND.EX` | 4 |
| ALU |  | `P2R` | 4 |
| ALU |  | `P2R.B1` | 4 |
| ALU |  | `SHF.R.U32.HI` | 4 |
| ALU |  | `LEA.HI.X.SX32` | 2 |
| ALU |  | `SHF.L.U32` | 2 |
| ALU |  | `IMNMX` | 1 |
| ALU |  | `ISETP.EQ.AND` | 1 |
| ALU |  | `ISETP.GE.OR` | 1 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `LEA.HI.SX32` | 1 |
| ALU |  | `SHF.L.U64.HI` | 1 |
| Shuffle |  | `SHFL.IDX` | 1 |
| Uniform |  | `ULDC.64` | 4 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 24 |
| Branch |  | `NOP` | 10 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 9 |
| Branch |  | `S2R` | 9 |
| Branch |  | `CS2R` | 4 |
| Branch |  | `EXIT` | 4 |
| Branch |  | `B2R.RESULT` | 2 |
| Branch |  | `BAR.RED.AND.DEFER_BLOCKING` | 2 |
| Branch |  | `WARPSYNC` | 2 |
| Branch |  | `BSSY` | 1 |
| Branch |  | `BSYNC` | 1 |
| Branch |  | `ERRBAR` | 1 |
| Branch |  | `YIELD` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `MOVM.16.MT88` | 8 |
| Other |  | `BMOV.32` | 2 |
| Other |  | `CCTL.IVALL` | 2 |
