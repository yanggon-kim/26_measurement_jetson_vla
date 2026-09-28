# SASS instruction inventory — `Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nn_align2_sm87.sass`

Total instructions: **808** (file has 808 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 106 | 13.1% |
| Tensor core | 8 | 1.0% |
| FMA | 176 | 21.8% |
| ALU | 326 | 40.3% |
| Shuffle | 1 | 0.1% |
| Uniform | 75 | 9.3% |
| Branch | 81 | 10.0% |
| Other | 35 | 4.3% |
| **Total** | **808** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 85 | 80.2% | 10.5% |
| Store | 21 | 19.8% | 2.6% |
| **LSU total** | **106** | **100.0%** | **13.1%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.LTC128B` | 36 |
| LSU | Load | `LD.E` | 32 |
| LSU | Load | `LDS.64` | 8 |
| LSU | Store | `STG.E` | 8 |
| LSU | Store | `STS.64` | 8 |
| LSU | Load | `LDG.E.64` | 4 |
| LSU | Store | `STS.128` | 4 |
| LSU | Load | `LDG.E` | 2 |
| LSU | Load | `LDG.E.STRONG.GPU` | 2 |
| LSU | Load | `MEMBAR.ALL.GPU` | 1 |
| LSU | Store | `STG.E.STRONG.GPU` | 1 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 8 |
| FMA |  | `IMAD` | 37 |
| FMA |  | `IMAD.MOV.U32` | 34 |
| FMA |  | `IMAD.IADD` | 17 |
| FMA |  | `FMNMX` | 16 |
| FMA |  | `FMUL` | 16 |
| FMA |  | `IMAD.SHL.U32` | 16 |
| FMA |  | `IMAD.WIDE.U32` | 13 |
| FMA |  | `FFMA` | 8 |
| FMA |  | `IMAD.WIDE` | 7 |
| FMA |  | `IMAD.U32` | 6 |
| FMA |  | `IMAD.X` | 3 |
| FMA |  | `FSEL` | 2 |
| FMA |  | `FSETP.NEU.AND` | 1 |
| ALU |  | `IADD3` | 61 |
| ALU |  | `ISETP.LT.AND` | 32 |
| ALU |  | `IADD3.X` | 31 |
| ALU |  | `SHF.R.S32.HI` | 23 |
| ALU |  | `LOP3.LUT` | 22 |
| ALU |  | `MOV` | 18 |
| ALU |  | `ISETP.NE.AND` | 14 |
| ALU |  | `PLOP3.LUT` | 14 |
| ALU |  | `ISETP.GE.AND` | 11 |
| ALU |  | `LEA` | 11 |
| ALU |  | `LEA.HI` | 11 |
| ALU |  | `LEA.HI.X` | 11 |
| ALU |  | `SEL` | 11 |
| ALU |  | `F2FP.BF16.PACK_AB` | 8 |
| ALU |  | `R2P` | 8 |
| ALU |  | `ISETP.GT.AND` | 7 |
| ALU |  | `ISETP.NE.U32.AND` | 5 |
| ALU |  | `ISETP.NE.AND.EX` | 4 |
| ALU |  | `P2R` | 4 |
| ALU |  | `P2R.B1` | 4 |
| ALU |  | `SHF.R.U32.HI` | 4 |
| ALU |  | `LEA.HI.SX32` | 2 |
| ALU |  | `LEA.HI.X.SX32` | 2 |
| ALU |  | `SHF.L.U32` | 2 |
| ALU |  | `IMNMX` | 1 |
| ALU |  | `ISETP.EQ.AND` | 1 |
| ALU |  | `ISETP.GE.OR` | 1 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `PRMT` | 1 |
| ALU |  | `SHF.L.U64.HI` | 1 |
| Shuffle |  | `SHFL.IDX` | 1 |
| Uniform |  | `UIADD3` | 39 |
| Uniform |  | `UIADD3.X` | 20 |
| Uniform |  | `ULDC.64` | 5 |
| Uniform |  | `ULDC` | 4 |
| Uniform |  | `USHF.L.U32` | 3 |
| Uniform |  | `UMOV` | 2 |
| Uniform |  | `USHF.L.U64.HI` | 2 |
| Branch |  | `BRA` | 24 |
| Branch |  | `CS2R` | 13 |
| Branch |  | `NOP` | 13 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 9 |
| Branch |  | `S2R` | 8 |
| Branch |  | `EXIT` | 4 |
| Branch |  | `B2R.RESULT` | 2 |
| Branch |  | `BAR.RED.AND.DEFER_BLOCKING` | 2 |
| Branch |  | `WARPSYNC` | 2 |
| Branch |  | `BSSY` | 1 |
| Branch |  | `BSYNC` | 1 |
| Branch |  | `ERRBAR` | 1 |
| Branch |  | `YIELD` | 1 |
| Other |  | `MOVM.16.MT88` | 16 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
| Other |  | `CCTL.IVALL` | 2 |
