# SASS instruction inventory — `Kernel2__cutlass_80_tensorop_bf16_s16816gemm_bf16_64x64_32x6_nn_align2_sm87.sass`

Total instructions: **1410** (file has 1416 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 243 | 17.2% |
| Tensor core | 16 | 1.1% |
| FMA | 351 | 24.9% |
| ALU | 586 | 41.6% |
| Shuffle | 1 | 0.1% |
| Uniform | 98 | 7.0% |
| Branch | 96 | 6.8% |
| Other | 19 | 1.3% |
| **Total** | **1410** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 178 | 73.3% | 12.6% |
| Store | 65 | 26.7% | 4.6% |
| **LSU total** | **243** | **100.0%** | **17.2%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDGSTS.E.LTC128B` | 96 |
| LSU | Load | `LDS.64` | 32 |
| LSU | Store | `STG.E` | 32 |
| LSU | Store | `STS.64` | 32 |
| LSU | Load | `LDG.E.LTC128B` | 16 |
| LSU | Load | `LDGDEPBAR` | 8 |
| LSU | Load | `LDSM.16.M88.4` | 6 |
| LSU | Load | `LDSM.16.MT88.4` | 6 |
| LSU | Load | `LDG.E.64` | 4 |
| LSU | Load | `LDS` | 4 |
| LSU | Load | `LDG.E` | 2 |
| LSU | Load | `LDG.E.STRONG.GPU` | 2 |
| LSU | Load | `LDC` | 1 |
| LSU | Load | `MEMBAR.ALL.GPU` | 1 |
| LSU | Store | `STG.E.STRONG.GPU` | 1 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 16 |
| FMA |  | `FMNMX` | 64 |
| FMA |  | `FMUL` | 64 |
| FMA |  | `IMAD` | 45 |
| FMA |  | `IMAD.MOV.U32` | 38 |
| FMA |  | `FFMA` | 32 |
| FMA |  | `IMAD.IADD` | 31 |
| FMA |  | `IMAD.SHL.U32` | 20 |
| FMA |  | `IMAD.U32` | 18 |
| FMA |  | `IMAD.X` | 16 |
| FMA |  | `IMAD.WIDE.U32` | 13 |
| FMA |  | `IMAD.WIDE` | 7 |
| FMA |  | `FSEL` | 2 |
| FMA |  | `FSETP.NEU.AND` | 1 |
| ALU |  | `IADD3` | 134 |
| ALU |  | `ISETP.LT.AND` | 68 |
| ALU |  | `LOP3.LUT` | 63 |
| ALU |  | `IADD3.X` | 59 |
| ALU |  | `SHF.R.S32.HI` | 36 |
| ALU |  | `F2FP.BF16.PACK_AB` | 32 |
| ALU |  | `ISETP.NE.AND` | 24 |
| ALU |  | `R2P` | 24 |
| ALU |  | `LEA.HI` | 22 |
| ALU |  | `SEL` | 18 |
| ALU |  | `LEA` | 15 |
| ALU |  | `PLOP3.LUT` | 15 |
| ALU |  | `MOV` | 14 |
| ALU |  | `LEA.HI.X` | 11 |
| ALU |  | `ISETP.GE.AND` | 9 |
| ALU |  | `ISETP.GT.AND` | 5 |
| ALU |  | `ISETP.NE.U32.AND` | 5 |
| ALU |  | `SHF.R.U32.HI` | 5 |
| ALU |  | `ISETP.NE.AND.EX` | 4 |
| ALU |  | `LEA.HI.SX32` | 4 |
| ALU |  | `P2R` | 4 |
| ALU |  | `P2R.B1` | 4 |
| ALU |  | `SHF.L.U32` | 3 |
| ALU |  | `ISETP.GE.U32.AND` | 2 |
| ALU |  | `IMNMX` | 1 |
| ALU |  | `IMNMX.U32` | 1 |
| ALU |  | `ISETP.EQ.AND` | 1 |
| ALU |  | `ISETP.GE.OR` | 1 |
| ALU |  | `ISETP.GT.U32.AND` | 1 |
| ALU |  | `PRMT` | 1 |
| Shuffle |  | `SHFL.IDX` | 1 |
| Uniform |  | `UIADD3` | 22 |
| Uniform |  | `ULDC.64` | 21 |
| Uniform |  | `UMOV` | 11 |
| Uniform |  | `UIADD3.X` | 8 |
| Uniform |  | `UIMAD` | 8 |
| Uniform |  | `USHF.L.U32` | 7 |
| Uniform |  | `UIMAD.WIDE.U32` | 6 |
| Uniform |  | `ULOP3.LUT` | 4 |
| Uniform |  | `USHF.R.S32.HI` | 3 |
| Uniform |  | `UISETP.NE.AND` | 2 |
| Uniform |  | `ULEA` | 2 |
| Uniform |  | `ULEA.HI` | 2 |
| Uniform |  | `UISETP.NE.U32.AND` | 1 |
| Uniform |  | `ULDC` | 1 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 22 |
| Branch |  | `BRA` | 22 |
| Branch |  | `CS2R` | 17 |
| Branch |  | `NOP` | 9 |
| Branch |  | `S2R` | 6 |
| Branch |  | `EXIT` | 4 |
| Branch |  | `DEPBAR.LE` | 3 |
| Branch |  | `B2R.RESULT` | 2 |
| Branch |  | `BAR.RED.AND.DEFER_BLOCKING` | 2 |
| Branch |  | `WARPSYNC` | 2 |
| Branch |  | `BRX` | 1 |
| Branch |  | `BSSY` | 1 |
| Branch |  | `BSYNC` | 1 |
| Branch |  | `ERRBAR` | 1 |
| Branch |  | `R2UR` | 1 |
| Branch |  | `S2UR` | 1 |
| Branch |  | `YIELD` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
| Other |  | `CCTL.IVALL` | 2 |
