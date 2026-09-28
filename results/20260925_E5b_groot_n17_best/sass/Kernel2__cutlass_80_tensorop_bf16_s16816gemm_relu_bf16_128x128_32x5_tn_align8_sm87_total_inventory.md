# SASS instruction inventory — `Kernel2__cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8_sm87.sass`

Total instructions: **5354** (file has 5360 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 634 | 11.8% |
| Tensor core | 64 | 1.2% |
| FMA | 1956 | 36.5% |
| ALU | 2001 | 37.4% |
| Shuffle | 1 | 0.0% |
| Uniform | 99 | 1.8% |
| Branch | 324 | 6.1% |
| Other | 275 | 5.1% |
| **Total** | **5354** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 249 | 39.3% | 4.7% |
| Store | 385 | 60.7% | 7.2% |
| **LSU total** | **634** | **100.0%** | **11.8%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Store | `STS.64` | 256 |
| LSU | Load | `LDS.128` | 128 |
| LSU | Store | `STG.E.128` | 64 |
| LSU | Load | `LDGSTS.E.BYPASS.LTC128B.128` | 40 |
| LSU | Load | `LDG.E.LTC128B.128` | 32 |
| LSU | Store | `ST.E.U8` | 32 |
| LSU | Store | `STG.E.U8` | 32 |
| LSU | Load | `LDSM.16.M88.4` | 24 |
| LSU | Load | `LDGDEPBAR` | 7 |
| LSU | Load | `LDG.E.64` | 6 |
| LSU | Load | `LDS` | 4 |
| LSU | Load | `LDG.E` | 2 |
| LSU | Load | `LDG.E.STRONG.GPU` | 2 |
| LSU | Load | `LD.E.128` | 1 |
| LSU | Load | `LDC` | 1 |
| LSU | Load | `LDG.E.128` | 1 |
| LSU | Load | `MEMBAR.ALL.GPU` | 1 |
| LSU | Store | `STG.E.STRONG.GPU` | 1 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 64 |
| FMA |  | `FFMA` | 512 |
| FMA |  | `FSETP.GEU.AND` | 512 |
| FMA |  | `FADD` | 256 |
| FMA |  | `FMUL` | 256 |
| FMA |  | `IMAD.U32` | 137 |
| FMA |  | `IMAD.SHL.U32` | 79 |
| FMA |  | `IMAD.MOV.U32` | 68 |
| FMA |  | `IMAD` | 63 |
| FMA |  | `IMAD.IADD` | 29 |
| FMA |  | `IMAD.WIDE.U32` | 17 |
| FMA |  | `IMAD.X` | 15 |
| FMA |  | `IMAD.WIDE` | 9 |
| FMA |  | `FSETP.NEU.AND` | 2 |
| FMA |  | `FSEL` | 1 |
| ALU |  | `PRMT` | 515 |
| ALU |  | `F2FP.BF16.PACK_AB` | 256 |
| ALU |  | `IADD3` | 243 |
| ALU |  | `LOP3.LUT` | 180 |
| ALU |  | `IADD3.X` | 138 |
| ALU |  | `P2R` | 135 |
| ALU |  | `ISETP.GE.AND` | 102 |
| ALU |  | `ISETP.LT.AND` | 89 |
| ALU |  | `ISETP.NE.AND` | 66 |
| ALU |  | `PLOP3.LUT` | 63 |
| ALU |  | `SHF.R.S32.HI` | 41 |
| ALU |  | `MOV` | 23 |
| ALU |  | `SEL` | 23 |
| ALU |  | `LEA.HI` | 20 |
| ALU |  | `ISETP.NE.U32.AND` | 17 |
| ALU |  | `ISETP.NE.AND.EX` | 16 |
| ALU |  | `LEA` | 14 |
| ALU |  | `LEA.HI.X` | 13 |
| ALU |  | `R2P` | 12 |
| ALU |  | `SHF.R.U32.HI` | 10 |
| ALU |  | `SHF.L.U32` | 6 |
| ALU |  | `ISETP.GT.AND` | 5 |
| ALU |  | `LEA.HI.SX32` | 4 |
| ALU |  | `ISETP.GE.U32.AND` | 2 |
| ALU |  | `SHF.R.U64` | 2 |
| ALU |  | `IMNMX` | 1 |
| ALU |  | `IMNMX.U32` | 1 |
| ALU |  | `ISETP.EQ.AND` | 1 |
| ALU |  | `ISETP.GE.OR` | 1 |
| ALU |  | `ISETP.GT.U32.AND` | 1 |
| ALU |  | `SHF.L.U64.HI` | 1 |
| Shuffle |  | `SHFL.IDX` | 1 |
| Uniform |  | `ULDC.64` | 44 |
| Uniform |  | `UMOV` | 15 |
| Uniform |  | `UIADD3` | 10 |
| Uniform |  | `USHF.L.U32` | 6 |
| Uniform |  | `UIMAD` | 5 |
| Uniform |  | `USHF.R.S32.HI` | 3 |
| Uniform |  | `UIMAD.WIDE.U32` | 2 |
| Uniform |  | `UISETP.NE.AND` | 2 |
| Uniform |  | `ULEA` | 2 |
| Uniform |  | `ULEA.HI` | 2 |
| Uniform |  | `ULOP3.LUT` | 2 |
| Uniform |  | `UPRMT` | 2 |
| Uniform |  | `USHF.L.U64.HI` | 2 |
| Uniform |  | `ULDC` | 1 |
| Uniform |  | `ULDC.U16` | 1 |
| Branch |  | `CS2R` | 85 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 69 |
| Branch |  | `BRA` | 62 |
| Branch |  | `BSSY` | 31 |
| Branch |  | `BSYNC` | 31 |
| Branch |  | `NOP` | 15 |
| Branch |  | `EXIT` | 11 |
| Branch |  | `S2R` | 7 |
| Branch |  | `DEPBAR.LE` | 3 |
| Branch |  | `B2R.RESULT` | 2 |
| Branch |  | `BAR.RED.AND.DEFER_BLOCKING` | 2 |
| Branch |  | `WARPSYNC` | 2 |
| Branch |  | `BRX` | 1 |
| Branch |  | `ERRBAR` | 1 |
| Branch |  | `R2UR` | 1 |
| Branch |  | `YIELD` | 1 |
| Other |  | `HMNMX2.BF16_V2` | 256 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
| Other |  | `CCTL.IVALL` | 2 |
