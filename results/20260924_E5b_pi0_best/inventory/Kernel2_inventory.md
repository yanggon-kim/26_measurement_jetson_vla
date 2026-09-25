# SASS instruction inventory — `Kernel2.sass`

Total instructions: **2266** (file has 2272 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 373 | 16.5% |
| Tensor core | 64 | 2.8% |
| FMA | 969 | 42.8% |
| ALU | 550 | 24.3% |
| Shuffle | 1 | 0.0% |
| Uniform | 116 | 5.1% |
| Branch | 174 | 7.7% |
| Other | 19 | 0.8% |
| **Total** | **2266** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 180 | 48.3% | 7.9% |
| Store | 193 | 51.7% | 8.5% |
| **LSU total** | **373** | **100.0%** | **16.5%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Store | `STS.64` | 128 |
| LSU | Load | `LDS.128` | 64 |
| LSU | Store | `STG.E.128` | 64 |
| LSU | Load | `LDGSTS.E.BYPASS.LTC128B.128` | 40 |
| LSU | Load | `LDG.E.LTC128B.128` | 32 |
| LSU | Load | `LDSM.16.M88.4` | 24 |
| LSU | Load | `LDGDEPBAR` | 6 |
| LSU | Load | `LDG.E.64` | 4 |
| LSU | Load | `LDS` | 4 |
| LSU | Load | `LDG.E` | 2 |
| LSU | Load | `LDG.E.STRONG.GPU` | 2 |
| LSU | Load | `LDC` | 1 |
| LSU | Load | `MEMBAR.ALL.GPU` | 1 |
| LSU | Store | `STG.E.STRONG.GPU` | 1 |
| Tensor core |  | `HMMA.1688.F32.TF32` | 64 |
| FMA |  | `FMNMX` | 256 |
| FMA |  | `FMUL` | 256 |
| FMA |  | `FFMA` | 128 |
| FMA |  | `FSETP.GEU.AND` | 96 |
| FMA |  | `IMAD.IADD.U32` | 96 |
| FMA |  | `IMAD` | 38 |
| FMA |  | `IMAD.MOV.U32` | 30 |
| FMA |  | `IMAD.IADD` | 24 |
| FMA |  | `IMAD.SHL.U32` | 21 |
| FMA |  | `IMAD.WIDE.U32` | 12 |
| FMA |  | `IMAD.WIDE` | 6 |
| FMA |  | `IMAD.U32` | 3 |
| FMA |  | `FSEL` | 2 |
| FMA |  | `FSETP.NEU.AND` | 1 |
| ALU |  | `ISETP.LT.AND` | 114 |
| ALU |  | `IADD3` | 96 |
| ALU |  | `IADD3.X` | 51 |
| ALU |  | `ISETP.NE.AND` | 43 |
| ALU |  | `LOP3.LUT` | 40 |
| ALU |  | `SHF.R.S32.HI` | 30 |
| ALU |  | `P2R` | 21 |
| ALU |  | `ISETP.GE.AND` | 20 |
| ALU |  | `LEA.HI` | 18 |
| ALU |  | `PLOP3.LUT` | 18 |
| ALU |  | `SEL` | 17 |
| ALU |  | `R2P` | 15 |
| ALU |  | `MOV` | 12 |
| ALU |  | `LEA` | 10 |
| ALU |  | `LEA.HI.X` | 7 |
| ALU |  | `ISETP.GT.AND` | 6 |
| ALU |  | `PRMT` | 6 |
| ALU |  | `SHF.R.U32.HI` | 6 |
| ALU |  | `ISETP.NE.U32.AND` | 5 |
| ALU |  | `ISETP.NE.AND.EX` | 4 |
| ALU |  | `SHF.L.U32` | 3 |
| ALU |  | `LEA.HI.SX32` | 2 |
| ALU |  | `P2R.B1` | 2 |
| ALU |  | `IMNMX.U32` | 1 |
| ALU |  | `ISETP.EQ.AND` | 1 |
| ALU |  | `ISETP.GE.OR` | 1 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| Shuffle |  | `SHFL.IDX` | 1 |
| Uniform |  | `UIADD3` | 25 |
| Uniform |  | `ULDC.64` | 16 |
| Uniform |  | `UMOV` | 15 |
| Uniform |  | `UIADD3.X` | 12 |
| Uniform |  | `UISETP.NE.AND` | 8 |
| Uniform |  | `USHF.L.U32` | 6 |
| Uniform |  | `UIMAD` | 5 |
| Uniform |  | `ULOP3.LUT` | 4 |
| Uniform |  | `USEL` | 4 |
| Uniform |  | `ULDC` | 3 |
| Uniform |  | `ULEA.HI` | 3 |
| Uniform |  | `USHF.R.S32.HI` | 3 |
| Uniform |  | `UIMAD.WIDE` | 2 |
| Uniform |  | `ULEA` | 2 |
| Uniform |  | `USHF.L.U64.HI` | 2 |
| Uniform |  | `UIMAD.WIDE.U32` | 1 |
| Uniform |  | `UISETP.GE.AND` | 1 |
| Uniform |  | `UISETP.GE.U32.AND` | 1 |
| Uniform |  | `UISETP.GT.U32.AND` | 1 |
| Uniform |  | `UISETP.LT.AND` | 1 |
| Uniform |  | `ULEA.HI.SX32` | 1 |
| Branch |  | `CS2R` | 72 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 38 |
| Branch |  | `BRA` | 23 |
| Branch |  | `NOP` | 12 |
| Branch |  | `S2R` | 5 |
| Branch |  | `DEPBAR.LE` | 4 |
| Branch |  | `EXIT` | 4 |
| Branch |  | `R2UR` | 3 |
| Branch |  | `B2R.RESULT` | 2 |
| Branch |  | `BAR.RED.AND.DEFER_BLOCKING` | 2 |
| Branch |  | `S2UR` | 2 |
| Branch |  | `WARPSYNC` | 2 |
| Branch |  | `BRX` | 1 |
| Branch |  | `BSSY` | 1 |
| Branch |  | `BSYNC` | 1 |
| Branch |  | `ERRBAR` | 1 |
| Branch |  | `YIELD` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
| Other |  | `CCTL.IVALL` | 2 |
