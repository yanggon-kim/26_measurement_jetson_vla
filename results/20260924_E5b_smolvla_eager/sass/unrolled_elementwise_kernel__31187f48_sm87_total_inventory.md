# SASS instruction inventory — `unrolled_elementwise_kernel__31187f48_sm87.sass`

Total instructions: **1888** (file has 1888 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 178 | 9.4% |
| FMA | 253 | 13.4% |
| ALU | 864 | 45.8% |
| Uniform | 2 | 0.1% |
| Branch | 558 | 29.6% |
| Other | 33 | 1.7% |
| **Total** | **1888** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 94 | 52.8% | 5.0% |
| Store | 84 | 47.2% | 4.4% |
| **LSU total** | **178** | **100.0%** | **9.4%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Store | `STG.E.U8` | 32 |
| LSU | Load | `LDG.E.U8` | 28 |
| LSU | Load | `LDG.E.U16` | 20 |
| LSU | Load | `LDG.E` | 16 |
| LSU | Load | `LDG.E.64` | 16 |
| LSU | Store | `STG.E` | 16 |
| LSU | Store | `STG.E.64` | 16 |
| LSU | Store | `STG.E.U16` | 16 |
| LSU | Load | `LDC.64` | 8 |
| LSU | Load | `LDG.E.S8` | 4 |
| LSU | Store | `STG.E.128` | 4 |
| LSU | Load | `LDC.U8` | 2 |
| FMA |  | `IMAD.MOV.U32` | 144 |
| FMA |  | `IMAD.SHL.U32` | 32 |
| FMA |  | `IMAD` | 21 |
| FMA |  | `FADD` | 20 |
| FMA |  | `HADD2.F32` | 8 |
| FMA |  | `IMAD.U32` | 8 |
| FMA |  | `IMAD.X` | 8 |
| FMA |  | `FMUL` | 4 |
| FMA |  | `FSETP.NEU.AND` | 4 |
| FMA |  | `IMAD.MOV` | 4 |
| ALU |  | `ISETP.NE.AND` | 200 |
| ALU |  | `LOP3.LUT` | 157 |
| ALU |  | `IADD3` | 90 |
| ALU |  | `SHF.R.U32.HI` | 72 |
| ALU |  | `ISETP.GT.AND` | 56 |
| ALU |  | `MOV` | 40 |
| ALU |  | `PRMT` | 32 |
| ALU |  | `ISETP.GT.U32.AND` | 24 |
| ALU |  | `ISETP.GE.U32.AND` | 20 |
| ALU |  | `SEL` | 16 |
| ALU |  | `F2I.TRUNC.NTZ` | 12 |
| ALU |  | `FLO.U32` | 12 |
| ALU |  | `SHF.L.U32` | 12 |
| ALU |  | `F2F.F32.F64` | 8 |
| ALU |  | `F2F.F64.F32` | 8 |
| ALU |  | `F2FP.PACK_AB` | 8 |
| ALU |  | `F2I.S64.TRUNC` | 8 |
| ALU |  | `F2I.U32.TRUNC.NTZ` | 8 |
| ALU |  | `I2F.S16` | 8 |
| ALU |  | `I2F.U16` | 8 |
| ALU |  | `IADD3.X` | 8 |
| ALU |  | `ISETP.GE.AND` | 8 |
| ALU |  | `LEA` | 8 |
| ALU |  | `PLOP3.LUT` | 8 |
| ALU |  | `SHF.R.S32.HI` | 8 |
| ALU |  | `F2FP.BF16.PACK_AB` | 4 |
| ALU |  | `F2I.U64.TRUNC` | 4 |
| ALU |  | `I2F.S64` | 4 |
| ALU |  | `I2F.U64` | 4 |
| ALU |  | `ISETP.EQ.U32.AND` | 4 |
| ALU |  | `LEA.HI` | 4 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 444 |
| Branch |  | `BSSY` | 33 |
| Branch |  | `BSYNC` | 33 |
| Branch |  | `EXIT` | 23 |
| Branch |  | `NOP` | 9 |
| Branch |  | `CALL.ABS.NOINC` | 8 |
| Branch |  | `CS2R` | 4 |
| Branch |  | `S2R` | 4 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `LEPC` | 8 |
| Other |  | `I2FP.F32.S32` | 4 |
| Other |  | `I2FP.F32.U32` | 4 |
| Other |  | `BMOV.32` | 2 |
