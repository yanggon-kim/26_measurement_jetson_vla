# SASS instruction inventory — `vectorized_elementwise_kernel__e1a0eced_sm87.sass`

Total instructions: **1184** (file has 1184 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 159 | 13.4% |
| FMA | 267 | 22.6% |
| ALU | 276 | 23.3% |
| XU | 256 | 21.6% |
| Uniform | 4 | 0.3% |
| Branch | 205 | 17.3% |
| Other | 17 | 1.4% |
| **Total** | **1184** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 128 | 80.5% | 10.8% |
| Store | 31 | 19.5% | 2.6% |
| **LSU total** | **159** | **100.0%** | **13.4%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.64.CONSTANT` | 97 |
| LSU | Store | `STL` | 17 |
| LSU | Load | `LDL` | 16 |
| LSU | Load | `LDG.E.64` | 8 |
| LSU | Store | `STG.E.64` | 8 |
| LSU | Load | `LDG.E.128` | 4 |
| LSU | Store | `STG.E.128` | 4 |
| LSU | Load | `LDL.64` | 3 |
| LSU | Store | `STL.64` | 2 |
| FMA |  | `IMAD.MOV.U32` | 134 |
| FMA |  | `FSEL` | 32 |
| FMA |  | `IMAD.WIDE.U32` | 22 |
| FMA |  | `IMAD.SHL.U32` | 21 |
| FMA |  | `IMAD.WIDE` | 19 |
| FMA |  | `IMAD.IADD` | 18 |
| FMA |  | `IMAD.X` | 9 |
| FMA |  | `IMAD` | 5 |
| FMA |  | `IMAD.HI.U32` | 4 |
| FMA |  | `IMAD.MOV` | 3 |
| ALU |  | `IADD3` | 54 |
| ALU |  | `LOP3.LUT` | 51 |
| ALU |  | `ISETP.GE.AND` | 24 |
| ALU |  | `ISETP.NE.AND` | 17 |
| ALU |  | `F2I.F64` | 16 |
| ALU |  | `I2F.F64` | 16 |
| ALU |  | `ISETP.EQ.AND` | 16 |
| ALU |  | `MOV` | 16 |
| ALU |  | `R2P` | 16 |
| ALU |  | `SHF.R.U32.HI` | 9 |
| ALU |  | `SEL` | 8 |
| ALU |  | `IADD3.X` | 7 |
| ALU |  | `SHF.L.U64.HI` | 6 |
| ALU |  | `SHF.R.U64` | 5 |
| ALU |  | `ISETP.NE.U32.AND` | 3 |
| ALU |  | `SHF.L.U32` | 3 |
| ALU |  | `ISETP.GT.AND` | 2 |
| ALU |  | `FLO.U32` | 1 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.GT.AND.EX` | 1 |
| ALU |  | `ISETP.GT.U32.AND` | 1 |
| ALU |  | `ISETP.NE.AND.EX` | 1 |
| ALU |  | `LEA.HI` | 1 |
| ALU |  | `LEA.HI.X` | 1 |
| XU |  | `DFMA` | 192 |
| XU |  | `DMUL` | 48 |
| XU |  | `DSETP.GE.AND` | 16 |
| Uniform |  | `ULDC.64` | 2 |
| Uniform |  | `UMOV` | 2 |
| Branch |  | `BRA` | 67 |
| Branch |  | `BSSY` | 43 |
| Branch |  | `BSYNC` | 43 |
| Branch |  | `CALL.REL.NOINC` | 16 |
| Branch |  | `S2R` | 11 |
| Branch |  | `CS2R` | 10 |
| Branch |  | `NOP` | 9 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `RET.REL.NODEC` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
