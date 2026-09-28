# SASS instruction inventory — `vectorized_elementwise_kernel__6c608a7c_sm87.sass`

Total instructions: **1232** (file has 1232 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 159 | 12.9% |
| FMA | 267 | 21.7% |
| ALU | 292 | 23.7% |
| XU | 256 | 20.8% |
| Uniform | 4 | 0.3% |
| Branch | 237 | 19.2% |
| Other | 17 | 1.4% |
| **Total** | **1232** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 128 | 80.5% | 10.4% |
| Store | 31 | 19.5% | 2.5% |
| **LSU total** | **159** | **100.0%** | **12.9%** |

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
| ALU |  | `IADD3` | 70 |
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
| Branch |  | `BSSY` | 59 |
| Branch |  | `BSYNC` | 59 |
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
