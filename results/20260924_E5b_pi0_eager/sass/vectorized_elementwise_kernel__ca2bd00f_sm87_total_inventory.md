# SASS instruction inventory — `vectorized_elementwise_kernel__ca2bd00f_sm87.sass`

Total instructions: **2336** (file has 2336 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 36 | 1.5% |
| FMA | 1067 | 45.7% |
| ALU | 911 | 39.0% |
| XU | 16 | 0.7% |
| Uniform | 66 | 2.8% |
| Branch | 207 | 8.9% |
| Other | 33 | 1.4% |
| **Total** | **2336** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 26 | 72.2% | 1.1% |
| Store | 10 | 27.8% | 0.4% |
| **LSU total** | **36** | **100.0%** | **1.5%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.CONSTANT` | 16 |
| LSU | Load | `LDG.E` | 8 |
| LSU | Store | `STG.E` | 8 |
| LSU | Load | `LDG.E.128` | 2 |
| LSU | Store | `STG.E.128` | 2 |
| FMA |  | `IMAD.MOV.U32` | 583 |
| FMA |  | `FFMA` | 144 |
| FMA |  | `FSEL` | 64 |
| FMA |  | `IMAD.SHL.U32` | 64 |
| FMA |  | `FMUL` | 48 |
| FMA |  | `IMAD.IADD` | 48 |
| FMA |  | `IMAD.WIDE.U32` | 34 |
| FMA |  | `IMAD.U32` | 32 |
| FMA |  | `FSETP.GE.AND` | 16 |
| FMA |  | `FSETP.NEU.AND` | 16 |
| FMA |  | `IMAD.MOV` | 16 |
| FMA |  | `IMAD.WIDE` | 2 |
| ALU |  | `ISETP.EQ.AND` | 336 |
| ALU |  | `IADD3` | 149 |
| ALU |  | `LOP3.LUT` | 144 |
| ALU |  | `SHF.R.U32.HI` | 80 |
| ALU |  | `ISETP.NE.AND` | 48 |
| ALU |  | `SHF.L.U32` | 33 |
| ALU |  | `ISETP.GE.AND` | 22 |
| ALU |  | `F2F.F32.F64` | 16 |
| ALU |  | `F2I.NTZ` | 16 |
| ALU |  | `I2F.F64.S64` | 16 |
| ALU |  | `IADD3.X` | 16 |
| ALU |  | `LEA.HI` | 16 |
| ALU |  | `SHF.L.U32.HI` | 16 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `MOV` | 1 |
| XU |  | `DMUL` | 16 |
| Uniform |  | `ULDC.64` | 17 |
| Uniform |  | `UMOV` | 17 |
| Uniform |  | `UIADD3` | 16 |
| Uniform |  | `UIADD3.X` | 16 |
| Branch |  | `BRA` | 96 |
| Branch |  | `BSSY` | 42 |
| Branch |  | `BSYNC` | 42 |
| Branch |  | `CS2R` | 11 |
| Branch |  | `NOP` | 8 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `I2FP.F32.S32` | 16 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
