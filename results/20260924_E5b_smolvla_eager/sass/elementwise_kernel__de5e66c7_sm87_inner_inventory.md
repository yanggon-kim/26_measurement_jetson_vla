<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `elementwise_kernel__de5e66c7_sm87.sass` (PC range 0x4cb74f00-0x4cb841f0)

Total instructions in PC range: **3888**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 276 | 7.1% |
| FMA | 1084 | 27.9% |
| ALU | 1552 | 39.9% |
| XU | 8 | 0.2% |
| Uniform | 2 | 0.1% |
| Branch | 937 | 24.1% |
| Other | 29 | 0.7% |
| **Total** | **3888** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 192 | 69.6% | 4.9% |
| Store | 84 | 30.4% | 2.2% |
| **LSU total** | **276** | **100.0%** | **7.1%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.U8` | 56 |
| LSU | Load | `LDG.E.U16` | 40 |
| LSU | Load | `LDG.E` | 32 |
| LSU | Load | `LDG.E.64` | 32 |
| LSU | Store | `STG.E.U8` | 32 |
| LSU | Store | `STG.E` | 16 |
| LSU | Store | `STG.E.64` | 16 |
| LSU | Store | `STG.E.U16` | 16 |
| LSU | Load | `LDC.64` | 12 |
| LSU | Load | `LDC.U8` | 12 |
| LSU | Load | `LDG.E.S8` | 8 |
| LSU | Store | `STG.E.128` | 4 |
| FMA |  | `IMAD` | 411 |
| FMA |  | `IMAD.MOV.U32` | 321 |
| FMA |  | `IMAD.MOV` | 104 |
| FMA |  | `IMAD.HI.U32` | 100 |
| FMA |  | `IMAD.SHL.U32` | 64 |
| FMA |  | `FADD` | 24 |
| FMA |  | `HADD2.F32` | 16 |
| FMA |  | `IMAD.U32` | 16 |
| FMA |  | `IMAD.X` | 12 |
| FMA |  | `FMUL` | 8 |
| FMA |  | `FSEL` | 8 |
| ALU |  | `ISETP.NE.AND` | 412 |
| ALU |  | `LOP3.LUT` | 220 |
| ALU |  | `SHF.R.U32.HI` | 188 |
| ALU |  | `IADD3` | 135 |
| ALU |  | `ISETP.GT.AND` | 84 |
| ALU |  | `F2F.F64.F32` | 80 |
| ALU |  | `MOV` | 60 |
| ALU |  | `F2F.F32.F64` | 40 |
| ALU |  | `PRMT` | 40 |
| ALU |  | `ISETP.GT.U32.AND` | 28 |
| ALU |  | `FLO.U32` | 24 |
| ALU |  | `ISETP.GE.U32.AND` | 24 |
| ALU |  | `SHF.L.U32` | 24 |
| ALU |  | `I2F.F64.S16` | 16 |
| ALU |  | `I2F.F64.U16` | 16 |
| ALU |  | `LEA` | 16 |
| ALU |  | `SEL` | 16 |
| ALU |  | `SHF.R.S32.HI` | 16 |
| ALU |  | `F2I.F64.TRUNC` | 12 |
| ALU |  | `IADD3.X` | 12 |
| ALU |  | `F2FP.PACK_AB` | 8 |
| ALU |  | `F2I.S64.F64.TRUNC` | 8 |
| ALU |  | `F2I.U32.F64.TRUNC` | 8 |
| ALU |  | `I2F.F64` | 8 |
| ALU |  | `I2F.F64.S64` | 8 |
| ALU |  | `I2F.F64.U32` | 8 |
| ALU |  | `I2F.F64.U64` | 8 |
| ALU |  | `LEA.HI` | 8 |
| ALU |  | `PLOP3.LUT` | 8 |
| ALU |  | `F2FP.BF16.PACK_AB` | 4 |
| ALU |  | `F2I.U64.F64.TRUNC` | 4 |
| ALU |  | `ISETP.EQ.U32.AND` | 4 |
| ALU |  | `ISETP.GE.AND` | 4 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| XU |  | `DMUL` | 4 |
| XU |  | `DSETP.NEU.AND` | 4 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 764 |
| Branch |  | `BSSY` | 58 |
| Branch |  | `BSYNC` | 58 |
| Branch |  | `EXIT` | 23 |
| Branch |  | `CALL.ABS.NOINC` | 12 |
| Branch |  | `CS2R` | 12 |
| Branch |  | `NOP` | 8 |
| Branch |  | `S2R` | 2 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `LEPC` | 12 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 52 ÷ 252 = 0.206. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 192 : 84
