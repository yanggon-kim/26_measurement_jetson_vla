<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `unrolled_elementwise_kernel__3282ef26_sm87.sass` (PC range 0x2223900-0x222b2f0)

Total instructions in PC range: **1952**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 178 | 9.1% |
| FMA | 256 | 13.1% |
| ALU | 908 | 46.5% |
| Uniform | 2 | 0.1% |
| Branch | 583 | 29.9% |
| Other | 25 | 1.3% |
| **Total** | **1952** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 94 | 52.8% | 4.8% |
| Store | 84 | 47.2% | 4.3% |
| **LSU total** | **178** | **100.0%** | **9.1%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Store | `STG.E.U8` | 32 |
| LSU | Load | `LDG.E.U8` | 28 |
| LSU | Load | `LDG.E` | 16 |
| LSU | Load | `LDG.E.64` | 16 |
| LSU | Load | `LDG.E.U16` | 16 |
| LSU | Store | `STG.E` | 16 |
| LSU | Store | `STG.E.64` | 16 |
| LSU | Store | `STG.E.U16` | 16 |
| LSU | Load | `LDC.64` | 8 |
| LSU | Load | `LDG.E.S16` | 4 |
| LSU | Load | `LDG.E.S8` | 4 |
| LSU | Store | `STG.E.128` | 4 |
| LSU | Load | `LDC.U8` | 2 |
| FMA |  | `IMAD.MOV.U32` | 151 |
| FMA |  | `IMAD.SHL.U32` | 32 |
| FMA |  | `IMAD` | 21 |
| FMA |  | `FADD` | 20 |
| FMA |  | `HADD2.F32` | 8 |
| FMA |  | `IMAD.U32` | 8 |
| FMA |  | `IMAD.X` | 8 |
| FMA |  | `FMUL` | 4 |
| FMA |  | `IMAD.MOV` | 4 |
| ALU |  | `ISETP.NE.AND` | 200 |
| ALU |  | `LOP3.LUT` | 157 |
| ALU |  | `IADD3` | 90 |
| ALU |  | `SHF.R.U32.HI` | 72 |
| ALU |  | `ISETP.GT.AND` | 56 |
| ALU |  | `F2I.S64.TRUNC` | 40 |
| ALU |  | `I2F.S64` | 40 |
| ALU |  | `MOV` | 40 |
| ALU |  | `PRMT` | 32 |
| ALU |  | `ISETP.GT.U32.AND` | 24 |
| ALU |  | `ISETP.GE.U32.AND` | 20 |
| ALU |  | `SHF.R.S32.HI` | 20 |
| ALU |  | `SEL` | 16 |
| ALU |  | `FLO.U32` | 12 |
| ALU |  | `SHF.L.U32` | 12 |
| ALU |  | `F2FP.PACK_AB` | 8 |
| ALU |  | `F2I.S64.F64.TRUNC` | 8 |
| ALU |  | `I2F.F64.S64` | 8 |
| ALU |  | `IADD3.X` | 8 |
| ALU |  | `ISETP.GE.AND` | 8 |
| ALU |  | `LEA` | 8 |
| ALU |  | `PLOP3.LUT` | 8 |
| ALU |  | `ISETP.NE.U32.AND` | 5 |
| ALU |  | `F2FP.BF16.PACK_AB` | 4 |
| ALU |  | `ISETP.EQ.U32.AND` | 4 |
| ALU |  | `ISETP.NE.AND.EX` | 4 |
| ALU |  | `LEA.HI` | 4 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 444 |
| Branch |  | `BSSY` | 42 |
| Branch |  | `BSYNC` | 42 |
| Branch |  | `EXIT` | 23 |
| Branch |  | `CS2R` | 11 |
| Branch |  | `NOP` | 9 |
| Branch |  | `CALL.ABS.NOINC` | 8 |
| Branch |  | `S2R` | 4 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `LEPC` | 8 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 32 ÷ 168 = 0.19. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 94 : 84
