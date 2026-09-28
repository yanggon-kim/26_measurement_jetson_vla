<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `triton_poi_fused_bitwise_and_cat_le_mul_scalar_tensor_unsqueeze_where_5_sm87.sass` (PC range 0x227af00-0x227b4f0)

Total instructions in PC range: **96**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 5 | 5.2% |
| FMA | 23 | 24.0% |
| ALU | 23 | 24.0% |
| Uniform | 2 | 2.1% |
| Branch | 26 | 27.1% |
| Other | 17 | 17.7% |
| **Total** | **96** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 4 | 80.0% | 4.2% |
| Store | 1 | 20.0% | 1.0% |
| **LSU total** | **5** | **100.0%** | **5.2%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.EL` | 2 |
| LSU | Load | `LDG.E.EL.64` | 1 |
| LSU | Load | `LDG.E.EL.U16` | 1 |
| LSU | Store | `STG.E.64` | 1 |
| FMA |  | `IMAD.MOV.U32` | 14 |
| FMA |  | `IMAD.WIDE` | 3 |
| FMA |  | `FSETP.GTU.AND` | 2 |
| FMA |  | `IMAD` | 2 |
| FMA |  | `IMAD.HI` | 1 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `SEL` | 6 |
| ALU |  | `LOP3.LUT` | 3 |
| ALU |  | `ISETP.GE.AND` | 2 |
| ALU |  | `ISETP.LT.AND` | 2 |
| ALU |  | `PRMT` | 2 |
| ALU |  | `SHF.R.U32.HI` | 2 |
| ALU |  | `IADD3` | 1 |
| ALU |  | `ISETP.GT.AND` | 1 |
| ALU |  | `ISETP.NE.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LEA.HI.SX32` | 1 |
| ALU |  | `LEA.HI.X.SX32` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 11 |
| Branch |  | `R2UR` | 8 |
| Branch |  | `BRA` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `S2R` | 2 |
| Branch |  | `CS2R` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 0 ÷ 5 = 0.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 4 : 1
