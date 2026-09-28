<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `triton_poi_fused_cat_5_sm87.sass` (PC range 0x2293b00-0x2293e70)

Total instructions in PC range: **56**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 3 | 5.4% |
| FMA | 7 | 12.5% |
| ALU | 9 | 16.1% |
| Uniform | 2 | 3.6% |
| Branch | 18 | 32.1% |
| Other | 17 | 30.4% |
| **Total** | **56** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 2 | 66.7% | 3.6% |
| Store | 1 | 33.3% | 1.8% |
| **LSU total** | **3** | **100.0%** | **5.4%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.64` | 2 |
| LSU | Store | `STG.E.64` | 1 |
| FMA |  | `IMAD.WIDE` | 3 |
| FMA |  | `IMAD.MOV.U32` | 2 |
| FMA |  | `IMAD.SHL.U32` | 2 |
| ALU |  | `IADD3` | 2 |
| ALU |  | `ISETP.GE.AND` | 2 |
| ALU |  | `SEL` | 2 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LOP3.LUT` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 10 |
| Branch |  | `BRA` | 2 |
| Branch |  | `CS2R` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `S2R` | 2 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 0 ÷ 3 = 0.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 2 : 1
