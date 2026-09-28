<!-- inner loop: backward BRA with the largest range (no per-PC counts available) -->
# SASS instruction inventory — `triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_2_sm87.sass` (PC range 0x2285990-0x2285a60)

Total instructions in PC range: **14**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 2 | 14.3% |
| FMA | 5 | 35.7% |
| ALU | 6 | 42.9% |
| Branch | 1 | 7.1% |
| **Total** | **14** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 50.0% | 7.1% |
| Store | 1 | 50.0% | 7.1% |
| **LSU total** | **2** | **100.0%** | **14.3%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.CONSTANT` | 1 |
| LSU | Store | `STL` | 1 |
| FMA |  | `IMAD.MOV.U32` | 3 |
| FMA |  | `IMAD.WIDE.U32` | 1 |
| FMA |  | `IMAD.X` | 1 |
| ALU |  | `IADD3` | 3 |
| ALU |  | `IADD3.X` | 1 |
| ALU |  | `ISETP.NE.AND.EX` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Branch |  | `BRA` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 0 ÷ 1 = 0.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 1 : 1
