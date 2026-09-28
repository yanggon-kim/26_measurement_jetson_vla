<!-- inner loop: backward BRA with the largest range (no per-PC counts available) -->
# SASS instruction inventory — `triton_poi_fused__to_copy_cat_cos_linspace_mul_pow_reciprocal_sin_unsqueeze_3_sm87.sass` (PC range 0x2279050-0x2279210)

Total instructions in PC range: **29**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 2 | 6.9% |
| FMA | 15 | 51.7% |
| ALU | 10 | 34.5% |
| Uniform | 1 | 3.4% |
| Branch | 1 | 3.4% |
| **Total** | **29** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 50.0% | 3.4% |
| Store | 1 | 50.0% | 3.4% |
| **LSU total** | **2** | **100.0%** | **6.9%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.64.CONSTANT` | 1 |
| LSU | Store | `STL.64` | 1 |
| FMA |  | `IMAD.MOV.U32` | 4 |
| FMA |  | `IMAD.X` | 4 |
| FMA |  | `IMAD` | 3 |
| FMA |  | `IMAD.HI.U32` | 3 |
| FMA |  | `IMAD.WIDE.U32` | 1 |
| ALU |  | `IADD3` | 5 |
| ALU |  | `IADD3.X` | 3 |
| ALU |  | `ISETP.NE.AND.EX` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Branch |  | `BRA` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 0 ÷ 1 = 0.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 1 : 1
