<!-- inner loop: backward BRA with the largest range (no per-PC counts available) -->
# SASS instruction inventory — `vectorized_elementwise_kernel__e1a0eced_sm87.sass` (PC range 0x22b6790-0x22b6920)

Total instructions in PC range: **26**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 2 | 7.7% |
| FMA | 14 | 53.8% |
| ALU | 8 | 30.8% |
| Uniform | 1 | 3.8% |
| Branch | 1 | 3.8% |
| **Total** | **26** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 50.0% | 3.8% |
| Store | 1 | 50.0% | 3.8% |
| **LSU total** | **2** | **100.0%** | **7.7%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.64.CONSTANT` | 1 |
| LSU | Store | `STL.64` | 1 |
| FMA |  | `IMAD.X` | 4 |
| FMA |  | `IMAD` | 3 |
| FMA |  | `IMAD.HI.U32` | 3 |
| FMA |  | `IMAD.MOV.U32` | 3 |
| FMA |  | `IMAD.WIDE.U32` | 1 |
| ALU |  | `IADD3` | 5 |
| ALU |  | `IADD3.X` | 2 |
| ALU |  | `ISETP.GE.AND` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Branch |  | `BRA` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 0 ÷ 1 = 0.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 1 : 1
