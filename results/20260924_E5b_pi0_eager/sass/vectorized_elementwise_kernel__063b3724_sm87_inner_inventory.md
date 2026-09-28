<!-- inner loop: backward BRA with the largest range (no per-PC counts available) -->
# SASS instruction inventory — `vectorized_elementwise_kernel__063b3724_sm87.sass` (PC range 0x2251e60-0x2251fd0)

Total instructions in PC range: **24**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 1 | 4.2% |
| FMA | 9 | 37.5% |
| ALU | 11 | 45.8% |
| Uniform | 2 | 8.3% |
| Branch | 1 | 4.2% |
| **Total** | **24** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 100.0% | 4.2% |
| **LSU total** | **1** | **100.0%** | **4.2%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.CONSTANT` | 1 |
| FMA |  | `IMAD.MOV.U32` | 6 |
| FMA |  | `IMAD.U32` | 2 |
| FMA |  | `IMAD.WIDE.U32` | 1 |
| ALU |  | `ISETP.EQ.AND` | 6 |
| ALU |  | `IADD3` | 3 |
| ALU |  | `IADD3.X` | 1 |
| ALU |  | `ISETP.NE.AND` | 1 |
| Uniform |  | `UIADD3` | 1 |
| Uniform |  | `UIADD3.X` | 1 |
| Branch |  | `BRA` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 0 ÷ 1 = 0.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 1 : 0
