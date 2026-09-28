<!-- inner loop: backward BRA 0x22d0ee0 -> 0x22d0b20: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `CatArrayBatchedCopy__ff2753b7_sm87.sass` (PC range 0x22d0b20-0x22d0ee0)

Total instructions in PC range: **61**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 2 | 3.3% |
| FMA | 29 | 47.5% |
| ALU | 29 | 47.5% |
| Branch | 1 | 1.6% |
| **Total** | **61** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 50.0% | 1.6% |
| Store | 1 | 50.0% | 1.6% |
| **LSU total** | **2** | **100.0%** | **3.3%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.U8` | 1 |
| LSU | Store | `STG.E.U8` | 1 |
| FMA |  | `IMAD` | 16 |
| FMA |  | `IMAD.MOV` | 5 |
| FMA |  | `IMAD.HI.U32` | 4 |
| FMA |  | `IMAD.IADD` | 3 |
| FMA |  | `IMAD.X` | 1 |
| ALU |  | `IADD3` | 14 |
| ALU |  | `ISETP.GE.U32.AND` | 9 |
| ALU |  | `SEL` | 4 |
| ALU |  | `IADD3.X` | 1 |
| ALU |  | `MOV` | 1 |
| Branch |  | `BRA` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 0 ÷ 2 = 0.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 1 : 1
