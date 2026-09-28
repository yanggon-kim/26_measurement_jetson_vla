<!-- inner loop: backward BRA 0x2242ca0 -> 0x2242900: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `CatArrayBatchedCopy__2eb9aed4_sm87.sass` (PC range 0x2242900-0x2242ca0)

Total instructions in PC range: **59**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 2 | 3.4% |
| FMA | 28 | 47.5% |
| ALU | 28 | 47.5% |
| Branch | 1 | 1.7% |
| **Total** | **59** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 50.0% | 1.7% |
| Store | 1 | 50.0% | 1.7% |
| **LSU total** | **2** | **100.0%** | **3.4%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E` | 1 |
| LSU | Store | `STG.E` | 1 |
| FMA |  | `IMAD` | 16 |
| FMA |  | `IMAD.HI.U32` | 4 |
| FMA |  | `IMAD.IADD` | 4 |
| FMA |  | `IMAD.MOV` | 2 |
| FMA |  | `IMAD.WIDE.U32` | 2 |
| ALU |  | `IADD3` | 14 |
| ALU |  | `ISETP.GE.U32.AND` | 9 |
| ALU |  | `SEL` | 4 |
| ALU |  | `MOV` | 1 |
| Branch |  | `BRA` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 0 ÷ 2 = 0.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 1 : 1
