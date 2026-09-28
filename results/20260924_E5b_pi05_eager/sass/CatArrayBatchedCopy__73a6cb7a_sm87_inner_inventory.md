<!-- inner loop: backward BRA 0x2272310 -> 0x2271df0: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `CatArrayBatchedCopy__73a6cb7a_sm87.sass` (PC range 0x2271df0-0x2272310)

Total instructions in PC range: **83**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 2 | 2.4% |
| FMA | 40 | 48.2% |
| ALU | 40 | 48.2% |
| Branch | 1 | 1.2% |
| **Total** | **83** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 50.0% | 1.2% |
| Store | 1 | 50.0% | 1.2% |
| **LSU total** | **2** | **100.0%** | **2.4%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E` | 1 |
| LSU | Store | `STG.E` | 1 |
| FMA |  | `IMAD` | 22 |
| FMA |  | `IMAD.MOV` | 7 |
| FMA |  | `IMAD.HI.U32` | 6 |
| FMA |  | `IMAD.IADD` | 3 |
| FMA |  | `IMAD.WIDE.U32` | 2 |
| ALU |  | `IADD3` | 20 |
| ALU |  | `ISETP.GE.U32.AND` | 13 |
| ALU |  | `SEL` | 6 |
| ALU |  | `MOV` | 1 |
| Branch |  | `BRA` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 0 ÷ 2 = 0.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 1 : 1
