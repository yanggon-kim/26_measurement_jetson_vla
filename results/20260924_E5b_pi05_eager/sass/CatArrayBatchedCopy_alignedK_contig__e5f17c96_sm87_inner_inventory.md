<!-- inner loop: backward BRA 0x22bd500 -> 0x22bd270: of the 3 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `CatArrayBatchedCopy_alignedK_contig__e5f17c96_sm87.sass` (PC range 0x22bd270-0x22bd500)

Total instructions in PC range: **42**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 3 | 7.1% |
| FMA | 20 | 47.6% |
| ALU | 18 | 42.9% |
| Branch | 1 | 2.4% |
| **Total** | **42** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 33.3% | 2.4% |
| Store | 2 | 66.7% | 4.8% |
| **LSU total** | **3** | **100.0%** | **7.1%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Store | `STG.E.64` | 2 |
| LSU | Load | `LDG.E.128` | 1 |
| FMA |  | `IMAD` | 10 |
| FMA |  | `IMAD.WIDE.U32` | 3 |
| FMA |  | `IMAD.HI.U32` | 2 |
| FMA |  | `IMAD.MOV` | 2 |
| FMA |  | `IMAD.MOV.U32` | 2 |
| FMA |  | `IMAD.IADD` | 1 |
| ALU |  | `IADD3` | 9 |
| ALU |  | `ISETP.GE.U32.AND` | 4 |
| ALU |  | `SEL` | 2 |
| ALU |  | `ISETP.GT.U32.AND` | 1 |
| ALU |  | `LEA` | 1 |
| ALU |  | `MOV` | 1 |
| Branch |  | `BRA` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 0 ÷ 3 = 0.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 1 : 2
