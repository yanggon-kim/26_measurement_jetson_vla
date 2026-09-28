<!-- inner loop: backward BRA 0x22e8380 -> 0x22e7710: of the 3 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `CatArrayBatchedCopy_alignedK_contig__8239fae5_sm87.sass` (PC range 0x22e7710-0x22e8380)

Total instructions in PC range: **200**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 5 | 2.5% |
| FMA | 98 | 49.0% |
| ALU | 93 | 46.5% |
| XU | 3 | 1.5% |
| Branch | 1 | 0.5% |
| **Total** | **200** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 20.0% | 0.5% |
| Store | 4 | 80.0% | 2.0% |
| **LSU total** | **5** | **100.0%** | **2.5%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Store | `STG.E` | 4 |
| LSU | Load | `LDG.E.128` | 1 |
| FMA |  | `IMAD` | 48 |
| FMA |  | `IMAD.MOV` | 19 |
| FMA |  | `IMAD.HI.U32` | 15 |
| FMA |  | `IMAD.IADD` | 7 |
| FMA |  | `IMAD.WIDE.U32` | 5 |
| FMA |  | `IMAD.MOV.U32` | 3 |
| FMA |  | `HFMA2.MMA` | 1 |
| ALU |  | `IADD3` | 44 |
| ALU |  | `ISETP.GE.U32.AND` | 24 |
| ALU |  | `SEL` | 12 |
| ALU |  | `F2I.FTZ.U32.TRUNC.NTZ` | 3 |
| ALU |  | `I2F.U32.RP` | 3 |
| ALU |  | `ISETP.NE.U32.AND` | 3 |
| ALU |  | `LOP3.LUT` | 3 |
| ALU |  | `ISETP.GT.U32.AND` | 1 |
| XU |  | `MUFU.RCP` | 3 |
| Branch |  | `BRA` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 4 ÷ 5 = 0.8. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 1 : 4
