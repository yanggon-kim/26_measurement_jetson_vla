<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `vectorized_elementwise_kernel__e54a6266_sm87.sass` (PC range 0x218b900-0x218c370)

Total instructions in PC range: **168**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 21 | 12.5% |
| FMA | 36 | 21.4% |
| ALU | 54 | 32.1% |
| Uniform | 2 | 1.2% |
| Branch | 38 | 22.6% |
| Other | 17 | 10.1% |
| **Total** | **168** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 4.8% | 0.6% |
| Store | 20 | 95.2% | 11.9% |
| **LSU total** | **21** | **100.0%** | **12.5%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Store | `STG.E.U8` | 16 |
| LSU | Store | `STG.E` | 4 |
| LSU | Load | `LDC.U8` | 1 |
| FMA |  | `IMAD.X` | 17 |
| FMA |  | `IMAD.IADD` | 16 |
| FMA |  | `IMAD.MOV.U32` | 1 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| FMA |  | `IMAD.WIDE` | 1 |
| ALU |  | `IADD3` | 33 |
| ALU |  | `ISETP.GE.AND` | 16 |
| ALU |  | `PRMT` | 3 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 16 |
| Branch |  | `NOP` | 10 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 0 ÷ 20 = 0.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 1 : 20
