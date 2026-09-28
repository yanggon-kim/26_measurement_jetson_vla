<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `vectorized_elementwise_kernel__3c617d75_sm87.sass` (PC range 0x2188700-0x2189270)

Total instructions in PC range: **184**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 20 | 10.9% |
| FMA | 55 | 29.9% |
| ALU | 51 | 27.7% |
| Uniform | 2 | 1.1% |
| Branch | 39 | 21.2% |
| Other | 17 | 9.2% |
| **Total** | **184** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 10 | 50.0% | 5.4% |
| Store | 10 | 50.0% | 5.4% |
| **LSU total** | **20** | **100.0%** | **10.9%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E` | 8 |
| LSU | Store | `STG.E` | 8 |
| LSU | Load | `LDG.E.128` | 2 |
| LSU | Store | `STG.E.128` | 2 |
| FMA |  | `IMAD.WIDE.U32` | 18 |
| FMA |  | `FFMA` | 16 |
| FMA |  | `IMAD.IADD` | 8 |
| FMA |  | `IMAD.MOV.U32` | 7 |
| FMA |  | `HFMA2.MMA` | 3 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `IADD3` | 23 |
| ALU |  | `ISETP.GE.AND` | 15 |
| ALU |  | `MOV` | 11 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 15 |
| Branch |  | `BRA` | 8 |
| Branch |  | `CS2R` | 4 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 19 ÷ 20 = 0.95. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 10 : 10
