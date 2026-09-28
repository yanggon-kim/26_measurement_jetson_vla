<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `vectorized_elementwise_kernel__e7288074_sm87.sass` (PC range 0x2190d00-0x2191470)

Total instructions in PC range: **120**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 10 | 8.3% |
| FMA | 33 | 27.5% |
| ALU | 30 | 25.0% |
| Uniform | 2 | 1.7% |
| Branch | 28 | 23.3% |
| Other | 17 | 14.2% |
| **Total** | **120** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Store | 10 | 100.0% | 8.3% |
| **LSU total** | **10** | **100.0%** | **8.3%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Store | `STG.E` | 8 |
| LSU | Store | `STG.E.128` | 2 |
| FMA |  | `IMAD.MOV.U32` | 16 |
| FMA |  | `IMAD.WIDE.U32` | 9 |
| FMA |  | `HFMA2.MMA` | 3 |
| FMA |  | `IMAD.IADD` | 3 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| FMA |  | `IMAD.WIDE` | 1 |
| ALU |  | `IADD3` | 13 |
| ALU |  | `ISETP.GE.AND` | 8 |
| ALU |  | `MOV` | 7 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 8 |
| Branch |  | `NOP` | 8 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 3 ÷ 10 = 0.3. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 0 : 10
