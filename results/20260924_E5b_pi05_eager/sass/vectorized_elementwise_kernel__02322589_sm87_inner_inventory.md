<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `vectorized_elementwise_kernel__02322589_sm87.sass` (PC range 0x2299b00-0x229a3f0)

Total instructions in PC range: **144**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 12 | 8.3% |
| FMA | 44 | 30.6% |
| ALU | 35 | 24.3% |
| Uniform | 2 | 1.4% |
| Branch | 34 | 23.6% |
| Other | 17 | 11.8% |
| **Total** | **144** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Store | 12 | 100.0% | 8.3% |
| **LSU total** | **12** | **100.0%** | **8.3%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Store | `STG.E.64` | 8 |
| LSU | Store | `STG.E.128` | 4 |
| FMA |  | `IMAD.MOV.U32` | 19 |
| FMA |  | `IMAD.WIDE.U32` | 9 |
| FMA |  | `IMAD.IADD` | 8 |
| FMA |  | `HFMA2.MMA` | 6 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| FMA |  | `IMAD.WIDE` | 1 |
| ALU |  | `MOV` | 17 |
| ALU |  | `IADD3` | 8 |
| ALU |  | `ISETP.GE.AND` | 8 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 14 |
| Branch |  | `BRA` | 8 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 6 ÷ 12 = 0.5. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 0 : 12
