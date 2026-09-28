<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `vectorized_elementwise_kernel__ee7915be_sm87.sass` (PC range 0x223ea00-0x223f870)

Total instructions in PC range: **232**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 24 | 10.3% |
| FMA | 79 | 34.1% |
| ALU | 71 | 30.6% |
| Uniform | 2 | 0.9% |
| Branch | 39 | 16.8% |
| Other | 17 | 7.3% |
| **Total** | **232** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 12 | 50.0% | 5.2% |
| Store | 12 | 50.0% | 5.2% |
| **LSU total** | **24** | **100.0%** | **10.3%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.64` | 8 |
| LSU | Store | `STG.E.64` | 8 |
| LSU | Load | `LDG.E.128` | 4 |
| LSU | Store | `STG.E.128` | 4 |
| FMA |  | `IMAD.WIDE.U32` | 34 |
| FMA |  | `IMAD.MOV.U32` | 17 |
| FMA |  | `IMAD.IADD` | 16 |
| FMA |  | `HFMA2.MMA` | 5 |
| FMA |  | `IMAD` | 4 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `IADD3` | 30 |
| ALU |  | `MOV` | 24 |
| ALU |  | `ISETP.GE.AND` | 15 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 11 |
| Branch |  | `BRA` | 8 |
| Branch |  | `CS2R` | 8 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 5 ÷ 24 = 0.208. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 12 : 12
