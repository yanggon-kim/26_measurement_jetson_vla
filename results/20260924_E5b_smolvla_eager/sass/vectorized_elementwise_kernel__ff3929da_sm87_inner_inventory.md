<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `vectorized_elementwise_kernel__ff3929da_sm87.sass` (PC range 0x218e700-0x218f2f0)

Total instructions in PC range: **192**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 20 | 10.4% |
| FMA | 58 | 30.2% |
| ALU | 60 | 31.2% |
| Uniform | 2 | 1.0% |
| Branch | 35 | 18.2% |
| Other | 17 | 8.9% |
| **Total** | **192** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 10 | 50.0% | 5.2% |
| Store | 10 | 50.0% | 5.2% |
| **LSU total** | **20** | **100.0%** | **10.4%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E` | 8 |
| LSU | Store | `STG.E.U16` | 8 |
| LSU | Load | `LDG.E.128` | 2 |
| LSU | Store | `STG.E.64` | 2 |
| FMA |  | `IMAD.MOV.U32` | 22 |
| FMA |  | `IMAD.WIDE.U32` | 18 |
| FMA |  | `IMAD.IADD` | 14 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `HFMA2.MMA` | 1 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `IADD3` | 23 |
| ALU |  | `ISETP.GE.AND` | 22 |
| ALU |  | `F2FP.BF16.PACK_AB` | 12 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `PRMT` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 12 |
| Branch |  | `BRA` | 8 |
| Branch |  | `CS2R` | 3 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `S2R` | 3 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 1 ÷ 20 = 0.05. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 10 : 10
