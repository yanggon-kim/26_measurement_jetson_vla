<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `elementwise_kernel__78798574_sm87.sass` (PC range 0x2237f00-0x223c470)

Total instructions in PC range: **1112**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 12 | 1.1% |
| FMA | 536 | 48.2% |
| ALU | 422 | 37.9% |
| Uniform | 2 | 0.2% |
| Branch | 123 | 11.1% |
| Other | 17 | 1.5% |
| **Total** | **1112** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 8 | 66.7% | 0.7% |
| Store | 4 | 33.3% | 0.4% |
| **LSU total** | **12** | **100.0%** | **1.1%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.U8` | 8 |
| LSU | Store | `STG.E.U8` | 4 |
| FMA |  | `IMAD` | 400 |
| FMA |  | `IMAD.HI.U32` | 100 |
| FMA |  | `HFMA2.MMA` | 36 |
| ALU |  | `IADD3` | 115 |
| ALU |  | `ISETP.NE.AND` | 108 |
| ALU |  | `SHF.R.U32.HI` | 100 |
| ALU |  | `MOV` | 77 |
| ALU |  | `IADD3.X` | 12 |
| ALU |  | `ISETP.GE.AND` | 4 |
| ALU |  | `SEL` | 4 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LEA` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 101 |
| Branch |  | `NOP` | 10 |
| Branch |  | `CS2R` | 4 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `S2R` | 2 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 36 ÷ 12 = 3.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 8 : 4
