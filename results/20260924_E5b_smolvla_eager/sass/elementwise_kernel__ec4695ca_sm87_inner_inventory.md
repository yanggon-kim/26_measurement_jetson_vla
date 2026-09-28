<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `elementwise_kernel__ec4695ca_sm87.sass` (PC range 0x222e900-0x2230cf0)

Total instructions in PC range: **576**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 6 | 1.0% |
| FMA | 274 | 47.6% |
| ALU | 209 | 36.3% |
| Uniform | 2 | 0.3% |
| Branch | 68 | 11.8% |
| Other | 17 | 3.0% |
| **Total** | **576** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 4 | 66.7% | 0.7% |
| Store | 2 | 33.3% | 0.3% |
| **LSU total** | **6** | **100.0%** | **1.0%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.64` | 4 |
| LSU | Store | `STG.E.64` | 2 |
| FMA |  | `IMAD` | 204 |
| FMA |  | `IMAD.HI.U32` | 50 |
| FMA |  | `HFMA2.MMA` | 18 |
| FMA |  | `IMAD.WIDE.U32` | 2 |
| ALU |  | `IADD3` | 59 |
| ALU |  | `ISETP.NE.AND` | 50 |
| ALU |  | `SHF.R.U32.HI` | 50 |
| ALU |  | `MOV` | 40 |
| ALU |  | `IADD3.X` | 6 |
| ALU |  | `ISETP.GE.AND` | 2 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LEA` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 51 |
| Branch |  | `NOP` | 9 |
| Branch |  | `CS2R` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `S2R` | 2 |
| Branch |  | `BSSY` | 1 |
| Branch |  | `BSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 18 ÷ 6 = 3.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 4 : 2
