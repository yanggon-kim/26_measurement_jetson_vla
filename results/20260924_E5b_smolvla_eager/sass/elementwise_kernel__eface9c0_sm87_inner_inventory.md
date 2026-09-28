<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `elementwise_kernel__eface9c0_sm87.sass` (PC range 0x4cb9ee00-0x4cba35f0)

Total instructions in PC range: **1152**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 12 | 1.0% |
| FMA | 552 | 47.9% |
| ALU | 430 | 37.3% |
| Uniform | 2 | 0.2% |
| Branch | 139 | 12.1% |
| Other | 17 | 1.5% |
| **Total** | **1152** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 8 | 66.7% | 0.7% |
| Store | 4 | 33.3% | 0.3% |
| **LSU total** | **12** | **100.0%** | **1.0%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E` | 8 |
| LSU | Store | `STG.E.U8` | 4 |
| FMA |  | `IMAD` | 400 |
| FMA |  | `IMAD.HI.U32` | 100 |
| FMA |  | `HFMA2.MMA` | 36 |
| FMA |  | `FSETP.GE.AND` | 4 |
| FMA |  | `FSETP.GT.AND` | 4 |
| FMA |  | `FSETP.LE.AND` | 4 |
| FMA |  | `FSETP.LT.AND` | 4 |
| ALU |  | `IADD3` | 115 |
| ALU |  | `ISETP.NE.AND` | 112 |
| ALU |  | `SHF.R.U32.HI` | 100 |
| ALU |  | `MOV` | 81 |
| ALU |  | `IADD3.X` | 12 |
| ALU |  | `ISETP.GE.AND` | 4 |
| ALU |  | `SEL` | 4 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LEA` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 117 |
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
