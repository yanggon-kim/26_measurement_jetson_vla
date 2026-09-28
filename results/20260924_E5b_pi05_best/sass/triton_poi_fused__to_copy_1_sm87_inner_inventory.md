<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `triton_poi_fused__to_copy_1_sm87.sass` (PC range 0x2283f00-0x2284270)

Total instructions in PC range: **56**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 2 | 3.6% |
| FMA | 5 | 8.9% |
| ALU | 7 | 12.5% |
| Uniform | 2 | 3.6% |
| Branch | 23 | 41.1% |
| Other | 17 | 30.4% |
| **Total** | **56** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 50.0% | 1.8% |
| Store | 1 | 50.0% | 1.8% |
| **LSU total** | **2** | **100.0%** | **3.6%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.128` | 1 |
| LSU | Store | `STG.E.64` | 1 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `HFMA2.MMA` | 1 |
| FMA |  | `IMAD` | 1 |
| FMA |  | `IMAD.MOV.U32` | 1 |
| ALU |  | `I2F.S64` | 2 |
| ALU |  | `ISETP.GE.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LOP3.LUT` | 1 |
| ALU |  | `MOV` | 1 |
| ALU |  | `SHF.L.U32` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 12 |
| Branch |  | `BRA` | 3 |
| Branch |  | `CS2R` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `S2R` | 2 |
| Branch |  | `BSSY` | 1 |
| Branch |  | `BSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 1 ÷ 2 = 0.5. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 1 : 1
