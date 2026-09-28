<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `triton_per_fused__softmax_add_mul_4_sm87.sass` (PC range 0x2292500-0x22931f0)

Total instructions in PC range: **208**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 14 | 6.7% |
| FMA | 107 | 51.4% |
| ALU | 18 | 8.7% |
| Shuffle | 16 | 7.7% |
| XU | 5 | 2.4% |
| Uniform | 4 | 1.9% |
| Branch | 27 | 13.0% |
| Other | 17 | 8.2% |
| **Total** | **208** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 8 | 57.1% | 3.8% |
| Store | 6 | 42.9% | 2.9% |
| **LSU total** | **14** | **100.0%** | **6.7%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDS` | 4 |
| LSU | Store | `STS` | 4 |
| LSU | Load | `LDG.E.64` | 2 |
| LSU | Load | `LDG.E.EL.64` | 2 |
| LSU | Store | `STG.E.64` | 2 |
| FMA |  | `FMUL` | 26 |
| FMA |  | `FADD` | 17 |
| FMA |  | `FSEL` | 17 |
| FMA |  | `FSETP.GT.AND` | 12 |
| FMA |  | `FSETP.NAN.AND` | 11 |
| FMA |  | `IMAD` | 6 |
| FMA |  | `FSETP.GEU.AND` | 5 |
| FMA |  | `FFMA` | 4 |
| FMA |  | `IMAD.WIDE` | 4 |
| FMA |  | `IMAD.MOV.U32` | 2 |
| FMA |  | `IMAD.HI.U32` | 1 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| FMA |  | `IMAD.WIDE.U32` | 1 |
| ALU |  | `LOP3.LUT` | 4 |
| ALU |  | `IADD3` | 3 |
| ALU |  | `ISETP.GE.U32.AND` | 2 |
| ALU |  | `SEL` | 2 |
| ALU |  | `SHF.R.U32.HI` | 2 |
| ALU |  | `IADD3.X` | 1 |
| ALU |  | `ISETP.NE.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LEA` | 1 |
| ALU |  | `LEA.HI.X` | 1 |
| Shuffle |  | `SHFL.BFLY` | 16 |
| XU |  | `MUFU.EX2` | 4 |
| XU |  | `MUFU.RCP` | 1 |
| Uniform |  | `UMOV` | 3 |
| Uniform |  | `ULDC.64` | 1 |
| Branch |  | `NOP` | 14 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 5 |
| Branch |  | `BRA` | 2 |
| Branch |  | `CS2R` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `S2R` | 2 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 52 ÷ 6 = 8.667. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 8 : 6
