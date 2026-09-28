<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `vectorized_elementwise_kernel__e8e91f66_sm87.sass` (PC range 0x4cb92300-0x4cb94e70)

Total instructions in PC range: **696**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 20 | 2.9% |
| FMA | 284 | 40.8% |
| ALU | 213 | 30.6% |
| XU | 34 | 4.9% |
| Uniform | 2 | 0.3% |
| Branch | 126 | 18.1% |
| Other | 17 | 2.4% |
| **Total** | **696** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 10 | 50.0% | 1.4% |
| Store | 10 | 50.0% | 1.4% |
| **LSU total** | **20** | **100.0%** | **2.9%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E` | 8 |
| LSU | Store | `STG.E` | 8 |
| LSU | Load | `LDG.E.128` | 2 |
| LSU | Store | `STG.E.128` | 2 |
| FMA |  | `FFMA` | 137 |
| FMA |  | `FFMA.RM` | 23 |
| FMA |  | `FFMA.SAT` | 22 |
| FMA |  | `HFMA2.MMA` | 22 |
| FMA |  | `IMAD.WIDE.U32` | 18 |
| FMA |  | `FADD` | 16 |
| FMA |  | `FCHK` | 16 |
| FMA |  | `IMAD.MOV.U32` | 12 |
| FMA |  | `FSETP.NEU.FTZ.AND` | 4 |
| FMA |  | `IMAD.IADD` | 4 |
| FMA |  | `FADD.FTZ` | 2 |
| FMA |  | `FSETP.GTU.FTZ.AND` | 2 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `FFMA.RP` | 1 |
| FMA |  | `FFMA.RZ` | 1 |
| FMA |  | `IMAD` | 1 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `MOV` | 86 |
| ALU |  | `IADD3` | 45 |
| ALU |  | `ISETP.GE.AND` | 26 |
| ALU |  | `LOP3.LUT` | 17 |
| ALU |  | `SHF.L.U32` | 17 |
| ALU |  | `SHF.R.U32.HI` | 5 |
| ALU |  | `PLOP3.LUT` | 4 |
| ALU |  | `ISETP.NE.AND` | 3 |
| ALU |  | `ISETP.GE.U32.AND` | 2 |
| ALU |  | `LEA` | 2 |
| ALU |  | `SEL` | 2 |
| ALU |  | `ISETP.GT.AND` | 1 |
| ALU |  | `ISETP.GT.U32.AND` | 1 |
| ALU |  | `ISETP.GT.U32.OR` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| XU |  | `MUFU.RCP` | 17 |
| XU |  | `MUFU.EX2` | 16 |
| XU |  | `MUFU.RSQ` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 48 |
| Branch |  | `BSSY` | 20 |
| Branch |  | `BSYNC` | 20 |
| Branch |  | `CALL.REL.NOINC` | 16 |
| Branch |  | `NOP` | 10 |
| Branch |  | `S2R` | 4 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `CS2R` | 2 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `RET.REL.NODEC` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 258 ÷ 20 = 12.9. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 10 : 10
