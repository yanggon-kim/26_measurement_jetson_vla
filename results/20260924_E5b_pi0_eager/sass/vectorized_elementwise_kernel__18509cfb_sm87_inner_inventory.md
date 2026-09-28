<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `vectorized_elementwise_kernel__18509cfb_sm87.sass` (PC range 0x2dd32e00-0x2dd34f70)

Total instructions in PC range: **536**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 24 | 4.5% |
| FMA | 144 | 26.9% |
| ALU | 116 | 21.6% |
| XU | 115 | 21.5% |
| Uniform | 2 | 0.4% |
| Branch | 118 | 22.0% |
| Other | 17 | 3.2% |
| **Total** | **536** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 12 | 50.0% | 2.2% |
| Store | 12 | 50.0% | 2.2% |
| **LSU total** | **24** | **100.0%** | **4.5%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.64` | 8 |
| LSU | Store | `STG.E.64` | 8 |
| LSU | Load | `LDG.E.128` | 4 |
| LSU | Store | `STG.E.128` | 4 |
| FMA |  | `IMAD.MOV.U32` | 91 |
| FMA |  | `IMAD.WIDE.U32` | 18 |
| FMA |  | `FSETP.GEU.AND` | 16 |
| FMA |  | `IMAD.IADD` | 16 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `IADD3` | 55 |
| ALU |  | `ISETP.GE.AND` | 22 |
| ALU |  | `LOP3.LUT` | 19 |
| ALU |  | `MOV` | 16 |
| ALU |  | `ISETP.GE.U32.AND` | 3 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| XU |  | `DFMA` | 93 |
| XU |  | `MUFU.RCP64H` | 18 |
| XU |  | `DMUL` | 3 |
| XU |  | `DSETP.GTU.AND` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `BRA` | 37 |
| Branch |  | `BSSY` | 19 |
| Branch |  | `BSYNC` | 19 |
| Branch |  | `CALL.REL.NOINC` | 16 |
| Branch |  | `NOP` | 9 |
| Branch |  | `CS2R` | 8 |
| Branch |  | `S2R` | 4 |
| Branch |  | `EXIT` | 3 |
| Branch |  | `BREAK` | 1 |
| Branch |  | `RET.REL.NODEC` | 1 |
| Branch |  | `WARPSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 114 ÷ 24 = 4.75. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 12 : 12
