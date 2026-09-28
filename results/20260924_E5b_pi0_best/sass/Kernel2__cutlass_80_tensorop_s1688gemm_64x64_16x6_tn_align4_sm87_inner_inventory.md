<!-- inner loop: backward BRA 0x2ef04680 -> 0x2ef03e70: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `Kernel2__cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4_sm87.sass` (PC range 0x2ef03e70-0x2ef04680)

Total instructions in PC range: **127**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 13 | 10.2% |
| Tensor core | 16 | 12.6% |
| FMA | 65 | 51.2% |
| ALU | 18 | 14.2% |
| Uniform | 12 | 9.4% |
| Branch | 3 | 2.4% |
| **Total** | **127** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 13 | 100.0% | 10.2% |
| **LSU total** | **13** | **100.0%** | **10.2%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDSM.16.M88.4` | 8 |
| LSU | Load | `LDGSTS.E.BYPASS.LTC128B.128` | 4 |
| LSU | Load | `LDGDEPBAR` | 1 |
| Tensor core |  | `HMMA.1688.F32.TF32` | 16 |
| FMA |  | `FSETP.GEU.AND` | 32 |
| FMA |  | `IMAD.IADD.U32` | 32 |
| FMA |  | `IMAD.MOV.U32` | 1 |
| ALU |  | `IADD3` | 7 |
| ALU |  | `R2P` | 4 |
| ALU |  | `IADD3.X` | 3 |
| ALU |  | `ISETP.NE.AND` | 2 |
| ALU |  | `ISETP.GT.AND` | 1 |
| ALU |  | `SEL` | 1 |
| Uniform |  | `UIADD3` | 6 |
| Uniform |  | `UMOV` | 3 |
| Uniform |  | `UISETP.NE.AND` | 2 |
| Uniform |  | `USEL` | 1 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 1 |
| Branch |  | `BRA` | 1 |
| Branch |  | `DEPBAR.LE` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 16 ÷ 4 = 4.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 13 : 0
