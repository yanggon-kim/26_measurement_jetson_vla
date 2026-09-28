<!-- inner loop: backward BRA 0x2eefa1f0 -> 0x2eef9140: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `Kernel2__cutlass_80_tensorop_s1688gemm_256x64_16x4_tn_align4_sm87.sass` (PC range 0x2eef9140-0x2eefa1f0)

Total instructions in PC range: **262**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 27 | 10.3% |
| Tensor core | 64 | 24.4% |
| FMA | 136 | 51.9% |
| ALU | 20 | 7.6% |
| Uniform | 12 | 4.6% |
| Branch | 3 | 1.1% |
| **Total** | **262** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 27 | 100.0% | 10.3% |
| **LSU total** | **27** | **100.0%** | **10.3%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDSM.16.M88.4` | 16 |
| LSU | Load | `LDGSTS.E.BYPASS.LTC128B.128` | 10 |
| LSU | Load | `LDGDEPBAR` | 1 |
| Tensor core |  | `HMMA.1688.F32.TF32` | 64 |
| FMA |  | `FSETP.GEU.AND` | 64 |
| FMA |  | `IMAD.IADD.U32` | 64 |
| FMA |  | `IMAD.MOV.U32` | 5 |
| FMA |  | `IMAD.WIDE.U32` | 3 |
| ALU |  | `IADD3` | 9 |
| ALU |  | `R2P` | 4 |
| ALU |  | `IADD3.X` | 3 |
| ALU |  | `ISETP.NE.AND` | 2 |
| ALU |  | `ISETP.GT.AND` | 1 |
| ALU |  | `SEL` | 1 |
| Uniform |  | `UIADD3` | 6 |
| Uniform |  | `UMOV` | 4 |
| Uniform |  | `UISETP.NE.AND` | 2 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 1 |
| Branch |  | `BRA` | 1 |
| Branch |  | `DEPBAR.LE` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 64 ÷ 10 = 6.4. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 27 : 0
