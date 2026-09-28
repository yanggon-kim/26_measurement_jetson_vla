<!-- inner loop: backward BRA 0x2ef29b10 -> 0x2ef29660: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `Kernel2__cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_nn_align1_sm87.sass` (PC range 0x2ef29660-0x2ef29b10)

Total instructions in PC range: **73**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 34 | 46.6% |
| Tensor core | 4 | 5.5% |
| FMA | 3 | 4.1% |
| ALU | 21 | 28.8% |
| Branch | 3 | 4.1% |
| Other | 8 | 11.0% |
| **Total** | **73** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 32 | 94.1% | 43.8% |
| Store | 2 | 5.9% | 2.7% |
| **LSU total** | **34** | **100.0%** | **46.6%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LD.E` | 16 |
| LSU | Load | `LDG.E.LTC128B.U16` | 16 |
| LSU | Store | `STS.128` | 2 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 4 |
| FMA |  | `IMAD.MOV.U32` | 3 |
| ALU |  | `PRMT` | 8 |
| ALU |  | `R2P` | 4 |
| ALU |  | `IADD3` | 3 |
| ALU |  | `IADD3.X` | 2 |
| ALU |  | `ISETP.GT.AND` | 2 |
| ALU |  | `ISETP.GE.AND` | 1 |
| ALU |  | `SEL` | 1 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 2 |
| Branch |  | `BRA` | 1 |
| Other |  | `MOVM.16.MT88` | 8 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 4 ÷ 32 = 0.125. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 32 : 2
