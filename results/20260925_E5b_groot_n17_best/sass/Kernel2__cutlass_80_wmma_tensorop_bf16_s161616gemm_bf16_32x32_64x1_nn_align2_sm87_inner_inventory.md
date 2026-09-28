<!-- inner loop: backward BRA 0x862382c0 -> 0x86237af0: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nn_align2_sm87.sass` (PC range 0x86237af0-0x862382c0)

Total instructions in PC range: **126**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 52 | 41.3% |
| Tensor core | 8 | 6.3% |
| FMA | 1 | 0.8% |
| ALU | 46 | 36.5% |
| Branch | 3 | 2.4% |
| Other | 16 | 12.7% |
| **Total** | **126** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 48 | 92.3% | 38.1% |
| Store | 4 | 7.7% | 3.2% |
| **LSU total** | **52** | **100.0%** | **41.3%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LD.E` | 32 |
| LSU | Load | `LDG.E.LTC128B` | 16 |
| LSU | Store | `STS.128` | 4 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 8 |
| FMA |  | `IMAD.MOV.U32` | 1 |
| ALU |  | `IADD3` | 20 |
| ALU |  | `IADD3.X` | 18 |
| ALU |  | `R2P` | 4 |
| ALU |  | `ISETP.GT.AND` | 2 |
| ALU |  | `ISETP.GE.AND` | 1 |
| ALU |  | `SEL` | 1 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 2 |
| Branch |  | `BRA` | 1 |
| Other |  | `MOVM.16.MT88` | 16 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 8 ÷ 48 = 0.167. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 48 : 4
