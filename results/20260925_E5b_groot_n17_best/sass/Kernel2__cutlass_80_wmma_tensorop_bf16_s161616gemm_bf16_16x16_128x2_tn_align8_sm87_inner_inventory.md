<!-- inner loop: backward BRA 0x86232340 -> 0x86231190: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8_sm87.sass` (PC range 0x86231190-0x86232340)

Total instructions in PC range: **282**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 96 | 34.0% |
| Tensor core | 16 | 5.7% |
| FMA | 3 | 1.1% |
| ALU | 78 | 27.7% |
| Uniform | 55 | 19.5% |
| Branch | 34 | 12.1% |
| **Total** | **282** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 80 | 83.3% | 28.4% |
| Store | 16 | 16.7% | 5.7% |
| **LSU total** | **96** | **100.0%** | **34.0%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LD.E` | 64 |
| LSU | Load | `LDG.E.LTC128B.128` | 16 |
| LSU | Store | `STS.128` | 16 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 16 |
| FMA |  | `IMAD.MOV.U32` | 3 |
| ALU |  | `IADD3` | 25 |
| ALU |  | `IADD3.X` | 20 |
| ALU |  | `LEA` | 12 |
| ALU |  | `LEA.HI.X` | 12 |
| ALU |  | `R2P` | 4 |
| ALU |  | `ISETP.GT.AND` | 2 |
| ALU |  | `ISETP.GE.AND` | 1 |
| ALU |  | `PLOP3.LUT` | 1 |
| ALU |  | `SEL` | 1 |
| Uniform |  | `UIADD3` | 33 |
| Uniform |  | `ULEA.HI.X.SX32` | 16 |
| Uniform |  | `UMOV` | 2 |
| Uniform |  | `UIADD3.X` | 1 |
| Uniform |  | `UISETP.NE.AND` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `ULOP3.LUT` | 1 |
| Branch |  | `CS2R` | 32 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 1 |
| Branch |  | `BRA` | 1 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 16 ÷ 80 = 0.2. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 80 : 16
