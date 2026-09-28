<!-- inner loop: backward BRA 0x86217a80 -> 0x86217240: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3) -->
# SASS instruction inventory — `Kernel2__cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8_sm87.sass` (PC range 0x86217240-0x86217a80)

Total instructions in PC range: **127**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 25 | 19.7% |
| Tensor core | 64 | 50.4% |
| FMA | 2 | 1.6% |
| ALU | 21 | 16.5% |
| Uniform | 12 | 9.4% |
| Branch | 3 | 2.4% |
| **Total** | **127** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 25 | 100.0% | 19.7% |
| **LSU total** | **25** | **100.0%** | **19.7%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDSM.16.M88.4` | 16 |
| LSU | Load | `LDGSTS.E.BYPASS.LTC128B.128` | 8 |
| LSU | Load | `LDGDEPBAR` | 1 |
| Tensor core |  | `HMMA.16816.F32.BF16` | 64 |
| FMA |  | `IMAD.MOV.U32` | 2 |
| ALU |  | `IADD3` | 9 |
| ALU |  | `IADD3.X` | 4 |
| ALU |  | `R2P` | 4 |
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

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 64 ÷ 8 = 8.0. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 25 : 0
