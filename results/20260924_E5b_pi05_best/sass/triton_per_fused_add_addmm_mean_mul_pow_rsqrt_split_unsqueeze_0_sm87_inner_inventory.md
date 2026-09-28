<!-- inner loop: no backward BRA: fully unrolled, the inner range is the whole kernel (§3) -->
# SASS instruction inventory — `triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_0_sm87.sass` (PC range 0x2282f00-0x22835f0)

Total instructions in PC range: **112**.

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 10 | 8.9% |
| FMA | 40 | 35.7% |
| ALU | 10 | 8.9% |
| Shuffle | 8 | 7.1% |
| XU | 1 | 0.9% |
| Uniform | 4 | 3.6% |
| Branch | 22 | 19.6% |
| Other | 17 | 15.2% |
| **Total** | **112** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 7 | 70.0% | 6.2% |
| Store | 3 | 30.0% | 2.7% |
| **LSU total** | **10** | **100.0%** | **8.9%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.EL.128` | 4 |
| LSU | Load | `LDS` | 2 |
| LSU | Store | `STS` | 2 |
| LSU | Load | `LDG.E.128` | 1 |
| LSU | Store | `STG.E.128` | 1 |
| FMA |  | `FADD` | 21 |
| FMA |  | `FFMA` | 8 |
| FMA |  | `FMUL` | 5 |
| FMA |  | `HFMA2.MMA` | 2 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `IMAD.WIDE.U32` | 2 |
| ALU |  | `LOP3.LUT` | 3 |
| ALU |  | `ISETP.GE.U32.AND` | 1 |
| ALU |  | `ISETP.NE.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LEA` | 1 |
| ALU |  | `MOV` | 1 |
| ALU |  | `SHF.L.U32` | 1 |
| ALU |  | `SHF.R.U32.HI` | 1 |
| Shuffle |  | `SHFL.BFLY` | 8 |
| XU |  | `MUFU.RSQ` | 1 |
| Uniform |  | `UMOV` | 3 |
| Uniform |  | `ULDC.64` | 1 |
| Branch |  | `NOP` | 15 |
| Branch |  | `BAR.SYNC.DEFER_BLOCKING` | 2 |
| Branch |  | `BRA` | 2 |
| Branch |  | `S2R` | 2 |
| Branch |  | `EXIT` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |

## Ratios (§4)

* compute-to-memory = (FP + tensor-core) ÷ (global loads + stores) = 37 ÷ 6 = 6.167. Integer arithmetic is left out: statically it cannot be told apart from address arithmetic, so the ratio is a lower bound on §4's definition.
* static load : store = 7 : 3
