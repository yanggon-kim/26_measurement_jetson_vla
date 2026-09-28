# SASS instruction inventory — `triton_poi_fused_addmm_silu_4_sm87.sass`

Total instructions: **64** (file has 64 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 3 | 4.7% |
| FMA | 16 | 25.0% |
| ALU | 5 | 7.8% |
| XU | 2 | 3.1% |
| Uniform | 2 | 3.1% |
| Branch | 19 | 29.7% |
| Other | 17 | 26.6% |
| **Total** | **64** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 2 | 66.7% | 3.1% |
| Store | 1 | 33.3% | 1.6% |
| **LSU total** | **3** | **100.0%** | **4.7%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E` | 2 |
| LSU | Store | `STG.E` | 1 |
| FMA |  | `FMUL` | 6 |
| FMA |  | `FADD` | 3 |
| FMA |  | `HFMA2.MMA` | 2 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `FSEL` | 1 |
| FMA |  | `FSETP.GEU.AND` | 1 |
| FMA |  | `FSETP.GT.AND` | 1 |
| ALU |  | `ISETP.GE.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LEA` | 1 |
| ALU |  | `LOP3.LUT` | 1 |
| ALU |  | `MOV` | 1 |
| XU |  | `MUFU.EX2` | 1 |
| XU |  | `MUFU.RCP` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 12 |
| Branch |  | `BRA` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `S2R` | 2 |
| Branch |  | `CS2R` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
