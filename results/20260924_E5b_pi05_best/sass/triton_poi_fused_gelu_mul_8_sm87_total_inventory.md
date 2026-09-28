# SASS instruction inventory — `triton_poi_fused_gelu_mul_8_sm87.sass`

Total instructions: **104** (file has 104 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 3 | 2.9% |
| FMA | 46 | 44.2% |
| ALU | 10 | 9.6% |
| XU | 4 | 3.8% |
| Uniform | 2 | 1.9% |
| Branch | 22 | 21.2% |
| Other | 17 | 16.3% |
| **Total** | **104** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 2 | 66.7% | 1.9% |
| Store | 1 | 33.3% | 1.0% |
| **LSU total** | **3** | **100.0%** | **2.9%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.64` | 2 |
| LSU | Store | `STG.E.64` | 1 |
| FMA |  | `FMUL` | 16 |
| FMA |  | `FFMA.FTZ` | 12 |
| FMA |  | `FADD` | 4 |
| FMA |  | `FSETP.GE.AND` | 4 |
| FMA |  | `FADD.FTZ` | 2 |
| FMA |  | `FFMA` | 2 |
| FMA |  | `FSEL` | 2 |
| FMA |  | `HFMA2.MMA` | 2 |
| FMA |  | `IMAD.WIDE` | 2 |
| ALU |  | `MOV` | 4 |
| ALU |  | `LOP3.LUT` | 3 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LEA` | 1 |
| ALU |  | `SHF.L.U32` | 1 |
| XU |  | `MUFU.EX2` | 2 |
| XU |  | `MUFU.RCP` | 2 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 9 |
| Branch |  | `BRA` | 6 |
| Branch |  | `BSSY` | 2 |
| Branch |  | `BSYNC` | 2 |
| Branch |  | `S2R` | 2 |
| Branch |  | `EXIT` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
