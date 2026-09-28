# SASS instruction inventory — `triton_poi_fused_add_cumsum_sub_2_sm87.sass`

Total instructions: **56** (file has 56 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 2 | 3.6% |
| FMA | 9 | 16.1% |
| ALU | 11 | 19.6% |
| Uniform | 2 | 3.6% |
| Branch | 15 | 26.8% |
| Other | 17 | 30.4% |
| **Total** | **56** | **100.0%** |

## LSU split (Load / Store / Atomic)

| Sub-class | Count | % of LSU | % of total |
|---|---:|---:|---:|
| Load | 1 | 50.0% | 1.8% |
| Store | 1 | 50.0% | 1.8% |
| **LSU total** | **2** | **100.0%** | **3.6%** |

## Per-opcode breakdown

| Category | Sub | Opcode | Count |
|---|---|---|---:|
| LSU | Load | `LDG.E.64` | 1 |
| LSU | Store | `STG.E.128` | 1 |
| FMA |  | `IMAD.MOV.U32` | 6 |
| FMA |  | `IMAD` | 1 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| FMA |  | `IMAD.WIDE` | 1 |
| ALU |  | `IADD3` | 5 |
| ALU |  | `LEA.HI.X.SX32` | 2 |
| ALU |  | `IADD3.X` | 1 |
| ALU |  | `ISETP.GE.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LOP3.LUT` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 9 |
| Branch |  | `BRA` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `S2R` | 2 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
