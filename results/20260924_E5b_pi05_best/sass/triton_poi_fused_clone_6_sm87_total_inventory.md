# SASS instruction inventory — `triton_poi_fused_clone_6_sm87.sass`

Total instructions: **56** (file has 56 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 2 | 3.6% |
| FMA | 10 | 17.9% |
| ALU | 11 | 19.6% |
| Uniform | 2 | 3.6% |
| Branch | 14 | 25.0% |
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
| LSU | Store | `STG.E.64` | 1 |
| FMA |  | `IMAD` | 3 |
| FMA |  | `IMAD.IADD` | 2 |
| FMA |  | `IMAD.MOV.U32` | 2 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `IMAD.SHL.U32` | 1 |
| ALU |  | `LEA.HI` | 3 |
| ALU |  | `LOP3.LUT` | 3 |
| ALU |  | `SHF.R.S32.HI` | 3 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `SGXT` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 9 |
| Branch |  | `BRA` | 2 |
| Branch |  | `S2R` | 2 |
| Branch |  | `EXIT` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
