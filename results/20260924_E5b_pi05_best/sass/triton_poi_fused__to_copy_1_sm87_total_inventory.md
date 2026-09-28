# SASS instruction inventory — `triton_poi_fused__to_copy_1_sm87.sass`

Total instructions: **56** (file has 56 lines; cuobjdump emits ~2 lines per instruction).

## Per-category subtotal

| Category | Count | % of total |
|---|---:|---:|
| LSU | 2 | 3.6% |
| FMA | 5 | 8.9% |
| ALU | 7 | 12.5% |
| Uniform | 2 | 3.6% |
| Branch | 23 | 41.1% |
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
| LSU | Load | `LDG.E.128` | 1 |
| LSU | Store | `STG.E.64` | 1 |
| FMA |  | `IMAD.WIDE` | 2 |
| FMA |  | `HFMA2.MMA` | 1 |
| FMA |  | `IMAD` | 1 |
| FMA |  | `IMAD.MOV.U32` | 1 |
| ALU |  | `I2F.S64` | 2 |
| ALU |  | `ISETP.GE.AND` | 1 |
| ALU |  | `ISETP.NE.U32.AND` | 1 |
| ALU |  | `LOP3.LUT` | 1 |
| ALU |  | `MOV` | 1 |
| ALU |  | `SHF.L.U32` | 1 |
| Uniform |  | `ULDC.64` | 1 |
| Uniform |  | `UMOV` | 1 |
| Branch |  | `NOP` | 12 |
| Branch |  | `BRA` | 3 |
| Branch |  | `CS2R` | 2 |
| Branch |  | `EXIT` | 2 |
| Branch |  | `S2R` | 2 |
| Branch |  | `BSSY` | 1 |
| Branch |  | `BSYNC` | 1 |
| Other |  | `BMOV.32.CLEAR` | 15 |
| Other |  | `BMOV.32` | 2 |
