# Hot loop — `ampere_sgemm_32x32_sliced1x4_tn` (sm87)

PC range 0x4f6cc020-0x4f6cd6e0; backward BRA 0x4f6cd6e0 -> 0x4f6cc020: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_sgemm_32x32_sliced1x4_tn_sm87.sass`; per-PC counts: `raw/ampere_sgemm_32x32_sliced1x4_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*4f6cc020*/  FFMA R1, R60, R68, R1                                      ; // x      2400  FMA
/*4f6cc030*/  R2P PR, R57, 0x78                                          ; // x      2400  ALU
/*4f6cc040*/  FFMA R0, R60, R69, R0                                      ; // x      2400  FMA
/*4f6cc050*/  FFMA R9, R61, R68, R9                                      ; // x      2400  FMA
/*4f6cc060*/  LDS.128 R64, [R55+0x90]                                    ; // x      2400  shared-memory load
/*4f6cc070*/  FFMA R8, R61, R69, R8                                      ; // x      2400  FMA
/*4f6cc080*/  FFMA R17, R62, R68, R17                                    ; // x      2400  FMA
/*4f6cc090*/  LDS.128 R76, [R56+0x90]                                    ; // x      2400  shared-memory load
/*4f6cc0a0*/  FFMA R16, R62, R69, R16                                    ; // x      2400  FMA
/*4f6cc0b0*/  FFMA R25, R63, R68, R25                                    ; // x      2400  FMA
/*4f6cc0c0*/  LDS.128 R80, [R56+0xd0]                                    ; // x      2400  shared-memory load
/*4f6cc0d0*/  FFMA R24, R63, R69, R24                                    ; // x      2400  FMA
/*4f6cc0e0*/  FFMA R27, R63, R70, R27                                    ; // x      2400  FMA
/*4f6cc0f0*/  LDG.E.LTC128B.CONSTANT R33, P3, [R50]                      ; // x      2400  global load (DRAM/L2: weights or activations)
/*4f6cc100*/  FFMA R26, R63, R71, R26                                    ; // x      2400  FMA
/*4f6cc110*/  FFMA R19, R62, R70, R19                                    ; // x      2400  FMA
/*4f6cc120*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      2400  ALU
/*4f6cc130*/  FFMA R18, R62, R71, R18                                    ; // x      2400  FMA
/*4f6cc140*/  FFMA R11, R61, R70, R11                                    ; // x      2400  FMA
/*4f6cc150*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      2400  ALU
/*4f6cc160*/  FFMA R10, R61, R71, R10                                    ; // x      2400  FMA
/*4f6cc170*/  FFMA R3, R60, R70, R3                                      ; // x      2400  FMA
/*4f6cc180*/  FFMA R2, R60, R71, R2                                      ; // x      2400  FMA
/*4f6cc190*/  FFMA R5, R60, R72, R5                                      ; // x      2400  FMA
/*4f6cc1a0*/  LDG.E.LTC128B.CONSTANT R34, P4, [R50]                      ; // x      2400  global load (DRAM/L2: weights or activations)
/*4f6cc1b0*/  FFMA R4, R60, R73, R4                                      ; // x      2400  FMA
/*4f6cc1c0*/  FFMA R13, R61, R72, R13                                    ; // x      2400  FMA
/*4f6cc1d0*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      2400  ALU
/*4f6cc1e0*/  FFMA R12, R61, R73, R12                                    ; // x      2400  FMA
/*4f6cc1f0*/  FFMA R21, R62, R72, R21                                    ; // x      2400  FMA
/*4f6cc200*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      2400  ALU
/*4f6cc210*/  FFMA R20, R62, R73, R20                                    ; // x      2400  FMA
/*4f6cc220*/  FFMA R29, R63, R72, R29                                    ; // x      2400  FMA
/*4f6cc230*/  FFMA R28, R63, R73, R28                                    ; // x      2400  FMA
/*4f6cc240*/  FFMA R31, R63, R74, R31                                    ; // x      2400  FMA
/*4f6cc250*/  LDG.E.LTC128B.CONSTANT R35, P5, [R50]                      ; // x      2400  global load (DRAM/L2: weights or activations)
/*4f6cc260*/  FFMA R30, R63, R75, R30                                    ; // x      2400  FMA
/*4f6cc270*/  FFMA R23, R62, R74, R23                                    ; // x      2400  FMA
/*4f6cc280*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      2400  ALU
/*4f6cc290*/  FFMA R22, R62, R75, R22                                    ; // x      2400  FMA
/*4f6cc2a0*/  FFMA R15, R61, R74, R15                                    ; // x      2400  FMA
/*4f6cc2b0*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      2400  ALU
/*4f6cc2c0*/  FFMA R14, R61, R75, R14                                    ; // x      2400  FMA
/*4f6cc2d0*/  FFMA R7, R60, R74, R7                                      ; // x      2400  FMA
/*4f6cc2e0*/  FFMA R6, R60, R75, R6                                      ; // x      2400  FMA
/*4f6cc2f0*/  FFMA R1, R64, R76, R1                                      ; // x      2400  FMA
/*4f6cc300*/  LDS.128 R60, [R55+0x120]                                   ; // x      2400  shared-memory load
/*4f6cc310*/  FFMA R0, R64, R77, R0                                      ; // x      2400  FMA
/*4f6cc320*/  FFMA R9, R65, R76, R9                                      ; // x      2400  FMA
/*4f6cc330*/  LDS.128 R68, [R56+0x120]                                   ; // x      2400  shared-memory load
/*4f6cc340*/  FFMA R8, R65, R77, R8                                      ; // x      2400  FMA
/*4f6cc350*/  FFMA R17, R66, R76, R17                                    ; // x      2400  FMA
/*4f6cc360*/  LDS.128 R72, [R56+0x160]                                   ; // x      2400  shared-memory load
/*4f6cc370*/  FFMA R16, R66, R77, R16                                    ; // x      2400  FMA
/*4f6cc380*/  FFMA R25, R67, R76, R25                                    ; // x      2400  FMA
/*4f6cc390*/  LDG.E.LTC128B.CONSTANT R36, P6, [R50]                      ; // x      2400  global load (DRAM/L2: weights or activations)
/*4f6cc3a0*/  FFMA R24, R67, R77, R24                                    ; // x      2400  FMA
/*4f6cc3b0*/  FFMA R27, R67, R78, R27                                    ; // x      2400  FMA
/*4f6cc3c0*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      2400  ALU
/*4f6cc3d0*/  FFMA R26, R67, R79, R26                                    ; // x      2400  FMA
/*4f6cc3e0*/  FFMA R19, R66, R78, R19                                    ; // x      2400  FMA
/*4f6cc3f0*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      2400  ALU
/*4f6cc400*/  FFMA R18, R66, R79, R18                                    ; // x      2400  FMA
/*4f6cc410*/  FFMA R11, R65, R78, R11                                    ; // x      2400  FMA
/*4f6cc420*/  FFMA R10, R65, R79, R10                                    ; // x      2400  FMA
/*4f6cc430*/  FFMA R3, R64, R78, R3                                      ; // x      2400  FMA
/*4f6cc440*/  R2P PR, R57.B1, 0x78                                       ; // x      2400  ALU
/*4f6cc450*/  FFMA R2, R64, R79, R2                                      ; // x      2400  FMA
/*4f6cc460*/  FFMA R5, R64, R80, R5                                      ; // x      2400  FMA
/*4f6cc470*/  FFMA R4, R64, R81, R4                                      ; // x      2400  FMA
/*4f6cc480*/  FFMA R13, R65, R80, R13                                    ; // x      2400  FMA
/*4f6cc490*/  FFMA R12, R65, R81, R12                                    ; // x      2400  FMA
/*4f6cc4a0*/  FFMA R21, R66, R80, R21                                    ; // x      2400  FMA
/*4f6cc4b0*/  FFMA R20, R66, R81, R20                                    ; // x      2400  FMA
/*4f6cc4c0*/  LDG.E.LTC128B.CONSTANT R37, P3, [R50]                      ; // x      2400  global load (DRAM/L2: weights or activations)
/*4f6cc4d0*/  FFMA R29, R67, R80, R29                                    ; // x      2400  FMA
/*4f6cc4e0*/  FFMA R28, R67, R81, R28                                    ; // x      2400  FMA
/*4f6cc4f0*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      2400  ALU
/*4f6cc500*/  FFMA R31, R67, R82, R31                                    ; // x      2400  FMA
/*4f6cc510*/  FFMA R30, R67, R83, R30                                    ; // x      2400  FMA
/*4f6cc520*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      2400  ALU
/*4f6cc530*/  FFMA R23, R66, R82, R23                                    ; // x      2400  FMA
/*4f6cc540*/  FFMA R22, R66, R83, R22                                    ; // x      2400  FMA
/*4f6cc550*/  FFMA R15, R65, R82, R15                                    ; // x      2400  FMA
/*4f6cc560*/  FFMA R14, R65, R83, R14                                    ; // x      2400  FMA
/*4f6cc570*/  LDG.E.LTC128B.CONSTANT R38, P4, [R50]                      ; // x      2400  global load (DRAM/L2: weights or activations)
/*4f6cc580*/  FFMA R7, R64, R82, R7                                      ; // x      2400  FMA
/*4f6cc590*/  FFMA R6, R64, R83, R6                                      ; // x      2400  FMA
/*4f6cc5a0*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      2400  ALU
/*4f6cc5b0*/  FFMA R1, R60, R68, R1                                      ; // x      2400  FMA
/*4f6cc5c0*/  FFMA R0, R60, R69, R0                                      ; // x      2400  FMA
/*4f6cc5d0*/  LDS.128 R64, [R55+0x1b0]                                   ; // x      2400  shared-memory load
/*4f6cc5e0*/  FFMA R9, R61, R68, R9                                      ; // x      2400  FMA
/*4f6cc5f0*/  FFMA R8, R61, R69, R8                                      ; // x      2400  FMA
/*4f6cc600*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      2400  ALU
/*4f6cc610*/  FFMA R17, R62, R68, R17                                    ; // x      2400  FMA
/*4f6cc620*/  FFMA R16, R62, R69, R16                                    ; // x      2400  FMA
/*4f6cc630*/  LDS.128 R76, [R56+0x1b0]                                   ; // x      2400  shared-memory load
/*4f6cc640*/  FFMA R25, R63, R68, R25                                    ; // x      2400  FMA
/*4f6cc650*/  FFMA R24, R63, R69, R24                                    ; // x      2400  FMA
/*4f6cc660*/  LDS.128 R80, [R56+0x1f0]                                   ; // x      2400  shared-memory load
/*4f6cc670*/  FFMA R27, R63, R70, R27                                    ; // x      2400  FMA
/*4f6cc680*/  FFMA R26, R63, R71, R26                                    ; // x      2400  FMA
/*4f6cc690*/  LDG.E.LTC128B.CONSTANT R39, P5, [R50]                      ; // x      2400  global load (DRAM/L2: weights or activations)
/*4f6cc6a0*/  FFMA R19, R62, R70, R19                                    ; // x      2400  FMA
/*4f6cc6b0*/  FFMA R18, R62, R71, R18                                    ; // x      2400  FMA
/*4f6cc6c0*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      2400  ALU
/*4f6cc6d0*/  FFMA R11, R61, R70, R11                                    ; // x      2400  FMA
/*4f6cc6e0*/  FFMA R10, R61, R71, R10                                    ; // x      2400  FMA
/*4f6cc6f0*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      2400  ALU
/*4f6cc700*/  FFMA R3, R60, R70, R3                                      ; // x      2400  FMA
/*4f6cc710*/  FFMA R2, R60, R71, R2                                      ; // x      2400  FMA
/*4f6cc720*/  FFMA R5, R60, R72, R5                                      ; // x      2400  FMA
/*4f6cc730*/  FFMA R4, R60, R73, R4                                      ; // x      2400  FMA
/*4f6cc740*/  LDG.E.LTC128B.CONSTANT R40, P6, [R50]                      ; // x      2400  global load (DRAM/L2: weights or activations)
/*4f6cc750*/  FFMA R13, R61, R72, R13                                    ; // x      2400  FMA
/*4f6cc760*/  FFMA R12, R61, R73, R12                                    ; // x      2400  FMA
/*4f6cc770*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      2400  ALU
/*4f6cc780*/  FFMA R21, R62, R72, R21                                    ; // x      2400  FMA
/*4f6cc790*/  FFMA R20, R62, R73, R20                                    ; // x      2400  FMA
/*4f6cc7a0*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      2400  ALU
/*4f6cc7b0*/  FFMA R29, R63, R72, R29                                    ; // x      2400  FMA
/*4f6cc7c0*/  FFMA R28, R63, R73, R28                                    ; // x      2400  FMA
/*4f6cc7d0*/  FFMA R31, R63, R74, R31                                    ; // x      2400  FMA
/*4f6cc7e0*/  FFMA R30, R63, R75, R30                                    ; // x      2400  FMA
/*4f6cc7f0*/  R2P PR, R57.B2, 0x78                                       ; // x      2400  ALU
/*4f6cc800*/  FFMA R23, R62, R74, R23                                    ; // x      2400  FMA
/*4f6cc810*/  FFMA R22, R62, R75, R22                                    ; // x      2400  FMA
/*4f6cc820*/  FFMA R15, R61, R74, R15                                    ; // x      2400  FMA
/*4f6cc830*/  FFMA R14, R61, R75, R14                                    ; // x      2400  FMA
/*4f6cc840*/  FFMA R7, R60, R74, R7                                      ; // x      2400  FMA
/*4f6cc850*/  FFMA R6, R60, R75, R6                                      ; // x      2400  FMA
/*4f6cc860*/  FFMA R1, R64, R76, R1                                      ; // x      2400  FMA
/*4f6cc870*/  LDS.128 R60, [R55+0x240]                                   ; // x      2400  shared-memory load
/*4f6cc880*/  FFMA R0, R64, R77, R0                                      ; // x      2400  FMA
/*4f6cc890*/  FFMA R9, R65, R76, R9                                      ; // x      2400  FMA
/*4f6cc8a0*/  LDS.128 R68, [R56+0x240]                                   ; // x      2400  shared-memory load
/*4f6cc8b0*/  FFMA R8, R65, R77, R8                                      ; // x      2400  FMA
/*4f6cc8c0*/  FFMA R17, R66, R76, R17                                    ; // x      2400  FMA
/*4f6cc8d0*/  LDS.128 R72, [R56+0x280]                                   ; // x      2400  shared-memory load
/*4f6cc8e0*/  FFMA R16, R66, R77, R16                                    ; // x      2400  FMA
/*4f6cc8f0*/  FFMA R25, R67, R76, R25                                    ; // x      2400  FMA
/*4f6cc900*/  LDG.E.LTC128B.CONSTANT R41, P3, [R50]                      ; // x      2400  global load (DRAM/L2: weights or activations)
/*4f6cc910*/  FFMA R24, R67, R77, R24                                    ; // x      2400  FMA
/*4f6cc920*/  FFMA R27, R67, R78, R27                                    ; // x      2400  FMA
/*4f6cc930*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      2400  ALU
/*4f6cc940*/  FFMA R26, R67, R79, R26                                    ; // x      2400  FMA
/*4f6cc950*/  FFMA R19, R66, R78, R19                                    ; // x      2400  FMA
/*4f6cc960*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      2400  ALU
/*4f6cc970*/  FFMA R18, R66, R79, R18                                    ; // x      2400  FMA
/*4f6cc980*/  FFMA R11, R65, R78, R11                                    ; // x      2400  FMA
/*4f6cc990*/  FFMA R10, R65, R79, R10                                    ; // x      2400  FMA
/*4f6cc9a0*/  FFMA R3, R64, R78, R3                                      ; // x      2400  FMA
/*4f6cc9b0*/  LDG.E.LTC128B.CONSTANT R42, P4, [R50]                      ; // x      2400  global load (DRAM/L2: weights or activations)
/*4f6cc9c0*/  FFMA R2, R64, R79, R2                                      ; // x      2400  FMA
/*4f6cc9d0*/  FFMA R5, R64, R80, R5                                      ; // x      2400  FMA
/*4f6cc9e0*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      2400  ALU
/*4f6cc9f0*/  FFMA R4, R64, R81, R4                                      ; // x      2400  FMA
/*4f6cca00*/  FFMA R13, R65, R80, R13                                    ; // x      2400  FMA
/*4f6cca10*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      2400  ALU
/*4f6cca20*/  FFMA R12, R65, R81, R12                                    ; // x      2400  FMA
/*4f6cca30*/  FFMA R21, R66, R80, R21                                    ; // x      2400  FMA
/*4f6cca40*/  FFMA R20, R66, R81, R20                                    ; // x      2400  FMA
/*4f6cca50*/  FFMA R29, R67, R80, R29                                    ; // x      2400  FMA
/*4f6cca60*/  LDG.E.LTC128B.CONSTANT R43, P5, [R50]                      ; // x      2400  global load (DRAM/L2: weights or activations)
/*4f6cca70*/  FFMA R28, R67, R81, R28                                    ; // x      2400  FMA
/*4f6cca80*/  FFMA R31, R67, R82, R31                                    ; // x      2400  FMA
/*4f6cca90*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      2400  ALU
/*4f6ccaa0*/  FFMA R30, R67, R83, R30                                    ; // x      2400  FMA
/*4f6ccab0*/  FFMA R23, R66, R82, R23                                    ; // x      2400  FMA
/*4f6ccac0*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      2400  ALU
/*4f6ccad0*/  FFMA R22, R66, R83, R22                                    ; // x      2400  FMA
/*4f6ccae0*/  FFMA R15, R65, R82, R15                                    ; // x      2400  FMA
/*4f6ccaf0*/  FFMA R14, R65, R83, R14                                    ; // x      2400  FMA
/*4f6ccb00*/  FFMA R7, R64, R82, R7                                      ; // x      2400  FMA
/*4f6ccb10*/  LDG.E.LTC128B.CONSTANT R44, P6, [R50]                      ; // x      2400  global load (DRAM/L2: weights or activations)
/*4f6ccb20*/  FFMA R6, R64, R83, R6                                      ; // x      2400  FMA
/*4f6ccb30*/  FFMA R1, R60, R68, R1                                      ; // x      2400  FMA
/*4f6ccb40*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      2400  ALU
/*4f6ccb50*/  FFMA R0, R60, R69, R0                                      ; // x      2400  FMA
/*4f6ccb60*/  FFMA R9, R61, R68, R9                                      ; // x      2400  FMA
/*4f6ccb70*/  LDS.128 R64, [R55+0x2d0]                                   ; // x      2400  shared-memory load
/*4f6ccb80*/  FFMA R8, R61, R69, R8                                      ; // x      2400  FMA
/*4f6ccb90*/  FFMA R17, R62, R68, R17                                    ; // x      2400  FMA
/*4f6ccba0*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      2400  ALU
/*4f6ccbb0*/  FFMA R16, R62, R69, R16                                    ; // x      2400  FMA
/*4f6ccbc0*/  FFMA R25, R63, R68, R25                                    ; // x      2400  FMA
/*4f6ccbd0*/  LDS.128 R76, [R56+0x2d0]                                   ; // x      2400  shared-memory load
/*4f6ccbe0*/  FFMA R24, R63, R69, R24                                    ; // x      2400  FMA
/*4f6ccbf0*/  FFMA R27, R63, R70, R27                                    ; // x      2400  FMA
/*4f6ccc00*/  LDS.128 R80, [R56+0x310]                                   ; // x      2400  shared-memory load
/*4f6ccc10*/  FFMA R26, R63, R71, R26                                    ; // x      2400  FMA
/*4f6ccc20*/  FFMA R19, R62, R70, R19                                    ; // x      2400  FMA
/*4f6ccc30*/  R2P PR, R57.B3, 0x78                                       ; // x      2400  ALU
/*4f6ccc40*/  FFMA R18, R62, R71, R18                                    ; // x      2400  FMA
/*4f6ccc50*/  FFMA R11, R61, R70, R11                                    ; // x      2400  FMA
/*4f6ccc60*/  FFMA R10, R61, R71, R10                                    ; // x      2400  FMA
/*4f6ccc70*/  FFMA R3, R60, R70, R3                                      ; // x      2400  FMA
/*4f6ccc80*/  FFMA R2, R60, R71, R2                                      ; // x      2400  FMA
/*4f6ccc90*/  FFMA R5, R60, R72, R5                                      ; // x      2400  FMA
```

(first 200 of 365 instructions shown; the rest are in the full dump)
