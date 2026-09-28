# Hot loop — `ampere_sgemm_32x32_sliced1x4_tn` (sm87)

PC range 0x4cb96620-0x4cb97ce0; backward BRA 0x4cb97ce0 -> 0x4cb96620: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_sgemm_32x32_sliced1x4_tn_sm87.sass`; per-PC counts: `raw/ampere_sgemm_32x32_sliced1x4_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*4cb96620*/  FFMA R1, R60, R68, R1                                      ; // x      4232  FMA
/*4cb96630*/  R2P PR, R57, 0x78                                          ; // x      4232  ALU
/*4cb96640*/  FFMA R0, R60, R69, R0                                      ; // x      4232  FMA
/*4cb96650*/  FFMA R9, R61, R68, R9                                      ; // x      4232  FMA
/*4cb96660*/  LDS.128 R64, [R55+0x90]                                    ; // x      4232  shared-memory load
/*4cb96670*/  FFMA R8, R61, R69, R8                                      ; // x      4232  FMA
/*4cb96680*/  FFMA R17, R62, R68, R17                                    ; // x      4232  FMA
/*4cb96690*/  LDS.128 R76, [R56+0x90]                                    ; // x      4232  shared-memory load
/*4cb966a0*/  FFMA R16, R62, R69, R16                                    ; // x      4232  FMA
/*4cb966b0*/  FFMA R25, R63, R68, R25                                    ; // x      4232  FMA
/*4cb966c0*/  LDS.128 R80, [R56+0xd0]                                    ; // x      4232  shared-memory load
/*4cb966d0*/  FFMA R24, R63, R69, R24                                    ; // x      4232  FMA
/*4cb966e0*/  FFMA R27, R63, R70, R27                                    ; // x      4232  FMA
/*4cb966f0*/  LDG.E.LTC128B.CONSTANT R33, P3, [R50]                      ; // x      4232  global load (DRAM/L2: weights or activations)
/*4cb96700*/  FFMA R26, R63, R71, R26                                    ; // x      4232  FMA
/*4cb96710*/  FFMA R19, R62, R70, R19                                    ; // x      4232  FMA
/*4cb96720*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      4232  ALU
/*4cb96730*/  FFMA R18, R62, R71, R18                                    ; // x      4232  FMA
/*4cb96740*/  FFMA R11, R61, R70, R11                                    ; // x      4232  FMA
/*4cb96750*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      4232  ALU
/*4cb96760*/  FFMA R10, R61, R71, R10                                    ; // x      4232  FMA
/*4cb96770*/  FFMA R3, R60, R70, R3                                      ; // x      4232  FMA
/*4cb96780*/  FFMA R2, R60, R71, R2                                      ; // x      4232  FMA
/*4cb96790*/  FFMA R5, R60, R72, R5                                      ; // x      4232  FMA
/*4cb967a0*/  LDG.E.LTC128B.CONSTANT R34, P4, [R50]                      ; // x      4232  global load (DRAM/L2: weights or activations)
/*4cb967b0*/  FFMA R4, R60, R73, R4                                      ; // x      4232  FMA
/*4cb967c0*/  FFMA R13, R61, R72, R13                                    ; // x      4232  FMA
/*4cb967d0*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      4232  ALU
/*4cb967e0*/  FFMA R12, R61, R73, R12                                    ; // x      4232  FMA
/*4cb967f0*/  FFMA R21, R62, R72, R21                                    ; // x      4232  FMA
/*4cb96800*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      4232  ALU
/*4cb96810*/  FFMA R20, R62, R73, R20                                    ; // x      4232  FMA
/*4cb96820*/  FFMA R29, R63, R72, R29                                    ; // x      4232  FMA
/*4cb96830*/  FFMA R28, R63, R73, R28                                    ; // x      4232  FMA
/*4cb96840*/  FFMA R31, R63, R74, R31                                    ; // x      4232  FMA
/*4cb96850*/  LDG.E.LTC128B.CONSTANT R35, P5, [R50]                      ; // x      4232  global load (DRAM/L2: weights or activations)
/*4cb96860*/  FFMA R30, R63, R75, R30                                    ; // x      4232  FMA
/*4cb96870*/  FFMA R23, R62, R74, R23                                    ; // x      4232  FMA
/*4cb96880*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      4232  ALU
/*4cb96890*/  FFMA R22, R62, R75, R22                                    ; // x      4232  FMA
/*4cb968a0*/  FFMA R15, R61, R74, R15                                    ; // x      4232  FMA
/*4cb968b0*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      4232  ALU
/*4cb968c0*/  FFMA R14, R61, R75, R14                                    ; // x      4232  FMA
/*4cb968d0*/  FFMA R7, R60, R74, R7                                      ; // x      4232  FMA
/*4cb968e0*/  FFMA R6, R60, R75, R6                                      ; // x      4232  FMA
/*4cb968f0*/  FFMA R1, R64, R76, R1                                      ; // x      4232  FMA
/*4cb96900*/  LDS.128 R60, [R55+0x120]                                   ; // x      4232  shared-memory load
/*4cb96910*/  FFMA R0, R64, R77, R0                                      ; // x      4232  FMA
/*4cb96920*/  FFMA R9, R65, R76, R9                                      ; // x      4232  FMA
/*4cb96930*/  LDS.128 R68, [R56+0x120]                                   ; // x      4232  shared-memory load
/*4cb96940*/  FFMA R8, R65, R77, R8                                      ; // x      4232  FMA
/*4cb96950*/  FFMA R17, R66, R76, R17                                    ; // x      4232  FMA
/*4cb96960*/  LDS.128 R72, [R56+0x160]                                   ; // x      4232  shared-memory load
/*4cb96970*/  FFMA R16, R66, R77, R16                                    ; // x      4232  FMA
/*4cb96980*/  FFMA R25, R67, R76, R25                                    ; // x      4232  FMA
/*4cb96990*/  LDG.E.LTC128B.CONSTANT R36, P6, [R50]                      ; // x      4232  global load (DRAM/L2: weights or activations)
/*4cb969a0*/  FFMA R24, R67, R77, R24                                    ; // x      4232  FMA
/*4cb969b0*/  FFMA R27, R67, R78, R27                                    ; // x      4232  FMA
/*4cb969c0*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      4232  ALU
/*4cb969d0*/  FFMA R26, R67, R79, R26                                    ; // x      4232  FMA
/*4cb969e0*/  FFMA R19, R66, R78, R19                                    ; // x      4232  FMA
/*4cb969f0*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      4232  ALU
/*4cb96a00*/  FFMA R18, R66, R79, R18                                    ; // x      4232  FMA
/*4cb96a10*/  FFMA R11, R65, R78, R11                                    ; // x      4232  FMA
/*4cb96a20*/  FFMA R10, R65, R79, R10                                    ; // x      4232  FMA
/*4cb96a30*/  FFMA R3, R64, R78, R3                                      ; // x      4232  FMA
/*4cb96a40*/  R2P PR, R57.B1, 0x78                                       ; // x      4232  ALU
/*4cb96a50*/  FFMA R2, R64, R79, R2                                      ; // x      4232  FMA
/*4cb96a60*/  FFMA R5, R64, R80, R5                                      ; // x      4232  FMA
/*4cb96a70*/  FFMA R4, R64, R81, R4                                      ; // x      4232  FMA
/*4cb96a80*/  FFMA R13, R65, R80, R13                                    ; // x      4232  FMA
/*4cb96a90*/  FFMA R12, R65, R81, R12                                    ; // x      4232  FMA
/*4cb96aa0*/  FFMA R21, R66, R80, R21                                    ; // x      4232  FMA
/*4cb96ab0*/  FFMA R20, R66, R81, R20                                    ; // x      4232  FMA
/*4cb96ac0*/  LDG.E.LTC128B.CONSTANT R37, P3, [R50]                      ; // x      4232  global load (DRAM/L2: weights or activations)
/*4cb96ad0*/  FFMA R29, R67, R80, R29                                    ; // x      4232  FMA
/*4cb96ae0*/  FFMA R28, R67, R81, R28                                    ; // x      4232  FMA
/*4cb96af0*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      4232  ALU
/*4cb96b00*/  FFMA R31, R67, R82, R31                                    ; // x      4232  FMA
/*4cb96b10*/  FFMA R30, R67, R83, R30                                    ; // x      4232  FMA
/*4cb96b20*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      4232  ALU
/*4cb96b30*/  FFMA R23, R66, R82, R23                                    ; // x      4232  FMA
/*4cb96b40*/  FFMA R22, R66, R83, R22                                    ; // x      4232  FMA
/*4cb96b50*/  FFMA R15, R65, R82, R15                                    ; // x      4232  FMA
/*4cb96b60*/  FFMA R14, R65, R83, R14                                    ; // x      4232  FMA
/*4cb96b70*/  LDG.E.LTC128B.CONSTANT R38, P4, [R50]                      ; // x      4232  global load (DRAM/L2: weights or activations)
/*4cb96b80*/  FFMA R7, R64, R82, R7                                      ; // x      4232  FMA
/*4cb96b90*/  FFMA R6, R64, R83, R6                                      ; // x      4232  FMA
/*4cb96ba0*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      4232  ALU
/*4cb96bb0*/  FFMA R1, R60, R68, R1                                      ; // x      4232  FMA
/*4cb96bc0*/  FFMA R0, R60, R69, R0                                      ; // x      4232  FMA
/*4cb96bd0*/  LDS.128 R64, [R55+0x1b0]                                   ; // x      4232  shared-memory load
/*4cb96be0*/  FFMA R9, R61, R68, R9                                      ; // x      4232  FMA
/*4cb96bf0*/  FFMA R8, R61, R69, R8                                      ; // x      4232  FMA
/*4cb96c00*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      4232  ALU
/*4cb96c10*/  FFMA R17, R62, R68, R17                                    ; // x      4232  FMA
/*4cb96c20*/  FFMA R16, R62, R69, R16                                    ; // x      4232  FMA
/*4cb96c30*/  LDS.128 R76, [R56+0x1b0]                                   ; // x      4232  shared-memory load
/*4cb96c40*/  FFMA R25, R63, R68, R25                                    ; // x      4232  FMA
/*4cb96c50*/  FFMA R24, R63, R69, R24                                    ; // x      4232  FMA
/*4cb96c60*/  LDS.128 R80, [R56+0x1f0]                                   ; // x      4232  shared-memory load
/*4cb96c70*/  FFMA R27, R63, R70, R27                                    ; // x      4232  FMA
/*4cb96c80*/  FFMA R26, R63, R71, R26                                    ; // x      4232  FMA
/*4cb96c90*/  LDG.E.LTC128B.CONSTANT R39, P5, [R50]                      ; // x      4232  global load (DRAM/L2: weights or activations)
/*4cb96ca0*/  FFMA R19, R62, R70, R19                                    ; // x      4232  FMA
/*4cb96cb0*/  FFMA R18, R62, R71, R18                                    ; // x      4232  FMA
/*4cb96cc0*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      4232  ALU
/*4cb96cd0*/  FFMA R11, R61, R70, R11                                    ; // x      4232  FMA
/*4cb96ce0*/  FFMA R10, R61, R71, R10                                    ; // x      4232  FMA
/*4cb96cf0*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      4232  ALU
/*4cb96d00*/  FFMA R3, R60, R70, R3                                      ; // x      4232  FMA
/*4cb96d10*/  FFMA R2, R60, R71, R2                                      ; // x      4232  FMA
/*4cb96d20*/  FFMA R5, R60, R72, R5                                      ; // x      4232  FMA
/*4cb96d30*/  FFMA R4, R60, R73, R4                                      ; // x      4232  FMA
/*4cb96d40*/  LDG.E.LTC128B.CONSTANT R40, P6, [R50]                      ; // x      4232  global load (DRAM/L2: weights or activations)
/*4cb96d50*/  FFMA R13, R61, R72, R13                                    ; // x      4232  FMA
/*4cb96d60*/  FFMA R12, R61, R73, R12                                    ; // x      4232  FMA
/*4cb96d70*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      4232  ALU
/*4cb96d80*/  FFMA R21, R62, R72, R21                                    ; // x      4232  FMA
/*4cb96d90*/  FFMA R20, R62, R73, R20                                    ; // x      4232  FMA
/*4cb96da0*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      4232  ALU
/*4cb96db0*/  FFMA R29, R63, R72, R29                                    ; // x      4232  FMA
/*4cb96dc0*/  FFMA R28, R63, R73, R28                                    ; // x      4232  FMA
/*4cb96dd0*/  FFMA R31, R63, R74, R31                                    ; // x      4232  FMA
/*4cb96de0*/  FFMA R30, R63, R75, R30                                    ; // x      4232  FMA
/*4cb96df0*/  R2P PR, R57.B2, 0x78                                       ; // x      4232  ALU
/*4cb96e00*/  FFMA R23, R62, R74, R23                                    ; // x      4232  FMA
/*4cb96e10*/  FFMA R22, R62, R75, R22                                    ; // x      4232  FMA
/*4cb96e20*/  FFMA R15, R61, R74, R15                                    ; // x      4232  FMA
/*4cb96e30*/  FFMA R14, R61, R75, R14                                    ; // x      4232  FMA
/*4cb96e40*/  FFMA R7, R60, R74, R7                                      ; // x      4232  FMA
/*4cb96e50*/  FFMA R6, R60, R75, R6                                      ; // x      4232  FMA
/*4cb96e60*/  FFMA R1, R64, R76, R1                                      ; // x      4232  FMA
/*4cb96e70*/  LDS.128 R60, [R55+0x240]                                   ; // x      4232  shared-memory load
/*4cb96e80*/  FFMA R0, R64, R77, R0                                      ; // x      4232  FMA
/*4cb96e90*/  FFMA R9, R65, R76, R9                                      ; // x      4232  FMA
/*4cb96ea0*/  LDS.128 R68, [R56+0x240]                                   ; // x      4232  shared-memory load
/*4cb96eb0*/  FFMA R8, R65, R77, R8                                      ; // x      4232  FMA
/*4cb96ec0*/  FFMA R17, R66, R76, R17                                    ; // x      4232  FMA
/*4cb96ed0*/  LDS.128 R72, [R56+0x280]                                   ; // x      4232  shared-memory load
/*4cb96ee0*/  FFMA R16, R66, R77, R16                                    ; // x      4232  FMA
/*4cb96ef0*/  FFMA R25, R67, R76, R25                                    ; // x      4232  FMA
/*4cb96f00*/  LDG.E.LTC128B.CONSTANT R41, P3, [R50]                      ; // x      4232  global load (DRAM/L2: weights or activations)
/*4cb96f10*/  FFMA R24, R67, R77, R24                                    ; // x      4232  FMA
/*4cb96f20*/  FFMA R27, R67, R78, R27                                    ; // x      4232  FMA
/*4cb96f30*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      4232  ALU
/*4cb96f40*/  FFMA R26, R67, R79, R26                                    ; // x      4232  FMA
/*4cb96f50*/  FFMA R19, R66, R78, R19                                    ; // x      4232  FMA
/*4cb96f60*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      4232  ALU
/*4cb96f70*/  FFMA R18, R66, R79, R18                                    ; // x      4232  FMA
/*4cb96f80*/  FFMA R11, R65, R78, R11                                    ; // x      4232  FMA
/*4cb96f90*/  FFMA R10, R65, R79, R10                                    ; // x      4232  FMA
/*4cb96fa0*/  FFMA R3, R64, R78, R3                                      ; // x      4232  FMA
/*4cb96fb0*/  LDG.E.LTC128B.CONSTANT R42, P4, [R50]                      ; // x      4232  global load (DRAM/L2: weights or activations)
/*4cb96fc0*/  FFMA R2, R64, R79, R2                                      ; // x      4232  FMA
/*4cb96fd0*/  FFMA R5, R64, R80, R5                                      ; // x      4232  FMA
/*4cb96fe0*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      4232  ALU
/*4cb96ff0*/  FFMA R4, R64, R81, R4                                      ; // x      4232  FMA
/*4cb97000*/  FFMA R13, R65, R80, R13                                    ; // x      4232  FMA
/*4cb97010*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      4232  ALU
/*4cb97020*/  FFMA R12, R65, R81, R12                                    ; // x      4232  FMA
/*4cb97030*/  FFMA R21, R66, R80, R21                                    ; // x      4232  FMA
/*4cb97040*/  FFMA R20, R66, R81, R20                                    ; // x      4232  FMA
/*4cb97050*/  FFMA R29, R67, R80, R29                                    ; // x      4232  FMA
/*4cb97060*/  LDG.E.LTC128B.CONSTANT R43, P5, [R50]                      ; // x      4232  global load (DRAM/L2: weights or activations)
/*4cb97070*/  FFMA R28, R67, R81, R28                                    ; // x      4232  FMA
/*4cb97080*/  FFMA R31, R67, R82, R31                                    ; // x      4232  FMA
/*4cb97090*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      4232  ALU
/*4cb970a0*/  FFMA R30, R67, R83, R30                                    ; // x      4232  FMA
/*4cb970b0*/  FFMA R23, R66, R82, R23                                    ; // x      4232  FMA
/*4cb970c0*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      4232  ALU
/*4cb970d0*/  FFMA R22, R66, R83, R22                                    ; // x      4232  FMA
/*4cb970e0*/  FFMA R15, R65, R82, R15                                    ; // x      4232  FMA
/*4cb970f0*/  FFMA R14, R65, R83, R14                                    ; // x      4232  FMA
/*4cb97100*/  FFMA R7, R64, R82, R7                                      ; // x      4232  FMA
/*4cb97110*/  LDG.E.LTC128B.CONSTANT R44, P6, [R50]                      ; // x      4232  global load (DRAM/L2: weights or activations)
/*4cb97120*/  FFMA R6, R64, R83, R6                                      ; // x      4232  FMA
/*4cb97130*/  FFMA R1, R60, R68, R1                                      ; // x      4232  FMA
/*4cb97140*/  LEA R50, P2, R49, R50, 0x3                                 ; // x      4232  ALU
/*4cb97150*/  FFMA R0, R60, R69, R0                                      ; // x      4232  FMA
/*4cb97160*/  FFMA R9, R61, R68, R9                                      ; // x      4232  FMA
/*4cb97170*/  LDS.128 R64, [R55+0x2d0]                                   ; // x      4232  shared-memory load
/*4cb97180*/  FFMA R8, R61, R69, R8                                      ; // x      4232  FMA
/*4cb97190*/  FFMA R17, R62, R68, R17                                    ; // x      4232  FMA
/*4cb971a0*/  LEA.HI.X R51, R49, R51, RZ, 0x3, P2                        ; // x      4232  ALU
/*4cb971b0*/  FFMA R16, R62, R69, R16                                    ; // x      4232  FMA
/*4cb971c0*/  FFMA R25, R63, R68, R25                                    ; // x      4232  FMA
/*4cb971d0*/  LDS.128 R76, [R56+0x2d0]                                   ; // x      4232  shared-memory load
/*4cb971e0*/  FFMA R24, R63, R69, R24                                    ; // x      4232  FMA
/*4cb971f0*/  FFMA R27, R63, R70, R27                                    ; // x      4232  FMA
/*4cb97200*/  LDS.128 R80, [R56+0x310]                                   ; // x      4232  shared-memory load
/*4cb97210*/  FFMA R26, R63, R71, R26                                    ; // x      4232  FMA
/*4cb97220*/  FFMA R19, R62, R70, R19                                    ; // x      4232  FMA
/*4cb97230*/  R2P PR, R57.B3, 0x78                                       ; // x      4232  ALU
/*4cb97240*/  FFMA R18, R62, R71, R18                                    ; // x      4232  FMA
/*4cb97250*/  FFMA R11, R61, R70, R11                                    ; // x      4232  FMA
/*4cb97260*/  FFMA R10, R61, R71, R10                                    ; // x      4232  FMA
/*4cb97270*/  FFMA R3, R60, R70, R3                                      ; // x      4232  FMA
/*4cb97280*/  FFMA R2, R60, R71, R2                                      ; // x      4232  FMA
/*4cb97290*/  FFMA R5, R60, R72, R5                                      ; // x      4232  FMA
```

(first 200 of 365 instructions shown; the rest are in the full dump)
