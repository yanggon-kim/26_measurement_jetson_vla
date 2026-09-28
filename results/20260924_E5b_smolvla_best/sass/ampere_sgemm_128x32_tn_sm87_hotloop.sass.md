# Hot loop — `ampere_sgemm_128x32_tn` (sm87)

PC range 0x4f6456b0-0x4f646410; backward BRA 0x4f646410 -> 0x4f6456b0: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_sgemm_128x32_tn_sm87.sass`; per-PC counts: `raw/ampere_sgemm_128x32_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*4f6456b0*/  STS [R26], R16                                             ; // x        96  shared-memory store
/*4f6456c0*/  STS [R26+0x80], R17                                        ; // x        96  shared-memory store
/*4f6456d0*/  STS [R26+0x100], R18                                       ; // x        96  shared-memory store
/*4f6456e0*/  STS [R26+0x180], R19                                       ; // x        96  shared-memory store
/*4f6456f0*/  STS [R23], R20                                             ; // x        96  shared-memory store
/*4f645700*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x        96  block barrier
/*4f645710*/  LOP3.LUT R26, R26, 0x2000, RZ, 0x3c, !PT                   ; // x        96  ALU
/*4f645720*/  LOP3.LUT R23, R23, 0x2000, RZ, 0x3c, !PT                   ; // x        96  ALU
/*4f645730*/  LDS.128 R32, [R22]                                         ; // x        96  shared-memory load
/*4f645740*/  LDS.128 R40, [R21]                                         ; // x        96  shared-memory load
/*4f645750*/  FFMA R1, R32, R40, R1                                      ; // x       384  FMA
/*4f645760*/  ISETP.GE.AND P1, PT, R30, 0x10, PT                         ; // x       384  ALU
/*4f645770*/  FFMA R0, R32, R41, R0                                      ; // x       384  FMA
/*4f645780*/  FFMA R5, R33, R40, R5                                      ; // x       384  FMA
/*4f645790*/  IADD3 R30, R30, -0x8, RZ                                   ; // x       384  ALU
/*4f6457a0*/  FFMA R4, R33, R41, R4                                      ; // x       384  FMA
/*4f6457b0*/  FFMA R9, R34, R40, R9                                      ; // x       384  FMA
/*4f6457c0*/  LDS.128 R36, [R22+0x210]                                   ; // x       384  shared-memory load
/*4f6457d0*/  FFMA R8, R34, R41, R8                                      ; // x       384  FMA
/*4f6457e0*/  FFMA R13, R35, R40, R13                                    ; // x       384  FMA
/*4f6457f0*/  LDS.128 R44, [R21+0x90]                                    ; // x       384  shared-memory load
/*4f645800*/  FFMA R12, R35, R41, R12                                    ; // x       384  FMA
/*4f645810*/  FFMA R15, R35, R42, R15                                    ; // x       384  FMA
/*4f645820*/  @!P1  R2P PR, RZ, 0x78                                     ; // x       384  ALU
/*4f645830*/  FFMA R14, R35, R43, R14                                    ; // x       384  FMA
/*4f645840*/  FFMA R11, R34, R42, R11                                    ; // x       384  FMA
/*4f645850*/  @P1   R2P PR, R31, 0x78                                    ; // x       384  ALU
/*4f645860*/  FFMA R10, R34, R43, R10                                    ; // x       384  FMA
/*4f645870*/  FFMA R7, R33, R42, R7                                      ; // x       384  FMA
/*4f645880*/  @P3   LDG.E.LTC128B.CONSTANT R16, [R28]                    ; // x       384  global load (DRAM/L2: weights or activations)
/*4f645890*/  FFMA R6, R33, R43, R6                                      ; // x       384  FMA
/*4f6458a0*/  FFMA R3, R32, R42, R3                                      ; // x       384  FMA
/*4f6458b0*/  FFMA R2, R32, R43, R2                                      ; // x       384  FMA
/*4f6458c0*/  FFMA R1, R36, R44, R1                                      ; // x       384  FMA
/*4f6458d0*/  LDS.128 R32, [R22+0x420]                                   ; // x       384  shared-memory load
/*4f6458e0*/  FFMA R0, R36, R45, R0                                      ; // x       384  FMA
/*4f6458f0*/  FFMA R5, R37, R44, R5                                      ; // x       384  FMA
/*4f645900*/  @P1   IADD3 R28, P2, R28, c[0x0][0x178], RZ                ; // x       384  ALU
/*4f645910*/  FFMA R4, R37, R45, R4                                      ; // x       384  FMA
/*4f645920*/  FFMA R9, R38, R44, R9                                      ; // x       384  FMA
/*4f645930*/  LDS.128 R40, [R21+0x120]                                   ; // x       384  shared-memory load
/*4f645940*/  FFMA R8, R38, R45, R8                                      ; // x       384  FMA
/*4f645950*/  FFMA R13, R39, R44, R13                                    ; // x       384  FMA
/*4f645960*/  @P1   IADD3.X R29, R29, c[0x0][0x17c], RZ, P2, !PT         ; // x       384  ALU
/*4f645970*/  FFMA R12, R39, R45, R12                                    ; // x       384  FMA
/*4f645980*/  FFMA R15, R39, R46, R15                                    ; // x       384  FMA
/*4f645990*/  @P4   LDG.E.LTC128B.CONSTANT R17, [R28]                    ; // x       384  global load (DRAM/L2: weights or activations)
/*4f6459a0*/  FFMA R14, R39, R47, R14                                    ; // x       384  FMA
/*4f6459b0*/  FFMA R11, R38, R46, R11                                    ; // x       384  FMA
/*4f6459c0*/  FFMA R10, R38, R47, R10                                    ; // x       384  FMA
/*4f6459d0*/  FFMA R7, R37, R46, R7                                      ; // x       384  FMA
/*4f6459e0*/  @P1   IADD3 R28, P2, R28, c[0x0][0x178], RZ                ; // x       384  ALU
/*4f6459f0*/  FFMA R6, R37, R47, R6                                      ; // x       384  FMA
/*4f645a00*/  FFMA R3, R36, R46, R3                                      ; // x       384  FMA
/*4f645a10*/  @P1   IADD3.X R29, R29, c[0x0][0x17c], RZ, P2, !PT         ; // x       384  ALU
/*4f645a20*/  FFMA R2, R36, R47, R2                                      ; // x       384  FMA
/*4f645a30*/  FFMA R1, R32, R40, R1                                      ; // x       384  FMA
/*4f645a40*/  LDS.128 R36, [R22+0x630]                                   ; // x       384  shared-memory load
/*4f645a50*/  FFMA R0, R32, R41, R0                                      ; // x       384  FMA
/*4f645a60*/  FFMA R5, R33, R40, R5                                      ; // x       384  FMA
/*4f645a70*/  LDS.128 R44, [R21+0x1b0]                                   ; // x       384  shared-memory load
/*4f645a80*/  FFMA R4, R33, R41, R4                                      ; // x       384  FMA
/*4f645a90*/  FFMA R9, R34, R40, R9                                      ; // x       384  FMA
/*4f645aa0*/  @P5   LDG.E.LTC128B.CONSTANT R18, [R28]                    ; // x       384  global load (DRAM/L2: weights or activations)
/*4f645ab0*/  FFMA R8, R34, R41, R8                                      ; // x       384  FMA
/*4f645ac0*/  FFMA R13, R35, R40, R13                                    ; // x       384  FMA
/*4f645ad0*/  FFMA R12, R35, R41, R12                                    ; // x       384  FMA
/*4f645ae0*/  FFMA R15, R35, R42, R15                                    ; // x       384  FMA
/*4f645af0*/  @P1   IADD3 R28, P2, R28, c[0x0][0x178], RZ                ; // x       384  ALU
/*4f645b00*/  FFMA R14, R35, R43, R14                                    ; // x       384  FMA
/*4f645b10*/  FFMA R11, R34, R42, R11                                    ; // x       384  FMA
/*4f645b20*/  @P1   IADD3.X R29, R29, c[0x0][0x17c], RZ, P2, !PT         ; // x       384  ALU
/*4f645b30*/  FFMA R10, R34, R43, R10                                    ; // x       384  FMA
/*4f645b40*/  FFMA R7, R33, R42, R7                                      ; // x       384  FMA
/*4f645b50*/  @P6   LDG.E.LTC128B.CONSTANT R19, [R28]                    ; // x       384  global load (DRAM/L2: weights or activations)
/*4f645b60*/  FFMA R6, R33, R43, R6                                      ; // x       384  FMA
/*4f645b70*/  FFMA R3, R32, R42, R3                                      ; // x       384  FMA
/*4f645b80*/  FFMA R2, R32, R43, R2                                      ; // x       384  FMA
/*4f645b90*/  FFMA R1, R36, R44, R1                                      ; // x       384  FMA
/*4f645ba0*/  LDS.128 R32, [R22+0x840]                                   ; // x       384  shared-memory load
/*4f645bb0*/  FFMA R0, R36, R45, R0                                      ; // x       384  FMA
/*4f645bc0*/  FFMA R5, R37, R44, R5                                      ; // x       384  FMA
/*4f645bd0*/  @P1   IADD3 R28, P2, R28, c[0x0][0x188], RZ                ; // x       384  ALU
/*4f645be0*/  FFMA R4, R37, R45, R4                                      ; // x       384  FMA
/*4f645bf0*/  FFMA R9, R38, R44, R9                                      ; // x       384  FMA
/*4f645c00*/  LDS.128 R40, [R21+0x240]                                   ; // x       384  shared-memory load
/*4f645c10*/  FFMA R8, R38, R45, R8                                      ; // x       384  FMA
/*4f645c20*/  FFMA R13, R39, R44, R13                                    ; // x       384  FMA
/*4f645c30*/  @P1   IADD3.X R29, R29, c[0x0][0x18c], RZ, P2, !PT         ; // x       384  ALU
/*4f645c40*/  FFMA R12, R39, R45, R12                                    ; // x       384  FMA
/*4f645c50*/  FFMA R15, R39, R46, R15                                    ; // x       384  FMA
/*4f645c60*/  @P1   R2P PR, R31.B1, 0x78                                 ; // x       384  ALU
/*4f645c70*/  FFMA R14, R39, R47, R14                                    ; // x       384  FMA
/*4f645c80*/  FFMA R11, R38, R46, R11                                    ; // x       384  FMA
/*4f645c90*/  @P3   LDG.E.LTC128B.CONSTANT R20, [R24]                    ; // x       384  global load (DRAM/L2: weights or activations)
/*4f645ca0*/  FFMA R10, R38, R47, R10                                    ; // x       384  FMA
/*4f645cb0*/  FFMA R7, R37, R46, R7                                      ; // x       384  FMA
/*4f645cc0*/  FFMA R6, R37, R47, R6                                      ; // x       384  FMA
/*4f645cd0*/  FFMA R3, R36, R46, R3                                      ; // x       384  FMA
/*4f645ce0*/  @P1   IADD3 R24, P2, R24, c[0x0][0x190], RZ                ; // x       384  ALU
/*4f645cf0*/  FFMA R2, R36, R47, R2                                      ; // x       384  FMA
/*4f645d00*/  FFMA R1, R32, R40, R1                                      ; // x       384  FMA
/*4f645d10*/  LDS.128 R36, [R22+0xa50]                                   ; // x       384  shared-memory load
/*4f645d20*/  FFMA R0, R32, R41, R0                                      ; // x       384  FMA
/*4f645d30*/  FFMA R5, R33, R40, R5                                      ; // x       384  FMA
/*4f645d40*/  @P1   IADD3.X R25, R25, c[0x0][0x194], RZ, P2, !PT         ; // x       384  ALU
/*4f645d50*/  FFMA R4, R33, R41, R4                                      ; // x       384  FMA
/*4f645d60*/  FFMA R9, R34, R40, R9                                      ; // x       384  FMA
/*4f645d70*/  LDS.128 R44, [R21+0x2d0]                                   ; // x       384  shared-memory load
/*4f645d80*/  FFMA R8, R34, R41, R8                                      ; // x       384  FMA
/*4f645d90*/  FFMA R13, R35, R40, R13                                    ; // x       384  FMA
/*4f645da0*/  FFMA R12, R35, R41, R12                                    ; // x       384  FMA
/*4f645db0*/  FFMA R15, R35, R42, R15                                    ; // x       384  FMA
/*4f645dc0*/  FFMA R14, R35, R43, R14                                    ; // x       384  FMA
/*4f645dd0*/  FFMA R11, R34, R42, R11                                    ; // x       384  FMA
/*4f645de0*/  FFMA R10, R34, R43, R10                                    ; // x       384  FMA
/*4f645df0*/  FFMA R7, R33, R42, R7                                      ; // x       384  FMA
/*4f645e00*/  FFMA R6, R33, R43, R6                                      ; // x       384  FMA
/*4f645e10*/  FFMA R3, R32, R42, R3                                      ; // x       384  FMA
/*4f645e20*/  FFMA R2, R32, R43, R2                                      ; // x       384  FMA
/*4f645e30*/  FFMA R1, R36, R44, R1                                      ; // x       384  FMA
/*4f645e40*/  LDS.128 R32, [R22+0xc60]                                   ; // x       384  shared-memory load
/*4f645e50*/  FFMA R0, R36, R45, R0                                      ; // x       384  FMA
/*4f645e60*/  FFMA R5, R37, R44, R5                                      ; // x       384  FMA
/*4f645e70*/  LDS.128 R40, [R21+0x360]                                   ; // x       384  shared-memory load
/*4f645e80*/  FFMA R4, R37, R45, R4                                      ; // x       384  FMA
/*4f645e90*/  FFMA R9, R38, R44, R9                                      ; // x       384  FMA
/*4f645ea0*/  FFMA R8, R38, R45, R8                                      ; // x       384  FMA
/*4f645eb0*/  FFMA R13, R39, R44, R13                                    ; // x       384  FMA
/*4f645ec0*/  FFMA R12, R39, R45, R12                                    ; // x       384  FMA
/*4f645ed0*/  FFMA R15, R39, R46, R15                                    ; // x       384  FMA
/*4f645ee0*/  @P1   STS [R26], R16                                       ; // x       384  shared-memory store
/*4f645ef0*/  FFMA R14, R39, R47, R14                                    ; // x       384  FMA
/*4f645f00*/  FFMA R11, R38, R46, R11                                    ; // x       384  FMA
/*4f645f10*/  @P1   STS [R26+0x80], R17                                  ; // x       384  shared-memory store
/*4f645f20*/  FFMA R10, R38, R47, R10                                    ; // x       384  FMA
/*4f645f30*/  FFMA R7, R37, R46, R7                                      ; // x       384  FMA
/*4f645f40*/  @P1   STS [R26+0x100], R18                                 ; // x       384  shared-memory store
/*4f645f50*/  FFMA R6, R37, R47, R6                                      ; // x       384  FMA
/*4f645f60*/  FFMA R3, R36, R46, R3                                      ; // x       384  FMA
/*4f645f70*/  @P1   STS [R26+0x180], R19                                 ; // x       384  shared-memory store
/*4f645f80*/  FFMA R2, R36, R47, R2                                      ; // x       384  FMA
/*4f645f90*/  FFMA R1, R32, R40, R1                                      ; // x       384  FMA
/*4f645fa0*/  LDS.128 R36, [R22+0xe70]                                   ; // x       384  shared-memory load
/*4f645fb0*/  FFMA R0, R32, R41, R0                                      ; // x       384  FMA
/*4f645fc0*/  FFMA R5, R33, R40, R5                                      ; // x       384  FMA
/*4f645fd0*/  LDS.128 R44, [R21+0x3f0]                                   ; // x       384  shared-memory load
/*4f645fe0*/  FFMA R4, R33, R41, R4                                      ; // x       384  FMA
/*4f645ff0*/  FFMA R9, R34, R40, R9                                      ; // x       384  FMA
/*4f646000*/  @P1   STS [R23], R20                                       ; // x       384  shared-memory store
/*4f646010*/  FFMA R8, R34, R41, R8                                      ; // x       384  FMA
/*4f646020*/  FFMA R13, R35, R40, R13                                    ; // x       384  FMA
/*4f646030*/  @P1   BAR.SYNC.DEFER_BLOCKING 0x0                          ; // x       384  block barrier
/*4f646040*/  FFMA R12, R35, R41, R12                                    ; // x       384  FMA
/*4f646050*/  FFMA R15, R35, R42, R15                                    ; // x       384  FMA
/*4f646060*/  FFMA R14, R35, R43, R14                                    ; // x       384  FMA
/*4f646070*/  FFMA R11, R34, R42, R11                                    ; // x       384  FMA
/*4f646080*/  FFMA R10, R34, R43, R10                                    ; // x       384  FMA
/*4f646090*/  FFMA R7, R33, R42, R7                                      ; // x       384  FMA
/*4f6460a0*/  FFMA R6, R33, R43, R6                                      ; // x       384  FMA
/*4f6460b0*/  FFMA R3, R32, R42, R3                                      ; // x       384  FMA
/*4f6460c0*/  FFMA R2, R32, R43, R2                                      ; // x       384  FMA
/*4f6460d0*/  FFMA R1, R36, R44, R1                                      ; // x       384  FMA
/*4f6460e0*/  LOP3.LUT R22, R22, 0x2000, RZ, 0x3c, !PT                   ; // x       384  ALU
/*4f6460f0*/  FFMA R0, R36, R45, R0                                      ; // x       384  FMA
/*4f646100*/  FFMA R5, R37, R44, R5                                      ; // x       384  FMA
/*4f646110*/  LOP3.LUT R21, R21, 0x2000, RZ, 0x3c, !PT                   ; // x       384  ALU
/*4f646120*/  FFMA R4, R37, R45, R4                                      ; // x       384  FMA
/*4f646130*/  FFMA R9, R38, R44, R9                                      ; // x       384  FMA
/*4f646140*/  @P1   LOP3.LUT R26, R26, 0x2000, RZ, 0x3c, !PT             ; // x       384  ALU
/*4f646150*/  FFMA R8, R38, R45, R8                                      ; // x       384  FMA
/*4f646160*/  FFMA R13, R39, R44, R13                                    ; // x       384  FMA
/*4f646170*/  @P1   LDS.128 R32, [R22]                                   ; // x       384  shared-memory load
/*4f646180*/  FFMA R12, R39, R45, R12                                    ; // x       384  FMA
/*4f646190*/  FFMA R15, R39, R46, R15                                    ; // x       384  FMA
/*4f6461a0*/  @P1   LDS.128 R40, [R21]                                   ; // x       384  shared-memory load
/*4f6461b0*/  FFMA R14, R39, R47, R14                                    ; // x       384  FMA
/*4f6461c0*/  FFMA R11, R38, R46, R11                                    ; // x       384  FMA
/*4f6461d0*/  @P1   LOP3.LUT R23, R23, 0x2000, RZ, 0x3c, !PT             ; // x       384  ALU
/*4f6461e0*/  FFMA R10, R38, R47, R10                                    ; // x       384  FMA
/*4f6461f0*/  FFMA R7, R37, R46, R7                                      ; // x       384  FMA
/*4f646200*/  FFMA R6, R37, R47, R6                                      ; // x       384  FMA
/*4f646210*/  FFMA R3, R36, R46, R3                                      ; // x       384  FMA
/*4f646220*/  FFMA R2, R36, R47, R2                                      ; // x       384  FMA
/*4f646230*/  @P1   BRA.U 0x24f645750                                    ; // x       384  branch
/*4f646240*/  LOP3.LUT P0, RZ, R30, 0x7, RZ, 0xc0, !PT                   ; // x        96  ALU
/*4f646250*/  @!P0  BRA.U 0x24f646420                                    ; // x        96  branch
/*4f646260*/  NOP                                                        ; // x         0  Branch
/*4f646270*/  S2R R36, SR_TID.X                                          ; // x         0  Branch
/*4f646280*/  LOP3.LUT R38, R36, 0x7, RZ, 0xc0, !PT                      ; // x         0  ALU
/*4f646290*/  ISETP.GT.AND P1, PT, R30, R38, PT                          ; // x         0  ALU
/*4f6462a0*/  @!P1  R2P PR, RZ, 0x78                                     ; // x         0  ALU
/*4f6462b0*/  @P1   R2P PR, R31, 0x78                                    ; // x         0  ALU
/*4f6462c0*/  @P3   LDG.E.LTC128B.CONSTANT R16, [R28]                    ; // x         0  global load (DRAM/L2: weights or activations)
/*4f6462d0*/  @!P3  MOV R16, RZ                                          ; // x         0  ALU
/*4f6462e0*/  @P1   IADD3 R28, P2, R28, c[0x0][0x178], RZ                ; // x         0  ALU
/*4f6462f0*/  @P1   IADD3.X R29, R29, c[0x0][0x17c], RZ, P2, !PT         ; // x         0  ALU
/*4f646300*/  @P4   LDG.E.LTC128B.CONSTANT R17, [R28]                    ; // x         0  global load (DRAM/L2: weights or activations)
/*4f646310*/  @!P4  MOV R17, RZ                                          ; // x         0  ALU
/*4f646320*/  @P1   IADD3 R28, P2, R28, c[0x0][0x178], RZ                ; // x         0  ALU
```

(first 200 of 215 instructions shown; the rest are in the full dump)
