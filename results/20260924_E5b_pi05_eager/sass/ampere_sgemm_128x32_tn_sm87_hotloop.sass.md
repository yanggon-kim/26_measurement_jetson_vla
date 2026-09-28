# Hot loop — `ampere_sgemm_128x32_tn` (sm87)

PC range 0x22763b0-0x2277110; backward BRA 0x2277110 -> 0x22763b0: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_sgemm_128x32_tn_sm87.sass`; per-PC counts: `raw/ampere_sgemm_128x32_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*022763b0*/  STS [R26], R16                                             ; // x       128  shared-memory store
/*022763c0*/  STS [R26+0x80], R17                                        ; // x       128  shared-memory store
/*022763d0*/  STS [R26+0x100], R18                                       ; // x       128  shared-memory store
/*022763e0*/  STS [R26+0x180], R19                                       ; // x       128  shared-memory store
/*022763f0*/  STS [R23], R20                                             ; // x       128  shared-memory store
/*02276400*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       128  block barrier
/*02276410*/  LOP3.LUT R26, R26, 0x2000, RZ, 0x3c, !PT                   ; // x       128  ALU
/*02276420*/  LOP3.LUT R23, R23, 0x2000, RZ, 0x3c, !PT                   ; // x       128  ALU
/*02276430*/  LDS.128 R32, [R22]                                         ; // x       128  shared-memory load
/*02276440*/  LDS.128 R40, [R21]                                         ; // x       128  shared-memory load
/*02276450*/  FFMA R1, R32, R40, R1                                      ; // x       512  FMA
/*02276460*/  ISETP.GE.AND P1, PT, R30, 0x10, PT                         ; // x       512  ALU
/*02276470*/  FFMA R0, R32, R41, R0                                      ; // x       512  FMA
/*02276480*/  FFMA R5, R33, R40, R5                                      ; // x       512  FMA
/*02276490*/  IADD3 R30, R30, -0x8, RZ                                   ; // x       512  ALU
/*022764a0*/  FFMA R4, R33, R41, R4                                      ; // x       512  FMA
/*022764b0*/  FFMA R9, R34, R40, R9                                      ; // x       512  FMA
/*022764c0*/  LDS.128 R36, [R22+0x210]                                   ; // x       512  shared-memory load
/*022764d0*/  FFMA R8, R34, R41, R8                                      ; // x       512  FMA
/*022764e0*/  FFMA R13, R35, R40, R13                                    ; // x       512  FMA
/*022764f0*/  LDS.128 R44, [R21+0x90]                                    ; // x       512  shared-memory load
/*02276500*/  FFMA R12, R35, R41, R12                                    ; // x       512  FMA
/*02276510*/  FFMA R15, R35, R42, R15                                    ; // x       512  FMA
/*02276520*/  @!P1  R2P PR, RZ, 0x78                                     ; // x       512  ALU
/*02276530*/  FFMA R14, R35, R43, R14                                    ; // x       512  FMA
/*02276540*/  FFMA R11, R34, R42, R11                                    ; // x       512  FMA
/*02276550*/  @P1   R2P PR, R31, 0x78                                    ; // x       512  ALU
/*02276560*/  FFMA R10, R34, R43, R10                                    ; // x       512  FMA
/*02276570*/  FFMA R7, R33, R42, R7                                      ; // x       512  FMA
/*02276580*/  @P3   LDG.E.LTC128B.CONSTANT R16, [R28]                    ; // x       512  global load (DRAM/L2: weights or activations)
/*02276590*/  FFMA R6, R33, R43, R6                                      ; // x       512  FMA
/*022765a0*/  FFMA R3, R32, R42, R3                                      ; // x       512  FMA
/*022765b0*/  FFMA R2, R32, R43, R2                                      ; // x       512  FMA
/*022765c0*/  FFMA R1, R36, R44, R1                                      ; // x       512  FMA
/*022765d0*/  LDS.128 R32, [R22+0x420]                                   ; // x       512  shared-memory load
/*022765e0*/  FFMA R0, R36, R45, R0                                      ; // x       512  FMA
/*022765f0*/  FFMA R5, R37, R44, R5                                      ; // x       512  FMA
/*02276600*/  @P1   IADD3 R28, P2, R28, c[0x0][0x178], RZ                ; // x       512  ALU
/*02276610*/  FFMA R4, R37, R45, R4                                      ; // x       512  FMA
/*02276620*/  FFMA R9, R38, R44, R9                                      ; // x       512  FMA
/*02276630*/  LDS.128 R40, [R21+0x120]                                   ; // x       512  shared-memory load
/*02276640*/  FFMA R8, R38, R45, R8                                      ; // x       512  FMA
/*02276650*/  FFMA R13, R39, R44, R13                                    ; // x       512  FMA
/*02276660*/  @P1   IADD3.X R29, R29, c[0x0][0x17c], RZ, P2, !PT         ; // x       512  ALU
/*02276670*/  FFMA R12, R39, R45, R12                                    ; // x       512  FMA
/*02276680*/  FFMA R15, R39, R46, R15                                    ; // x       512  FMA
/*02276690*/  @P4   LDG.E.LTC128B.CONSTANT R17, [R28]                    ; // x       512  global load (DRAM/L2: weights or activations)
/*022766a0*/  FFMA R14, R39, R47, R14                                    ; // x       512  FMA
/*022766b0*/  FFMA R11, R38, R46, R11                                    ; // x       512  FMA
/*022766c0*/  FFMA R10, R38, R47, R10                                    ; // x       512  FMA
/*022766d0*/  FFMA R7, R37, R46, R7                                      ; // x       512  FMA
/*022766e0*/  @P1   IADD3 R28, P2, R28, c[0x0][0x178], RZ                ; // x       512  ALU
/*022766f0*/  FFMA R6, R37, R47, R6                                      ; // x       512  FMA
/*02276700*/  FFMA R3, R36, R46, R3                                      ; // x       512  FMA
/*02276710*/  @P1   IADD3.X R29, R29, c[0x0][0x17c], RZ, P2, !PT         ; // x       512  ALU
/*02276720*/  FFMA R2, R36, R47, R2                                      ; // x       512  FMA
/*02276730*/  FFMA R1, R32, R40, R1                                      ; // x       512  FMA
/*02276740*/  LDS.128 R36, [R22+0x630]                                   ; // x       512  shared-memory load
/*02276750*/  FFMA R0, R32, R41, R0                                      ; // x       512  FMA
/*02276760*/  FFMA R5, R33, R40, R5                                      ; // x       512  FMA
/*02276770*/  LDS.128 R44, [R21+0x1b0]                                   ; // x       512  shared-memory load
/*02276780*/  FFMA R4, R33, R41, R4                                      ; // x       512  FMA
/*02276790*/  FFMA R9, R34, R40, R9                                      ; // x       512  FMA
/*022767a0*/  @P5   LDG.E.LTC128B.CONSTANT R18, [R28]                    ; // x       512  global load (DRAM/L2: weights or activations)
/*022767b0*/  FFMA R8, R34, R41, R8                                      ; // x       512  FMA
/*022767c0*/  FFMA R13, R35, R40, R13                                    ; // x       512  FMA
/*022767d0*/  FFMA R12, R35, R41, R12                                    ; // x       512  FMA
/*022767e0*/  FFMA R15, R35, R42, R15                                    ; // x       512  FMA
/*022767f0*/  @P1   IADD3 R28, P2, R28, c[0x0][0x178], RZ                ; // x       512  ALU
/*02276800*/  FFMA R14, R35, R43, R14                                    ; // x       512  FMA
/*02276810*/  FFMA R11, R34, R42, R11                                    ; // x       512  FMA
/*02276820*/  @P1   IADD3.X R29, R29, c[0x0][0x17c], RZ, P2, !PT         ; // x       512  ALU
/*02276830*/  FFMA R10, R34, R43, R10                                    ; // x       512  FMA
/*02276840*/  FFMA R7, R33, R42, R7                                      ; // x       512  FMA
/*02276850*/  @P6   LDG.E.LTC128B.CONSTANT R19, [R28]                    ; // x       512  global load (DRAM/L2: weights or activations)
/*02276860*/  FFMA R6, R33, R43, R6                                      ; // x       512  FMA
/*02276870*/  FFMA R3, R32, R42, R3                                      ; // x       512  FMA
/*02276880*/  FFMA R2, R32, R43, R2                                      ; // x       512  FMA
/*02276890*/  FFMA R1, R36, R44, R1                                      ; // x       512  FMA
/*022768a0*/  LDS.128 R32, [R22+0x840]                                   ; // x       512  shared-memory load
/*022768b0*/  FFMA R0, R36, R45, R0                                      ; // x       512  FMA
/*022768c0*/  FFMA R5, R37, R44, R5                                      ; // x       512  FMA
/*022768d0*/  @P1   IADD3 R28, P2, R28, c[0x0][0x188], RZ                ; // x       512  ALU
/*022768e0*/  FFMA R4, R37, R45, R4                                      ; // x       512  FMA
/*022768f0*/  FFMA R9, R38, R44, R9                                      ; // x       512  FMA
/*02276900*/  LDS.128 R40, [R21+0x240]                                   ; // x       512  shared-memory load
/*02276910*/  FFMA R8, R38, R45, R8                                      ; // x       512  FMA
/*02276920*/  FFMA R13, R39, R44, R13                                    ; // x       512  FMA
/*02276930*/  @P1   IADD3.X R29, R29, c[0x0][0x18c], RZ, P2, !PT         ; // x       512  ALU
/*02276940*/  FFMA R12, R39, R45, R12                                    ; // x       512  FMA
/*02276950*/  FFMA R15, R39, R46, R15                                    ; // x       512  FMA
/*02276960*/  @P1   R2P PR, R31.B1, 0x78                                 ; // x       512  ALU
/*02276970*/  FFMA R14, R39, R47, R14                                    ; // x       512  FMA
/*02276980*/  FFMA R11, R38, R46, R11                                    ; // x       512  FMA
/*02276990*/  @P3   LDG.E.LTC128B.CONSTANT R20, [R24]                    ; // x       512  global load (DRAM/L2: weights or activations)
/*022769a0*/  FFMA R10, R38, R47, R10                                    ; // x       512  FMA
/*022769b0*/  FFMA R7, R37, R46, R7                                      ; // x       512  FMA
/*022769c0*/  FFMA R6, R37, R47, R6                                      ; // x       512  FMA
/*022769d0*/  FFMA R3, R36, R46, R3                                      ; // x       512  FMA
/*022769e0*/  @P1   IADD3 R24, P2, R24, c[0x0][0x190], RZ                ; // x       512  ALU
/*022769f0*/  FFMA R2, R36, R47, R2                                      ; // x       512  FMA
/*02276a00*/  FFMA R1, R32, R40, R1                                      ; // x       512  FMA
/*02276a10*/  LDS.128 R36, [R22+0xa50]                                   ; // x       512  shared-memory load
/*02276a20*/  FFMA R0, R32, R41, R0                                      ; // x       512  FMA
/*02276a30*/  FFMA R5, R33, R40, R5                                      ; // x       512  FMA
/*02276a40*/  @P1   IADD3.X R25, R25, c[0x0][0x194], RZ, P2, !PT         ; // x       512  ALU
/*02276a50*/  FFMA R4, R33, R41, R4                                      ; // x       512  FMA
/*02276a60*/  FFMA R9, R34, R40, R9                                      ; // x       512  FMA
/*02276a70*/  LDS.128 R44, [R21+0x2d0]                                   ; // x       512  shared-memory load
/*02276a80*/  FFMA R8, R34, R41, R8                                      ; // x       512  FMA
/*02276a90*/  FFMA R13, R35, R40, R13                                    ; // x       512  FMA
/*02276aa0*/  FFMA R12, R35, R41, R12                                    ; // x       512  FMA
/*02276ab0*/  FFMA R15, R35, R42, R15                                    ; // x       512  FMA
/*02276ac0*/  FFMA R14, R35, R43, R14                                    ; // x       512  FMA
/*02276ad0*/  FFMA R11, R34, R42, R11                                    ; // x       512  FMA
/*02276ae0*/  FFMA R10, R34, R43, R10                                    ; // x       512  FMA
/*02276af0*/  FFMA R7, R33, R42, R7                                      ; // x       512  FMA
/*02276b00*/  FFMA R6, R33, R43, R6                                      ; // x       512  FMA
/*02276b10*/  FFMA R3, R32, R42, R3                                      ; // x       512  FMA
/*02276b20*/  FFMA R2, R32, R43, R2                                      ; // x       512  FMA
/*02276b30*/  FFMA R1, R36, R44, R1                                      ; // x       512  FMA
/*02276b40*/  LDS.128 R32, [R22+0xc60]                                   ; // x       512  shared-memory load
/*02276b50*/  FFMA R0, R36, R45, R0                                      ; // x       512  FMA
/*02276b60*/  FFMA R5, R37, R44, R5                                      ; // x       512  FMA
/*02276b70*/  LDS.128 R40, [R21+0x360]                                   ; // x       512  shared-memory load
/*02276b80*/  FFMA R4, R37, R45, R4                                      ; // x       512  FMA
/*02276b90*/  FFMA R9, R38, R44, R9                                      ; // x       512  FMA
/*02276ba0*/  FFMA R8, R38, R45, R8                                      ; // x       512  FMA
/*02276bb0*/  FFMA R13, R39, R44, R13                                    ; // x       512  FMA
/*02276bc0*/  FFMA R12, R39, R45, R12                                    ; // x       512  FMA
/*02276bd0*/  FFMA R15, R39, R46, R15                                    ; // x       512  FMA
/*02276be0*/  @P1   STS [R26], R16                                       ; // x       512  shared-memory store
/*02276bf0*/  FFMA R14, R39, R47, R14                                    ; // x       512  FMA
/*02276c00*/  FFMA R11, R38, R46, R11                                    ; // x       512  FMA
/*02276c10*/  @P1   STS [R26+0x80], R17                                  ; // x       512  shared-memory store
/*02276c20*/  FFMA R10, R38, R47, R10                                    ; // x       512  FMA
/*02276c30*/  FFMA R7, R37, R46, R7                                      ; // x       512  FMA
/*02276c40*/  @P1   STS [R26+0x100], R18                                 ; // x       512  shared-memory store
/*02276c50*/  FFMA R6, R37, R47, R6                                      ; // x       512  FMA
/*02276c60*/  FFMA R3, R36, R46, R3                                      ; // x       512  FMA
/*02276c70*/  @P1   STS [R26+0x180], R19                                 ; // x       512  shared-memory store
/*02276c80*/  FFMA R2, R36, R47, R2                                      ; // x       512  FMA
/*02276c90*/  FFMA R1, R32, R40, R1                                      ; // x       512  FMA
/*02276ca0*/  LDS.128 R36, [R22+0xe70]                                   ; // x       512  shared-memory load
/*02276cb0*/  FFMA R0, R32, R41, R0                                      ; // x       512  FMA
/*02276cc0*/  FFMA R5, R33, R40, R5                                      ; // x       512  FMA
/*02276cd0*/  LDS.128 R44, [R21+0x3f0]                                   ; // x       512  shared-memory load
/*02276ce0*/  FFMA R4, R33, R41, R4                                      ; // x       512  FMA
/*02276cf0*/  FFMA R9, R34, R40, R9                                      ; // x       512  FMA
/*02276d00*/  @P1   STS [R23], R20                                       ; // x       512  shared-memory store
/*02276d10*/  FFMA R8, R34, R41, R8                                      ; // x       512  FMA
/*02276d20*/  FFMA R13, R35, R40, R13                                    ; // x       512  FMA
/*02276d30*/  @P1   BAR.SYNC.DEFER_BLOCKING 0x0                          ; // x       512  block barrier
/*02276d40*/  FFMA R12, R35, R41, R12                                    ; // x       512  FMA
/*02276d50*/  FFMA R15, R35, R42, R15                                    ; // x       512  FMA
/*02276d60*/  FFMA R14, R35, R43, R14                                    ; // x       512  FMA
/*02276d70*/  FFMA R11, R34, R42, R11                                    ; // x       512  FMA
/*02276d80*/  FFMA R10, R34, R43, R10                                    ; // x       512  FMA
/*02276d90*/  FFMA R7, R33, R42, R7                                      ; // x       512  FMA
/*02276da0*/  FFMA R6, R33, R43, R6                                      ; // x       512  FMA
/*02276db0*/  FFMA R3, R32, R42, R3                                      ; // x       512  FMA
/*02276dc0*/  FFMA R2, R32, R43, R2                                      ; // x       512  FMA
/*02276dd0*/  FFMA R1, R36, R44, R1                                      ; // x       512  FMA
/*02276de0*/  LOP3.LUT R22, R22, 0x2000, RZ, 0x3c, !PT                   ; // x       512  ALU
/*02276df0*/  FFMA R0, R36, R45, R0                                      ; // x       512  FMA
/*02276e00*/  FFMA R5, R37, R44, R5                                      ; // x       512  FMA
/*02276e10*/  LOP3.LUT R21, R21, 0x2000, RZ, 0x3c, !PT                   ; // x       512  ALU
/*02276e20*/  FFMA R4, R37, R45, R4                                      ; // x       512  FMA
/*02276e30*/  FFMA R9, R38, R44, R9                                      ; // x       512  FMA
/*02276e40*/  @P1   LOP3.LUT R26, R26, 0x2000, RZ, 0x3c, !PT             ; // x       512  ALU
/*02276e50*/  FFMA R8, R38, R45, R8                                      ; // x       512  FMA
/*02276e60*/  FFMA R13, R39, R44, R13                                    ; // x       512  FMA
/*02276e70*/  @P1   LDS.128 R32, [R22]                                   ; // x       512  shared-memory load
/*02276e80*/  FFMA R12, R39, R45, R12                                    ; // x       512  FMA
/*02276e90*/  FFMA R15, R39, R46, R15                                    ; // x       512  FMA
/*02276ea0*/  @P1   LDS.128 R40, [R21]                                   ; // x       512  shared-memory load
/*02276eb0*/  FFMA R14, R39, R47, R14                                    ; // x       512  FMA
/*02276ec0*/  FFMA R11, R38, R46, R11                                    ; // x       512  FMA
/*02276ed0*/  @P1   LOP3.LUT R23, R23, 0x2000, RZ, 0x3c, !PT             ; // x       512  ALU
/*02276ee0*/  FFMA R10, R38, R47, R10                                    ; // x       512  FMA
/*02276ef0*/  FFMA R7, R37, R46, R7                                      ; // x       512  FMA
/*02276f00*/  FFMA R6, R37, R47, R6                                      ; // x       512  FMA
/*02276f10*/  FFMA R3, R36, R46, R3                                      ; // x       512  FMA
/*02276f20*/  FFMA R2, R36, R47, R2                                      ; // x       512  FMA
/*02276f30*/  @P1   BRA.U 0x202276450                                    ; // x       512  branch
/*02276f40*/  LOP3.LUT P0, RZ, R30, 0x7, RZ, 0xc0, !PT                   ; // x       128  ALU
/*02276f50*/  @!P0  BRA.U 0x202277120                                    ; // x       128  branch
/*02276f60*/  NOP                                                        ; // x         0  Branch
/*02276f70*/  S2R R36, SR_TID.X                                          ; // x         0  Branch
/*02276f80*/  LOP3.LUT R38, R36, 0x7, RZ, 0xc0, !PT                      ; // x         0  ALU
/*02276f90*/  ISETP.GT.AND P1, PT, R30, R38, PT                          ; // x         0  ALU
/*02276fa0*/  @!P1  R2P PR, RZ, 0x78                                     ; // x         0  ALU
/*02276fb0*/  @P1   R2P PR, R31, 0x78                                    ; // x         0  ALU
/*02276fc0*/  @P3   LDG.E.LTC128B.CONSTANT R16, [R28]                    ; // x         0  global load (DRAM/L2: weights or activations)
/*02276fd0*/  @!P3  MOV R16, RZ                                          ; // x         0  ALU
/*02276fe0*/  @P1   IADD3 R28, P2, R28, c[0x0][0x178], RZ                ; // x         0  ALU
/*02276ff0*/  @P1   IADD3.X R29, R29, c[0x0][0x17c], RZ, P2, !PT         ; // x         0  ALU
/*02277000*/  @P4   LDG.E.LTC128B.CONSTANT R17, [R28]                    ; // x         0  global load (DRAM/L2: weights or activations)
/*02277010*/  @!P4  MOV R17, RZ                                          ; // x         0  ALU
/*02277020*/  @P1   IADD3 R28, P2, R28, c[0x0][0x178], RZ                ; // x         0  ALU
```

(first 200 of 215 instructions shown; the rest are in the full dump)
