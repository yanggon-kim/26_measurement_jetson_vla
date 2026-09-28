# Hot loop — `vectorized_elementwise_kernel__e8e91f66` (sm87)

PC range 0x2dd51600-0x2dd54170; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/vectorized_elementwise_kernel__e8e91f66_sm87.sass`; per-PC counts: `raw/vectorized_elementwise_kernel__e8e91f66_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*2dd51600*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x       200  ALU
/*2dd51610*/  @P0   BRA 0x42dd51740                                      ; // x       200  branch
/*2dd51620*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*2dd51630*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*2dd51640*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*2dd51650*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*2dd51660*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*2dd51670*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*2dd51680*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*2dd51690*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*2dd516a0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*2dd516b0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*2dd516c0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*2dd516d0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*2dd516e0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*2dd516f0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*2dd51700*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*2dd51710*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*2dd51720*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*2dd51730*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*2dd51740*/  MOV R1, c[0x0][0x28]                                       ; // x       200  ALU
/*2dd51750*/  S2R R2, SR_CTAID.X                                         ; // x       200  Branch
/*2dd51760*/  ULDC.64 UR4, c[0x0][0x118]                                 ; // x       200  Uniform
/*2dd51770*/  SHF.L.U32 R2, R2, 0xa, RZ                                  ; // x       200  ALU
/*2dd51780*/  IADD3 R3, -R2, c[0x0][0x160], RZ                           ; // x       200  ALU
/*2dd51790*/  ISETP.GE.U32.AND P0, PT, R3, 0x400, PT                     ; // x       200  ALU
/*2dd517a0*/  @!P0  BRA 0x42dd52550                                      ; // x       200  branch
/*2dd517b0*/  S2R R15, SR_TID.X                                          ; // x       200  Branch
/*2dd517c0*/  HFMA2.MMA R17, -RZ, RZ, 0, 2.384185791015625e-07           ; // x       200  FMA
/*2dd517d0*/  IMAD.WIDE R4, R2, R17, c[0x0][0x170]                       ; // x       200  FMA
/*2dd517e0*/  IMAD.WIDE.U32 R4, R15, 0x10, R4                            ; // x       200  FMA
/*2dd517f0*/  LDG.E.128 R8, [R4.64]                                      ; // x       200  global load (DRAM/L2: weights or activations)
/*2dd51800*/  LDG.E.128 R4, [R4.64+0x800]                                ; // x       200  global load (DRAM/L2: weights or activations)
/*2dd51810*/  HFMA2.MMA R18, -RZ, RZ, 3.7421875, 0                       ; // x       200  FMA
/*2dd51820*/  MOV R24, 0x3bbb989d                                        ; // x       200  ALU
/*2dd51830*/  BSSY B2, 0x42dd51af0                                       ; // x       200  Branch
/*2dd51840*/  FFMA.SAT R0, -R11, R24, 0.5                                ; // x       200  FMA
/*2dd51850*/  FFMA.SAT R25, -R8, R24, 0.5                                ; // x       200  FMA
/*2dd51860*/  FFMA.RM R3, R0, R18, 12582913                              ; // x       200  FMA
/*2dd51870*/  FFMA.RM R25, R25, R18, 12582913                            ; // x       200  FMA
/*2dd51880*/  FADD R0, R3, -12583039                                     ; // x       200  FMA
/*2dd51890*/  SHF.L.U32 R12, R3, 0x17, RZ                                ; // x       200  ALU
/*2dd518a0*/  FFMA.SAT R21, -R4, R24, 0.5                                ; // x       200  FMA
/*2dd518b0*/  FFMA.SAT R22, -R5, R24, 0.5                                ; // x       200  FMA
/*2dd518c0*/  FFMA R0, -R11, 1.4426950216293334961, -R0                  ; // x       200  FMA
/*2dd518d0*/  FFMA.SAT R23, -R6, R24, 0.5                                ; // x       200  FMA
/*2dd518e0*/  FFMA.RM R21, R21, R18, 12582913                            ; // x       200  FMA
/*2dd518f0*/  FFMA.RM R22, R22, R18, 12582913                            ; // x       200  FMA
/*2dd51900*/  FFMA R0, -R11, 1.925963033500011079e-08, R0                ; // x       200  FMA
/*2dd51910*/  FFMA.RM R23, R23, R18, 12582913                            ; // x       200  FMA
/*2dd51920*/  MUFU.EX2 R13, R0                                           ; // x       200  special-function unit
/*2dd51930*/  FFMA.SAT R0, -R10, R24, 0.5                                ; // x       200  FMA
/*2dd51940*/  FFMA.RM R0, R0, R18, 12582913                              ; // x       200  FMA
/*2dd51950*/  FFMA R16, R12, R13, 1                                      ; // x       200  FMA
/*2dd51960*/  FADD R3, R0, -12583039                                     ; // x       200  FMA
/*2dd51970*/  SHF.L.U32 R13, R0, 0x17, RZ                                ; // x       200  ALU
/*2dd51980*/  MUFU.RCP R14, R16                                          ; // x       200  special-function unit
/*2dd51990*/  FFMA R3, -R10, 1.4426950216293334961, -R3                  ; // x       200  FMA
/*2dd519a0*/  FFMA R3, -R10, 1.925963033500011079e-08, R3                ; // x       200  FMA
/*2dd519b0*/  FCHK P0, R11, R16                                          ; // x       200  FMA
/*2dd519c0*/  MUFU.EX2 R12, R3                                           ; // x       200  special-function unit
/*2dd519d0*/  FFMA R0, -R16, R14, 1                                      ; // x       200  FMA
/*2dd519e0*/  FFMA R0, R14, R0, R14                                      ; // x       200  FMA
/*2dd519f0*/  FFMA.SAT R14, -R9, R24, 0.5                                ; // x       200  FMA
/*2dd51a00*/  FFMA.SAT R24, -R7, R24, 0.5                                ; // x       200  FMA
/*2dd51a10*/  IMAD.WIDE.U32 R2, R2, R17, c[0x0][0x168]                   ; // x       200  FMA
/*2dd51a20*/  FFMA.RM R14, R14, R18, 12582913                            ; // x       200  FMA
/*2dd51a30*/  FFMA.RM R24, R24, R18, 12582913                            ; // x       200  FMA
/*2dd51a40*/  IMAD.WIDE R2, R15, 0x10, R2                                ; // x       200  FMA
/*2dd51a50*/  FFMA R15, R11, R0, RZ                                      ; // x       200  FMA
/*2dd51a60*/  FFMA R13, R13, R12, 1                                      ; // x       200  FMA
/*2dd51a70*/  FFMA R12, -R16, R15, R11                                   ; // x       200  FMA
/*2dd51a80*/  FFMA R15, R0, R12, R15                                     ; // x       200  FMA
/*2dd51a90*/  @!P0  BRA 0x42dd51ae0                                      ; // x       200  branch
/*2dd51aa0*/  IMAD.MOV.U32 R0, RZ, RZ, R16                               ; // x         0  FMA
/*2dd51ab0*/  MOV R16, 0x4d0                                             ; // x         0  ALU
/*2dd51ac0*/  CALL.REL.NOINC 0x42dd53a50                                 ; // x         0  Branch
/*2dd51ad0*/  MOV R15, R12                                               ; // x         0  ALU
/*2dd51ae0*/  BSYNC B2                                                   ; // x       200  Branch
/*2dd51af0*/  MUFU.RCP R0, R13                                           ; // x       200  special-function unit
/*2dd51b00*/  FADD R14, R14, -12583039                                   ; // x       200  FMA
/*2dd51b10*/  HFMA2.MMA R12, -RZ, RZ, 0.96630859375, -0.0022525787353515625 ; // x       200  FMA
/*2dd51b20*/  MOV R16, 0x437c0000                                        ; // x       200  ALU
/*2dd51b30*/  BSSY B2, 0x42dd51c80                                       ; // x       200  Branch
/*2dd51b40*/  FFMA R14, -R9, 1.4426950216293334961, -R14                 ; // x       200  FMA
/*2dd51b50*/  FCHK P0, R10, R13                                          ; // x       200  FMA
/*2dd51b60*/  FFMA R14, -R9, 1.925963033500011079e-08, R14               ; // x       200  FMA
/*2dd51b70*/  FFMA.SAT R12, -R9, R12, 0.5                                ; // x       200  FMA
/*2dd51b80*/  MUFU.EX2 R26, R14                                          ; // x       200  special-function unit
/*2dd51b90*/  FFMA R11, -R13, R0, 1                                      ; // x       200  FMA
/*2dd51ba0*/  FFMA.RM R12, R12, R16, 12582913                            ; // x       200  FMA
/*2dd51bb0*/  FFMA R11, R0, R11, R0                                      ; // x       200  FMA
/*2dd51bc0*/  SHF.L.U32 R12, R12, 0x17, RZ                               ; // x       200  ALU
/*2dd51bd0*/  FFMA R14, R10, R11, RZ                                     ; // x       200  FMA
/*2dd51be0*/  FFMA R0, -R13, R14, R10                                    ; // x       200  FMA
/*2dd51bf0*/  FFMA R26, R26, R12, 1                                      ; // x       200  FMA
/*2dd51c00*/  FFMA R14, R11, R0, R14                                     ; // x       200  FMA
/*2dd51c10*/  @!P0  BRA 0x42dd51c70                                      ; // x       200  branch
/*2dd51c20*/  MOV R11, R10                                               ; // x         0  ALU
/*2dd51c30*/  MOV R0, R13                                                ; // x         0  ALU
/*2dd51c40*/  MOV R16, 0x660                                             ; // x         0  ALU
/*2dd51c50*/  CALL.REL.NOINC 0x42dd53a50                                 ; // x         0  Branch
/*2dd51c60*/  MOV R14, R12                                               ; // x         0  ALU
/*2dd51c70*/  BSYNC B2                                                   ; // x       200  Branch
/*2dd51c80*/  MUFU.RCP R0, R26                                           ; // x       200  special-function unit
/*2dd51c90*/  FADD R25, R25, -12583039                                   ; // x       200  FMA
/*2dd51ca0*/  HFMA2.MMA R12, -RZ, RZ, 3.7421875, 0                       ; // x       200  FMA
/*2dd51cb0*/  IMAD.MOV.U32 R11, RZ, RZ, 0x3bbb989d                       ; // x       200  FMA
/*2dd51cc0*/  BSSY B2, 0x42dd51e10                                       ; // x       200  Branch
/*2dd51cd0*/  FFMA R25, -R8, 1.4426950216293334961, -R25                 ; // x       200  FMA
/*2dd51ce0*/  FFMA.SAT R11, -R8, R11, 0.5                                ; // x       200  FMA
/*2dd51cf0*/  FCHK P0, R9, R26                                           ; // x       200  FMA
/*2dd51d00*/  FFMA R25, -R8, 1.925963033500011079e-08, R25               ; // x       200  FMA
/*2dd51d10*/  MUFU.EX2 R25, R25                                          ; // x       200  special-function unit
/*2dd51d20*/  FFMA R10, R0, -R26, 1                                      ; // x       200  FMA
/*2dd51d30*/  FFMA R0, R0, R10, R0                                       ; // x       200  FMA
/*2dd51d40*/  FFMA.RM R10, R11, R12, 12582913                            ; // x       200  FMA
/*2dd51d50*/  FFMA R13, R9, R0, RZ                                       ; // x       200  FMA
/*2dd51d60*/  SHF.L.U32 R10, R10, 0x17, RZ                               ; // x       200  ALU
/*2dd51d70*/  FFMA R11, R13, -R26, R9                                    ; // x       200  FMA
/*2dd51d80*/  FFMA R10, R25, R10, 1                                      ; // x       200  FMA
/*2dd51d90*/  FFMA R13, R0, R11, R13                                     ; // x       200  FMA
/*2dd51da0*/  @!P0  BRA 0x42dd51e00                                      ; // x       200  branch
/*2dd51db0*/  MOV R11, R9                                                ; // x         0  ALU
/*2dd51dc0*/  MOV R0, R26                                                ; // x         0  ALU
/*2dd51dd0*/  MOV R16, 0x7f0                                             ; // x         0  ALU
/*2dd51de0*/  CALL.REL.NOINC 0x42dd53a50                                 ; // x         0  Branch
/*2dd51df0*/  MOV R13, R12                                               ; // x         0  ALU
/*2dd51e00*/  BSYNC B2                                                   ; // x       200  Branch
/*2dd51e10*/  MUFU.RCP R0, R10                                           ; // x       200  special-function unit
/*2dd51e20*/  FADD R24, R24, -12583039                                   ; // x       200  FMA
/*2dd51e30*/  HFMA2.MMA R11, -RZ, RZ, 0.96630859375, -0.0022525787353515625 ; // x       200  FMA
/*2dd51e40*/  MOV R12, 0x437c0000                                        ; // x       200  ALU
/*2dd51e50*/  BSSY B2, 0x42dd51f90                                       ; // x       200  Branch
/*2dd51e60*/  FFMA R24, -R7, 1.4426950216293334961, -R24                 ; // x       200  FMA
/*2dd51e70*/  FCHK P0, R8, R10                                           ; // x       200  FMA
/*2dd51e80*/  FFMA R24, -R7, 1.925963033500011079e-08, R24               ; // x       200  FMA
/*2dd51e90*/  FFMA.SAT R11, -R7, R11, 0.5                                ; // x       200  FMA
/*2dd51ea0*/  MUFU.EX2 R24, R24                                          ; // x       200  special-function unit
/*2dd51eb0*/  FFMA R9, R0, -R10, 1                                       ; // x       200  FMA
/*2dd51ec0*/  FFMA R0, R0, R9, R0                                        ; // x       200  FMA
/*2dd51ed0*/  FFMA.RM R9, R11, R12, 12582913                             ; // x       200  FMA
/*2dd51ee0*/  FFMA R12, R8, R0, RZ                                       ; // x       200  FMA
/*2dd51ef0*/  IMAD.SHL.U32 R9, R9, 0x800000, RZ                          ; // x       200  FMA
/*2dd51f00*/  FFMA R11, R12, -R10, R8                                    ; // x       200  FMA
/*2dd51f10*/  FFMA R9, R24, R9, 1                                        ; // x       200  FMA
/*2dd51f20*/  FFMA R12, R0, R11, R12                                     ; // x       200  FMA
/*2dd51f30*/  @!P0  BRA 0x42dd51f80                                      ; // x       200  branch
/*2dd51f40*/  MOV R11, R8                                                ; // x         0  ALU
/*2dd51f50*/  MOV R0, R10                                                ; // x         0  ALU
/*2dd51f60*/  MOV R16, 0x980                                             ; // x         0  ALU
/*2dd51f70*/  CALL.REL.NOINC 0x42dd53a50                                 ; // x         0  Branch
/*2dd51f80*/  BSYNC B2                                                   ; // x       200  Branch
/*2dd51f90*/  MUFU.RCP R0, R9                                            ; // x       200  special-function unit
/*2dd51fa0*/  FADD R23, R23, -12583039                                   ; // x       200  FMA
/*2dd51fb0*/  HFMA2.MMA R10, -RZ, RZ, 0.96630859375, -0.0022525787353515625 ; // x       200  FMA
/*2dd51fc0*/  MOV R11, 0x437c0000                                        ; // x       200  ALU
/*2dd51fd0*/  STG.E.128 [R2.64], R12                                     ; // x       200  global store
/*2dd51fe0*/  FFMA R23, -R6, 1.4426950216293334961, -R23                 ; // x       200  FMA
/*2dd51ff0*/  BSSY B2, 0x42dd52130                                       ; // x       200  Branch
/*2dd52000*/  FCHK P0, R7, R9                                            ; // x       200  FMA
/*2dd52010*/  FFMA R23, -R6, 1.925963033500011079e-08, R23               ; // x       200  FMA
/*2dd52020*/  FFMA.SAT R10, -R6, R10, 0.5                                ; // x       200  FMA
/*2dd52030*/  MUFU.EX2 R23, R23                                          ; // x       200  special-function unit
/*2dd52040*/  FFMA R8, R0, -R9, 1                                        ; // x       200  FMA
/*2dd52050*/  FFMA R0, R0, R8, R0                                        ; // x       200  FMA
/*2dd52060*/  FFMA.RM R8, R10, R11, 12582913                             ; // x       200  FMA
/*2dd52070*/  FFMA R15, R7, R0, RZ                                       ; // x       200  FMA
/*2dd52080*/  SHF.L.U32 R8, R8, 0x17, RZ                                 ; // x       200  ALU
/*2dd52090*/  FFMA R10, R15, -R9, R7                                     ; // x       200  FMA
/*2dd520a0*/  FFMA R8, R23, R8, 1                                        ; // x       200  FMA
/*2dd520b0*/  FFMA R15, R0, R10, R15                                     ; // x       200  FMA
/*2dd520c0*/  @!P0  BRA 0x42dd52120                                      ; // x       200  branch
/*2dd520d0*/  MOV R11, R7                                                ; // x         0  ALU
/*2dd520e0*/  MOV R0, R9                                                 ; // x         0  ALU
/*2dd520f0*/  MOV R16, 0xb10                                             ; // x         0  ALU
/*2dd52100*/  CALL.REL.NOINC 0x42dd53a50                                 ; // x         0  Branch
/*2dd52110*/  IMAD.MOV.U32 R15, RZ, RZ, R12                              ; // x         0  FMA
/*2dd52120*/  BSYNC B2                                                   ; // x       200  Branch
/*2dd52130*/  MUFU.RCP R0, R8                                            ; // x       200  special-function unit
/*2dd52140*/  FADD R22, R22, -12583039                                   ; // x       200  FMA
/*2dd52150*/  HFMA2.MMA R9, -RZ, RZ, 0.96630859375, -0.0022525787353515625 ; // x       200  FMA
/*2dd52160*/  MOV R10, 0x437c0000                                        ; // x       200  ALU
/*2dd52170*/  BSSY B2, 0x42dd522c0                                       ; // x       200  Branch
/*2dd52180*/  FFMA R22, -R5, 1.4426950216293334961, -R22                 ; // x       200  FMA
/*2dd52190*/  FCHK P0, R6, R8                                            ; // x       200  FMA
/*2dd521a0*/  FFMA R22, -R5, 1.925963033500011079e-08, R22               ; // x       200  FMA
/*2dd521b0*/  FFMA.SAT R9, -R5, R9, 0.5                                  ; // x       200  FMA
/*2dd521c0*/  MUFU.EX2 R22, R22                                          ; // x       200  special-function unit
/*2dd521d0*/  FFMA R7, R0, -R8, 1                                        ; // x       200  FMA
/*2dd521e0*/  FFMA R0, R0, R7, R0                                        ; // x       200  FMA
/*2dd521f0*/  FFMA.RM R7, R9, R10, 12582913                              ; // x       200  FMA
/*2dd52200*/  FFMA R14, R6, R0, RZ                                       ; // x       200  FMA
/*2dd52210*/  SHF.L.U32 R7, R7, 0x17, RZ                                 ; // x       200  ALU
/*2dd52220*/  FFMA R9, R14, -R8, R6                                      ; // x       200  FMA
/*2dd52230*/  FFMA R7, R22, R7, 1                                        ; // x       200  FMA
/*2dd52240*/  FFMA R14, R0, R9, R14                                      ; // x       200  FMA
/*2dd52250*/  @!P0  BRA 0x42dd522b0                                      ; // x       200  branch
/*2dd52260*/  MOV R11, R6                                                ; // x         0  ALU
/*2dd52270*/  MOV R0, R8                                                 ; // x         0  ALU
```

(first 200 of 696 instructions shown; the rest are in the full dump)
