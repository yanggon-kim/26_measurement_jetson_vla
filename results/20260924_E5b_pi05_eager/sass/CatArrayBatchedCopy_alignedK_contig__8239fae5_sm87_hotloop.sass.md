# Hot loop — `CatArrayBatchedCopy_alignedK_contig__8239fae5` (sm87)

PC range 0x22e7710-0x22e8380; backward BRA 0x22e8380 -> 0x22e7710: of the 3 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/CatArrayBatchedCopy_alignedK_contig__8239fae5_sm87.sass`; per-PC counts: `raw/CatArrayBatchedCopy_alignedK_contig__8239fae5_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*022e7710*/  IMAD.WIDE.U32 R4, R11, 0x4, R2                             ; // x      2036  FMA
/*022e7720*/  LDG.E.128 R4, [R4.64]                                      ; // x      2036  global load (DRAM/L2: weights or activations)
/*022e7730*/  I2F.U32.RP R19, R16                                        ; // x      2036  ALU
/*022e7740*/  IMAD.MOV R17, RZ, RZ, -R16                                 ; // x      2036  FMA
/*022e7750*/  IADD3 R21, R11, 0x3, RZ                                    ; // x      2036  ALU
/*022e7760*/  LOP3.LUT R31, RZ, R16, RZ, 0x33, !PT                       ; // x      2036  ALU
/*022e7770*/  I2F.U32.RP R25, R15                                        ; // x      2036  ALU
/*022e7780*/  MUFU.RCP R19, R19                                          ; // x      2036  special-function unit
/*022e7790*/  MUFU.RCP R25, R25                                          ; // x      2036  special-function unit
/*022e77a0*/  IADD3 R8, R19, 0xffffffe, RZ                               ; // x      2036  ALU
/*022e77b0*/  I2F.U32.RP R30, R14                                        ; // x      2036  ALU
/*022e77c0*/  IADD3 R19, R11, 0x2, RZ                                    ; // x      2036  ALU
/*022e77d0*/  F2I.FTZ.U32.TRUNC.NTZ R9, R8                               ; // x      2036  ALU
/*022e77e0*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x      2036  FMA
/*022e77f0*/  IMAD R17, R17, R9, RZ                                      ; // x      2036  FMA
/*022e7800*/  IMAD.HI.U32 R26, R9, R17, R8                               ; // x      2036  FMA
/*022e7810*/  IADD3 R17, R11, 0x1, RZ                                    ; // x      2036  ALU
/*022e7820*/  IADD3 R8, R25, 0xffffffe, RZ                               ; // x      2036  ALU
/*022e7830*/  IMAD.HI.U32 R20, R26, R17, RZ                              ; // x      2036  FMA
/*022e7840*/  F2I.FTZ.U32.TRUNC.NTZ R9, R8                               ; // x      2036  ALU
/*022e7850*/  IMAD.HI.U32 R24, R26, R11, RZ                              ; // x      2036  FMA
/*022e7860*/  IADD3 R27, -R20, RZ, RZ                                    ; // x      2036  ALU
/*022e7870*/  IMAD.HI.U32 R22, R26, R19, RZ                              ; // x      2036  FMA
/*022e7880*/  HFMA2.MMA R8, -RZ, RZ, 0, 0                                ; // x      2036  FMA
/*022e7890*/  IMAD.MOV R23, RZ, RZ, -R24                                 ; // x      2036  FMA
/*022e78a0*/  IMAD.MOV R29, RZ, RZ, -R22                                 ; // x      2036  FMA
/*022e78b0*/  IMAD R27, R16, R27, R17                                    ; // x      2036  FMA
/*022e78c0*/  IMAD R23, R16, R23, R11                                    ; // x      2036  FMA
/*022e78d0*/  IMAD R29, R16, R29, R19                                    ; // x      2036  FMA
/*022e78e0*/  ISETP.GE.U32.AND P1, PT, R27, R16, PT                      ; // x      2036  ALU
/*022e78f0*/  IMAD.HI.U32 R26, R26, R21, RZ                              ; // x      2036  FMA
/*022e7900*/  ISETP.GE.U32.AND P2, PT, R23, R16, PT                      ; // x      2036  ALU
/*022e7910*/  ISETP.GE.U32.AND P0, PT, R29, R16, PT                      ; // x      2036  ALU
/*022e7920*/  IMAD.MOV R25, RZ, RZ, -R26                                 ; // x      2036  FMA
/*022e7930*/  IMAD R25, R16, R25, R21                                    ; // x      2036  FMA
/*022e7940*/  @P1   IADD3 R27, -R16, R27, RZ                             ; // x      2036  ALU
/*022e7950*/  @P2   IMAD.IADD R23, R23, 0x1, -R16                        ; // x      2036  FMA
/*022e7960*/  ISETP.GE.U32.AND P6, PT, R25, R16, PT                      ; // x      2036  ALU
/*022e7970*/  @P0   IADD3 R29, -R16, R29, RZ                             ; // x      2036  ALU
/*022e7980*/  ISETP.GE.U32.AND P3, PT, R27, R16, PT                      ; // x      2036  ALU
/*022e7990*/  IMAD.MOV R27, RZ, RZ, -R15                                 ; // x      2036  FMA
/*022e79a0*/  ISETP.GE.U32.AND P5, PT, R23, R16, PT                      ; // x      2036  ALU
/*022e79b0*/  ISETP.GE.U32.AND P4, PT, R29, R16, PT                      ; // x      2036  ALU
/*022e79c0*/  IMAD.MOV.U32 R29, RZ, RZ, R27                              ; // x      2036  FMA
/*022e79d0*/  @P1   IADD3 R20, R20, 0x1, RZ                              ; // x      2036  ALU
/*022e79e0*/  @P2   IADD3 R24, R24, 0x1, RZ                              ; // x      2036  ALU
/*022e79f0*/  IMAD R29, R29, R9, RZ                                      ; // x      2036  FMA
/*022e7a00*/  @P0   IADD3 R22, R22, 0x1, RZ                              ; // x      2036  ALU
/*022e7a10*/  @P6   IMAD.IADD R25, R25, 0x1, -R16                        ; // x      2036  FMA
/*022e7a20*/  ISETP.NE.U32.AND P1, PT, R16, RZ, PT                       ; // x      2036  ALU
/*022e7a30*/  IMAD.HI.U32 R29, R9, R29, R8                               ; // x      2036  FMA
/*022e7a40*/  @P3   IADD3 R20, R20, 0x1, RZ                              ; // x      2036  ALU
/*022e7a50*/  MUFU.RCP R9, R30                                           ; // x      2036  special-function unit
/*022e7a60*/  @P5   IADD3 R24, R24, 0x1, RZ                              ; // x      2036  ALU
/*022e7a70*/  @P4   IADD3 R22, R22, 0x1, RZ                              ; // x      2036  ALU
/*022e7a80*/  SEL R20, R31, R20, !P1                                     ; // x      2036  ALU
/*022e7a90*/  SEL R24, R31, R24, !P1                                     ; // x      2036  ALU
/*022e7aa0*/  SEL R22, R31, R22, !P1                                     ; // x      2036  ALU
/*022e7ab0*/  IMAD.HI.U32 R23, R29, R20, RZ                              ; // x      2036  FMA
/*022e7ac0*/  ISETP.GE.U32.AND P2, PT, R25, R16, PT                      ; // x      2036  ALU
/*022e7ad0*/  @P6   IADD3 R26, R26, 0x1, RZ                              ; // x      2036  ALU
/*022e7ae0*/  IMAD.HI.U32 R27, R29, R24, RZ                              ; // x      2036  FMA
/*022e7af0*/  IADD3 R9, R9, 0xffffffe, RZ                                ; // x      2036  ALU
/*022e7b00*/  IMAD.HI.U32 R25, R29, R22, RZ                              ; // x      2036  FMA
/*022e7b10*/  IMAD.MOV R28, RZ, RZ, -R23                                 ; // x      2036  FMA
/*022e7b20*/  IADD3 R8, -R25, RZ, RZ                                     ; // x      2036  ALU
/*022e7b30*/  IMAD.MOV R32, RZ, RZ, -R27                                 ; // x      2036  FMA
/*022e7b40*/  @P2   IADD3 R26, R26, 0x1, RZ                              ; // x      2036  ALU
/*022e7b50*/  IMAD R28, R15, R28, R20                                    ; // x      2036  FMA
/*022e7b60*/  F2I.FTZ.U32.TRUNC.NTZ R9, R9                               ; // x      2036  ALU
/*022e7b70*/  IMAD R30, R15, R32, R24                                    ; // x      2036  FMA
/*022e7b80*/  SEL R26, R31, R26, !P1                                     ; // x      2036  ALU
/*022e7b90*/  IMAD R8, R15, R8, R22                                      ; // x      2036  FMA
/*022e7ba0*/  ISETP.GE.U32.AND P0, PT, R28, R15, PT                      ; // x      2036  ALU
/*022e7bb0*/  ISETP.GE.U32.AND P3, PT, R30, R15, PT                      ; // x      2036  ALU
/*022e7bc0*/  IMAD.HI.U32 R29, R29, R26, RZ                              ; // x      2036  FMA
/*022e7bd0*/  ISETP.GE.U32.AND P2, PT, R8, R15, PT                       ; // x      2036  ALU
/*022e7be0*/  IADD3 R31, RZ, -R14, RZ                                    ; // x      2036  ALU
/*022e7bf0*/  LOP3.LUT R32, RZ, R15, RZ, 0x33, !PT                       ; // x      2036  ALU
/*022e7c00*/  IMAD R31, R31, R9, RZ                                      ; // x      2036  FMA
/*022e7c10*/  @P0   IADD3 R23, R23, 0x1, RZ                              ; // x      2036  ALU
/*022e7c20*/  @P0   IMAD.IADD R28, R28, 0x1, -R15                        ; // x      2036  FMA
/*022e7c30*/  @P3   IADD3 R30, -R15, R30, RZ                             ; // x      2036  ALU
/*022e7c40*/  @P2   IMAD.IADD R8, R8, 0x1, -R15                          ; // x      2036  FMA
/*022e7c50*/  @P3   IADD3 R27, R27, 0x1, RZ                              ; // x      2036  ALU
/*022e7c60*/  ISETP.GE.U32.AND P4, PT, R28, R15, PT                      ; // x      2036  ALU
/*022e7c70*/  IMAD.MOV R28, RZ, RZ, -R29                                 ; // x      2036  FMA
/*022e7c80*/  ISETP.GE.U32.AND P6, PT, R30, R15, PT                      ; // x      2036  ALU
/*022e7c90*/  ISETP.GE.U32.AND P5, PT, R8, R15, PT                       ; // x      2036  ALU
/*022e7ca0*/  IMAD R28, R15, R28, R26                                    ; // x      2036  FMA
/*022e7cb0*/  ISETP.NE.U32.AND P0, PT, R15, RZ, PT                       ; // x      2036  ALU
/*022e7cc0*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x      2036  FMA
/*022e7cd0*/  @P2   IADD3 R25, R25, 0x1, RZ                              ; // x      2036  ALU
/*022e7ce0*/  ISETP.GE.U32.AND P1, PT, R28, R15, PT                      ; // x      2036  ALU
/*022e7cf0*/  IMAD.HI.U32 R30, R9, R31, R8                               ; // x      2036  FMA
/*022e7d00*/  @P4   IADD3 R23, R23, 0x1, RZ                              ; // x      2036  ALU
/*022e7d10*/  @P6   IADD3 R27, R27, 0x1, RZ                              ; // x      2036  ALU
/*022e7d20*/  @P5   IADD3 R25, R25, 0x1, RZ                              ; // x      2036  ALU
/*022e7d30*/  SEL R27, R32, R27, !P0                                     ; // x      2036  ALU
/*022e7d40*/  SEL R9, R32, R23, !P0                                      ; // x      2036  ALU
/*022e7d50*/  SEL R23, R32, R25, !P0                                     ; // x      2036  ALU
/*022e7d60*/  @P1   IMAD.IADD R28, R28, 0x1, -R15                        ; // x      2036  FMA
/*022e7d70*/  @P1   IADD3 R29, R29, 0x1, RZ                              ; // x      2036  ALU
/*022e7d80*/  IMAD.HI.U32 R25, R30, R27, RZ                              ; // x      2036  FMA
/*022e7d90*/  ISETP.GE.U32.AND P4, PT, R28, R15, PT                      ; // x      2036  ALU
/*022e7da0*/  IMAD.MOV R31, RZ, RZ, -R25                                 ; // x      2036  FMA
/*022e7db0*/  IMAD.HI.U32 R8, R30, R9, RZ                                ; // x      2036  FMA
/*022e7dc0*/  IMAD R31, R14, R31, R27                                    ; // x      2036  FMA
/*022e7dd0*/  IADD3 R33, -R8, RZ, RZ                                     ; // x      2036  ALU
/*022e7de0*/  IMAD.HI.U32 R28, R30, R23, RZ                              ; // x      2036  FMA
/*022e7df0*/  ISETP.GE.U32.AND P3, PT, R31, R14, PT                      ; // x      2036  ALU
/*022e7e00*/  IMAD R33, R14, R33, R9                                     ; // x      2036  FMA
/*022e7e10*/  @P4   IADD3 R29, R29, 0x1, RZ                              ; // x      2036  ALU
/*022e7e20*/  IMAD.MOV R35, RZ, RZ, -R28                                 ; // x      2036  FMA
/*022e7e30*/  SEL R29, R32, R29, !P0                                     ; // x      2036  ALU
/*022e7e40*/  IMAD R35, R14, R35, R23                                    ; // x      2036  FMA
/*022e7e50*/  ISETP.GE.U32.AND P2, PT, R33, R14, PT                      ; // x      2036  ALU
/*022e7e60*/  IMAD.MOV R32, RZ, RZ, -R20                                 ; // x      2036  FMA
/*022e7e70*/  IMAD.HI.U32 R30, R30, R29, RZ                              ; // x      2036  FMA
/*022e7e80*/  ISETP.GE.U32.AND P1, PT, R35, R14, PT                      ; // x      2036  ALU
/*022e7e90*/  @P3   IADD3 R25, R25, 0x1, RZ                              ; // x      2036  ALU
/*022e7ea0*/  @P3   IMAD.IADD R31, R31, 0x1, -R14                        ; // x      2036  FMA
/*022e7eb0*/  IMAD R17, R16, R32, R17                                    ; // x      2036  FMA
/*022e7ec0*/  IMAD.MOV R32, RZ, RZ, -R22                                 ; // x      2036  FMA
/*022e7ed0*/  ISETP.GE.U32.AND P6, PT, R31, R14, PT                      ; // x      2036  ALU
/*022e7ee0*/  IMAD R17, R17, c[0x0][0xc24], RZ                           ; // x      2036  FMA
/*022e7ef0*/  IADD3 R31, -R30, RZ, RZ                                    ; // x      2036  ALU
/*022e7f00*/  IMAD R19, R16, R32, R19                                    ; // x      2036  FMA
/*022e7f10*/  @P2   IADD3 R33, -R14, R33, RZ                             ; // x      2036  ALU
/*022e7f20*/  @P1   IMAD.IADD R35, R35, 0x1, -R14                        ; // x      2036  FMA
/*022e7f30*/  @P2   IADD3 R8, R8, 0x1, RZ                                ; // x      2036  ALU
/*022e7f40*/  IMAD R31, R14, R31, R29                                    ; // x      2036  FMA
/*022e7f50*/  ISETP.GE.U32.AND P5, PT, R33, R14, PT                      ; // x      2036  ALU
/*022e7f60*/  IMAD.MOV R32, RZ, RZ, -R9                                  ; // x      2036  FMA
/*022e7f70*/  IADD3 R33, -R26, RZ, RZ                                    ; // x      2036  ALU
/*022e7f80*/  IMAD R19, R19, c[0x0][0xc24], RZ                           ; // x      2036  FMA
/*022e7f90*/  ISETP.GE.U32.AND P0, PT, R31, R14, PT                      ; // x      2036  ALU
/*022e7fa0*/  IMAD R20, R15, R32, R20                                    ; // x      2036  FMA
/*022e7fb0*/  ISETP.GE.U32.AND P4, PT, R35, R14, PT                      ; // x      2036  ALU
/*022e7fc0*/  IMAD R21, R16, R33, R21                                    ; // x      2036  FMA
/*022e7fd0*/  IADD3 R33, -R29, RZ, RZ                                    ; // x      2036  ALU
/*022e7fe0*/  IMAD.MOV R32, RZ, RZ, -R23                                 ; // x      2036  FMA
/*022e7ff0*/  @P1   IADD3 R28, R28, 0x1, RZ                              ; // x      2036  ALU
/*022e8000*/  IMAD R21, R21, c[0x0][0xc24], RZ                           ; // x      2036  FMA
/*022e8010*/  ISETP.NE.U32.AND P1, PT, R14, RZ, PT                       ; // x      2036  ALU
/*022e8020*/  IMAD R26, R15, R33, R26                                    ; // x      2036  FMA
/*022e8030*/  @P6   IADD3 R25, R25, 0x1, RZ                              ; // x      2036  ALU
/*022e8040*/  IMAD R17, R20, c[0x0][0xc20], R17                          ; // x      2036  FMA
/*022e8050*/  @P5   IADD3 R8, R8, 0x1, RZ                                ; // x      2036  ALU
/*022e8060*/  IMAD R20, R26, c[0x0][0xc20], R21                          ; // x      2036  FMA
/*022e8070*/  @P0   IADD3 R31, -R14, R31, RZ                             ; // x      2036  ALU
/*022e8080*/  IMAD R22, R15, R32, R22                                    ; // x      2036  FMA
/*022e8090*/  @P0   IADD3 R30, R30, 0x1, RZ                              ; // x      2036  ALU
/*022e80a0*/  IMAD.MOV R32, RZ, RZ, -R27                                 ; // x      2036  FMA
/*022e80b0*/  ISETP.GE.U32.AND P2, PT, R31, R14, PT                      ; // x      2036  ALU
/*022e80c0*/  IMAD.MOV R31, RZ, RZ, -R24                                 ; // x      2036  FMA
/*022e80d0*/  LOP3.LUT R21, RZ, R14, RZ, 0x33, !PT                       ; // x      2036  ALU
/*022e80e0*/  IMAD R24, R15, R32, R24                                    ; // x      2036  FMA
/*022e80f0*/  @P4   IADD3 R28, R28, 0x1, RZ                              ; // x      2036  ALU
/*022e8100*/  IMAD R31, R16, R31, R11                                    ; // x      2036  FMA
/*022e8110*/  SEL R25, R21, R25, !P1                                     ; // x      2036  ALU
/*022e8120*/  IMAD R19, R22, c[0x0][0xc20], R19                          ; // x      2036  FMA
/*022e8130*/  SEL R8, R21, R8, !P1                                       ; // x      2036  ALU
/*022e8140*/  IMAD R31, R31, c[0x0][0xc24], RZ                           ; // x      2036  FMA
/*022e8150*/  SEL R28, R21, R28, !P1                                     ; // x      2036  ALU
/*022e8160*/  IMAD R11, R18, 0x4, R11                                    ; // x      2036  FMA
/*022e8170*/  @P2   IADD3 R30, R30, 0x1, RZ                              ; // x      2036  ALU
/*022e8180*/  IMAD.MOV R22, RZ, RZ, -R8                                  ; // x      2036  FMA
/*022e8190*/  IADD3 R26, -R28, RZ, RZ                                    ; // x      2036  ALU
/*022e81a0*/  IMAD R24, R24, c[0x0][0xc20], R31                          ; // x      2036  FMA
/*022e81b0*/  SEL R30, R21, R30, !P1                                     ; // x      2036  ALU
/*022e81c0*/  IMAD.MOV R21, RZ, RZ, -R25                                 ; // x      2036  FMA
/*022e81d0*/  IMAD R22, R14, R22, R9                                     ; // x      2036  FMA
/*022e81e0*/  IMAD.MOV R31, RZ, RZ, -R30                                 ; // x      2036  FMA
/*022e81f0*/  IMAD R27, R14, R21, R27                                    ; // x      2036  FMA
/*022e8200*/  IMAD R26, R14, R26, R23                                    ; // x      2036  FMA
/*022e8210*/  IMAD R29, R14, R31, R29                                    ; // x      2036  FMA
/*022e8220*/  IMAD R24, R27, c[0x0][0xc1c], R24                          ; // x      2036  FMA
/*022e8230*/  IMAD R17, R22, c[0x0][0xc1c], R17                          ; // x      2036  FMA
/*022e8240*/  IMAD R19, R26, c[0x0][0xc1c], R19                          ; // x      2036  FMA
/*022e8250*/  IMAD R29, R29, c[0x0][0xc1c], R20                          ; // x      2036  FMA
/*022e8260*/  IMAD R25, R25, c[0x0][0xc18], R24                          ; // x      2036  FMA
/*022e8270*/  IMAD R17, R8, c[0x0][0xc18], R17                           ; // x      2036  FMA
/*022e8280*/  IMAD R19, R28, c[0x0][0xc18], R19                          ; // x      2036  FMA
/*022e8290*/  IMAD R29, R30, c[0x0][0xc18], R29                          ; // x      2036  FMA
/*022e82a0*/  IMAD R8, R0, c[0x0][0xc2c], R25                            ; // x      2036  FMA
/*022e82b0*/  IMAD R20, R0, c[0x0][0xc2c], R17                           ; // x      2036  FMA
/*022e82c0*/  IADD3 R17, R11, 0x4, RZ                                    ; // x      2036  ALU
/*022e82d0*/  IMAD R22, R0, c[0x0][0xc2c], R19                           ; // x      2036  FMA
/*022e82e0*/  IMAD R24, R0, c[0x0][0xc2c], R29                           ; // x      2036  FMA
/*022e82f0*/  ISETP.GT.U32.AND P0, PT, R17, R12, PT                      ; // x      2036  ALU
/*022e8300*/  IMAD.WIDE.U32 R8, R8, R13, c[0x0][0x160]                   ; // x      2036  FMA
/*022e8310*/  IMAD.WIDE.U32 R20, R20, R13, c[0x0][0x160]                 ; // x      2036  FMA
/*022e8320*/  STG.E [R8.64], R4                                          ; // x      2036  global store
/*022e8330*/  IMAD.WIDE.U32 R22, R22, R13, c[0x0][0x160]                 ; // x      2036  FMA
/*022e8340*/  STG.E [R20.64], R5                                         ; // x      2036  global store
/*022e8350*/  IMAD.WIDE.U32 R24, R24, R13, c[0x0][0x160]                 ; // x      2036  FMA
/*022e8360*/  STG.E [R22.64], R6                                         ; // x      2036  global store
/*022e8370*/  STG.E [R24.64], R7                                         ; // x      2036  global store
/*022e8380*/  @!P0  BRA 0x2022e7710                                      ; // x      2036  branch
```
