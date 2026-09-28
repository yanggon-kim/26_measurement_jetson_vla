# Hot loop — `CatArrayBatchedCopy__ff2753b7` (sm87)

PC range 0x22d0b20-0x22d0ee0; backward BRA 0x22d0ee0 -> 0x22d0b20: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/CatArrayBatchedCopy__ff2753b7_sm87.sass`; per-PC counts: `raw/CatArrayBatchedCopy__ff2753b7_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*022d0b20*/  IMAD.HI.U32 R7, R15, R12, RZ                               ; // x      1592  FMA
/*022d0b30*/  IADD3 R6, -R7, RZ, RZ                                      ; // x      1592  ALU
/*022d0b40*/  IMAD R6, R13, R6, R12                                      ; // x      1592  FMA
/*022d0b50*/  ISETP.GE.U32.AND P4, PT, R6, R13, PT                       ; // x      1592  ALU
/*022d0b60*/  @P4   IMAD.IADD R6, R6, 0x1, -R13                          ; // x      1592  FMA
/*022d0b70*/  @P4   IADD3 R7, R7, 0x1, RZ                                ; // x      1592  ALU
/*022d0b80*/  ISETP.GE.U32.AND P5, PT, R6, R13, PT                       ; // x      1592  ALU
/*022d0b90*/  @P5   IADD3 R7, R7, 0x1, RZ                                ; // x      1592  ALU
/*022d0ba0*/  SEL R6, R22, R7, !P2                                       ; // x      1592  ALU
/*022d0bb0*/  IMAD.HI.U32 R7, R18, R6, RZ                                ; // x      1592  FMA
/*022d0bc0*/  IMAD.MOV R24, RZ, RZ, -R7                                  ; // x      1592  FMA
/*022d0bd0*/  IMAD R24, R11, R24, R6                                     ; // x      1592  FMA
/*022d0be0*/  ISETP.GE.U32.AND P4, PT, R24, R11, PT                      ; // x      1592  ALU
/*022d0bf0*/  @P4   IADD3 R24, -R11, R24, RZ                             ; // x      1592  ALU
/*022d0c00*/  @P4   IADD3 R7, R7, 0x1, RZ                                ; // x      1592  ALU
/*022d0c10*/  ISETP.GE.U32.AND P5, PT, R24, R11, PT                      ; // x      1592  ALU
/*022d0c20*/  IMAD.MOV R24, RZ, RZ, -R6                                  ; // x      1592  FMA
/*022d0c30*/  @P5   IADD3 R7, R7, 0x1, RZ                                ; // x      1592  ALU
/*022d0c40*/  SEL R23, R14, R7, !P3                                      ; // x      1592  ALU
/*022d0c50*/  IMAD R7, R13, R24, R12                                     ; // x      1592  FMA
/*022d0c60*/  IMAD.MOV R25, RZ, RZ, -R23                                 ; // x      1592  FMA
/*022d0c70*/  IMAD R24, R16, R7, RZ                                      ; // x      1592  FMA
/*022d0c80*/  IMAD R6, R11, R25, R6                                      ; // x      1592  FMA
/*022d0c90*/  IMAD R6, R3, R6, R24                                       ; // x      1592  FMA
/*022d0ca0*/  IMAD R7, R2, R23, R6                                       ; // x      1592  FMA
/*022d0cb0*/  IADD3 R6, P4, R4, R7, RZ                                   ; // x      1592  ALU
/*022d0cc0*/  IADD3.X R7, RZ, R5, RZ, P4, !PT                            ; // x      1592  ALU
/*022d0cd0*/  LDG.E.U8 R23, [R6.64]                                      ; // x      1592  global load (DRAM/L2: weights or activations)
/*022d0ce0*/  IMAD.HI.U32 R24, R17, R12, RZ                              ; // x      1592  FMA
/*022d0cf0*/  IMAD.MOV R25, RZ, RZ, -R24                                 ; // x      1592  FMA
/*022d0d00*/  IMAD R26, R9, R25, R12                                     ; // x      1592  FMA
/*022d0d10*/  ISETP.GE.U32.AND P4, PT, R26, R9, PT                       ; // x      1592  ALU
/*022d0d20*/  @P4   IMAD.IADD R26, R26, 0x1, -R9                         ; // x      1592  FMA
/*022d0d30*/  @P4   IADD3 R24, R24, 0x1, RZ                              ; // x      1592  ALU
/*022d0d40*/  ISETP.GE.U32.AND P5, PT, R26, R9, PT                       ; // x      1592  ALU
/*022d0d50*/  @P5   IADD3 R24, R24, 0x1, RZ                              ; // x      1592  ALU
/*022d0d60*/  SEL R25, R19, R24, !P0                                     ; // x      1592  ALU
/*022d0d70*/  IMAD.HI.U32 R24, R20, R25, RZ                              ; // x      1592  FMA
/*022d0d80*/  IADD3 R6, -R24, RZ, RZ                                     ; // x      1592  ALU
/*022d0d90*/  IMAD R7, R10, R6, R25                                      ; // x      1592  FMA
/*022d0da0*/  IMAD.MOV R6, RZ, RZ, -R25                                  ; // x      1592  FMA
/*022d0db0*/  ISETP.GE.U32.AND P4, PT, R7, R10, PT                       ; // x      1592  ALU
/*022d0dc0*/  IMAD R6, R9, R6, R12                                       ; // x      1592  FMA
/*022d0dd0*/  IMAD R6, R6, c[0x0][0xec0], RZ                             ; // x      1592  FMA
/*022d0de0*/  @P4   IMAD.IADD R7, R7, 0x1, -R10                          ; // x      1592  FMA
/*022d0df0*/  @P4   IADD3 R24, R24, 0x1, RZ                              ; // x      1592  ALU
/*022d0e00*/  ISETP.GE.U32.AND P5, PT, R7, R10, PT                       ; // x      1592  ALU
/*022d0e10*/  @P5   IADD3 R24, R24, 0x1, RZ                              ; // x      1592  ALU
/*022d0e20*/  SEL R24, R21, R24, !P1                                     ; // x      1592  ALU
/*022d0e30*/  IADD3 R7, -R24, RZ, RZ                                     ; // x      1592  ALU
/*022d0e40*/  IMAD R7, R10, R7, R25                                      ; // x      1592  FMA
/*022d0e50*/  MOV R25, c[0x0][0xc]                                       ; // x      1592  ALU
/*022d0e60*/  IMAD R7, R7, c[0x0][0xebc], R6                             ; // x      1592  FMA
/*022d0e70*/  IMAD R12, R25, c[0x0][0x0], R12                            ; // x      1592  FMA
/*022d0e80*/  IMAD R7, R24, c[0x0][0xeb8], R7                            ; // x      1592  FMA
/*022d0e90*/  IMAD R7, R8, c[0x0][0xecc], R7                             ; // x      1592  FMA
/*022d0ea0*/  IADD3 R6, P4, R7, c[0x0][0x160], RZ                        ; // x      1592  ALU
/*022d0eb0*/  IMAD.X R7, RZ, RZ, c[0x0][0x164], P4                       ; // x      1592  FMA
/*022d0ec0*/  ISETP.GE.U32.AND P4, PT, R12, R0, PT                       ; // x      1592  ALU
/*022d0ed0*/  STG.E.U8 [R6.64], R23                                      ; // x      1592  global store
/*022d0ee0*/  @!P4  BRA 0x2022d0b20                                      ; // x      1592  branch
```
