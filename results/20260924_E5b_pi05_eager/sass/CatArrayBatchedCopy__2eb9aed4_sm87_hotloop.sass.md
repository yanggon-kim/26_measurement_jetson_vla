# Hot loop — `CatArrayBatchedCopy__2eb9aed4` (sm87)

PC range 0x2242900-0x2242ca0; backward BRA 0x2242ca0 -> 0x2242900: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/CatArrayBatchedCopy__2eb9aed4_sm87.sass`; per-PC counts: `raw/CatArrayBatchedCopy__2eb9aed4_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*02242900*/  IMAD.HI.U32 R6, R20, R13, RZ                               ; // x       400  FMA
/*02242910*/  IADD3 R7, -R6, RZ, RZ                                      ; // x       400  ALU
/*02242920*/  IMAD R7, R12, R7, R13                                      ; // x       400  FMA
/*02242930*/  ISETP.GE.U32.AND P4, PT, R7, R12, PT                       ; // x       400  ALU
/*02242940*/  @P4   IMAD.IADD R7, R7, 0x1, -R12                          ; // x       400  FMA
/*02242950*/  @P4   IADD3 R6, R6, 0x1, RZ                                ; // x       400  ALU
/*02242960*/  ISETP.GE.U32.AND P5, PT, R7, R12, PT                       ; // x       400  ALU
/*02242970*/  @P5   IADD3 R6, R6, 0x1, RZ                                ; // x       400  ALU
/*02242980*/  SEL R6, R23, R6, !P2                                       ; // x       400  ALU
/*02242990*/  IADD3 R7, -R6, RZ, RZ                                      ; // x       400  ALU
/*022429a0*/  IMAD.HI.U32 R24, R19, R6, RZ                               ; // x       400  FMA
/*022429b0*/  IMAD R7, R12, R7, R13                                      ; // x       400  FMA
/*022429c0*/  IADD3 R26, -R24, RZ, RZ                                    ; // x       400  ALU
/*022429d0*/  IMAD R26, R15, R26, R6                                     ; // x       400  FMA
/*022429e0*/  ISETP.GE.U32.AND P4, PT, R26, R15, PT                      ; // x       400  ALU
/*022429f0*/  @P4   IMAD.IADD R26, R26, 0x1, -R15                        ; // x       400  FMA
/*02242a00*/  @P4   IADD3 R24, R24, 0x1, RZ                              ; // x       400  ALU
/*02242a10*/  ISETP.GE.U32.AND P5, PT, R26, R15, PT                      ; // x       400  ALU
/*02242a20*/  @P5   IADD3 R24, R24, 0x1, RZ                              ; // x       400  ALU
/*02242a30*/  SEL R25, R17, R24, !P3                                     ; // x       400  ALU
/*02242a40*/  IMAD R24, R18, R7, RZ                                      ; // x       400  FMA
/*02242a50*/  IMAD.MOV R26, RZ, RZ, -R25                                 ; // x       400  FMA
/*02242a60*/  IMAD R6, R15, R26, R6                                      ; // x       400  FMA
/*02242a70*/  IMAD R6, R3, R6, R24                                       ; // x       400  FMA
/*02242a80*/  IMAD R7, R2, R25, R6                                       ; // x       400  FMA
/*02242a90*/  IMAD.WIDE.U32 R6, R7, 0x4, R4                              ; // x       400  FMA
/*02242aa0*/  LDG.E R25, [R6.64]                                         ; // x       400  global load (DRAM/L2: weights or activations)
/*02242ab0*/  IMAD.HI.U32 R24, R14, R13, RZ                              ; // x       400  FMA
/*02242ac0*/  IADD3 R26, -R24, RZ, RZ                                    ; // x       400  ALU
/*02242ad0*/  IMAD R27, R10, R26, R13                                    ; // x       400  FMA
/*02242ae0*/  ISETP.GE.U32.AND P4, PT, R27, R10, PT                      ; // x       400  ALU
/*02242af0*/  @P4   IMAD.IADD R27, R27, 0x1, -R10                        ; // x       400  FMA
/*02242b00*/  @P4   IADD3 R24, R24, 0x1, RZ                              ; // x       400  ALU
/*02242b10*/  ISETP.GE.U32.AND P5, PT, R27, R10, PT                      ; // x       400  ALU
/*02242b20*/  @P5   IADD3 R24, R24, 0x1, RZ                              ; // x       400  ALU
/*02242b30*/  SEL R24, R21, R24, !P0                                     ; // x       400  ALU
/*02242b40*/  IMAD.HI.U32 R27, R16, R24, RZ                              ; // x       400  FMA
/*02242b50*/  IADD3 R6, -R27, RZ, RZ                                     ; // x       400  ALU
/*02242b60*/  IMAD R6, R11, R6, R24                                      ; // x       400  FMA
/*02242b70*/  ISETP.GE.U32.AND P4, PT, R6, R11, PT                       ; // x       400  ALU
/*02242b80*/  @P4   IMAD.IADD R6, R6, 0x1, -R11                          ; // x       400  FMA
/*02242b90*/  @P4   IADD3 R27, R27, 0x1, RZ                              ; // x       400  ALU
/*02242ba0*/  ISETP.GE.U32.AND P5, PT, R6, R11, PT                       ; // x       400  ALU
/*02242bb0*/  IADD3 R6, -R24, RZ, RZ                                     ; // x       400  ALU
/*02242bc0*/  IMAD R6, R10, R6, R13                                      ; // x       400  FMA
/*02242bd0*/  IMAD R6, R6, c[0x0][0xec0], RZ                             ; // x       400  FMA
/*02242be0*/  @P5   IADD3 R27, R27, 0x1, RZ                              ; // x       400  ALU
/*02242bf0*/  SEL R27, R22, R27, !P1                                     ; // x       400  ALU
/*02242c00*/  IMAD.MOV R7, RZ, RZ, -R27                                  ; // x       400  FMA
/*02242c10*/  IMAD R7, R11, R7, R24                                      ; // x       400  FMA
/*02242c20*/  MOV R24, c[0x0][0xc]                                       ; // x       400  ALU
/*02242c30*/  IMAD R6, R7, c[0x0][0xebc], R6                             ; // x       400  FMA
/*02242c40*/  IMAD R13, R24, c[0x0][0x0], R13                            ; // x       400  FMA
/*02242c50*/  IMAD R6, R27, c[0x0][0xeb8], R6                            ; // x       400  FMA
/*02242c60*/  ISETP.GE.U32.AND P4, PT, R13, R8, PT                       ; // x       400  ALU
/*02242c70*/  IMAD R7, R9, c[0x0][0xecc], R6                             ; // x       400  FMA
/*02242c80*/  IMAD.WIDE.U32 R6, R7, R0, c[0x0][0x160]                    ; // x       400  FMA
/*02242c90*/  STG.E [R6.64], R25                                         ; // x       400  global store
/*02242ca0*/  @!P4  BRA 0x202242900                                      ; // x       400  branch
```
