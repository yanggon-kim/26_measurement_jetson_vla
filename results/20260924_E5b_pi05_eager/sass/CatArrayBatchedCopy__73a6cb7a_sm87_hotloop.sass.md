# Hot loop — `CatArrayBatchedCopy__73a6cb7a` (sm87)

PC range 0x2271df0-0x2272310; backward BRA 0x2272310 -> 0x2271df0: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/CatArrayBatchedCopy__73a6cb7a_sm87.sass`; per-PC counts: `raw/CatArrayBatchedCopy__73a6cb7a_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*02271df0*/  IMAD.HI.U32 R9, R0, R19, RZ                                ; // x      3200  FMA
/*02271e00*/  IADD3 R8, -R9, RZ, RZ                                      ; // x      3200  ALU
/*02271e10*/  IMAD R11, R18, R8, R19                                     ; // x      3200  FMA
/*02271e20*/  ISETP.GE.U32.AND P6, PT, R11, R18, PT                      ; // x      3200  ALU
/*02271e30*/  @P6   IMAD.IADD R11, R11, 0x1, -R18                        ; // x      3200  FMA
/*02271e40*/  @P6   IADD3 R9, R9, 0x1, RZ                                ; // x      3200  ALU
/*02271e50*/  ISETP.GE.U32.AND P6, PT, R11, R18, PT                      ; // x      3200  ALU
/*02271e60*/  @P6   IADD3 R9, R9, 0x1, RZ                                ; // x      3200  ALU
/*02271e70*/  SEL R10, R28, R9, !P0                                      ; // x      3200  ALU
/*02271e80*/  IMAD.HI.U32 R8, R24, R10, RZ                               ; // x      3200  FMA
/*02271e90*/  IMAD.MOV R9, RZ, RZ, -R8                                   ; // x      3200  FMA
/*02271ea0*/  IMAD R34, R17, R9, R10                                     ; // x      3200  FMA
/*02271eb0*/  ISETP.GE.U32.AND P6, PT, R34, R17, PT                      ; // x      3200  ALU
/*02271ec0*/  @P6   IADD3 R34, -R17, R34, RZ                             ; // x      3200  ALU
/*02271ed0*/  @P6   IADD3 R8, R8, 0x1, RZ                                ; // x      3200  ALU
/*02271ee0*/  ISETP.GE.U32.AND P6, PT, R34, R17, PT                      ; // x      3200  ALU
/*02271ef0*/  IMAD.HI.U32 R34, R26, R19, RZ                              ; // x      3200  FMA
/*02271f00*/  @P6   IADD3 R8, R8, 0x1, RZ                                ; // x      3200  ALU
/*02271f10*/  SEL R11, R29, R8, !P1                                      ; // x      3200  ALU
/*02271f20*/  IMAD.HI.U32 R33, R25, R11, RZ                              ; // x      3200  FMA
/*02271f30*/  IMAD.MOV R8, RZ, RZ, -R33                                  ; // x      3200  FMA
/*02271f40*/  IMAD R9, R16, R8, R11                                      ; // x      3200  FMA
/*02271f50*/  IADD3 R8, -R34, RZ, RZ                                     ; // x      3200  ALU
/*02271f60*/  ISETP.GE.U32.AND P6, PT, R9, R16, PT                       ; // x      3200  ALU
/*02271f70*/  IMAD R8, R15, R8, R19                                      ; // x      3200  FMA
/*02271f80*/  @P6   IMAD.IADD R9, R9, 0x1, -R16                          ; // x      3200  FMA
/*02271f90*/  @P6   IADD3 R33, R33, 0x1, RZ                              ; // x      3200  ALU
/*02271fa0*/  ISETP.GE.U32.AND P6, PT, R9, R16, PT                       ; // x      3200  ALU
/*02271fb0*/  @P6   IADD3 R33, R33, 0x1, RZ                              ; // x      3200  ALU
/*02271fc0*/  ISETP.GE.U32.AND P6, PT, R8, R15, PT                       ; // x      3200  ALU
/*02271fd0*/  @P6   IMAD.IADD R8, R8, 0x1, -R15                          ; // x      3200  FMA
/*02271fe0*/  @P6   IADD3 R34, R34, 0x1, RZ                              ; // x      3200  ALU
/*02271ff0*/  ISETP.GE.U32.AND P6, PT, R8, R15, PT                       ; // x      3200  ALU
/*02272000*/  IADD3 R8, -R11, RZ, RZ                                     ; // x      3200  ALU
/*02272010*/  IMAD R8, R17, R8, R10                                      ; // x      3200  FMA
/*02272020*/  IMAD.MOV R10, RZ, RZ, -R10                                 ; // x      3200  FMA
/*02272030*/  IMAD R10, R18, R10, R19                                    ; // x      3200  FMA
/*02272040*/  @P6   IADD3 R34, R34, 0x1, RZ                              ; // x      3200  ALU
/*02272050*/  IMAD R35, R10, c[0x0][0xec4], RZ                           ; // x      3200  FMA
/*02272060*/  SEL R10, R30, R33, !P2                                     ; // x      3200  ALU
/*02272070*/  SEL R9, R31, R34, !P3                                      ; // x      3200  ALU
/*02272080*/  IMAD R8, R8, c[0x0][0xec0], R35                            ; // x      3200  FMA
/*02272090*/  IADD3 R34, -R10, RZ, RZ                                    ; // x      3200  ALU
/*022720a0*/  IMAD.HI.U32 R33, R12, R9, RZ                               ; // x      3200  FMA
/*022720b0*/  IMAD R35, R16, R34, R11                                    ; // x      3200  FMA
/*022720c0*/  IMAD.MOV R11, RZ, RZ, -R33                                 ; // x      3200  FMA
/*022720d0*/  IMAD.MOV R34, RZ, RZ, -R9                                  ; // x      3200  FMA
/*022720e0*/  IMAD R11, R14, R11, R9                                     ; // x      3200  FMA
/*022720f0*/  IMAD R34, R15, R34, R19                                    ; // x      3200  FMA
/*02272100*/  ISETP.GE.U32.AND P6, PT, R11, R14, PT                      ; // x      3200  ALU
/*02272110*/  @P6   IADD3 R11, -R14, R11, RZ                             ; // x      3200  ALU
/*02272120*/  @P6   IADD3 R33, R33, 0x1, RZ                              ; // x      3200  ALU
/*02272130*/  ISETP.GE.U32.AND P6, PT, R11, R14, PT                      ; // x      3200  ALU
/*02272140*/  IMAD R11, R35, c[0x0][0xebc], R8                           ; // x      3200  FMA
/*02272150*/  @P6   IADD3 R33, R33, 0x1, RZ                              ; // x      3200  ALU
/*02272160*/  SEL R8, R32, R33, !P4                                      ; // x      3200  ALU
/*02272170*/  IADD3 R33, -R8, RZ, RZ                                     ; // x      3200  ALU
/*02272180*/  IMAD.HI.U32 R36, R27, R8, RZ                               ; // x      3200  FMA
/*02272190*/  IMAD R33, R14, R33, R9                                     ; // x      3200  FMA
/*022721a0*/  IMAD R9, R7, R34, RZ                                       ; // x      3200  FMA
/*022721b0*/  IMAD R34, R6, R33, R9                                      ; // x      3200  FMA
/*022721c0*/  IMAD.MOV R9, RZ, RZ, -R36                                  ; // x      3200  FMA
/*022721d0*/  IMAD R9, R21, R9, R8                                       ; // x      3200  FMA
/*022721e0*/  ISETP.GE.U32.AND P6, PT, R9, R21, PT                       ; // x      3200  ALU
/*022721f0*/  @P6   IADD3 R9, -R21, R9, RZ                               ; // x      3200  ALU
/*02272200*/  @P6   IADD3 R36, R36, 0x1, RZ                              ; // x      3200  ALU
/*02272210*/  ISETP.GE.U32.AND P6, PT, R9, R21, PT                       ; // x      3200  ALU
/*02272220*/  @P6   IADD3 R36, R36, 0x1, RZ                              ; // x      3200  ALU
/*02272230*/  SEL R9, R13, R36, !P5                                      ; // x      3200  ALU
/*02272240*/  IMAD.MOV R33, RZ, RZ, -R9                                  ; // x      3200  FMA
/*02272250*/  IMAD R8, R21, R33, R8                                      ; // x      3200  FMA
/*02272260*/  IMAD R8, R5, R8, R34                                       ; // x      3200  FMA
/*02272270*/  IMAD R9, R4, R9, R8                                        ; // x      3200  FMA
/*02272280*/  IMAD.WIDE.U32 R8, R9, 0x4, R2                              ; // x      3200  FMA
/*02272290*/  LDG.E R9, [R8.64]                                          ; // x      3200  global load (DRAM/L2: weights or activations)
/*022722a0*/  IMAD R11, R10, c[0x0][0xeb8], R11                          ; // x      3200  FMA
/*022722b0*/  MOV R34, c[0x0][0xc]                                       ; // x      3200  ALU
/*022722c0*/  IMAD R10, R20, c[0x0][0xecc], R11                          ; // x      3200  FMA
/*022722d0*/  IMAD R19, R34, c[0x0][0x0], R19                            ; // x      3200  FMA
/*022722e0*/  IMAD.WIDE.U32 R10, R10, R23, c[0x0][0x160]                 ; // x      3200  FMA
/*022722f0*/  ISETP.GE.U32.AND P6, PT, R19, R22, PT                      ; // x      3200  ALU
/*02272300*/  STG.E [R10.64], R9                                         ; // x      3200  global store
/*02272310*/  @!P6  BRA 0x202271df0                                      ; // x      3200  branch
```
