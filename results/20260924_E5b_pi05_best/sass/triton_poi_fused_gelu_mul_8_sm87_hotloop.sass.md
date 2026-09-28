# Hot loop — `triton_poi_fused_gelu_mul_8` (sm87)

PC range 0x2296300-0x2296970; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/triton_poi_fused_gelu_mul_8_sm87.sass`; per-PC counts: `raw/triton_poi_fused_gelu_mul_8_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*02296300*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x      3200  ALU
/*02296310*/  @P0   BRA 0x202296440                                      ; // x      3200  branch
/*02296320*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*02296330*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*02296340*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*02296350*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*02296360*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*02296370*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*02296380*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*02296390*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*022963a0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*022963b0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*022963c0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*022963d0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*022963e0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*022963f0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*02296400*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*02296410*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*02296420*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*02296430*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*02296440*/  MOV R1, c[0x0][0x28]                                       ; // x      3200  ALU
/*02296450*/  S2R R0, SR_TID.X                                           ; // x      3200  Branch
/*02296460*/  HFMA2.MMA R7, -RZ, RZ, 0, 2.384185791015625e-07            ; // x      3200  FMA
/*02296470*/  ULDC.64 UR4, c[0x0][0x118]                                 ; // x      3200  Uniform
/*02296480*/  S2R R3, SR_CTAID.X                                         ; // x      3200  Branch
/*02296490*/  SHF.L.U32 R0, R0, 0x1, RZ                                  ; // x      3200  ALU
/*022964a0*/  LOP3.LUT R0, R0, 0x1fe, RZ, 0xc0, !PT                      ; // x      3200  ALU
/*022964b0*/  LEA R0, R3, R0, 0x9                                        ; // x      3200  ALU
/*022964c0*/  IMAD.WIDE R2, R0, R7, c[0x0][0x160]                        ; // x      3200  FMA
/*022964d0*/  LDG.E.64 R4, [R2.64]                                       ; // x      3200  global load (DRAM/L2: weights or activations)
/*022964e0*/  IMAD.WIDE R6, R0, R7, c[0x0][0x168]                        ; // x      3200  FMA
/*022964f0*/  LDG.E.64 R6, [R6.64]                                       ; // x      3200  global load (DRAM/L2: weights or activations)
/*02296500*/  BSSY B0, 0x2022966e0                                       ; // x      3200  Branch
/*02296510*/  FMUL R9, R4, R4                                            ; // x      3200  FMA
/*02296520*/  FMUL R0, R5, R5                                            ; // x      3200  FMA
/*02296530*/  FMUL R9, R4, R9                                            ; // x      3200  FMA
/*02296540*/  FMUL R0, R5, R0                                            ; // x      3200  FMA
/*02296550*/  FFMA R9, R9, 0.044714998453855514526, R4                   ; // x      3200  FMA
/*02296560*/  FFMA R0, R0, 0.044714998453855514526, R5                   ; // x      3200  FMA
/*02296570*/  FMUL R9, R9, 0.79788458347320556641                        ; // x      3200  FMA
/*02296580*/  FMUL R8, R0, 0.79788458347320556641                        ; // x      3200  FMA
/*02296590*/  FADD.FTZ R10, |R9|, -RZ                                    ; // x      3200  FMA
/*022965a0*/  FSETP.GE.AND P0, PT, R10, 0.60000002384185791016, PT       ; // x      3200  FMA
/*022965b0*/  @!P0  BRA 0x202296660                                      ; // x      3200  branch
/*022965c0*/  FMUL R0, R10, 2.8853900432586669922                        ; // x      3200  FMA
/*022965d0*/  MOV R12, 0x3f800000                                        ; // x      3200  ALU
/*022965e0*/  FSETP.GE.AND P0, PT, R10, 9.010913848876953125, PT         ; // x      3200  FMA
/*022965f0*/  MUFU.EX2 R0, R0                                            ; // x      3200  special-function unit
/*02296600*/  FADD R11, R0, 1                                            ; // x      3200  FMA
/*02296610*/  MUFU.RCP R11, R11                                          ; // x      3200  special-function unit
/*02296620*/  FFMA.FTZ R12, R11, -2, R12                                 ; // x      3200  FMA
/*02296630*/  FSEL R12, R12, 1, !P0                                      ; // x      3200  FMA
/*02296640*/  LOP3.LUT R9, R12, 0x80000000, R9, 0xf8, !PT                ; // x      3200  ALU
/*02296650*/  BRA 0x2022966d0                                            ; // x      3200  branch
/*02296660*/  MOV R0, 0x3c80f082                                         ; // x      3200  ALU
/*02296670*/  FMUL R11, R9, R9                                           ; // x      3200  FMA
/*02296680*/  FFMA.FTZ R0, R11, R0, -0.052303962409496307373             ; // x      3200  FMA
/*02296690*/  FFMA.FTZ R0, R11, R0, 0.1331529766321182251                ; // x      3200  FMA
/*022966a0*/  FFMA.FTZ R0, R11, R0, -0.33332768082618713379              ; // x      3200  FMA
/*022966b0*/  FFMA.FTZ R0, R11, R0, RZ                                   ; // x      3200  FMA
/*022966c0*/  FFMA.FTZ R9, R9, R0, R9                                    ; // x      3200  FMA
/*022966d0*/  BSYNC B0                                                   ; // x      6400  Branch
/*022966e0*/  FADD.FTZ R12, |R8|, -RZ                                    ; // x      3200  FMA
/*022966f0*/  BSSY B0, 0x202296840                                       ; // x      3200  Branch
/*02296700*/  FSETP.GE.AND P0, PT, R12, 0.60000002384185791016, PT       ; // x      3200  FMA
/*02296710*/  @!P0  BRA 0x2022967c0                                      ; // x      3200  branch
/*02296720*/  FMUL R0, R12, 2.8853900432586669922                        ; // x      3200  FMA
/*02296730*/  HFMA2.MMA R11, -RZ, RZ, 1.875, 0                           ; // x      3200  FMA
/*02296740*/  FSETP.GE.AND P0, PT, R12, 9.010913848876953125, PT         ; // x      3200  FMA
/*02296750*/  MUFU.EX2 R0, R0                                            ; // x      3200  special-function unit
/*02296760*/  FADD R10, R0, 1                                            ; // x      3200  FMA
/*02296770*/  MUFU.RCP R10, R10                                          ; // x      3200  special-function unit
/*02296780*/  FFMA.FTZ R11, R10, -2, R11                                 ; // x      3200  FMA
/*02296790*/  FSEL R11, R11, 1, !P0                                      ; // x      3200  FMA
/*022967a0*/  LOP3.LUT R8, R11, 0x80000000, R8, 0xf8, !PT                ; // x      3200  ALU
/*022967b0*/  BRA 0x202296830                                            ; // x      3200  branch
/*022967c0*/  MOV R0, 0x3c80f082                                         ; // x      3200  ALU
/*022967d0*/  FMUL R11, R8, R8                                           ; // x      3200  FMA
/*022967e0*/  FFMA.FTZ R0, R11, R0, -0.052303962409496307373             ; // x      3200  FMA
/*022967f0*/  FFMA.FTZ R0, R11, R0, 0.1331529766321182251                ; // x      3200  FMA
/*02296800*/  FFMA.FTZ R0, R11, R0, -0.33332768082618713379              ; // x      3200  FMA
/*02296810*/  FFMA.FTZ R11, R11, R0, RZ                                  ; // x      3200  FMA
/*02296820*/  FFMA.FTZ R8, R8, R11, R8                                   ; // x      3200  FMA
/*02296830*/  BSYNC B0                                                   ; // x      6400  Branch
/*02296840*/  FMUL R5, R5, 0.5                                           ; // x      3200  FMA
/*02296850*/  FMUL R4, R4, 0.5                                           ; // x      3200  FMA
/*02296860*/  FADD R9, R9, 1                                             ; // x      3200  FMA
/*02296870*/  FADD R8, R8, 1                                             ; // x      3200  FMA
/*02296880*/  FMUL R9, R4, R9                                            ; // x      3200  FMA
/*02296890*/  FMUL R8, R5, R8                                            ; // x      3200  FMA
/*022968a0*/  FMUL R6, R6, R9                                            ; // x      3200  FMA
/*022968b0*/  FMUL R7, R7, R8                                            ; // x      3200  FMA
/*022968c0*/  STG.E.64 [R2.64], R6                                       ; // x      3200  global store
/*022968d0*/  EXIT                                                       ; // x      3200  Branch
/*022968e0*/  BRA 0x2022968e0                                            ; // x         0  branch
/*022968f0*/  NOP                                                        ; // x         0  Branch
/*02296900*/  NOP                                                        ; // x         0  Branch
/*02296910*/  NOP                                                        ; // x         0  Branch
/*02296920*/  NOP                                                        ; // x         0  Branch
/*02296930*/  NOP                                                        ; // x         0  Branch
/*02296940*/  NOP                                                        ; // x         0  Branch
/*02296950*/  NOP                                                        ; // x         0  Branch
/*02296960*/  NOP                                                        ; // x         0  Branch
/*02296970*/  NOP                                                        ; // x         0  Branch
```
