# Hot loop — `Kernel2__cutlass_80_tensorop_bf16_s16816gemm_bf16_64x64_32x6_nn_align2` (sm87)

PC range 0x86245d00-0x8624b570; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/Kernel2__cutlass_80_tensorop_bf16_s16816gemm_bf16_64x64_32x6_nn_align2_sm87.sass`; per-PC counts: `raw/Kernel2__cutlass_80_tensorop_bf16_s16816gemm_bf16_64x64_32x6_nn_align2_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*86245d00*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x        64  ALU
/*86245d10*/  @P0   BRA 0x386245e40                                      ; // x        64  branch
/*86245d20*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*86245d30*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*86245d40*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*86245d50*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*86245d60*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*86245d70*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*86245d80*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*86245d90*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*86245da0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*86245db0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*86245dc0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*86245dd0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*86245de0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*86245df0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*86245e00*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*86245e10*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*86245e20*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*86245e30*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*86245e40*/  MOV R1, c[0x0][0x28]                                       ; // x        64  ALU
/*86245e50*/  S2R R2, SR_CTAID.Y                                         ; // x        64  Branch
/*86245e60*/  IMAD.MOV.U32 R0, RZ, RZ, -0x1                              ; // x        64  FMA
/*86245e70*/  S2R R11, SR_CTAID.X                                        ; // x        64  Branch
/*86245e80*/  SHF.L.U32 R0, R0, c[0x0][0x178], RZ                        ; // x        64  ALU
/*86245e90*/  SHF.L.U32 R2, R2, c[0x0][0x178], RZ                        ; // x        64  ALU
/*86245ea0*/  LOP3.LUT R5, R11, R0, RZ, 0x30, !PT                        ; // x        64  ALU
/*86245eb0*/  SHF.R.S32.HI R11, RZ, c[0x0][0x178], R11                   ; // x        64  ALU
/*86245ec0*/  IMAD.IADD R5, R2, 0x1, R5                                  ; // x        64  FMA
/*86245ed0*/  ISETP.GE.AND P0, PT, R5, c[0x0][0x170], PT                 ; // x        64  ALU
/*86245ee0*/  ISETP.GE.OR P0, PT, R11, c[0x0][0x16c], P0                 ; // x        64  ALU
/*86245ef0*/  @P0   EXIT                                                 ; // x        64  Branch
/*86245f00*/  S2R R69, SR_CTAID.Z                                        ; // x        48  Branch
/*86245f10*/  IMAD.MOV.U32 R0, RZ, RZ, c[0x0][0x17c]                     ; // x        48  FMA
/*86245f20*/  MOV R52, c[0x0][0x278]                                     ; // x        48  ALU
/*86245f30*/  IMAD.MOV.U32 R53, RZ, RZ, c[0x0][0x27c]                    ; // x        48  FMA
/*86245f40*/  MOV R54, c[0x0][0x280]                                     ; // x        48  ALU
/*86245f50*/  IMAD.MOV.U32 R55, RZ, RZ, c[0x0][0x284]                    ; // x        48  FMA
/*86245f60*/  ISETP.GE.U32.AND P0, PT, R0, 0x2, PT                       ; // x        48  ALU
/*86245f70*/  @!P0  BRA 0x386246130                                      ; // x        48  branch
/*86245f80*/  ISETP.NE.AND P0, PT, R0, 0x2, PT                           ; // x         0  ALU
/*86245f90*/  MOV R14, c[0x0][0x168]                                     ; // x         0  ALU
/*86245fa0*/  @!P0  BRA 0x386246040                                      ; // x         0  branch
/*86245fb0*/  ISETP.NE.AND P0, PT, R0, 0x3, PT                           ; // x         0  ALU
/*86245fc0*/  MOV R3, RZ                                                 ; // x         0  ALU
/*86245fd0*/  @P0   BRA 0x386246180                                      ; // x         0  branch
/*86245fe0*/  MOV R54, 0x8                                               ; // x         0  ALU
/*86245ff0*/  IMAD.WIDE R52, R69, R54, c[0x0][0x278]                     ; // x         0  FMA
/*86246000*/  IMAD.WIDE R54, R69, R54, c[0x0][0x280]                     ; // x         0  FMA
/*86246010*/  LDG.E.64 R52, [R52]                                        ; // x         0  global load (DRAM/L2: weights or activations)
/*86246020*/  LDG.E.64 R54, [R54]                                        ; // x         0  global load (DRAM/L2: weights or activations)
/*86246030*/  BRA 0x386246180                                            ; // x         0  branch
/*86246040*/  SHF.R.S32.HI R0, RZ, 0x1f, R69                             ; // x         0  ALU
/*86246050*/  IMAD.WIDE.U32 R6, R69, c[0x0][0x298], RZ                   ; // x         0  FMA
/*86246060*/  IMAD R4, R0, c[0x0][0x298], RZ                             ; // x         0  FMA
/*86246070*/  LEA R52, P1, R6, c[0x0][0x278], 0x1                        ; // x         0  ALU
/*86246080*/  IMAD R0, R0, c[0x0][0x2a0], RZ                             ; // x         0  FMA
/*86246090*/  IMAD.WIDE.U32 R2, R69, c[0x0][0x2a0], RZ                   ; // x         0  FMA
/*862460a0*/  IMAD R53, R69, c[0x0][0x29c], R4                           ; // x         0  FMA
/*862460b0*/  LEA R54, P0, R2, c[0x0][0x280], 0x1                        ; // x         0  ALU
/*862460c0*/  IMAD R55, R69, c[0x0][0x2a4], R0                           ; // x         0  FMA
/*862460d0*/  IADD3 R53, R7, R53, RZ                                     ; // x         0  ALU
/*862460e0*/  IADD3 R55, R3, R55, RZ                                     ; // x         0  ALU
/*862460f0*/  LEA.HI.X R53, R6, c[0x0][0x27c], R53, 0x1, P1              ; // x         0  ALU
/*86246100*/  MOV R3, RZ                                                 ; // x         0  ALU
/*86246110*/  LEA.HI.X R55, R2, c[0x0][0x284], R55, 0x1, P0              ; // x         0  ALU
/*86246120*/  BRA 0x386246180                                            ; // x         0  branch
/*86246130*/  IADD3 R14, R69, 0x1, RZ                                    ; // x        48  ALU
/*86246140*/  IMAD R3, R69, c[0x0][0x184], RZ                            ; // x        48  FMA
/*86246150*/  ISETP.GE.AND P0, PT, R14, c[0x0][0x174], PT                ; // x        48  ALU
/*86246160*/  IMAD R14, R14, c[0x0][0x184], RZ                           ; // x        48  FMA
/*86246170*/  SEL R14, R14, c[0x0][0x168], !P0                           ; // x        48  ALU
/*86246180*/  S2R R2, SR_TID.X                                           ; // x        48  Branch
/*86246190*/  IMAD.IADD R9, R14, 0x1, -R3                                ; // x        48  FMA
/*862461a0*/  ULDC.64 UR4, c[0x0][0x1a0]                                 ; // x        48  Uniform
/*862461b0*/  CS2R R30, SRZ                                              ; // x        48  Branch
/*862461c0*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x        48  block barrier
/*862461d0*/  CS2R R28, SRZ                                              ; // x        48  Branch
/*862461e0*/  SHF.R.S32.HI R62, RZ, 0x1f, R9                             ; // x        48  ALU
/*862461f0*/  CS2R R34, SRZ                                              ; // x        48  Branch
/*86246200*/  CS2R R32, SRZ                                              ; // x        48  Branch
/*86246210*/  LEA.HI R62, R62, R9, RZ, 0x5                               ; // x        48  ALU
/*86246220*/  ULDC.64 UR8, c[0x0][0x1b0]                                 ; // x        48  Uniform
/*86246230*/  ULDC.64 UR6, c[0x0][0x1a0]                                 ; // x        48  Uniform
/*86246240*/  LOP3.LUT R62, R62, 0xffffffe0, RZ, 0xc0, !PT               ; // x        48  ALU
/*86246250*/  ULDC.64 UR12, c[0x0][0x1c0]                                ; // x        48  Uniform
/*86246260*/  ULDC.64 UR14, c[0x0][0x1a0]                                ; // x        48  Uniform
/*86246270*/  IMAD.IADD R62, R9, 0x1, -R62                               ; // x        48  FMA
/*86246280*/  SHF.R.S32.HI R7, RZ, 0x1f, R2                              ; // x        48  ALU
/*86246290*/  ISETP.NE.AND P0, PT, R62, RZ, PT                           ; // x        48  ALU
/*862462a0*/  LEA.HI R8, R7, R2, RZ, 0x5                                 ; // x        48  ALU
/*862462b0*/  SEL R62, R62, 0x20, P0                                     ; // x        48  ALU
/*862462c0*/  LOP3.LUT R7, R8, 0xffffffe0, RZ, 0xc0, !PT                 ; // x        48  ALU
/*862462d0*/  SHF.R.S32.HI R8, RZ, 0x5, R8                               ; // x        48  ALU
/*862462e0*/  IMAD.IADD R13, R62, 0x1, R3                                ; // x        48  FMA
/*862462f0*/  SHF.R.S32.HI R63, RZ, 0x1f, R62                            ; // x        48  ALU
/*86246300*/  IMAD.IADD R7, R2, 0x1, -R7                                 ; // x        48  FMA
/*86246310*/  IMAD.SHL.U32 R16, R8, 0x10, RZ                             ; // x        48  FMA
/*86246320*/  IMNMX R13, R13, R14, PT                                    ; // x        48  ALU
/*86246330*/  IMAD.SHL.U32 R8, R8, 0x8, RZ                               ; // x        48  FMA
/*86246340*/  SHF.R.S32.HI R0, RZ, 0x1f, R7                              ; // x        48  ALU
/*86246350*/  LEA.HI R9, R0, R7, RZ, 0x2                                 ; // x        48  ALU
/*86246360*/  LOP3.LUT R10, R9, 0xfffffffc, RZ, 0xc0, !PT                ; // x        48  ALU
/*86246370*/  LEA.HI.SX32 R16, R9, R16, 0x1e                             ; // x        48  ALU
/*86246380*/  LEA.HI R9, R0, R7, RZ, 0x3                                 ; // x        48  ALU
/*86246390*/  IMAD.IADD R10, R7, 0x1, -R10                               ; // x        48  FMA
/*862463a0*/  IMAD R60, R11, 0x40, R16                                   ; // x        48  FMA
/*862463b0*/  LOP3.LUT R4, R9, 0xfffffff8, RZ, 0xc0, !PT                 ; // x        48  ALU
/*862463c0*/  IMAD R64, R10, 0x8, R3                                     ; // x        48  FMA
/*862463d0*/  LEA.HI.SX32 R8, R9, R8, 0x1d                               ; // x        48  ALU
/*862463e0*/  ISETP.GE.AND P6, PT, R60, c[0x0][0x160], PT                ; // x        48  ALU
/*862463f0*/  IMAD.IADD R4, R7, 0x1, -R4                                 ; // x        48  FMA
/*86246400*/  IADD3 R12, R64, 0x2, RZ                                    ; // x        48  ALU
/*86246410*/  IMAD.IADD R56, R8, 0x1, R3                                 ; // x        48  FMA
/*86246420*/  IADD3 R0, R64, 0x4, RZ                                     ; // x        48  ALU
/*86246430*/  IMAD R5, R5, 0x8, R4                                       ; // x        48  FMA
/*86246440*/  IADD3 R6, R64, 0x6, RZ                                     ; // x        48  ALU
/*86246450*/  IADD3 R11, R60, 0x8, RZ                                    ; // x        48  ALU
/*86246460*/  IMAD.SHL.U32 R58, R5, 0x8, RZ                              ; // x        48  FMA
/*86246470*/  ISETP.LT.AND P0, PT, R64, R13, !P6                         ; // x        48  ALU
/*86246480*/  ISETP.LT.AND P1, PT, R12, R13, !P6                         ; // x        48  ALU
/*86246490*/  ISETP.LT.AND P2, PT, R0, R13, !P6                          ; // x        48  ALU
/*862464a0*/  ISETP.LT.AND P3, PT, R6, R13, !P6                          ; // x        48  ALU
/*862464b0*/  ISETP.GE.AND P5, PT, R11, c[0x0][0x160], PT                ; // x        48  ALU
/*862464c0*/  P2R R26, PR, RZ, 0xf                                       ; // x        48  ALU
/*862464d0*/  ISETP.LT.AND P1, PT, R12, R13, !P5                         ; // x        48  ALU
/*862464e0*/  LEA.HI R12, R16, R16, RZ, 0x1                              ; // x        48  ALU
/*862464f0*/  ISETP.LT.AND P2, PT, R0, R13, !P5                          ; // x        48  ALU
/*86246500*/  ISETP.LT.AND P3, PT, R6, R13, !P5                          ; // x        48  ALU
/*86246510*/  ISETP.LT.AND P0, PT, R64, R13, !P5                         ; // x        48  ALU
/*86246520*/  ISETP.GE.AND P4, PT, R56, R13, PT                          ; // x        48  ALU
/*86246530*/  IADD3 R7, R58, 0x2, RZ                                     ; // x        48  ALU
/*86246540*/  IADD3 R6, R58, 0x4, RZ                                     ; // x        48  ALU
/*86246550*/  IADD3 R5, R58, 0x6, RZ                                     ; // x        48  ALU
/*86246560*/  IADD3 R0, R14, 0x1f, -R3                                   ; // x        48  ALU
/*86246570*/  LOP3.LUT R9, R12, 0x3ffffffe, RZ, 0xc0, !PT                ; // x        48  ALU
/*86246580*/  P2R.B1 R26, PR, R26, 0xf                                   ; // x        48  ALU
/*86246590*/  ISETP.LT.AND P0, PT, R58, c[0x0][0x164], !P4               ; // x        48  ALU
/*862465a0*/  IMAD.IADD R9, R16, 0x1, -R9                                ; // x        48  FMA
/*862465b0*/  ISETP.LT.AND P1, PT, R7, c[0x0][0x164], !P4                ; // x        48  ALU
/*862465c0*/  ISETP.LT.AND P2, PT, R6, c[0x0][0x164], !P4                ; // x        48  ALU
/*862465d0*/  IMAD R9, R9, 0x4, R10                                      ; // x        48  FMA
/*862465e0*/  ISETP.LT.AND P3, PT, R5, c[0x0][0x164], !P4                ; // x        48  ALU
/*862465f0*/  IADD3 R3, R0, 0x1f, RZ                                     ; // x        48  ALU
/*86246600*/  IADD3 R14, R16, 0x8, RZ                                    ; // x        48  ALU
/*86246610*/  P2R R22, PR, RZ, 0xf                                       ; // x        48  ALU
/*86246620*/  ISETP.GT.U32.AND P4, PT, R3, 0x3e, PT                      ; // x        48  ALU
/*86246630*/  ISETP.GE.U32.AND P0, PT, R3, 0x3f, PT                      ; // x        48  ALU
/*86246640*/  LEA.HI R3, R14, R14, RZ, 0x1                               ; // x        48  ALU
/*86246650*/  SHF.R.S32.HI R57, RZ, 0x1f, R60                            ; // x        48  ALU
/*86246660*/  LOP3.LUT R27, R3, 0x3ffffffe, RZ, 0xc0, !PT                ; // x        48  ALU
/*86246670*/  SHF.R.S32.HI R19, RZ, 0x1f, R4                             ; // x        48  ALU
/*86246680*/  IMAD R57, R57, c[0x0][0x198], RZ                           ; // x        48  FMA
/*86246690*/  SHF.R.S32.HI R23, RZ, 0x1, R12                             ; // x        48  ALU
/*862466a0*/  IMAD.IADD R27, R14, 0x1, -R27                              ; // x        48  FMA
/*862466b0*/  SHF.R.S32.HI R65, RZ, 0x1f, R64                            ; // x        48  ALU
/*862466c0*/  IMAD R57, R60, c[0x0][0x19c], R57                          ; // x        48  FMA
/*862466d0*/  IADD3 R11, R56, 0x4, RZ                                    ; // x        48  ALU
/*862466e0*/  IMAD R27, R27, 0x4, R10                                    ; // x        48  FMA
/*862466f0*/  SHF.R.S32.HI R12, RZ, 0x1f, R12                            ; // x        48  ALU
/*86246700*/  IMAD.WIDE.U32 R14, R60, c[0x0][0x198], R64                 ; // x        48  FMA
/*86246710*/  SHF.R.S32.HI R16, RZ, 0x1f, R9                             ; // x        48  ALU
/*86246720*/  LEA.HI R19, R19, R4, RZ, 0x2                               ; // x        48  ALU
/*86246730*/  IMAD.IADD R10, R15, 0x1, R57                               ; // x        48  FMA
/*86246740*/  ISETP.GE.AND P3, PT, R11, R13, PT                          ; // x        48  ALU
/*86246750*/  LEA.HI R12, R12, R23, RZ, 0x2                              ; // x        48  ALU
/*86246760*/  LEA.HI R16, R16, R9, RZ, 0x2                               ; // x        48  ALU
/*86246770*/  SHF.R.S32.HI R13, RZ, 0x1f, R8                             ; // x        48  ALU
/*86246780*/  LOP3.LUT R17, R19, 0xfffffffc, RZ, 0xc0, !PT               ; // x        48  ALU
/*86246790*/  LOP3.LUT R12, R12, 0x1ffffffc, RZ, 0xc0, !PT               ; // x        48  ALU
/*862467a0*/  LOP3.LUT R16, R16, 0xfffffffc, RZ, 0xc0, !PT               ; // x        48  ALU
/*862467b0*/  IMAD.IADD R17, R4, 0x1, -R17                               ; // x        48  FMA
/*862467c0*/  LEA R24, P1, R14, R52, 0x1                                 ; // x        48  ALU
/*862467d0*/  IMAD.IADD R12, R23, 0x1, -R12                              ; // x        48  FMA
/*862467e0*/  LEA.HI R13, R13, R8, RZ, 0x3                               ; // x        48  ALU
/*862467f0*/  IMAD.IADD R9, R9, 0x1, -R16                                ; // x        48  FMA
/*86246800*/  IADD3 R4, R8, 0x4, RZ                                      ; // x        48  ALU
/*86246810*/  LEA.HI.X R25, R14, R53, R10, 0x1, P1                       ; // x        48  ALU
/*86246820*/  LOP3.LUT R13, R13, 0xfffffff8, RZ, 0xc0, !PT               ; // x        48  ALU
/*86246830*/  SHF.R.S32.HI R21, RZ, 0x1, R3                              ; // x        48  ALU
/*86246840*/  SHF.R.S32.HI R10, RZ, 0x1f, R3                             ; // x        48  ALU
/*86246850*/  IMAD.IADD R20, R8, 0x1, -R13                               ; // x        48  FMA
/*86246860*/  SHF.R.S32.HI R14, RZ, 0x1f, R27                            ; // x        48  ALU
/*86246870*/  SHF.R.S32.HI R11, RZ, 0x1f, R4                             ; // x        48  ALU
/*86246880*/  LOP3.LUT R9, R9, R12, RZ, 0x3c, !PT                        ; // x        48  ALU
/*86246890*/  LEA.HI R10, R10, R21, RZ, 0x2                              ; // x        48  ALU
/*862468a0*/  LEA.HI R14, R14, R27, RZ, 0x2                              ; // x        48  ALU
/*862468b0*/  IMAD.IADD R16, R16, 0x1, R9                                ; // x        48  FMA
/*862468c0*/  LEA.HI R11, R11, R4, RZ, 0x3                               ; // x        48  ALU
/*862468d0*/  LOP3.LUT R10, R10, 0x1ffffffc, RZ, 0xc0, !PT               ; // x        48  ALU
/*862468e0*/  IMAD R16, R23, 0x30, R16                                   ; // x        48  FMA
/*862468f0*/  LOP3.LUT R14, R14, 0xfffffffc, RZ, 0xc0, !PT               ; // x        48  ALU
/*86246900*/  IMAD.MOV.U32 R23, RZ, RZ, RZ                               ; // x        48  FMA
/*86246910*/  SHF.R.S32.HI R13, RZ, 0x1f, R20                            ; // x        48  ALU
/*86246920*/  IMAD.IADD R15, R21, 0x1, -R10                              ; // x        48  FMA
/*86246930*/  LOP3.LUT R11, R11, 0xfffffff8, RZ, 0xc0, !PT               ; // x        48  ALU
/*86246940*/  IMAD.IADD R12, R27, 0x1, -R14                              ; // x        48  FMA
/*86246950*/  SEL R26, R26, RZ, P0                                       ; // x        48  ALU
/*86246960*/  IMAD.SHL.U32 R16, R16, 0x8, RZ                             ; // x        48  FMA
/*86246970*/  ISETP.LT.AND P0, PT, R58, c[0x0][0x164], !P3               ; // x        48  ALU
```

(first 200 of 1416 instructions shown; the rest are in the full dump)
