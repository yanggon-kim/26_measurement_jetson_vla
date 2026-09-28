# Hot loop — `triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_7` (sm87)

PC range 0x2295300-0x2295a70; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_7_sm87.sass`; per-PC counts: `raw/triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_7_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*02295300*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x       400  ALU
/*02295310*/  @P0   BRA 0x202295440                                      ; // x       400  branch
/*02295320*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*02295330*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*02295340*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*02295350*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*02295360*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*02295370*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*02295380*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*02295390*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*022953a0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*022953b0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*022953c0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*022953d0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*022953e0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*022953f0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*02295400*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*02295410*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*02295420*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*02295430*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*02295440*/  MOV R1, c[0x0][0x28]                                       ; // x       400  ALU
/*02295450*/  S2R R2, SR_TID.X                                           ; // x       400  Branch
/*02295460*/  HFMA2.MMA R3, -RZ, RZ, 0, 2.384185791015625e-07            ; // x       400  FMA
/*02295470*/  UMOV UR4, 0x0                                              ; // x       400  Uniform
/*02295480*/  S2R R5, SR_CTAID.X                                         ; // x       400  Branch
/*02295490*/  UMOV UR5, 0x14f00000                                       ; // x       400  Uniform
/*022954a0*/  ULDC.64 UR6, c[0x0][0x118]                                 ; // x       400  Uniform
/*022954b0*/  SHF.L.U32 R22, R2, 0x2, RZ                                 ; // x       400  ALU
/*022954c0*/  LOP3.LUT R22, R22, 0x3fc, RZ, 0xc0, !PT                    ; // x       400  ALU
/*022954d0*/  LEA R20, R5, R22, 0xa                                      ; // x       400  ALU
/*022954e0*/  IMAD.WIDE.U32 R12, R22, R3, c[0x0][0x170]                  ; // x       400  FMA
/*022954f0*/  IMAD.WIDE.U32 R16, R22, R3, c[0x0][0x178]                  ; // x       400  FMA
/*02295500*/  LDG.E.EL.128 R12desc, [UR4][R12.64+0x2000]                 ; // x       400  global load (DRAM/L2: weights or activations)
/*02295510*/  IMAD.WIDE R4, R20, R3, c[0x0][0x160]                       ; // x       400  FMA
/*02295520*/  LDG.E.EL.128 R16desc, [UR4][R16.64+0x2000]                 ; // x       400  global load (DRAM/L2: weights or activations)
/*02295530*/  IMAD.WIDE R8, R20, R3, c[0x0][0x168]                       ; // x       400  FMA
/*02295540*/  LDG.E.128 R4, [R4.64]                                      ; // x       400  global load (DRAM/L2: weights or activations)
/*02295550*/  LDG.E.128 R8, [R8.64]                                      ; // x       400  global load (DRAM/L2: weights or activations)
/*02295560*/  IMAD.WIDE.U32 R26, R22, R3, c[0x0][0x180]                  ; // x       400  FMA
/*02295570*/  IMAD.WIDE.U32 R28, R22, R3, c[0x0][0x188]                  ; // x       400  FMA
/*02295580*/  LOP3.LUT P0, RZ, R2, 0x1f, RZ, 0xc0, !PT                   ; // x       400  ALU
/*02295590*/  ISETP.GE.U32.AND P1, PT, R2, 0x8, PT                       ; // x       400  ALU
/*022955a0*/  FADD R24, R13, R17                                         ; // x       400  FMA
/*022955b0*/  FADD R23, R12, R16                                         ; // x       400  FMA
/*022955c0*/  FADD R25, R14, R18                                         ; // x       400  FMA
/*022955d0*/  FFMA R21, R9, R24, R5                                      ; // x       400  FMA
/*022955e0*/  FFMA R23, R8, R23, R4                                      ; // x       400  FMA
/*022955f0*/  FMUL R14, R21, R21                                         ; // x       400  FMA
/*02295600*/  FADD R12, R15, R19                                         ; // x       400  FMA
/*02295610*/  FFMA R24, R10, R25, R6                                     ; // x       400  FMA
/*02295620*/  LDG.E.EL.128 R16desc, [UR4][R28.64+0x1000]                 ; // x       400  global load (DRAM/L2: weights or activations)
/*02295630*/  FFMA R13, R23, R23, R14                                    ; // x       400  FMA
/*02295640*/  FFMA R25, R11, R12, R7                                     ; // x       400  FMA
/*02295650*/  LDG.E.EL.128 R8desc, [UR4][R28.64]                         ; // x       400  global load (DRAM/L2: weights or activations)
/*02295660*/  FFMA R4, R24, R24, R13                                     ; // x       400  FMA
/*02295670*/  LDG.E.EL.128 R12desc, [UR4][R26.64+0x1000]                 ; // x       400  global load (DRAM/L2: weights or activations)
/*02295680*/  FFMA R30, R25, R25, R4                                     ; // x       400  FMA
/*02295690*/  SHFL.BFLY PT, R5, R30, 0x10, 0x1f                          ; // x       400  warp shuffle
/*022956a0*/  FADD R31, R30, R5                                          ; // x       400  FMA
/*022956b0*/  LDG.E.EL.128 R4desc, [UR4][R26.64]                         ; // x       400  global load (DRAM/L2: weights or activations)
/*022956c0*/  SHFL.BFLY PT, R22, R31, 0x8, 0x1f                          ; // x       400  warp shuffle
/*022956d0*/  FADD R22, R31, R22                                         ; // x       400  FMA
/*022956e0*/  SHFL.BFLY PT, R33, R22, 0x4, 0x1f                          ; // x       400  warp shuffle
/*022956f0*/  FADD R33, R22, R33                                         ; // x       400  FMA
/*02295700*/  SHFL.BFLY PT, R30, R33, 0x2, 0x1f                          ; // x       400  warp shuffle
/*02295710*/  FADD R30, R33, R30                                         ; // x       400  FMA
/*02295720*/  SHFL.BFLY PT, R35, R30, 0x1, 0x1f                          ; // x       400  warp shuffle
/*02295730*/  SHF.R.U32.HI R31, RZ, 0x3, R2                              ; // x       400  ALU
/*02295740*/  LOP3.LUT R26, R31, 0x1c, RZ, 0xc0, !PT                     ; // x       400  ALU
/*02295750*/  FADD R35, R30, R35                                         ; // x       400  FMA
/*02295760*/  @!P0  STS [R26], R35                                       ; // x       400  shared-memory store
/*02295770*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       400  block barrier
/*02295780*/  @!P1  LDS R0, [R2.X4]                                      ; // x       400  shared-memory load
/*02295790*/  SHFL.BFLY PT, R27, R0, 0x4, 0x1f                           ; // x       400  warp shuffle
/*022957a0*/  FADD R27, R0, R27                                          ; // x       400  FMA
/*022957b0*/  SHFL.BFLY PT, R22, R27, 0x2, 0x1f                          ; // x       400  warp shuffle
/*022957c0*/  ISETP.NE.AND P0, PT, R2, RZ, PT                            ; // x       400  ALU
/*022957d0*/  FADD R22, R27, R22                                         ; // x       400  FMA
/*022957e0*/  SHFL.BFLY PT, R29, R22, 0x1, 0x1f                          ; // x       400  warp shuffle
/*022957f0*/  FADD R29, R22, R29                                         ; // x       400  FMA
/*02295800*/  @!P0  STS [R2.X4], R29                                     ; // x       400  shared-memory store
/*02295810*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       400  block barrier
/*02295820*/  LDS R26, [RZ]                                              ; // x       400  shared-memory load
/*02295830*/  HFMA2.MMA R31, -RZ, RZ, 0.8125, 0                          ; // x       400  FMA
/*02295840*/  FADD R9, R5, R9                                            ; // x       400  FMA
/*02295850*/  FADD R8, R4, R8                                            ; // x       400  FMA
/*02295860*/  FADD R10, R6, R10                                          ; // x       400  FMA
/*02295870*/  FADD R11, R7, R11                                          ; // x       400  FMA
/*02295880*/  FADD R27, R12, R16                                         ; // x       400  FMA
/*02295890*/  FADD R12, R13, R17                                         ; // x       400  FMA
/*022958a0*/  FADD R17, R14, R18                                         ; // x       400  FMA
/*022958b0*/  FADD R0, R9, 1                                             ; // x       400  FMA
/*022958c0*/  FADD R14, R15, R19                                         ; // x       400  FMA
/*022958d0*/  FADD R13, R8, 1                                            ; // x       400  FMA
/*022958e0*/  FADD R9, R10, 1                                            ; // x       400  FMA
/*022958f0*/  FADD R2, R11, 1                                            ; // x       400  FMA
/*02295900*/  FADD R26, RZ, R26                                          ; // x       400  FMA
/*02295910*/  FFMA R26, R26, R31, 9.9999999747524270788e-07              ; // x       400  FMA
/*02295920*/  MUFU.RSQ R26, R26                                          ; // x       400  special-function unit
/*02295930*/  FMUL R4, R23, R26                                          ; // x       400  FMA
/*02295940*/  FMUL R5, R21, R26                                          ; // x       400  FMA
/*02295950*/  FMUL R6, R24, R26                                          ; // x       400  FMA
/*02295960*/  FMUL R7, R25, R26                                          ; // x       400  FMA
/*02295970*/  FFMA R4, R4, R13, R27                                      ; // x       400  FMA
/*02295980*/  FFMA R5, R5, R0, R12                                       ; // x       400  FMA
/*02295990*/  FFMA R6, R6, R9, R17                                       ; // x       400  FMA
/*022959a0*/  FFMA R7, R7, R2, R14                                       ; // x       400  FMA
/*022959b0*/  IMAD.WIDE R20, R20, R3, c[0x0][0x190]                      ; // x       400  FMA
/*022959c0*/  STG.E.128 [R20.64], R4                                     ; // x       400  global store
/*022959d0*/  EXIT                                                       ; // x       400  Branch
/*022959e0*/  BRA 0x2022959e0                                            ; // x         0  branch
/*022959f0*/  NOP                                                        ; // x         0  Branch
/*02295a00*/  NOP                                                        ; // x         0  Branch
/*02295a10*/  NOP                                                        ; // x         0  Branch
/*02295a20*/  NOP                                                        ; // x         0  Branch
/*02295a30*/  NOP                                                        ; // x         0  Branch
/*02295a40*/  NOP                                                        ; // x         0  Branch
/*02295a50*/  NOP                                                        ; // x         0  Branch
/*02295a60*/  NOP                                                        ; // x         0  Branch
/*02295a70*/  NOP                                                        ; // x         0  Branch
```
