# Hot loop — `triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_9` (sm87)

PC range 0x2297200-0x2297af0; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_9_sm87.sass`; per-PC counts: `raw/triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_9_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*02297200*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x       400  ALU
/*02297210*/  @P0   BRA 0x202297340                                      ; // x       400  branch
/*02297220*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*02297230*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*02297240*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*02297250*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*02297260*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*02297270*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*02297280*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*02297290*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*022972a0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*022972b0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*022972c0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*022972d0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*022972e0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*022972f0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*02297300*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*02297310*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*02297320*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*02297330*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*02297340*/  MOV R1, c[0x0][0x28]                                       ; // x       400  ALU
/*02297350*/  S2R R2, SR_TID.X                                           ; // x       400  Branch
/*02297360*/  HFMA2.MMA R33, -RZ, RZ, 0, 2.384185791015625e-07           ; // x       400  FMA
/*02297370*/  UMOV UR4, 0x0                                              ; // x       400  Uniform
/*02297380*/  S2R R3, SR_CTAID.X                                         ; // x       400  Branch
/*02297390*/  UMOV UR5, 0x14f00000                                       ; // x       400  Uniform
/*022973a0*/  ULDC.64 UR6, c[0x0][0x118]                                 ; // x       400  Uniform
/*022973b0*/  SHF.L.U32 R2, R2, 0x2, RZ                                  ; // x       400  ALU
/*022973c0*/  LOP3.LUT R2, R2, 0x3fc, RZ, 0xc0, !PT                      ; // x       400  ALU
/*022973d0*/  LEA R3, R3, R2, 0xa                                        ; // x       400  ALU
/*022973e0*/  IMAD.WIDE.U32 R6, R2, R33, c[0x0][0x170]                   ; // x       400  FMA
/*022973f0*/  IMAD.WIDE.U32 R8, R2, R33, c[0x0][0x178]                   ; // x       400  FMA
/*02297400*/  LDG.E.EL.128 R4desc, [UR4][R6.64+0x2000]                   ; // x       400  global load (DRAM/L2: weights or activations)
/*02297410*/  IMAD.WIDE.U32 R12, R2, R33, c[0x0][0x188]                  ; // x       400  FMA
/*02297420*/  LDG.E.EL.128 R8desc, [UR4][R8.64+0x2000]                   ; // x       400  global load (DRAM/L2: weights or activations)
/*02297430*/  IMAD.WIDE.U32 R24, R2, R33, c[0x0][0x190]                  ; // x       400  FMA
/*02297440*/  LDG.E.EL.128 R12desc, [UR4][R12.64+0x2000]                 ; // x       400  global load (DRAM/L2: weights or activations)
/*02297450*/  IMAD.WIDE R16, R3, R33, c[0x0][0x168]                      ; // x       400  FMA
/*02297460*/  LDG.E.EL.128 R24desc, [UR4][R24.64+0x2000]                 ; // x       400  global load (DRAM/L2: weights or activations)
/*02297470*/  IMAD.WIDE R20, R3, R33, c[0x0][0x160]                      ; // x       400  FMA
/*02297480*/  LDG.E.128 R16, [R16.64]                                    ; // x       400  global load (DRAM/L2: weights or activations)
/*02297490*/  IMAD.WIDE R30, R3, R33, c[0x0][0x180]                      ; // x       400  FMA
/*022974a0*/  LDG.E.128 R20, [R20.64]                                    ; // x       400  global load (DRAM/L2: weights or activations)
/*022974b0*/  LDG.E.128 R28, [R30.64]                                    ; // x       400  global load (DRAM/L2: weights or activations)
/*022974c0*/  S2R R34, SR_TID.X                                          ; // x       400  Branch
/*022974d0*/  LOP3.LUT P0, RZ, R34, 0x1f, RZ, 0xc0, !PT                  ; // x       400  ALU
/*022974e0*/  ISETP.GE.U32.AND P1, PT, R34, 0x8, PT                      ; // x       400  ALU
/*022974f0*/  FADD R32, R5, R9                                           ; // x       400  FMA
/*02297500*/  FADD R5, R4, R8                                            ; // x       400  FMA
/*02297510*/  FADD R4, R7, R11                                           ; // x       400  FMA
/*02297520*/  FADD R9, R6, R10                                           ; // x       400  FMA
/*02297530*/  FADD R7, R13, R25                                          ; // x       400  FMA
/*02297540*/  FADD R12, R12, R24                                         ; // x       400  FMA
/*02297550*/  FFMA R32, R21, R32, R17                                    ; // x       400  FMA
/*02297560*/  FFMA R5, R20, R5, R16                                      ; // x       400  FMA
/*02297570*/  FFMA R29, R29, R7, R32                                     ; // x       400  FMA
/*02297580*/  FFMA R28, R28, R12, R5                                     ; // x       400  FMA
/*02297590*/  FFMA R9, R22, R9, R18                                      ; // x       400  FMA
/*022975a0*/  FADD R14, R14, R26                                         ; // x       400  FMA
/*022975b0*/  FMUL R5, R29, R29                                          ; // x       400  FMA
/*022975c0*/  FFMA R4, R23, R4, R19                                      ; // x       400  FMA
/*022975d0*/  FADD R15, R15, R27                                         ; // x       400  FMA
/*022975e0*/  FFMA R30, R30, R14, R9                                     ; // x       400  FMA
/*022975f0*/  FFMA R5, R28, R28, R5                                      ; // x       400  FMA
/*02297600*/  FFMA R31, R31, R15, R4                                     ; // x       400  FMA
/*02297610*/  FFMA R4, R30, R30, R5                                      ; // x       400  FMA
/*02297620*/  FFMA R21, R31, R31, R4                                     ; // x       400  FMA
/*02297630*/  SHFL.BFLY PT, R4, R21, 0x10, 0x1f                          ; // x       400  warp shuffle
/*02297640*/  HFMA2.MMA R20, -RZ, RZ, 0, 2.384185791015625e-07           ; // x       400  FMA
/*02297650*/  IMAD.WIDE.U32 R22, R2, R33, c[0x0][0x198]                  ; // x       400  FMA
/*02297660*/  LDG.E.EL.128 R12desc, [UR4][R22.64]                        ; // x       400  global load (DRAM/L2: weights or activations)
/*02297670*/  IMAD.WIDE.U32 R24, R2, R20, c[0x0][0x1a0]                  ; // x       400  FMA
/*02297680*/  LDG.E.EL.128 R8desc, [UR4][R22.64+0x1000]                  ; // x       400  global load (DRAM/L2: weights or activations)
/*02297690*/  LDG.E.EL.128 R16desc, [UR4][R24.64]                        ; // x       400  global load (DRAM/L2: weights or activations)
/*022976a0*/  FADD R2, R21, R4                                           ; // x       400  FMA
/*022976b0*/  LDG.E.EL.128 R4desc, [UR4][R24.64+0x1000]                  ; // x       400  global load (DRAM/L2: weights or activations)
/*022976c0*/  SHFL.BFLY PT, R27, R2, 0x8, 0x1f                           ; // x       400  warp shuffle
/*022976d0*/  FADD R27, R2, R27                                          ; // x       400  FMA
/*022976e0*/  SHFL.BFLY PT, R26, R27, 0x4, 0x1f                          ; // x       400  warp shuffle
/*022976f0*/  FADD R26, R27, R26                                         ; // x       400  FMA
/*02297700*/  SHFL.BFLY PT, R21, R26, 0x2, 0x1f                          ; // x       400  warp shuffle
/*02297710*/  FADD R21, R26, R21                                         ; // x       400  FMA
/*02297720*/  SHFL.BFLY PT, R32, R21, 0x1, 0x1f                          ; // x       400  warp shuffle
/*02297730*/  SHF.R.U32.HI R2, RZ, 0x3, R34                              ; // x       400  ALU
/*02297740*/  LOP3.LUT R23, R2, 0x1c, RZ, 0xc0, !PT                      ; // x       400  ALU
/*02297750*/  FADD R32, R21, R32                                         ; // x       400  FMA
/*02297760*/  @!P0  STS [R23], R32                                       ; // x       400  shared-memory store
/*02297770*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       400  block barrier
/*02297780*/  @!P1  LDS R0, [R34.X4]                                     ; // x       400  shared-memory load
/*02297790*/  SHFL.BFLY PT, R25, R0, 0x4, 0x1f                           ; // x       400  warp shuffle
/*022977a0*/  FADD R25, R0, R25                                          ; // x       400  FMA
/*022977b0*/  SHFL.BFLY PT, R2, R25, 0x2, 0x1f                           ; // x       400  warp shuffle
/*022977c0*/  ISETP.NE.AND P0, PT, R34, RZ, PT                           ; // x       400  ALU
/*022977d0*/  FADD R2, R25, R2                                           ; // x       400  FMA
/*022977e0*/  SHFL.BFLY PT, R21, R2, 0x1, 0x1f                           ; // x       400  warp shuffle
/*022977f0*/  FADD R21, R2, R21                                          ; // x       400  FMA
/*02297800*/  @!P0  STS [R34.X4], R21                                    ; // x       400  shared-memory store
/*02297810*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       400  block barrier
/*02297820*/  LDS R22, [RZ]                                              ; // x       400  shared-memory load
/*02297830*/  MOV R23, 0x3a800000                                        ; // x       400  ALU
/*02297840*/  FADD R12, R12, R16                                         ; // x       400  FMA
/*02297850*/  FADD R15, R15, R19                                         ; // x       400  FMA
/*02297860*/  FADD R13, R13, R17                                         ; // x       400  FMA
/*02297870*/  FADD R14, R14, R18                                         ; // x       400  FMA
/*02297880*/  FADD R19, R10, R6                                          ; // x       400  FMA
/*02297890*/  FADD R17, R8, R4                                           ; // x       400  FMA
/*022978a0*/  FADD R16, R9, R5                                           ; // x       400  FMA
/*022978b0*/  FADD R10, R11, R7                                          ; // x       400  FMA
/*022978c0*/  FADD R9, R12, 1                                            ; // x       400  FMA
/*022978d0*/  FADD R8, R15, 1                                            ; // x       400  FMA
/*022978e0*/  FADD R2, R13, 1                                            ; // x       400  FMA
/*022978f0*/  FADD R22, RZ, R22                                          ; // x       400  FMA
/*02297900*/  FFMA R0, R22, R23, 9.9999999747524270788e-07               ; // x       400  FMA
/*02297910*/  MUFU.RSQ R0, R0                                            ; // x       400  special-function unit
/*02297920*/  FADD R11, R14, 1                                           ; // x       400  FMA
/*02297930*/  FMUL R4, R28, R0                                           ; // x       400  FMA
/*02297940*/  FMUL R7, R31, R0                                           ; // x       400  FMA
/*02297950*/  FMUL R5, R29, R0                                           ; // x       400  FMA
/*02297960*/  FMUL R6, R30, R0                                           ; // x       400  FMA
/*02297970*/  FFMA R4, R4, R9, R17                                       ; // x       400  FMA
/*02297980*/  FFMA R7, R7, R8, R10                                       ; // x       400  FMA
/*02297990*/  IMAD.WIDE R8, R3, R20, c[0x0][0x160]                       ; // x       400  FMA
/*022979a0*/  FFMA R5, R5, R2, R16                                       ; // x       400  FMA
/*022979b0*/  FFMA R6, R6, R11, R19                                      ; // x       400  FMA
/*022979c0*/  IMAD.WIDE R20, R3, R20, c[0x0][0x1a8]                      ; // x       400  FMA
/*022979d0*/  STG.E.128 [R8.64], R28                                     ; // x       400  global store
/*022979e0*/  STG.E.128 [R20.64], R4                                     ; // x       400  global store
/*022979f0*/  EXIT                                                       ; // x       400  Branch
/*02297a00*/  BRA 0x202297a00                                            ; // x         0  branch
/*02297a10*/  NOP                                                        ; // x         0  Branch
/*02297a20*/  NOP                                                        ; // x         0  Branch
/*02297a30*/  NOP                                                        ; // x         0  Branch
/*02297a40*/  NOP                                                        ; // x         0  Branch
/*02297a50*/  NOP                                                        ; // x         0  Branch
/*02297a60*/  NOP                                                        ; // x         0  Branch
/*02297a70*/  NOP                                                        ; // x         0  Branch
/*02297a80*/  NOP                                                        ; // x         0  Branch
/*02297a90*/  NOP                                                        ; // x         0  Branch
/*02297aa0*/  NOP                                                        ; // x         0  Branch
/*02297ab0*/  NOP                                                        ; // x         0  Branch
/*02297ac0*/  NOP                                                        ; // x         0  Branch
/*02297ad0*/  NOP                                                        ; // x         0  Branch
/*02297ae0*/  NOP                                                        ; // x         0  Branch
/*02297af0*/  NOP                                                        ; // x         0  Branch
```
