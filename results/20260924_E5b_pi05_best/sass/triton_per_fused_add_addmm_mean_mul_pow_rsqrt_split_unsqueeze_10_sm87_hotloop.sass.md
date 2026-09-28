# Hot loop — `triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_10` (sm87)

PC range 0x2298400-0x2298cf0; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_10_sm87.sass`; per-PC counts: `raw/triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_10_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*02298400*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x       400  ALU
/*02298410*/  @P0   BRA 0x202298540                                      ; // x       400  branch
/*02298420*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*02298430*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*02298440*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*02298450*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*02298460*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*02298470*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*02298480*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*02298490*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*022984a0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*022984b0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*022984c0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*022984d0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*022984e0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*022984f0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*02298500*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*02298510*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*02298520*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*02298530*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*02298540*/  MOV R1, c[0x0][0x28]                                       ; // x       400  ALU
/*02298550*/  S2R R2, SR_TID.X                                           ; // x       400  Branch
/*02298560*/  HFMA2.MMA R33, -RZ, RZ, 0, 2.384185791015625e-07           ; // x       400  FMA
/*02298570*/  UMOV UR4, 0x0                                              ; // x       400  Uniform
/*02298580*/  S2R R3, SR_CTAID.X                                         ; // x       400  Branch
/*02298590*/  UMOV UR5, 0x14f00000                                       ; // x       400  Uniform
/*022985a0*/  ULDC.64 UR6, c[0x0][0x118]                                 ; // x       400  Uniform
/*022985b0*/  SHF.L.U32 R2, R2, 0x2, RZ                                  ; // x       400  ALU
/*022985c0*/  LOP3.LUT R2, R2, 0x3fc, RZ, 0xc0, !PT                      ; // x       400  ALU
/*022985d0*/  LEA R3, R3, R2, 0xa                                        ; // x       400  ALU
/*022985e0*/  IMAD.WIDE.U32 R6, R2, R33, c[0x0][0x170]                   ; // x       400  FMA
/*022985f0*/  IMAD.WIDE.U32 R8, R2, R33, c[0x0][0x178]                   ; // x       400  FMA
/*02298600*/  LDG.E.EL.128 R4desc, [UR4][R6.64+0x2000]                   ; // x       400  global load (DRAM/L2: weights or activations)
/*02298610*/  IMAD.WIDE.U32 R12, R2, R33, c[0x0][0x188]                  ; // x       400  FMA
/*02298620*/  LDG.E.EL.128 R8desc, [UR4][R8.64+0x2000]                   ; // x       400  global load (DRAM/L2: weights or activations)
/*02298630*/  IMAD.WIDE.U32 R24, R2, R33, c[0x0][0x190]                  ; // x       400  FMA
/*02298640*/  LDG.E.EL.128 R12desc, [UR4][R12.64+0x2000]                 ; // x       400  global load (DRAM/L2: weights or activations)
/*02298650*/  IMAD.WIDE R16, R3, R33, c[0x0][0x160]                      ; // x       400  FMA
/*02298660*/  LDG.E.EL.128 R24desc, [UR4][R24.64+0x2000]                 ; // x       400  global load (DRAM/L2: weights or activations)
/*02298670*/  IMAD.WIDE R20, R3, R33, c[0x0][0x168]                      ; // x       400  FMA
/*02298680*/  LDG.E.128 R16, [R16.64]                                    ; // x       400  global load (DRAM/L2: weights or activations)
/*02298690*/  IMAD.WIDE R30, R3, R33, c[0x0][0x180]                      ; // x       400  FMA
/*022986a0*/  LDG.E.128 R20, [R20.64]                                    ; // x       400  global load (DRAM/L2: weights or activations)
/*022986b0*/  LDG.E.128 R28, [R30.64]                                    ; // x       400  global load (DRAM/L2: weights or activations)
/*022986c0*/  S2R R34, SR_TID.X                                          ; // x       400  Branch
/*022986d0*/  LOP3.LUT P0, RZ, R34, 0x1f, RZ, 0xc0, !PT                  ; // x       400  ALU
/*022986e0*/  ISETP.GE.U32.AND P1, PT, R34, 0x8, PT                      ; // x       400  ALU
/*022986f0*/  FADD R32, R5, R9                                           ; // x       400  FMA
/*02298700*/  FADD R5, R4, R8                                            ; // x       400  FMA
/*02298710*/  FADD R4, R7, R11                                           ; // x       400  FMA
/*02298720*/  FADD R9, R6, R10                                           ; // x       400  FMA
/*02298730*/  FADD R7, R13, R25                                          ; // x       400  FMA
/*02298740*/  FADD R12, R12, R24                                         ; // x       400  FMA
/*02298750*/  FFMA R32, R21, R32, R17                                    ; // x       400  FMA
/*02298760*/  FFMA R5, R20, R5, R16                                      ; // x       400  FMA
/*02298770*/  FFMA R29, R29, R7, R32                                     ; // x       400  FMA
/*02298780*/  FFMA R28, R28, R12, R5                                     ; // x       400  FMA
/*02298790*/  FFMA R9, R22, R9, R18                                      ; // x       400  FMA
/*022987a0*/  FADD R14, R14, R26                                         ; // x       400  FMA
/*022987b0*/  FMUL R5, R29, R29                                          ; // x       400  FMA
/*022987c0*/  FFMA R4, R23, R4, R19                                      ; // x       400  FMA
/*022987d0*/  FADD R15, R15, R27                                         ; // x       400  FMA
/*022987e0*/  FFMA R30, R30, R14, R9                                     ; // x       400  FMA
/*022987f0*/  FFMA R5, R28, R28, R5                                      ; // x       400  FMA
/*02298800*/  FFMA R31, R31, R15, R4                                     ; // x       400  FMA
/*02298810*/  FFMA R4, R30, R30, R5                                      ; // x       400  FMA
/*02298820*/  FFMA R21, R31, R31, R4                                     ; // x       400  FMA
/*02298830*/  SHFL.BFLY PT, R4, R21, 0x10, 0x1f                          ; // x       400  warp shuffle
/*02298840*/  HFMA2.MMA R20, -RZ, RZ, 0, 2.384185791015625e-07           ; // x       400  FMA
/*02298850*/  IMAD.WIDE.U32 R22, R2, R33, c[0x0][0x198]                  ; // x       400  FMA
/*02298860*/  LDG.E.EL.128 R12desc, [UR4][R22.64]                        ; // x       400  global load (DRAM/L2: weights or activations)
/*02298870*/  IMAD.WIDE.U32 R24, R2, R20, c[0x0][0x1a0]                  ; // x       400  FMA
/*02298880*/  LDG.E.EL.128 R8desc, [UR4][R22.64+0x1000]                  ; // x       400  global load (DRAM/L2: weights or activations)
/*02298890*/  LDG.E.EL.128 R16desc, [UR4][R24.64]                        ; // x       400  global load (DRAM/L2: weights or activations)
/*022988a0*/  FADD R2, R21, R4                                           ; // x       400  FMA
/*022988b0*/  LDG.E.EL.128 R4desc, [UR4][R24.64+0x1000]                  ; // x       400  global load (DRAM/L2: weights or activations)
/*022988c0*/  SHFL.BFLY PT, R27, R2, 0x8, 0x1f                           ; // x       400  warp shuffle
/*022988d0*/  FADD R27, R2, R27                                          ; // x       400  FMA
/*022988e0*/  SHFL.BFLY PT, R26, R27, 0x4, 0x1f                          ; // x       400  warp shuffle
/*022988f0*/  FADD R26, R27, R26                                         ; // x       400  FMA
/*02298900*/  SHFL.BFLY PT, R21, R26, 0x2, 0x1f                          ; // x       400  warp shuffle
/*02298910*/  FADD R21, R26, R21                                         ; // x       400  FMA
/*02298920*/  SHFL.BFLY PT, R32, R21, 0x1, 0x1f                          ; // x       400  warp shuffle
/*02298930*/  SHF.R.U32.HI R2, RZ, 0x3, R34                              ; // x       400  ALU
/*02298940*/  LOP3.LUT R23, R2, 0x1c, RZ, 0xc0, !PT                      ; // x       400  ALU
/*02298950*/  FADD R32, R21, R32                                         ; // x       400  FMA
/*02298960*/  @!P0  STS [R23], R32                                       ; // x       400  shared-memory store
/*02298970*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       400  block barrier
/*02298980*/  @!P1  LDS R0, [R34.X4]                                     ; // x       400  shared-memory load
/*02298990*/  SHFL.BFLY PT, R25, R0, 0x4, 0x1f                           ; // x       400  warp shuffle
/*022989a0*/  FADD R25, R0, R25                                          ; // x       400  FMA
/*022989b0*/  SHFL.BFLY PT, R2, R25, 0x2, 0x1f                           ; // x       400  warp shuffle
/*022989c0*/  ISETP.NE.AND P0, PT, R34, RZ, PT                           ; // x       400  ALU
/*022989d0*/  FADD R2, R25, R2                                           ; // x       400  FMA
/*022989e0*/  SHFL.BFLY PT, R21, R2, 0x1, 0x1f                           ; // x       400  warp shuffle
/*022989f0*/  FADD R21, R2, R21                                          ; // x       400  FMA
/*02298a00*/  @!P0  STS [R34.X4], R21                                    ; // x       400  shared-memory store
/*02298a10*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       400  block barrier
/*02298a20*/  LDS R22, [RZ]                                              ; // x       400  shared-memory load
/*02298a30*/  MOV R23, 0x3a800000                                        ; // x       400  ALU
/*02298a40*/  FADD R12, R12, R16                                         ; // x       400  FMA
/*02298a50*/  FADD R15, R15, R19                                         ; // x       400  FMA
/*02298a60*/  FADD R13, R13, R17                                         ; // x       400  FMA
/*02298a70*/  FADD R14, R14, R18                                         ; // x       400  FMA
/*02298a80*/  FADD R19, R10, R6                                          ; // x       400  FMA
/*02298a90*/  FADD R17, R8, R4                                           ; // x       400  FMA
/*02298aa0*/  FADD R16, R9, R5                                           ; // x       400  FMA
/*02298ab0*/  FADD R10, R11, R7                                          ; // x       400  FMA
/*02298ac0*/  FADD R9, R12, 1                                            ; // x       400  FMA
/*02298ad0*/  FADD R8, R15, 1                                            ; // x       400  FMA
/*02298ae0*/  FADD R2, R13, 1                                            ; // x       400  FMA
/*02298af0*/  FADD R22, RZ, R22                                          ; // x       400  FMA
/*02298b00*/  FFMA R0, R22, R23, 9.9999999747524270788e-07               ; // x       400  FMA
/*02298b10*/  MUFU.RSQ R0, R0                                            ; // x       400  special-function unit
/*02298b20*/  FADD R11, R14, 1                                           ; // x       400  FMA
/*02298b30*/  FMUL R4, R28, R0                                           ; // x       400  FMA
/*02298b40*/  FMUL R7, R31, R0                                           ; // x       400  FMA
/*02298b50*/  FMUL R5, R29, R0                                           ; // x       400  FMA
/*02298b60*/  FMUL R6, R30, R0                                           ; // x       400  FMA
/*02298b70*/  FFMA R4, R4, R9, R17                                       ; // x       400  FMA
/*02298b80*/  FFMA R7, R7, R8, R10                                       ; // x       400  FMA
/*02298b90*/  IMAD.WIDE R8, R3, R20, c[0x0][0x160]                       ; // x       400  FMA
/*02298ba0*/  FFMA R5, R5, R2, R16                                       ; // x       400  FMA
/*02298bb0*/  FFMA R6, R6, R11, R19                                      ; // x       400  FMA
/*02298bc0*/  IMAD.WIDE R20, R3, R20, c[0x0][0x1a8]                      ; // x       400  FMA
/*02298bd0*/  STG.E.128 [R8.64], R28                                     ; // x       400  global store
/*02298be0*/  STG.E.128 [R20.64], R4                                     ; // x       400  global store
/*02298bf0*/  EXIT                                                       ; // x       400  Branch
/*02298c00*/  BRA 0x202298c00                                            ; // x         0  branch
/*02298c10*/  NOP                                                        ; // x         0  Branch
/*02298c20*/  NOP                                                        ; // x         0  Branch
/*02298c30*/  NOP                                                        ; // x         0  Branch
/*02298c40*/  NOP                                                        ; // x         0  Branch
/*02298c50*/  NOP                                                        ; // x         0  Branch
/*02298c60*/  NOP                                                        ; // x         0  Branch
/*02298c70*/  NOP                                                        ; // x         0  Branch
/*02298c80*/  NOP                                                        ; // x         0  Branch
/*02298c90*/  NOP                                                        ; // x         0  Branch
/*02298ca0*/  NOP                                                        ; // x         0  Branch
/*02298cb0*/  NOP                                                        ; // x         0  Branch
/*02298cc0*/  NOP                                                        ; // x         0  Branch
/*02298cd0*/  NOP                                                        ; // x         0  Branch
/*02298ce0*/  NOP                                                        ; // x         0  Branch
/*02298cf0*/  NOP                                                        ; // x         0  Branch
```
