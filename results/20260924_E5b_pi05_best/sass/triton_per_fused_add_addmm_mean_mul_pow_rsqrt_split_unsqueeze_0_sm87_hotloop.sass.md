# Hot loop — `triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_0` (sm87)

PC range 0x2282f00-0x22835f0; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_0_sm87.sass`; per-PC counts: `raw/triton_per_fused_add_addmm_mean_mul_pow_rsqrt_split_unsqueeze_0_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*02282f00*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x       400  ALU
/*02282f10*/  @P0   BRA 0x202283040                                      ; // x       400  branch
/*02282f20*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*02282f30*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*02282f40*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*02282f50*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*02282f60*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*02282f70*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*02282f80*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*02282f90*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*02282fa0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*02282fb0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*02282fc0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*02282fd0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*02282fe0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*02282ff0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*02283000*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*02283010*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*02283020*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*02283030*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*02283040*/  MOV R1, c[0x0][0x28]                                       ; // x       400  ALU
/*02283050*/  S2R R24, SR_TID.X                                          ; // x       400  Branch
/*02283060*/  ULDC.64 UR4, c[0x0][0x118]                                 ; // x       400  Uniform
/*02283070*/  S2R R4, SR_CTAID.X                                         ; // x       400  Branch
/*02283080*/  SHF.L.U32 R0, R24, 0x2, RZ                                 ; // x       400  ALU
/*02283090*/  LOP3.LUT R29, R0, 0x3fc, RZ, 0xc0, !PT                     ; // x       400  ALU
/*022830a0*/  HFMA2.MMA R0, -RZ, RZ, 0, 2.384185791015625e-07            ; // x       400  FMA
/*022830b0*/  LEA R3, R4, R29, 0xa                                       ; // x       400  ALU
/*022830c0*/  IMAD.WIDE R4, R3, R0, c[0x0][0x160]                        ; // x       400  FMA
/*022830d0*/  LDG.E.128 R4, [R4.64]                                      ; // x       400  global load (DRAM/L2: weights or activations)
/*022830e0*/  IMAD.WIDE.U32 R26, R29, R0, c[0x0][0x168]                  ; // x       400  FMA
/*022830f0*/  UMOV UR6, 0x0                                              ; // x       400  Uniform
/*02283100*/  UMOV UR7, 0x14f00000                                       ; // x       400  Uniform
/*02283110*/  IMAD.WIDE.U32 R28, R29, R0, c[0x0][0x170]                  ; // x       400  FMA
/*02283120*/  LDG.E.EL.128 R16desc, [UR6][R26.64+0x1000]                 ; // x       400  global load (DRAM/L2: weights or activations)
/*02283130*/  LDG.E.EL.128 R12desc, [UR6][R28.64]                        ; // x       400  global load (DRAM/L2: weights or activations)
/*02283140*/  LDG.E.EL.128 R20desc, [UR6][R28.64+0x1000]                 ; // x       400  global load (DRAM/L2: weights or activations)
/*02283150*/  LOP3.LUT P0, RZ, R24, 0x1f, RZ, 0xc0, !PT                  ; // x       400  ALU
/*02283160*/  SHF.R.U32.HI R33, RZ, 0x3, R24                             ; // x       400  ALU
/*02283170*/  LOP3.LUT R33, R33, 0x1c, RZ, 0xc0, !PT                     ; // x       400  ALU
/*02283180*/  ISETP.GE.U32.AND P1, PT, R24, 0x8, PT                      ; // x       400  ALU
/*02283190*/  FMUL R9, R5, R5                                            ; // x       400  FMA
/*022831a0*/  FFMA R9, R4, R4, R9                                        ; // x       400  FMA
/*022831b0*/  FFMA R8, R6, R6, R9                                        ; // x       400  FMA
/*022831c0*/  FFMA R25, R7, R7, R8                                       ; // x       400  FMA
/*022831d0*/  SHFL.BFLY PT, R8, R25, 0x10, 0x1f                          ; // x       400  warp shuffle
/*022831e0*/  FADD R30, R25, R8                                          ; // x       400  FMA
/*022831f0*/  LDG.E.EL.128 R8desc, [UR6][R26.64]                         ; // x       400  global load (DRAM/L2: weights or activations)
/*02283200*/  SHFL.BFLY PT, R31, R30, 0x8, 0x1f                          ; // x       400  warp shuffle
/*02283210*/  FADD R31, R30, R31                                         ; // x       400  FMA
/*02283220*/  SHFL.BFLY PT, R32, R31, 0x4, 0x1f                          ; // x       400  warp shuffle
/*02283230*/  FADD R32, R31, R32                                         ; // x       400  FMA
/*02283240*/  SHFL.BFLY PT, R25, R32, 0x2, 0x1f                          ; // x       400  warp shuffle
/*02283250*/  FADD R25, R32, R25                                         ; // x       400  FMA
/*02283260*/  SHFL.BFLY PT, R30, R25, 0x1, 0x1f                          ; // x       400  warp shuffle
/*02283270*/  FADD R30, R25, R30                                         ; // x       400  FMA
/*02283280*/  @!P0  STS [R33], R30                                       ; // x       400  shared-memory store
/*02283290*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       400  block barrier
/*022832a0*/  @!P1  LDS R2, [R24.X4]                                     ; // x       400  shared-memory load
/*022832b0*/  SHFL.BFLY PT, R27, R2, 0x4, 0x1f                           ; // x       400  warp shuffle
/*022832c0*/  FADD R27, R2, R27                                          ; // x       400  FMA
/*022832d0*/  SHFL.BFLY PT, R26, R27, 0x2, 0x1f                          ; // x       400  warp shuffle
/*022832e0*/  ISETP.NE.AND P0, PT, R24, RZ, PT                           ; // x       400  ALU
/*022832f0*/  FADD R26, R27, R26                                         ; // x       400  FMA
/*02283300*/  SHFL.BFLY PT, R25, R26, 0x1, 0x1f                          ; // x       400  warp shuffle
/*02283310*/  FADD R25, R26, R25                                         ; // x       400  FMA
/*02283320*/  @!P0  STS [R24.X4], R25                                    ; // x       400  shared-memory store
/*02283330*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       400  block barrier
/*02283340*/  LDS R28, [RZ]                                              ; // x       400  shared-memory load
/*02283350*/  HFMA2.MMA R29, -RZ, RZ, 0.8125, 0                          ; // x       400  FMA
/*02283360*/  FADD R8, R8, R12                                           ; // x       400  FMA
/*02283370*/  FADD R9, R9, R13                                           ; // x       400  FMA
/*02283380*/  FADD R10, R10, R14                                         ; // x       400  FMA
/*02283390*/  FADD R11, R11, R15                                         ; // x       400  FMA
/*022833a0*/  FADD R12, R17, R21                                         ; // x       400  FMA
/*022833b0*/  FADD R13, R8, 1                                            ; // x       400  FMA
/*022833c0*/  FADD R2, R9, 1                                             ; // x       400  FMA
/*022833d0*/  FADD R15, R16, R20                                         ; // x       400  FMA
/*022833e0*/  FADD R17, R18, R22                                         ; // x       400  FMA
/*022833f0*/  FADD R14, R19, R23                                         ; // x       400  FMA
/*02283400*/  FADD R9, R10, 1                                            ; // x       400  FMA
/*02283410*/  FADD R8, R11, 1                                            ; // x       400  FMA
/*02283420*/  FADD R28, RZ, R28                                          ; // x       400  FMA
/*02283430*/  FFMA R28, R28, R29, 9.9999999747524270788e-07              ; // x       400  FMA
/*02283440*/  MUFU.RSQ R28, R28                                          ; // x       400  special-function unit
/*02283450*/  FMUL R5, R5, R28                                           ; // x       400  FMA
/*02283460*/  FMUL R4, R4, R28                                           ; // x       400  FMA
/*02283470*/  FMUL R6, R6, R28                                           ; // x       400  FMA
/*02283480*/  FMUL R7, R7, R28                                           ; // x       400  FMA
/*02283490*/  FFMA R5, R5, R2, R12                                       ; // x       400  FMA
/*022834a0*/  FFMA R4, R4, R13, R15                                      ; // x       400  FMA
/*022834b0*/  FFMA R6, R6, R9, R17                                       ; // x       400  FMA
/*022834c0*/  FFMA R7, R7, R8, R14                                       ; // x       400  FMA
/*022834d0*/  IMAD.WIDE R2, R3, R0, c[0x0][0x178]                        ; // x       400  FMA
/*022834e0*/  STG.E.128 [R2.64], R4                                      ; // x       400  global store
/*022834f0*/  EXIT                                                       ; // x       400  Branch
/*02283500*/  BRA 0x202283500                                            ; // x         0  branch
/*02283510*/  NOP                                                        ; // x         0  Branch
/*02283520*/  NOP                                                        ; // x         0  Branch
/*02283530*/  NOP                                                        ; // x         0  Branch
/*02283540*/  NOP                                                        ; // x         0  Branch
/*02283550*/  NOP                                                        ; // x         0  Branch
/*02283560*/  NOP                                                        ; // x         0  Branch
/*02283570*/  NOP                                                        ; // x         0  Branch
/*02283580*/  NOP                                                        ; // x         0  Branch
/*02283590*/  NOP                                                        ; // x         0  Branch
/*022835a0*/  NOP                                                        ; // x         0  Branch
/*022835b0*/  NOP                                                        ; // x         0  Branch
/*022835c0*/  NOP                                                        ; // x         0  Branch
/*022835d0*/  NOP                                                        ; // x         0  Branch
/*022835e0*/  NOP                                                        ; // x         0  Branch
/*022835f0*/  NOP                                                        ; // x         0  Branch
```
