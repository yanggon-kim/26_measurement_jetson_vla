# Hot loop — `triton_per_fused__softmax_add_mul_4` (sm87)

PC range 0x2292500-0x22931f0; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/triton_per_fused__softmax_add_mul_4_sm87.sass`; per-PC counts: `raw/triton_per_fused__softmax_add_mul_4_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*02292500*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x      3200  ALU
/*02292510*/  @P0   BRA 0x202292640                                      ; // x      3200  branch
/*02292520*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*02292530*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*02292540*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*02292550*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*02292560*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*02292570*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*02292580*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*02292590*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*022925a0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*022925b0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*022925c0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*022925d0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*022925e0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*022925f0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*02292600*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*02292610*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*02292620*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*02292630*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*02292640*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]                      ; // x      3200  FMA
/*02292650*/  S2R R19, SR_CTAID.X                                        ; // x      3200  Branch
/*02292660*/  IMAD.MOV.U32 R4, RZ, RZ, 0x4                               ; // x      3200  FMA
/*02292670*/  ULDC.64 UR4, c[0x0][0x118]                                 ; // x      3200  Uniform
/*02292680*/  S2R R5, SR_TID.X                                           ; // x      3200  Branch
/*02292690*/  UMOV UR6, 0x0                                              ; // x      3200  Uniform
/*022926a0*/  UMOV UR7, 0x14f00000                                       ; // x      3200  Uniform
/*022926b0*/  IMAD.HI.U32 R0, R19, 0x51eb851f, RZ                        ; // x      3200  FMA
/*022926c0*/  IMAD.SHL.U32 R2, R5, 0x2, RZ                               ; // x      3200  FMA
/*022926d0*/  SHF.R.U32.HI R0, RZ, 0x4, R0                               ; // x      3200  ALU
/*022926e0*/  LOP3.LUT R8, R2, 0x1fe, RZ, 0xc0, !PT                      ; // x      3200  ALU
/*022926f0*/  IMAD R7, R0, -0x32, R19                                    ; // x      3200  FMA
/*02292700*/  LOP3.LUT R12, R8, 0x200, RZ, 0xfc, !PT                     ; // x      3200  ALU
/*02292710*/  IMAD R7, R7, 0x3fa, RZ                                     ; // x      3200  FMA
/*02292720*/  IMAD R17, R19, 0x3fa, R8                                   ; // x      3200  FMA
/*02292730*/  IMAD R19, R19, 0x3fa, R12                                  ; // x      3200  FMA
/*02292740*/  IADD3 R3, P0, R8, R7, RZ                                   ; // x      3200  ALU
/*02292750*/  IMAD.WIDE R16, R17, R4, c[0x0][0x160]                      ; // x      3200  FMA
/*02292760*/  IADD3 R13, R8, R7, RZ                                      ; // x      3200  ALU
/*02292770*/  IADD3.X R10, RZ, RZ, RZ, P0, !PT                           ; // x      3200  ALU
/*02292780*/  CS2R R8, SRZ                                               ; // x      3200  Branch
/*02292790*/  LEA R2, P0, R3, c[0x0][0x168], 0x2                         ; // x      3200  ALU
/*022927a0*/  IMAD.WIDE.U32 R20, R13, R4, c[0x0][0x168]                  ; // x      3200  FMA
/*022927b0*/  LDG.E.64 R16, [R16.64]                                     ; // x      3200  global load (DRAM/L2: weights or activations)
/*022927c0*/  LEA.HI.X R3, R3, c[0x0][0x16c], R10, 0x2, P0               ; // x      3200  ALU
/*022927d0*/  IMAD.WIDE R18, R19, R4, c[0x0][0x160]                      ; // x      3200  FMA
/*022927e0*/  ISETP.GE.U32.AND P0, PT, R12, 0x3fa, PT                    ; // x      3200  ALU
/*022927f0*/  LDG.E.EL.64 R20desc, [UR6][R20.64]                         ; // x      3200  global load (DRAM/L2: weights or activations)
/*02292800*/  CS2R R10, SRZ                                              ; // x      3200  Branch
/*02292810*/  @!P0  LDG.E.64 R8, [R18.64]                                ; // x      3200  global load (DRAM/L2: weights or activations)
/*02292820*/  @!P0  LDG.E.EL.64 R10desc, [UR6][R2.64+0x800]              ; // x      3200  global load (DRAM/L2: weights or activations)
/*02292830*/  SHF.R.U32.HI R3, RZ, 0x3, R5                               ; // x      3200  ALU
/*02292840*/  FFMA R22, R16, 0.0625, R20                                 ; // x      3200  FMA
/*02292850*/  FFMA R15, R17, 0.0625, R21                                 ; // x      3200  FMA
/*02292860*/  FSETP.NAN.AND P2, PT, R22, R22, PT                         ; // x      3200  FMA
/*02292870*/  FSETP.GT.AND P1, PT, R22, R15, PT                          ; // x      3200  FMA
/*02292880*/  FSEL R17, R22, R15, P2                                     ; // x      3200  FMA
/*02292890*/  FFMA R10, R8, 0.0625, R10                                  ; // x      3200  FMA
/*022928a0*/  FSEL R17, R22, R17, P1                                     ; // x      3200  FMA
/*022928b0*/  FSEL R8, R10, -INF , !P0                                   ; // x      3200  FMA
/*022928c0*/  FSETP.GT.AND P1, PT, R17, R8, PT                           ; // x      3200  FMA
/*022928d0*/  FFMA R16, R9, 0.0625, R11                                  ; // x      3200  FMA
/*022928e0*/  FSETP.NAN.AND P2, PT, R17, R17, PT                         ; // x      3200  FMA
/*022928f0*/  FSEL R2, R16, -INF , !P0                                   ; // x      3200  FMA
/*02292900*/  @!P1  FSEL R17, R17, R8, P2                                ; // x      3200  FMA
/*02292910*/  FSETP.GT.AND P1, PT, R17, R2, PT                           ; // x      3200  FMA
/*02292920*/  FSETP.NAN.AND P2, PT, R17, R17, PT                         ; // x      3200  FMA
/*02292930*/  @!P1  FSEL R17, R17, R2, P2                                ; // x      3200  FMA
/*02292940*/  SHFL.BFLY PT, R2, R17, 0x10, 0x1f                          ; // x      3200  warp shuffle
/*02292950*/  FSETP.NAN.AND P2, PT, R17, R17, PT                         ; // x      3200  FMA
/*02292960*/  FSETP.GT.AND P1, PT, R17, R2, PT                           ; // x      3200  FMA
/*02292970*/  @!P1  FSEL R17, R17, R2, P2                                ; // x      3200  FMA
/*02292980*/  SHFL.BFLY PT, R2, R17, 0x8, 0x1f                           ; // x      3200  warp shuffle
/*02292990*/  FSETP.NAN.AND P2, PT, R17, R17, PT                         ; // x      3200  FMA
/*022929a0*/  FSETP.GT.AND P1, PT, R17, R2, PT                           ; // x      3200  FMA
/*022929b0*/  @!P1  FSEL R17, R17, R2, P2                                ; // x      3200  FMA
/*022929c0*/  SHFL.BFLY PT, R2, R17, 0x4, 0x1f                           ; // x      3200  warp shuffle
/*022929d0*/  FSETP.NAN.AND P2, PT, R17, R17, PT                         ; // x      3200  FMA
/*022929e0*/  FSETP.GT.AND P1, PT, R17, R2, PT                           ; // x      3200  FMA
/*022929f0*/  @!P1  FSEL R17, R17, R2, P2                                ; // x      3200  FMA
/*02292a00*/  SHFL.BFLY PT, R2, R17, 0x2, 0x1f                           ; // x      3200  warp shuffle
/*02292a10*/  FSETP.NAN.AND P2, PT, R17, R17, PT                         ; // x      3200  FMA
/*02292a20*/  FSETP.GT.AND P1, PT, R17, R2, PT                           ; // x      3200  FMA
/*02292a30*/  @!P1  FSEL R17, R17, R2, P2                                ; // x      3200  FMA
/*02292a40*/  SHFL.BFLY PT, R2, R17, 0x1, 0x1f                           ; // x      3200  warp shuffle
/*02292a50*/  LOP3.LUT P1, RZ, R5, 0x1f, RZ, 0xc0, !PT                   ; // x      3200  ALU
/*02292a60*/  FSETP.NAN.AND P4, PT, R17, R17, PT                         ; // x      3200  FMA
/*02292a70*/  FSETP.GT.AND P3, PT, R17, R2, PT                           ; // x      3200  FMA
/*02292a80*/  @!P3  SEL R17, R17, R2, P4                                 ; // x      3200  ALU
/*02292a90*/  LOP3.LUT R2, R3, 0x1c, RZ, 0xc0, !PT                       ; // x      3200  ALU
/*02292aa0*/  @!P1  STS [R2], R17                                        ; // x      3200  shared-memory store
/*02292ab0*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x      3200  block barrier
/*02292ac0*/  ISETP.GE.U32.AND P2, PT, R5, 0x8, PT                       ; // x      3200  ALU
/*02292ad0*/  @!P2  LDS R14, [R5.X4]                                     ; // x      3200  shared-memory load
/*02292ae0*/  SHFL.BFLY PT, R3, R14, 0x4, 0x1f                           ; // x      3200  warp shuffle
/*02292af0*/  FSETP.NAN.AND P4, PT, R14, R14, PT                         ; // x      3200  FMA
/*02292b00*/  FSETP.GT.AND P3, PT, R14, R3, PT                           ; // x      3200  FMA
/*02292b10*/  FSEL R3, R14, R3, P3                                       ; // x      3200  FMA
/*02292b20*/  FSEL R8, R14, R3, P4                                       ; // x      3200  FMA
/*02292b30*/  SHFL.BFLY PT, R3, R8, 0x2, 0x1f                            ; // x      3200  warp shuffle
/*02292b40*/  FSETP.NAN.AND P4, PT, R8, R8, PT                           ; // x      3200  FMA
/*02292b50*/  FSETP.GT.AND P3, PT, R8, R3, PT                            ; // x      3200  FMA
/*02292b60*/  @!P3  FSEL R8, R8, R3, P4                                  ; // x      3200  FMA
/*02292b70*/  SHFL.BFLY PT, R3, R8, 0x1, 0x1f                            ; // x      3200  warp shuffle
/*02292b80*/  ISETP.NE.AND P3, PT, R5, RZ, PT                            ; // x      3200  ALU
/*02292b90*/  FSETP.NAN.AND P5, PT, R8, R8, PT                           ; // x      3200  FMA
/*02292ba0*/  FSETP.GT.AND P4, PT, R8, R3, PT                            ; // x      3200  FMA
/*02292bb0*/  @!P4  SEL R8, R8, R3, P5                                   ; // x      3200  ALU
/*02292bc0*/  @!P3  STS [R5.X4], R8                                      ; // x      3200  shared-memory store
/*02292bd0*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x      3200  block barrier
/*02292be0*/  LDS R3, [RZ]                                               ; // x      3200  shared-memory load
/*02292bf0*/  FADD R9, RZ, R3                                            ; // x      3200  FMA
/*02292c00*/  FADD R22, R22, -R9                                         ; // x      3200  FMA
/*02292c10*/  FMUL R22, R22, 1.4426950216293334961                       ; // x      3200  FMA
/*02292c20*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x      3200  block barrier
/*02292c30*/  FSETP.GEU.AND P6, PT, R22, -126, PT                        ; // x      3200  FMA
/*02292c40*/  FADD R10, R10, -R9                                         ; // x      3200  FMA
/*02292c50*/  FADD R15, R15, -R9                                         ; // x      3200  FMA
/*02292c60*/  @!P6  FMUL R22, R22, 0.5                                   ; // x      3200  FMA
/*02292c70*/  MUFU.EX2 R3, R22                                           ; // x      3200  special-function unit
/*02292c80*/  FADD R9, R16, -R9                                          ; // x      3200  FMA
/*02292c90*/  FMUL R11, R10, 1.4426950216293334961                       ; // x      3200  FMA
/*02292ca0*/  FMUL R15, R15, 1.4426950216293334961                       ; // x      3200  FMA
/*02292cb0*/  FMUL R14, R9, 1.4426950216293334961                        ; // x      3200  FMA
/*02292cc0*/  FSETP.GEU.AND P5, PT, R11, -126, PT                        ; // x      3200  FMA
/*02292cd0*/  FSETP.GEU.AND P4, PT, R15, -126, PT                        ; // x      3200  FMA
/*02292ce0*/  @!P6  FMUL R3, R3, R3                                      ; // x      3200  FMA
/*02292cf0*/  FSETP.GEU.AND P6, PT, R14, -126, PT                        ; // x      3200  FMA
/*02292d00*/  @!P5  FMUL R11, R11, 0.5                                   ; // x      3200  FMA
/*02292d10*/  @!P4  FMUL R15, R15, 0.5                                   ; // x      3200  FMA
/*02292d20*/  MUFU.EX2 R8, R11                                           ; // x      3200  special-function unit
/*02292d30*/  @!P6  FMUL R14, R14, 0.5                                   ; // x      3200  FMA
/*02292d40*/  MUFU.EX2 R10, R15                                          ; // x      3200  special-function unit
/*02292d50*/  MUFU.EX2 R9, R14                                           ; // x      3200  special-function unit
/*02292d60*/  @!P5  FMUL R8, R8, R8                                      ; // x      3200  FMA
/*02292d70*/  @!P4  FMUL R10, R10, R10                                   ; // x      3200  FMA
/*02292d80*/  FSEL R16, R8, RZ, !P0                                      ; // x      3200  FMA
/*02292d90*/  FADD R19, R3, R10                                          ; // x      3200  FMA
/*02292da0*/  @!P6  FMUL R9, R9, R9                                      ; // x      3200  FMA
/*02292db0*/  FADD R16, R16, R19                                         ; // x      3200  FMA
/*02292dc0*/  FSEL R17, R9, RZ, !P0                                      ; // x      3200  FMA
/*02292dd0*/  FADD R16, R17, R16                                         ; // x      3200  FMA
/*02292de0*/  SHFL.BFLY PT, R11, R16, 0x10, 0x1f                         ; // x      3200  warp shuffle
/*02292df0*/  FADD R11, R16, R11                                         ; // x      3200  FMA
/*02292e00*/  SHFL.BFLY PT, R14, R11, 0x8, 0x1f                          ; // x      3200  warp shuffle
/*02292e10*/  FADD R14, R11, R14                                         ; // x      3200  FMA
/*02292e20*/  SHFL.BFLY PT, R15, R14, 0x4, 0x1f                          ; // x      3200  warp shuffle
/*02292e30*/  FADD R15, R14, R15                                         ; // x      3200  FMA
/*02292e40*/  SHFL.BFLY PT, R18, R15, 0x2, 0x1f                          ; // x      3200  warp shuffle
/*02292e50*/  FADD R18, R15, R18                                         ; // x      3200  FMA
/*02292e60*/  SHFL.BFLY PT, R17, R18, 0x1, 0x1f                          ; // x      3200  warp shuffle
/*02292e70*/  FADD R17, R18, R17                                         ; // x      3200  FMA
/*02292e80*/  @!P1  STS [R2], R17                                        ; // x      3200  shared-memory store
/*02292e90*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x      3200  block barrier
/*02292ea0*/  @!P2  LDS R6, [R5.X4]                                      ; // x      3200  shared-memory load
/*02292eb0*/  SHFL.BFLY PT, R11, R6, 0x4, 0x1f                           ; // x      3200  warp shuffle
/*02292ec0*/  FADD R11, R6, R11                                          ; // x      3200  FMA
/*02292ed0*/  SHFL.BFLY PT, R14, R11, 0x2, 0x1f                          ; // x      3200  warp shuffle
/*02292ee0*/  FADD R14, R11, R14                                         ; // x      3200  FMA
/*02292ef0*/  SHFL.BFLY PT, R15, R14, 0x1, 0x1f                          ; // x      3200  warp shuffle
/*02292f00*/  FADD R16, R14, R15                                         ; // x      3200  FMA
/*02292f10*/  @!P3  STS [R5.X4], R16                                     ; // x      3200  shared-memory store
/*02292f20*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x      3200  block barrier
/*02292f30*/  LDS R2, [RZ]                                               ; // x      3200  shared-memory load
/*02292f40*/  FMUL R11, R10, 0.25                                        ; // x      3200  FMA
/*02292f50*/  FADD R15, RZ, R2                                           ; // x      3200  FMA
/*02292f60*/  FSETP.GT.AND P1, PT, |R15|, 8.50705917302346158658e+37, PT ; // x      3200  FMA
/*02292f70*/  FSETP.GEU.AND P2, PT, |R15|, 1.175494350822287508e-38, PT  ; // x      3200  FMA
/*02292f80*/  @P1   FMUL R15, R15, 0.25                                  ; // x      3200  FMA
/*02292f90*/  @!P2  FMUL R15, R15, 16777216                              ; // x      3200  FMA
/*02292fa0*/  FMUL R2, R3, 0.25                                          ; // x      3200  FMA
/*02292fb0*/  MUFU.RCP R15, R15                                          ; // x      3200  special-function unit
/*02292fc0*/  FSEL R6, R2, R3, P1                                        ; // x      3200  FMA
/*02292fd0*/  FSEL R14, R11, R10, P1                                     ; // x      3200  FMA
/*02292fe0*/  IMAD R3, R0, 0xc6e0, R13                                   ; // x      3200  FMA
/*02292ff0*/  @!P2  FMUL R6, R6, 16777216                                ; // x      3200  FMA
/*02293000*/  @!P2  FMUL R14, R14, 16777216                              ; // x      3200  FMA
/*02293010*/  IADD3 R7, R12, R7, RZ                                      ; // x      3200  ALU
/*02293020*/  IMAD.WIDE R2, R3, R4, c[0x0][0x170]                        ; // x      3200  FMA
/*02293030*/  @P1   FMUL R8, R8, 0.25                                    ; // x      3200  FMA
/*02293040*/  @P1   FMUL R9, R9, 0.25                                    ; // x      3200  FMA
/*02293050*/  FMUL R10, R15, R6                                          ; // x      3200  FMA
/*02293060*/  FMUL R11, R15, R14                                         ; // x      3200  FMA
/*02293070*/  @!P2  FMUL R8, R8, 16777216                                ; // x      3200  FMA
/*02293080*/  @!P2  FMUL R9, R9, 16777216                                ; // x      3200  FMA
/*02293090*/  IMAD R5, R0, 0xc6e0, R7                                    ; // x      3200  FMA
/*022930a0*/  STG.E.64 [R2.64], R10                                      ; // x      3200  global store
/*022930b0*/  FMUL R8, R15, R8                                           ; // x      3200  FMA
/*022930c0*/  FMUL R9, R15, R9                                           ; // x      3200  FMA
/*022930d0*/  IMAD.WIDE R4, R5, R4, c[0x0][0x170]                        ; // x      3200  FMA
/*022930e0*/  @P0   EXIT                                                 ; // x      3200  Branch
/*022930f0*/  STG.E.64 [R4.64], R8                                       ; // x      3200  global store
/*02293100*/  EXIT                                                       ; // x      3200  Branch
/*02293110*/  BRA 0x202293110                                            ; // x         0  branch
/*02293120*/  NOP                                                        ; // x         0  Branch
/*02293130*/  NOP                                                        ; // x         0  Branch
/*02293140*/  NOP                                                        ; // x         0  Branch
/*02293150*/  NOP                                                        ; // x         0  Branch
/*02293160*/  NOP                                                        ; // x         0  Branch
/*02293170*/  NOP                                                        ; // x         0  Branch
```

(first 200 of 208 instructions shown; the rest are in the full dump)
