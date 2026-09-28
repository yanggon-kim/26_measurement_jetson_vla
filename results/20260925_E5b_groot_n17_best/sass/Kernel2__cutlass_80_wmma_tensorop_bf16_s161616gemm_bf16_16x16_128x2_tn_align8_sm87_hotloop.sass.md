# Hot loop — `Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8` (sm87)

PC range 0x86231190-0x86232340; backward BRA 0x86232340 -> 0x86231190: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8_sm87.sass`; per-PC counts: `raw/Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*86231190*/  R2P PR, R105, 0xf                                          ; // x       192  ALU
/*862311a0*/  CS2R R36, SRZ                                              ; // x       192  Branch
/*862311b0*/  CS2R R38, SRZ                                              ; // x       192  Branch
/*862311c0*/  IADD3 R40, P4, R110, c[0x0][0x1a0], RZ                     ; // x       192  ALU
/*862311d0*/  ULDC.64 UR6, c[0x0][0x18]                                  ; // x       192  Uniform
/*862311e0*/  CS2R R56, SRZ                                              ; // x       192  Branch
/*862311f0*/  CS2R R58, SRZ                                              ; // x       192  Branch
/*86231200*/  IADD3.X R41, R111, c[0x0][0x1a4], RZ, P4, !PT              ; // x       192  ALU
/*86231210*/  CS2R R48, SRZ                                              ; // x       192  Branch
/*86231220*/  CS2R R50, SRZ                                              ; // x       192  Branch
/*86231230*/  HMMA.16816.F32.BF16 R4, R12, R2, R4                        ; // x       192  tensor-core MMA
/*86231240*/  CS2R R52, SRZ                                              ; // x       192  Branch
/*86231250*/  CS2R R54, SRZ                                              ; // x       192  Branch
/*86231260*/  @P0   LDG.E.LTC128B.128 R36, [R110.64]                     ; // x       192  global load (DRAM/L2: weights or activations)
/*86231270*/  UIADD3 UR10, UR18, 0x20, URZ                               ; // x       192  Uniform
/*86231280*/  IADD3 R64, P0, R40, c[0x0][0x1a0], RZ                      ; // x       192  ALU
/*86231290*/  CS2R R44, SRZ                                              ; // x       192  Branch
/*862312a0*/  @P1   LDG.E.LTC128B.128 R56, [R40.64]                      ; // x       192  global load (DRAM/L2: weights or activations)
/*862312b0*/  UIADD3 UR11, UP0, UR10, UR6, URZ                           ; // x       192  Uniform
/*862312c0*/  IADD3.X R65, R41, c[0x0][0x1a4], RZ, P0, !PT               ; // x       192  ALU
/*862312d0*/  HMMA.16816.F32.BF16 R8, R12, R16, R8                       ; // x       192  tensor-core MMA
/*862312e0*/  UIADD3 UR12, UR18, 0x40, URZ                               ; // x       192  Uniform
/*862312f0*/  CS2R R46, SRZ                                              ; // x       192  Branch
/*86231300*/  IADD3 R42, P0, R64, c[0x0][0x1a0], RZ                      ; // x       192  ALU
/*86231310*/  @P2   LDG.E.LTC128B.128 R52, [R64.64]                      ; // x       192  global load (DRAM/L2: weights or activations)
/*86231320*/  ULEA.HI.X.SX32 UR10, UR10, UR7, 0x1, UP0                   ; // x       192  Uniform
/*86231330*/  CS2R R96, SRZ                                              ; // x       192  Branch
/*86231340*/  IADD3.X R43, R65, c[0x0][0x1a4], RZ, P0, !PT               ; // x       192  ALU
/*86231350*/  CS2R R98, SRZ                                              ; // x       192  Branch
/*86231360*/  CS2R R92, SRZ                                              ; // x       192  Branch
/*86231370*/  UIADD3 UR13, UP1, UR12, UR6, URZ                           ; // x       192  Uniform
/*86231380*/  @P3   LDG.E.LTC128B.128 R48, [R42.64]                      ; // x       192  global load (DRAM/L2: weights or activations)
/*86231390*/  R2P PR, R105.B1, 0xf                                       ; // x       192  ALU
/*862313a0*/  CS2R R94, SRZ                                              ; // x       192  Branch
/*862313b0*/  IADD3 R2, P5, R102, UR11, RZ                               ; // x       192  ALU
/*862313c0*/  CS2R R66, SRZ                                              ; // x       192  Branch
/*862313d0*/  IADD3 R62, P4, R42, c[0x0][0x1a0], RZ                      ; // x       192  ALU
/*862313e0*/  CS2R R76, SRZ                                              ; // x       192  Branch
/*862313f0*/  IADD3.X R3, R101, UR10, RZ, P5, !PT                        ; // x       192  ALU
/*86231400*/  CS2R R78, SRZ                                              ; // x       192  Branch
/*86231410*/  IADD3.X R63, R43, c[0x0][0x1a4], RZ, P4, !PT               ; // x       192  ALU
/*86231420*/  CS2R R40, SRZ                                              ; // x       192  Branch
/*86231430*/  IADD3 R60, P4, R62, c[0x0][0x1a0], RZ                      ; // x       192  ALU
/*86231440*/  LD.E R12, [R2.64]                                          ; // x       192  LSU / Load
/*86231450*/  CS2R R74, SRZ                                              ; // x       192  Branch
/*86231460*/  CS2R R88, SRZ                                              ; // x       192  Branch
/*86231470*/  IADD3.X R61, R63, c[0x0][0x1a4], RZ, P4, !PT               ; // x       192  ALU
/*86231480*/  LD.E R13, [R2.64+0x1080]                                   ; // x       192  LSU / Load
/*86231490*/  IADD3 R70, P4, R68, c[0x0][0x1c0], RZ                      ; // x       192  ALU
/*862314a0*/  CS2R R64, SRZ                                              ; // x       192  Branch
/*862314b0*/  CS2R R90, SRZ                                              ; // x       192  Branch
/*862314c0*/  LD.E R14, [R2.64+0x10]                                     ; // x       192  LSU / Load
/*862314d0*/  IADD3.X R71, R69, c[0x0][0x1c4], RZ, P4, !PT               ; // x       192  ALU
/*862314e0*/  CS2R R86, SRZ                                              ; // x       192  Branch
/*862314f0*/  ULEA.HI.X.SX32 UR12, UR12, UR7, 0x1, UP1                   ; // x       192  Uniform
/*86231500*/  LD.E R15, [R2.64+0x1090]                                   ; // x       192  LSU / Load
/*86231510*/  CS2R R42, SRZ                                              ; // x       192  Branch
/*86231520*/  UIADD3 UR10, UR16, 0x20, URZ                               ; // x       192  Uniform
/*86231530*/  IMAD.MOV.U32 R100, RZ, RZ, R104                            ; // x       192  FMA
/*86231540*/  @P0   LDG.E.LTC128B.128 R44, [R62.64]                      ; // x       192  global load (DRAM/L2: weights or activations)
/*86231550*/  IADD3 R72, P0, R60, c[0x0][0x1a0], RZ                      ; // x       192  ALU
/*86231560*/  UIADD3 UR11, UP0, UR10, UR6, URZ                           ; // x       192  Uniform
/*86231570*/  IMAD.MOV.U32 R35, RZ, RZ, R103                             ; // x       192  FMA
/*86231580*/  @P1   LDG.E.LTC128B.128 R40, [R60.64]                      ; // x       192  global load (DRAM/L2: weights or activations)
/*86231590*/  IADD3.X R73, R61, c[0x0][0x1a4], RZ, P0, !PT               ; // x       192  ALU
/*862315a0*/  ULEA.HI.X.SX32 UR10, UR10, UR7, 0x1, UP0                   ; // x       192  Uniform
/*862315b0*/  IADD3 R110, P0, R72, c[0x0][0x1a0], RZ                     ; // x       192  ALU
/*862315c0*/  UIADD3 UR15, UR16, 0x40, URZ                               ; // x       192  Uniform
/*862315d0*/  IADD3 R107, R107, -0x1, RZ                                 ; // x       192  ALU
/*862315e0*/  @P2   LDG.E.LTC128B.128 R96, [R72.64]                      ; // x       192  global load (DRAM/L2: weights or activations)
/*862315f0*/  IADD3.X R111, R73, c[0x0][0x1a4], RZ, P0, !PT              ; // x       192  ALU
/*86231600*/  UMOV UR8, UR16                                             ; // x       192  Uniform
/*86231610*/  UMOV UR9, UR18                                             ; // x       192  Uniform
/*86231620*/  @P3   LDG.E.LTC128B.128 R92, [R110.64]                     ; // x       192  global load (DRAM/L2: weights or activations)
/*86231630*/  R2P PR, R106, 0xf                                          ; // x       192  ALU
/*86231640*/  IADD3 R18, P4, R102, UR11, RZ                              ; // x       192  ALU
/*86231650*/  UIADD3 UR11, UP0, UR6, 0x2100, URZ                         ; // x       192  Uniform
/*86231660*/  IADD3.X R19, R101, UR10, RZ, P4, !PT                       ; // x       192  ALU
/*86231670*/  UIADD3.X UR10, URZ, UR7, URZ, UP0, !UPT                    ; // x       192  Uniform
/*86231680*/  CS2R R62, SRZ                                              ; // x       192  Branch
/*86231690*/  UIADD3 UR14, UP0, UR15, UR11, URZ                          ; // x       192  Uniform
/*862316a0*/  LD.E R2, [R18.64+0x2100]                                   ; // x       192  LSU / Load
/*862316b0*/  CS2R R60, SRZ                                              ; // x       192  Branch
/*862316c0*/  ULEA.HI.X.SX32 UR15, UR15, UR10, 0x1, UP0                  ; // x       192  Uniform
/*862316d0*/  LD.E R3, [R18.64+0x2110]                                   ; // x       192  LSU / Load
/*862316e0*/  CS2R R72, SRZ                                              ; // x       192  Branch
/*862316f0*/  LD.E R16, [R18.64+0x3180]                                  ; // x       192  LSU / Load
/*86231700*/  LD.E R17, [R18.64+0x3190]                                  ; // x       192  LSU / Load
/*86231710*/  @P0   LDG.E.LTC128B.128 R64, [R68.64]                      ; // x       192  global load (DRAM/L2: weights or activations)
/*86231720*/  IADD3 R84, P0, R70, c[0x0][0x1c0], RZ                      ; // x       192  ALU
/*86231730*/  @P1   LDG.E.LTC128B.128 R60, [R70.64]                      ; // x       192  global load (DRAM/L2: weights or activations)
/*86231740*/  IADD3.X R85, R71, c[0x0][0x1c4], RZ, P0, !PT               ; // x       192  ALU
/*86231750*/  IADD3 R82, P0, R84, c[0x0][0x1c0], RZ                      ; // x       192  ALU
/*86231760*/  @P2   LDG.E.LTC128B.128 R76, [R84.64]                      ; // x       192  global load (DRAM/L2: weights or activations)
/*86231770*/  IADD3.X R83, R85, c[0x0][0x1c4], RZ, P0, !PT               ; // x       192  ALU
/*86231780*/  @P3   LDG.E.LTC128B.128 R72, [R82.64]                      ; // x       192  global load (DRAM/L2: weights or activations)
/*86231790*/  R2P PR, R106.B1, 0xf                                       ; // x       192  ALU
/*862317a0*/  IADD3 R80, P4, R82, c[0x0][0x1c0], RZ                      ; // x       192  ALU
/*862317b0*/  IADD3.X R81, R83, c[0x0][0x1c4], RZ, P4, !PT               ; // x       192  ALU
/*862317c0*/  CS2R R68, SRZ                                              ; // x       192  Branch
/*862317d0*/  ISETP.GT.AND P4, PT, R107, 0x2, PT                         ; // x       192  ALU
/*862317e0*/  CS2R R70, SRZ                                              ; // x       192  Branch
/*862317f0*/  @P0   LDG.E.LTC128B.128 R68, [R80.64]                      ; // x       192  global load (DRAM/L2: weights or activations)
/*86231800*/  IADD3 R116, P0, R80, c[0x0][0x1c0], RZ                     ; // x       192  ALU
/*86231810*/  CS2R R84, SRZ                                              ; // x       192  Branch
/*86231820*/  @!P4  IMAD.MOV.U32 R106, RZ, RZ, RZ                        ; // x       192  FMA
/*86231830*/  IADD3.X R117, R81, c[0x0][0x1c4], RZ, P0, !PT              ; // x       192  ALU
/*86231840*/  CS2R R82, SRZ                                              ; // x       192  Branch
/*86231850*/  IADD3 R114, P0, R116, c[0x0][0x1c0], RZ                    ; // x       192  ALU
/*86231860*/  @P1   LDG.E.LTC128B.128 R88, [R116.64]                     ; // x       192  global load (DRAM/L2: weights or activations)
/*86231870*/  IADD3.X R115, R117, c[0x0][0x1c4], RZ, P0, !PT             ; // x       192  ALU
/*86231880*/  IADD3 R112, P0, R114, c[0x0][0x1c0], RZ                    ; // x       192  ALU
/*86231890*/  LEA R28, P1, R108, UR13, 0x2                               ; // x       192  ALU
/*862318a0*/  @P2   LDG.E.LTC128B.128 R84, [R114.64]                     ; // x       192  global load (DRAM/L2: weights or activations)
/*862318b0*/  IADD3.X R113, R115, c[0x0][0x1c4], RZ, P0, !PT             ; // x       192  ALU
/*862318c0*/  LEA R26, P0, R108, UR14, 0x2                               ; // x       192  ALU
/*862318d0*/  LEA.HI.X R29, R108, UR12, R109, 0x2, P1                    ; // x       192  ALU
/*862318e0*/  UIADD3 UR12, UR18, 0x60, URZ                               ; // x       192  Uniform
/*862318f0*/  LEA.HI.X R27, R108, UR15, R109, 0x2, P0                    ; // x       192  ALU
/*86231900*/  UIADD3 UR15, UR16, 0x60, URZ                               ; // x       192  Uniform
/*86231910*/  CS2R R80, SRZ                                              ; // x       192  Branch
/*86231920*/  LD.E R20, [R28.64]                                         ; // x       192  LSU / Load
/*86231930*/  UIADD3 UR13, UP1, UR12, UR6, URZ                           ; // x       192  Uniform
/*86231940*/  ISETP.GE.AND P2, PT, R107, 0x3, PT                         ; // x       192  ALU
/*86231950*/  UIADD3 UR14, UP0, UR15, UR11, URZ                          ; // x       192  Uniform
/*86231960*/  LD.E R21, [R28.64+0x1080]                                  ; // x       192  LSU / Load
/*86231970*/  ULEA.HI.X.SX32 UR12, UR12, UR7, 0x1, UP1                   ; // x       192  Uniform
/*86231980*/  SEL R105, R105, RZ, P2                                     ; // x       192  ALU
/*86231990*/  ULEA.HI.X.SX32 UR15, UR15, UR10, 0x1, UP0                  ; // x       192  Uniform
/*862319a0*/  LD.E R22, [R28.64+0x10]                                    ; // x       192  LSU / Load
/*862319b0*/  ISETP.GT.AND P2, PT, R107, 0x1, PT                         ; // x       192  ALU
/*862319c0*/  LD.E R23, [R28.64+0x1090]                                  ; // x       192  LSU / Load
/*862319d0*/  LD.E R18, [R26.64]                                         ; // x       192  LSU / Load
/*862319e0*/  LD.E R19, [R26.64+0x10]                                    ; // x       192  LSU / Load
/*862319f0*/  LD.E R24, [R26.64+0x1080]                                  ; // x       192  LSU / Load
/*86231a00*/  LD.E R25, [R26.64+0x1090]                                  ; // x       192  LSU / Load
/*86231a10*/  @P3   LDG.E.LTC128B.128 R80, [R112.64]                     ; // x       192  global load (DRAM/L2: weights or activations)
/*86231a20*/  LEA R28, P1, R108, UR13, 0x2                               ; // x       192  ALU
/*86231a30*/  LEA.HI.X R29, R108, UR12, R109, 0x2, P1                    ; // x       192  ALU
/*86231a40*/  UIADD3 UR12, UR18, 0x80, URZ                               ; // x       192  Uniform
/*86231a50*/  UIADD3 UR13, UP1, UR12, UR6, URZ                           ; // x       192  Uniform
/*86231a60*/  ULEA.HI.X.SX32 UR12, UR12, UR7, 0x1, UP1                   ; // x       192  Uniform
/*86231a70*/  LEA R26, P0, R108, UR14, 0x2                               ; // x       192  ALU
/*86231a80*/  LEA.HI.X R27, R108, UR15, R109, 0x2, P0                    ; // x       192  ALU
/*86231a90*/  UIADD3 UR15, UR16, 0x80, URZ                               ; // x       192  Uniform
/*86231aa0*/  UIADD3 UR14, UP0, UR15, UR11, URZ                          ; // x       192  Uniform
/*86231ab0*/  ULEA.HI.X.SX32 UR15, UR15, UR10, 0x1, UP0                  ; // x       192  Uniform
/*86231ac0*/  HMMA.16816.F32.BF16 R4, R12, R2, R4                        ; // x       192  tensor-core MMA
/*86231ad0*/  LD.E R2, [R26.64]                                          ; // x       192  LSU / Load
/*86231ae0*/  LD.E R3, [R26.64+0x10]                                     ; // x       192  LSU / Load
/*86231af0*/  HMMA.16816.F32.BF16 R8, R12, R16, R8                       ; // x       192  tensor-core MMA
/*86231b00*/  LD.E R12, [R28.64]                                         ; // x       192  LSU / Load
/*86231b10*/  LD.E R13, [R28.64+0x1080]                                  ; // x       192  LSU / Load
/*86231b20*/  LD.E R14, [R28.64+0x10]                                    ; // x       192  LSU / Load
/*86231b30*/  LD.E R15, [R28.64+0x1090]                                  ; // x       192  LSU / Load
/*86231b40*/  LD.E R16, [R26.64+0x1080]                                  ; // x       192  LSU / Load
/*86231b50*/  LD.E R17, [R26.64+0x1090]                                  ; // x       192  LSU / Load
/*86231b60*/  LEA R28, P1, R108, UR13, 0x2                               ; // x       192  ALU
/*86231b70*/  LEA.HI.X R29, R108, UR12, R109, 0x2, P1                    ; // x       192  ALU
/*86231b80*/  UIADD3 UR12, UR18, 0xa0, URZ                               ; // x       192  Uniform
/*86231b90*/  LEA R26, P0, R108, UR14, 0x2                               ; // x       192  ALU
/*86231ba0*/  UIADD3 UR13, UP1, UR12, UR6, URZ                           ; // x       192  Uniform
/*86231bb0*/  LEA.HI.X R27, R108, UR15, R109, 0x2, P0                    ; // x       192  ALU
/*86231bc0*/  UIADD3 UR15, UR16, 0xa0, URZ                               ; // x       192  Uniform
/*86231bd0*/  ULEA.HI.X.SX32 UR12, UR12, UR7, 0x1, UP1                   ; // x       192  Uniform
/*86231be0*/  UIADD3 UR14, UP0, UR15, UR11, URZ                          ; // x       192  Uniform
/*86231bf0*/  ULEA.HI.X.SX32 UR15, UR15, UR10, 0x1, UP0                  ; // x       192  Uniform
/*86231c00*/  HMMA.16816.F32.BF16 R4, R20, R18, R4                       ; // x       192  tensor-core MMA
/*86231c10*/  LD.E R18, [R26.64]                                         ; // x       192  LSU / Load
/*86231c20*/  LD.E R19, [R26.64+0x10]                                    ; // x       192  LSU / Load
/*86231c30*/  HMMA.16816.F32.BF16 R8, R20, R24, R8                       ; // x       192  tensor-core MMA
/*86231c40*/  LD.E R20, [R28.64]                                         ; // x       192  LSU / Load
/*86231c50*/  LD.E R21, [R28.64+0x1080]                                  ; // x       192  LSU / Load
/*86231c60*/  LD.E R22, [R28.64+0x10]                                    ; // x       192  LSU / Load
/*86231c70*/  LD.E R23, [R28.64+0x1090]                                  ; // x       192  LSU / Load
/*86231c80*/  LD.E R24, [R26.64+0x1080]                                  ; // x       192  LSU / Load
/*86231c90*/  LD.E R25, [R26.64+0x1090]                                  ; // x       192  LSU / Load
/*86231ca0*/  LEA R28, P1, R108, UR13, 0x2                               ; // x       192  ALU
/*86231cb0*/  LEA.HI.X R29, R108, UR12, R109, 0x2, P1                    ; // x       192  ALU
/*86231cc0*/  UIADD3 UR12, UR18, 0xc0, URZ                               ; // x       192  Uniform
/*86231cd0*/  LEA R26, P0, R108, UR14, 0x2                               ; // x       192  ALU
/*86231ce0*/  UIADD3 UR13, UP1, UR12, UR6, URZ                           ; // x       192  Uniform
/*86231cf0*/  LEA.HI.X R27, R108, UR15, R109, 0x2, P0                    ; // x       192  ALU
/*86231d00*/  UIADD3 UR15, UR16, 0xc0, URZ                               ; // x       192  Uniform
/*86231d10*/  ULEA.HI.X.SX32 UR12, UR12, UR7, 0x1, UP1                   ; // x       192  Uniform
/*86231d20*/  UIADD3 UR14, UP0, UR15, UR11, URZ                          ; // x       192  Uniform
/*86231d30*/  ULEA.HI.X.SX32 UR15, UR15, UR10, 0x1, UP0                  ; // x       192  Uniform
/*86231d40*/  HMMA.16816.F32.BF16 R4, R12, R2, R4                        ; // x       192  tensor-core MMA
/*86231d50*/  LD.E R2, [R26.64]                                          ; // x       192  LSU / Load
/*86231d60*/  LD.E R3, [R26.64+0x10]                                     ; // x       192  LSU / Load
/*86231d70*/  HMMA.16816.F32.BF16 R8, R12, R16, R8                       ; // x       192  tensor-core MMA
/*86231d80*/  LD.E R12, [R28.64]                                         ; // x       192  LSU / Load
/*86231d90*/  LD.E R13, [R28.64+0x1080]                                  ; // x       192  LSU / Load
/*86231da0*/  LD.E R14, [R28.64+0x10]                                    ; // x       192  LSU / Load
/*86231db0*/  LD.E R15, [R28.64+0x1090]                                  ; // x       192  LSU / Load
/*86231dc0*/  LD.E R16, [R26.64+0x1080]                                  ; // x       192  LSU / Load
/*86231dd0*/  LD.E R17, [R26.64+0x1090]                                  ; // x       192  LSU / Load
/*86231de0*/  LEA R28, P1, R108, UR13, 0x2                               ; // x       192  ALU
/*86231df0*/  UIADD3 UR13, UR16, 0xe0, URZ                               ; // x       192  Uniform
/*86231e00*/  LEA.HI.X R29, R108, UR12, R109, 0x2, P1                    ; // x       192  ALU
```

(first 200 of 284 instructions shown; the rest are in the full dump)
