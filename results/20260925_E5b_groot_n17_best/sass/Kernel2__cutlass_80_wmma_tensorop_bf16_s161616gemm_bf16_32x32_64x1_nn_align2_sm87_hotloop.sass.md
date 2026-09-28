# Hot loop — `Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nn_align2` (sm87)

PC range 0x86237af0-0x862382c0; backward BRA 0x862382c0 -> 0x86237af0: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nn_align2_sm87.sass`; per-PC counts: `raw/Kernel2__cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nn_align2_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*86237af0*/  STS.128 [R59], R36                                         ; // x      1152  shared-memory store
/*86237b00*/  R2P PR, R54, 0xf                                           ; // x      1152  ALU
/*86237b10*/  IADD3 R62, P5, P4, R49, c[0x0][0x1a0], -R43                ; // x      1152  ALU
/*86237b20*/  STS.128 [R59+0x900], R32                                   ; // x      1152  shared-memory store
/*86237b30*/  IADD3 R52, R52, -0x1, RZ                                   ; // x      1152  ALU
/*86237b40*/  IADD3.X R63, R48, c[0x0][0x1a4], ~R42, P5, P4              ; // x      1152  ALU
/*86237b50*/  STS.128 [R53+0x1200], R28                                  ; // x      1152  shared-memory store
/*86237b60*/  STS.128 [R53+0x1c00], R24                                  ; // x      1152  shared-memory store
/*86237b70*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x      1152  block barrier
/*86237b80*/  LD.E R2, [R16.64+0x1200]                                   ; // x      1152  LSU / Load
/*86237b90*/  LD.E R3, [R16.64+0x1480]                                   ; // x      1152  LSU / Load
/*86237ba0*/  LD.E R12, [R18.64]                                         ; // x      1152  LSU / Load
/*86237bb0*/  LD.E R13, [R18.64+0x480]                                   ; // x      1152  LSU / Load
/*86237bc0*/  LD.E R14, [R18.64+0x10]                                    ; // x      1152  LSU / Load
/*86237bd0*/  LD.E R15, [R18.64+0x490]                                   ; // x      1152  LSU / Load
/*86237be0*/  LD.E R20, [R16.64+0x1210]                                  ; // x      1152  LSU / Load
/*86237bf0*/  LD.E R0, [R16.64+0x1490]                                   ; // x      1152  LSU / Load
/*86237c00*/  MOVM.16.MT88 R2, R2                                        ; // x      1152  Other
/*86237c10*/  MOVM.16.MT88 R3, R3                                        ; // x      1152  Other
/*86237c20*/  HMMA.16816.F32.BF16 R4, R12, R2, R4                        ; // x      1152  tensor-core MMA
/*86237c30*/  MOVM.16.MT88 R2, R20                                       ; // x      1152  Other
/*86237c40*/  MOVM.16.MT88 R3, R0                                        ; // x      1152  Other
/*86237c50*/  LD.E R20, [R16.64+0x1710]                                  ; // x      1152  LSU / Load
/*86237c60*/  LD.E R0, [R16.64+0x1990]                                   ; // x      1152  LSU / Load
/*86237c70*/  HMMA.16816.F32.BF16 R8, R12, R2, R8                        ; // x      1152  tensor-core MMA
/*86237c80*/  LD.E R2, [R16.64+0x1700]                                   ; // x      1152  LSU / Load
/*86237c90*/  LD.E R3, [R16.64+0x1980]                                   ; // x      1152  LSU / Load
/*86237ca0*/  LD.E R12, [R18.64+0x20]                                    ; // x      1152  LSU / Load
/*86237cb0*/  LD.E R13, [R18.64+0x4a0]                                   ; // x      1152  LSU / Load
/*86237cc0*/  LD.E R14, [R18.64+0x30]                                    ; // x      1152  LSU / Load
/*86237cd0*/  LD.E R15, [R18.64+0x4b0]                                   ; // x      1152  LSU / Load
/*86237ce0*/  MOVM.16.MT88 R2, R2                                        ; // x      1152  Other
/*86237cf0*/  MOVM.16.MT88 R3, R3                                        ; // x      1152  Other
/*86237d00*/  HMMA.16816.F32.BF16 R4, R12, R2, R4                        ; // x      1152  tensor-core MMA
/*86237d10*/  MOVM.16.MT88 R2, R20                                       ; // x      1152  Other
/*86237d20*/  MOVM.16.MT88 R3, R0                                        ; // x      1152  Other
/*86237d30*/  LD.E R20, [R16.64+0x1c10]                                  ; // x      1152  LSU / Load
/*86237d40*/  LD.E R0, [R16.64+0x1e90]                                   ; // x      1152  LSU / Load
/*86237d50*/  HMMA.16816.F32.BF16 R8, R12, R2, R8                        ; // x      1152  tensor-core MMA
/*86237d60*/  LD.E R2, [R16.64+0x1c00]                                   ; // x      1152  LSU / Load
/*86237d70*/  LD.E R3, [R16.64+0x1e80]                                   ; // x      1152  LSU / Load
/*86237d80*/  LD.E R12, [R18.64+0x40]                                    ; // x      1152  LSU / Load
/*86237d90*/  LD.E R13, [R18.64+0x4c0]                                   ; // x      1152  LSU / Load
/*86237da0*/  LD.E R14, [R18.64+0x50]                                    ; // x      1152  LSU / Load
/*86237db0*/  LD.E R15, [R18.64+0x4d0]                                   ; // x      1152  LSU / Load
/*86237dc0*/  MOVM.16.MT88 R2, R2                                        ; // x      1152  Other
/*86237dd0*/  MOVM.16.MT88 R3, R3                                        ; // x      1152  Other
/*86237de0*/  HMMA.16816.F32.BF16 R4, R12, R2, R4                        ; // x      1152  tensor-core MMA
/*86237df0*/  MOVM.16.MT88 R2, R20                                       ; // x      1152  Other
/*86237e00*/  MOVM.16.MT88 R3, R0                                        ; // x      1152  Other
/*86237e10*/  LD.E R20, [R16.64+0x2110]                                  ; // x      1152  LSU / Load
/*86237e20*/  LD.E R0, [R16.64+0x2390]                                   ; // x      1152  LSU / Load
/*86237e30*/  HMMA.16816.F32.BF16 R8, R12, R2, R8                        ; // x      1152  tensor-core MMA
/*86237e40*/  LD.E R2, [R16.64+0x2100]                                   ; // x      1152  LSU / Load
/*86237e50*/  LD.E R3, [R16.64+0x2380]                                   ; // x      1152  LSU / Load
/*86237e60*/  LD.E R12, [R18.64+0x60]                                    ; // x      1152  LSU / Load
/*86237e70*/  LD.E R13, [R18.64+0x4e0]                                   ; // x      1152  LSU / Load
/*86237e80*/  LD.E R14, [R18.64+0x70]                                    ; // x      1152  LSU / Load
/*86237e90*/  LD.E R15, [R18.64+0x4f0]                                   ; // x      1152  LSU / Load
/*86237ea0*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x      1152  block barrier
/*86237eb0*/  @P0   LDG.E.LTC128B R36, [R62.64]                          ; // x      1152  global load (DRAM/L2: weights or activations)
/*86237ec0*/  MOVM.16.MT88 R2, R2                                        ; // x      1152  Other
/*86237ed0*/  MOVM.16.MT88 R3, R3                                        ; // x      1152  Other
/*86237ee0*/  HMMA.16816.F32.BF16 R4, R12, R2, R4                        ; // x      1152  tensor-core MMA
/*86237ef0*/  MOVM.16.MT88 R2, R20                                       ; // x      1152  Other
/*86237f00*/  MOVM.16.MT88 R3, R0                                        ; // x      1152  Other
/*86237f10*/  HMMA.16816.F32.BF16 R8, R12, R2, R8                        ; // x      1152  tensor-core MMA
/*86237f20*/  IADD3 R14, P6, R49, UR32, RZ                               ; // x      1152  ALU
/*86237f30*/  IADD3 R12, P5, R49, UR33, RZ                               ; // x      1152  ALU
/*86237f40*/  IADD3 R2, P4, R49, UR34, RZ                                ; // x      1152  ALU
/*86237f50*/  IADD3.X R15, R48, UR20, RZ, P6, !PT                        ; // x      1152  ALU
/*86237f60*/  IADD3.X R13, R48, UR19, RZ, P5, !PT                        ; // x      1152  ALU
/*86237f70*/  IADD3.X R3, R48, UR18, RZ, P4, !PT                         ; // x      1152  ALU
/*86237f80*/  @P1   LDG.E.LTC128B R37, [R14.64]                          ; // x      1152  global load (DRAM/L2: weights or activations)
/*86237f90*/  @P2   LDG.E.LTC128B R38, [R12.64]                          ; // x      1152  global load (DRAM/L2: weights or activations)
/*86237fa0*/  @P3   LDG.E.LTC128B R39, [R2.64]                           ; // x      1152  global load (DRAM/L2: weights or activations)
/*86237fb0*/  R2P PR, R54.B1, 0xf                                        ; // x      1152  ALU
/*86237fc0*/  IADD3 R62, P4, R49, UR23, RZ                               ; // x      1152  ALU
/*86237fd0*/  IADD3.X R63, R48, UR26, RZ, P4, !PT                        ; // x      1152  ALU
/*86237fe0*/  @P0   LDG.E.LTC128B R32, [R62.64]                          ; // x      1152  global load (DRAM/L2: weights or activations)
/*86237ff0*/  IADD3 R14, P6, R49, UR25, RZ                               ; // x      1152  ALU
/*86238000*/  IADD3 R12, P5, R49, UR22, RZ                               ; // x      1152  ALU
/*86238010*/  IADD3.X R15, R48, UR35, RZ, P6, !PT                        ; // x      1152  ALU
/*86238020*/  IADD3 R2, P4, R49, UR21, RZ                                ; // x      1152  ALU
/*86238030*/  IADD3.X R13, R48, UR24, RZ, P5, !PT                        ; // x      1152  ALU
/*86238040*/  @P1   LDG.E.LTC128B R33, [R14.64]                          ; // x      1152  global load (DRAM/L2: weights or activations)
/*86238050*/  IADD3.X R3, R48, UR6, RZ, P4, !PT                          ; // x      1152  ALU
/*86238060*/  @P2   LDG.E.LTC128B R34, [R12.64]                          ; // x      1152  global load (DRAM/L2: weights or activations)
/*86238070*/  @P3   LDG.E.LTC128B R35, [R2.64]                           ; // x      1152  global load (DRAM/L2: weights or activations)
/*86238080*/  R2P PR, R55, 0xf                                           ; // x      1152  ALU
/*86238090*/  IADD3 RZ, P5, P4, R47, c[0x0][0x1c0], -R45                 ; // x      1152  ALU
/*862380a0*/  IADD3 R62, R47, c[0x0][0x1c0], -R21                        ; // x      1152  ALU
/*862380b0*/  IADD3.X R63, R46, c[0x0][0x1c4], ~R44, P5, P4              ; // x      1152  ALU
/*862380c0*/  @P0   LDG.E.LTC128B R28, [R62.64]                          ; // x      1152  global load (DRAM/L2: weights or activations)
/*862380d0*/  IADD3 R14, P6, R47, UR31, RZ                               ; // x      1152  ALU
/*862380e0*/  IADD3 R12, P5, R47, UR30, RZ                               ; // x      1152  ALU
/*862380f0*/  IADD3.X R15, R46, UR10, RZ, P6, !PT                        ; // x      1152  ALU
/*86238100*/  IADD3 R2, P4, R47, UR29, RZ                                ; // x      1152  ALU
/*86238110*/  IADD3.X R13, R46, UR9, RZ, P5, !PT                         ; // x      1152  ALU
/*86238120*/  @P1   LDG.E.LTC128B R29, [R14.64]                          ; // x      1152  global load (DRAM/L2: weights or activations)
/*86238130*/  IADD3.X R3, R46, UR8, RZ, P4, !PT                          ; // x      1152  ALU
/*86238140*/  @P2   LDG.E.LTC128B R30, [R12.64]                          ; // x      1152  global load (DRAM/L2: weights or activations)
/*86238150*/  @P3   LDG.E.LTC128B R31, [R2.64]                           ; // x      1152  global load (DRAM/L2: weights or activations)
/*86238160*/  R2P PR, R55.B1, 0xf                                        ; // x      1152  ALU
/*86238170*/  IADD3 R62, P4, R47, UR28, RZ                               ; // x      1152  ALU
/*86238180*/  IADD3.X R63, R46, UR17, RZ, P4, !PT                        ; // x      1152  ALU
/*86238190*/  @P0   LDG.E.LTC128B R24, [R62.64]                          ; // x      1152  global load (DRAM/L2: weights or activations)
/*862381a0*/  IADD3 R14, P6, R47, UR16, RZ                               ; // x      1152  ALU
/*862381b0*/  IADD3 R12, P5, R47, UR14, RZ                               ; // x      1152  ALU
/*862381c0*/  IADD3.X R15, R46, UR15, RZ, P6, !PT                        ; // x      1152  ALU
/*862381d0*/  IADD3 R2, P4, R47, UR12, RZ                                ; // x      1152  ALU
/*862381e0*/  IADD3.X R13, R46, UR13, RZ, P5, !PT                        ; // x      1152  ALU
/*862381f0*/  @P1   LDG.E.LTC128B R25, [R14.64]                          ; // x      1152  global load (DRAM/L2: weights or activations)
/*86238200*/  IADD3.X R3, R46, UR11, RZ, P4, !PT                         ; // x      1152  ALU
/*86238210*/  ISETP.GT.AND P4, PT, R52, 0x1, PT                          ; // x      1152  ALU
/*86238220*/  @P2   LDG.E.LTC128B R26, [R12.64]                          ; // x      1152  global load (DRAM/L2: weights or activations)
/*86238230*/  ISETP.GT.AND P5, PT, R52, 0x2, PT                          ; // x      1152  ALU
/*86238240*/  IADD3 R49, P1, P0, R49, c[0x0][0x1a0], R23                 ; // x      1152  ALU
/*86238250*/  @P3   LDG.E.LTC128B R27, [R2.64]                           ; // x      1152  global load (DRAM/L2: weights or activations)
/*86238260*/  IADD3 R47, P3, P2, R47, c[0x0][0x1c0], R41                 ; // x      1152  ALU
/*86238270*/  ISETP.GE.AND P6, PT, R52, 0x3, PT                          ; // x      1152  ALU
/*86238280*/  IADD3.X R46, R46, c[0x0][0x1c4], R40, P3, P2               ; // x      1152  ALU
/*86238290*/  IADD3.X R48, R48, c[0x0][0x1a4], R22, P1, P0               ; // x      1152  ALU
/*862382a0*/  SEL R54, R54, RZ, P6                                       ; // x      1152  ALU
/*862382b0*/  @!P5  IMAD.MOV.U32 R55, RZ, RZ, RZ                         ; // x      1152  FMA
/*862382c0*/  @P4   BRA 0x386237af0                                      ; // x      1152  branch
```
