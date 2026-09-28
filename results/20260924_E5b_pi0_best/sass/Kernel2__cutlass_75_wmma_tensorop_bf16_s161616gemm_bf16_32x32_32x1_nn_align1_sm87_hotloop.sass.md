# Hot loop — `Kernel2__cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_nn_align1` (sm87)

PC range 0x2ef29660-0x2ef29b10; backward BRA 0x2ef29b10 -> 0x2ef29660: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/Kernel2__cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_nn_align1_sm87.sass`; per-PC counts: `raw/Kernel2__cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_nn_align1_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*2ef29660*/  PRMT R15, R30, 0x5410, R29                                 ; // x     14336  ALU
/*2ef29670*/  PRMT R14, R32, 0x5410, R31                                 ; // x     14336  ALU
/*2ef29680*/  PRMT R13, R34, 0x5410, R33                                 ; // x     14336  ALU
/*2ef29690*/  PRMT R12, R36, 0x5410, R35                                 ; // x     14336  ALU
/*2ef296a0*/  PRMT R27, R38, 0x5410, R37                                 ; // x     14336  ALU
/*2ef296b0*/  PRMT R26, R40, 0x5410, R39                                 ; // x     14336  ALU
/*2ef296c0*/  STS.128 [R48+0xa00], R12                                   ; // x     14336  shared-memory store
/*2ef296d0*/  PRMT R25, R42, 0x5410, R41                                 ; // x     14336  ALU
/*2ef296e0*/  PRMT R24, R44, 0x5410, R43                                 ; // x     14336  ALU
/*2ef296f0*/  R2P PR, R23, 0xf                                           ; // x     14336  ALU
/*2ef29700*/  IADD3 R22, R22, -0x1, RZ                                   ; // x     14336  ALU
/*2ef29710*/  STS.128 [R48], R24                                         ; // x     14336  shared-memory store
/*2ef29720*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x     14336  block barrier
/*2ef29730*/  LD.E R2, [R18.64+0xa00]                                    ; // x     14336  LSU / Load
/*2ef29740*/  IMAD.MOV.U32 R24, RZ, RZ, R51                              ; // x     14336  FMA
/*2ef29750*/  LD.E R3, [R18.64+0xc80]                                    ; // x     14336  LSU / Load
/*2ef29760*/  IMAD.MOV.U32 R25, RZ, RZ, R50                              ; // x     14336  FMA
/*2ef29770*/  LD.E R16, [R18.64+0xa10]                                   ; // x     14336  LSU / Load
/*2ef29780*/  LD.E R17, [R18.64+0xc90]                                   ; // x     14336  LSU / Load
/*2ef29790*/  LD.E R12, [R20.64]                                         ; // x     14336  LSU / Load
/*2ef297a0*/  LD.E R13, [R20.64+0x280]                                   ; // x     14336  LSU / Load
/*2ef297b0*/  LD.E R14, [R20.64+0x10]                                    ; // x     14336  LSU / Load
/*2ef297c0*/  LD.E R15, [R20.64+0x290]                                   ; // x     14336  LSU / Load
/*2ef297d0*/  MOVM.16.MT88 R2, R2                                        ; // x     14336  Other
/*2ef297e0*/  MOVM.16.MT88 R3, R3                                        ; // x     14336  Other
/*2ef297f0*/  MOVM.16.MT88 R16, R16                                      ; // x     14336  Other
/*2ef29800*/  MOVM.16.MT88 R17, R17                                      ; // x     14336  Other
/*2ef29810*/  HMMA.16816.F32.BF16 R4, R12, R2, R4                        ; // x     14336  tensor-core MMA
/*2ef29820*/  LD.E R2, [R18.64+0xf00]                                    ; // x     14336  LSU / Load
/*2ef29830*/  LD.E R3, [R18.64+0x1180]                                   ; // x     14336  LSU / Load
/*2ef29840*/  HMMA.16816.F32.BF16 R8, R12, R16, R8                       ; // x     14336  tensor-core MMA
/*2ef29850*/  LD.E R16, [R18.64+0xf10]                                   ; // x     14336  LSU / Load
/*2ef29860*/  LD.E R17, [R18.64+0x1190]                                  ; // x     14336  LSU / Load
/*2ef29870*/  LD.E R12, [R20.64+0x20]                                    ; // x     14336  LSU / Load
/*2ef29880*/  LD.E R13, [R20.64+0x2a0]                                   ; // x     14336  LSU / Load
/*2ef29890*/  LD.E R14, [R20.64+0x30]                                    ; // x     14336  LSU / Load
/*2ef298a0*/  LD.E R15, [R20.64+0x2b0]                                   ; // x     14336  LSU / Load
/*2ef298b0*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x     14336  block barrier
/*2ef298c0*/  @P0   LDG.E.LTC128B.U16 R44, [R52.64]                      ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef298d0*/  @P1   LDG.E.LTC128B.U16 R43, [R52.64+0x2]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef298e0*/  @P2   LDG.E.LTC128B.U16 R42, [R52.64+0x4]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef298f0*/  @P3   LDG.E.LTC128B.U16 R41, [R52.64+0x6]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef29900*/  R2P PR, R23.B1, 0xf                                        ; // x     14336  ALU
/*2ef29910*/  @!UPT UIADD3 URZ, URZ, URZ, URZ                            ; // x     14336  Uniform
/*2ef29920*/  @P0   LDG.E.LTC128B.U16 R40, [R52.64+0x8]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef29930*/  @P1   LDG.E.LTC128B.U16 R39, [R52.64+0xa]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef29940*/  @P2   LDG.E.LTC128B.U16 R38, [R52.64+0xc]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef29950*/  @P3   LDG.E.LTC128B.U16 R37, [R52.64+0xe]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef29960*/  R2P PR, R28, 0xf                                           ; // x     14336  ALU
/*2ef29970*/  @!UPT UIADD3 URZ, URZ, URZ, URZ                            ; // x     14336  Uniform
/*2ef29980*/  @P0   LDG.E.LTC128B.U16 R36, [R24.64]                      ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef29990*/  @P1   LDG.E.LTC128B.U16 R35, [R24.64+0x2]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef299a0*/  @P2   LDG.E.LTC128B.U16 R34, [R24.64+0x4]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef299b0*/  @P3   LDG.E.LTC128B.U16 R33, [R24.64+0x6]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef299c0*/  R2P PR, R28.B1, 0xf                                        ; // x     14336  ALU
/*2ef299d0*/  ISETP.GE.AND P4, PT, R22, 0x3, PT                          ; // x     14336  ALU
/*2ef299e0*/  SEL R23, R23, RZ, P4                                       ; // x     14336  ALU
/*2ef299f0*/  @P0   LDG.E.LTC128B.U16 R32, [R24.64+0x8]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef29a00*/  IADD3 R52, P0, R52, c[0x0][0x1a8], RZ                      ; // x     14336  ALU
/*2ef29a10*/  @P1   LDG.E.LTC128B.U16 R31, [R24.64+0xa]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef29a20*/  IADD3 R51, P1, R24, c[0x0][0x1c8], RZ                      ; // x     14336  ALU
/*2ef29a30*/  IADD3.X R53, R53, c[0x0][0x1ac], RZ, P0, !PT               ; // x     14336  ALU
/*2ef29a40*/  @P2   LDG.E.LTC128B.U16 R30, [R24.64+0xc]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef29a50*/  ISETP.GT.AND P2, PT, R22, 0x1, PT                          ; // x     14336  ALU
/*2ef29a60*/  IADD3.X R50, R25, c[0x0][0x1cc], RZ, P1, !PT               ; // x     14336  ALU
/*2ef29a70*/  @P3   LDG.E.LTC128B.U16 R29, [R24.64+0xe]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef29a80*/  ISETP.GT.AND P3, PT, R22, 0x2, PT                          ; // x     14336  ALU
/*2ef29a90*/  @!UPT UIADD3 URZ, URZ, URZ, URZ                            ; // x     14336  Uniform
/*2ef29aa0*/  @!P3  IMAD.MOV.U32 R28, RZ, RZ, RZ                         ; // x     14336  FMA
/*2ef29ab0*/  MOVM.16.MT88 R2, R2                                        ; // x     14336  Other
/*2ef29ac0*/  MOVM.16.MT88 R3, R3                                        ; // x     14336  Other
/*2ef29ad0*/  MOVM.16.MT88 R16, R16                                      ; // x     14336  Other
/*2ef29ae0*/  MOVM.16.MT88 R17, R17                                      ; // x     14336  Other
/*2ef29af0*/  HMMA.16816.F32.BF16 R4, R12, R2, R4                        ; // x     14336  tensor-core MMA
/*2ef29b00*/  HMMA.16816.F32.BF16 R8, R12, R16, R8                       ; // x     14336  tensor-core MMA
/*2ef29b10*/  @P2   BRA 0x42ef29660                                      ; // x     14336  branch
```
