# Hot loop — `Kernel2__cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_tn_align1` (sm87)

PC range 0x2ef25f10-0x2ef26350; backward BRA 0x2ef26350 -> 0x2ef25f10: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/Kernel2__cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_tn_align1_sm87.sass`; per-PC counts: `raw/Kernel2__cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_tn_align1_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*2ef25f10*/  PRMT R15, R33, 0x5410, R32                                 ; // x     14336  ALU
/*2ef25f20*/  PRMT R14, R35, 0x5410, R34                                 ; // x     14336  ALU
/*2ef25f30*/  PRMT R13, R37, 0x5410, R36                                 ; // x     14336  ALU
/*2ef25f40*/  PRMT R12, R39, 0x5410, R38                                 ; // x     14336  ALU
/*2ef25f50*/  PRMT R27, R41, 0x5410, R40                                 ; // x     14336  ALU
/*2ef25f60*/  PRMT R26, R43, 0x5410, R42                                 ; // x     14336  ALU
/*2ef25f70*/  STS.128 [R51+0xa00], R12                                   ; // x     14336  shared-memory store
/*2ef25f80*/  PRMT R25, R45, 0x5410, R44                                 ; // x     14336  ALU
/*2ef25f90*/  PRMT R24, R47, 0x5410, R46                                 ; // x     14336  ALU
/*2ef25fa0*/  R2P PR, R30, 0xf                                           ; // x     14336  ALU
/*2ef25fb0*/  IADD3 R29, R29, -0x1, RZ                                   ; // x     14336  ALU
/*2ef25fc0*/  STS.128 [R51], R24                                         ; // x     14336  shared-memory store
/*2ef25fd0*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x     14336  block barrier
/*2ef25fe0*/  LD.E R2, [R18.64+0xa00]                                    ; // x     14336  LSU / Load
/*2ef25ff0*/  IMAD.MOV.U32 R24, RZ, RZ, R53                              ; // x     14336  FMA
/*2ef26000*/  LD.E R3, [R18.64+0xa10]                                    ; // x     14336  LSU / Load
/*2ef26010*/  IMAD.MOV.U32 R25, RZ, RZ, R23                              ; // x     14336  FMA
/*2ef26020*/  IMAD.MOV.U32 R23, RZ, RZ, R0                               ; // x     14336  FMA
/*2ef26030*/  LD.E R12, [R20.64]                                         ; // x     14336  LSU / Load
/*2ef26040*/  LD.E R13, [R20.64+0x280]                                   ; // x     14336  LSU / Load
/*2ef26050*/  LD.E R14, [R20.64+0x10]                                    ; // x     14336  LSU / Load
/*2ef26060*/  LD.E R15, [R20.64+0x290]                                   ; // x     14336  LSU / Load
/*2ef26070*/  LD.E R16, [R18.64+0xc80]                                   ; // x     14336  LSU / Load
/*2ef26080*/  LD.E R17, [R18.64+0xc90]                                   ; // x     14336  LSU / Load
/*2ef26090*/  HMMA.16816.F32.BF16 R4, R12, R2, R4                        ; // x     14336  tensor-core MMA
/*2ef260a0*/  LD.E R2, [R18.64+0xa20]                                    ; // x     14336  LSU / Load
/*2ef260b0*/  LD.E R3, [R18.64+0xa30]                                    ; // x     14336  LSU / Load
/*2ef260c0*/  HMMA.16816.F32.BF16 R8, R12, R16, R8                       ; // x     14336  tensor-core MMA
/*2ef260d0*/  LD.E R12, [R20.64+0x20]                                    ; // x     14336  LSU / Load
/*2ef260e0*/  LD.E R13, [R20.64+0x2a0]                                   ; // x     14336  LSU / Load
/*2ef260f0*/  LD.E R14, [R20.64+0x30]                                    ; // x     14336  LSU / Load
/*2ef26100*/  LD.E R15, [R20.64+0x2b0]                                   ; // x     14336  LSU / Load
/*2ef26110*/  LD.E R16, [R18.64+0xca0]                                   ; // x     14336  LSU / Load
/*2ef26120*/  LD.E R17, [R18.64+0xcb0]                                   ; // x     14336  LSU / Load
/*2ef26130*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x     14336  block barrier
/*2ef26140*/  @P0   LDG.E.LTC128B.U16 R47, [R24.64]                      ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef26150*/  @P1   LDG.E.LTC128B.U16 R46, [R24.64+0x2]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef26160*/  @P2   LDG.E.LTC128B.U16 R45, [R24.64+0x4]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef26170*/  @P3   LDG.E.LTC128B.U16 R44, [R24.64+0x6]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef26180*/  R2P PR, R30.B1, 0xf                                        ; // x     14336  ALU
/*2ef26190*/  @!UPT UIADD3 URZ, URZ, URZ, URZ                            ; // x     14336  Uniform
/*2ef261a0*/  @P0   LDG.E.LTC128B.U16 R43, [R24.64+0x8]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef261b0*/  @P1   LDG.E.LTC128B.U16 R42, [R24.64+0xa]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef261c0*/  @P2   LDG.E.LTC128B.U16 R41, [R24.64+0xc]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef261d0*/  @P3   LDG.E.LTC128B.U16 R40, [R24.64+0xe]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef261e0*/  R2P PR, R31, 0xf                                           ; // x     14336  ALU
/*2ef261f0*/  @!UPT UIADD3 URZ, URZ, URZ, URZ                            ; // x     14336  Uniform
/*2ef26200*/  @P0   LDG.E.LTC128B.U16 R39, [R22.64]                      ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef26210*/  @P1   LDG.E.LTC128B.U16 R38, [R22.64+0x2]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef26220*/  @P2   LDG.E.LTC128B.U16 R37, [R22.64+0x4]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef26230*/  @P3   LDG.E.LTC128B.U16 R36, [R22.64+0x6]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef26240*/  R2P PR, R31.B1, 0xf                                        ; // x     14336  ALU
/*2ef26250*/  ISETP.GE.AND P4, PT, R29, 0x3, PT                          ; // x     14336  ALU
/*2ef26260*/  SEL R30, R30, RZ, P4                                       ; // x     14336  ALU
/*2ef26270*/  @P0   LDG.E.LTC128B.U16 R35, [R22.64+0x8]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef26280*/  IADD3 R53, P0, R24, c[0x0][0x1a8], RZ                      ; // x     14336  ALU
/*2ef26290*/  @P1   LDG.E.LTC128B.U16 R34, [R22.64+0xa]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef262a0*/  @P2   LDG.E.LTC128B.U16 R33, [R22.64+0xc]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef262b0*/  ISETP.GT.AND P2, PT, R29, 0x1, PT                          ; // x     14336  ALU
/*2ef262c0*/  @P3   LDG.E.LTC128B.U16 R32, [R22.64+0xe]                  ; // x     14336  global load (DRAM/L2: weights or activations)
/*2ef262d0*/  ISETP.GT.AND P3, PT, R29, 0x2, PT                          ; // x     14336  ALU
/*2ef262e0*/  @!UPT UIADD3 URZ, URZ, URZ, URZ                            ; // x     14336  Uniform
/*2ef262f0*/  @!P3  IMAD.MOV.U32 R31, RZ, RZ, RZ                         ; // x     14336  FMA
/*2ef26300*/  IADD3 R22, P1, R22, c[0x0][0x1c8], RZ                      ; // x     14336  ALU
/*2ef26310*/  IADD3.X R0, R23, c[0x0][0x1cc], RZ, P1, !PT                ; // x     14336  ALU
/*2ef26320*/  IADD3.X R23, R25, c[0x0][0x1ac], RZ, P0, !PT               ; // x     14336  ALU
/*2ef26330*/  HMMA.16816.F32.BF16 R4, R12, R2, R4                        ; // x     14336  tensor-core MMA
/*2ef26340*/  HMMA.16816.F32.BF16 R8, R12, R16, R8                       ; // x     14336  tensor-core MMA
/*2ef26350*/  @P2   BRA 0x42ef25f10                                      ; // x     14336  branch
```
