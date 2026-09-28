# Hot loop — `Kernel2__cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4` (sm87)

PC range 0x2ef03e70-0x2ef04680; backward BRA 0x2ef04680 -> 0x2ef03e70: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/Kernel2__cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4_sm87.sass`; per-PC counts: `raw/Kernel2__cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*2ef03e70*/  HMMA.1688.F32.TF32 R4, R36, R40, R4                        ; // x      4096  tensor-core MMA
/*2ef03e80*/  R2P PR, R80, 0x1                                           ; // x      4096  ALU
/*2ef03e90*/  LDSM.16.M88.4 R56, [R78+UR9]                               ; // x      4096  shared-memory matrix load (tensor-core operand)
/*2ef03ea0*/  UIADD3 UR8, UR9, 0x80, URZ                                 ; // x      4096  Uniform
/*2ef03eb0*/  IADD3 R85, R77, UR7, RZ                                    ; // x      4096  ALU
/*2ef03ec0*/  UIADD3 UR6, UR10, 0x80, URZ                                ; // x      4096  Uniform
/*2ef03ed0*/  R2P PR, R80, 0x2                                           ; // x      4096  ALU
/*2ef03ee0*/  UIADD3 UR18, UR18, 0x1, URZ                                ; // x      4096  Uniform
/*2ef03ef0*/  HMMA.1688.F32.TF32 R8, R36, R42, R8                        ; // x      4096  tensor-core MMA
/*2ef03f00*/  IADD3 R81, R75, UR7, RZ                                    ; // x      4096  ALU
/*2ef03f10*/  LDSM.16.M88.4 R52, [R79+UR10]                              ; // x      4096  shared-memory matrix load (tensor-core operand)
/*2ef03f20*/  UISETP.NE.AND UP1, UPT, UR18, 0x6, UPT                     ; // x      4096  Uniform
/*2ef03f30*/  IADD3 R84, R76, 0x4, RZ                                    ; // x      4096  ALU
/*2ef03f40*/  UIADD3 UR4, UR4, 0x1, URZ                                  ; // x      4096  Uniform
/*2ef03f50*/  IADD3 R76, R76, -0x1, RZ                                   ; // x      4096  ALU
/*2ef03f60*/  UIADD3 UR5, UR7, -0x280, URZ                               ; // x      4096  Uniform
/*2ef03f70*/  HMMA.1688.F32.TF32 R12, R36, R44, R12                      ; // x      4096  tensor-core MMA
/*2ef03f80*/  UISETP.NE.AND UP0, UPT, UR4, 0x6, UPT                      ; // x      4096  Uniform
/*2ef03f90*/  @P0   LDGSTS.E.BYPASS.LTC128B.128 [R77+UR7], [R68.64]      ; // x      4096  async global->shared copy (tile staging)
/*2ef03fa0*/  R2P PR, R73, 0x1                                           ; // x      4096  ALU
/*2ef03fb0*/  USEL UR4, UR4, URZ, UP0                                    ; // x      4096  Uniform
/*2ef03fc0*/  @!UP1 UIADD3 UR8, UR9, -0x280, URZ                         ; // x      4096  Uniform
/*2ef03fd0*/  HMMA.1688.F32.TF32 R16, R36, R46, R16                      ; // x      4096  tensor-core MMA
/*2ef03fe0*/  @!UP1 UIADD3 UR6, UR10, -0x280, URZ                        ; // x      4096  Uniform
/*2ef03ff0*/  @P1   LDGSTS.E.BYPASS.LTC128B.128 [R75+UR7], [R70.64]      ; // x      4096  async global->shared copy (tile staging)
/*2ef04000*/  R2P PR, R73, 0x2                                           ; // x      4096  ALU
/*2ef04010*/  @UP0  UIADD3 UR5, UR7, 0x80, URZ                           ; // x      4096  Uniform
/*2ef04020*/  @!UP1 UMOV UR18, URZ                                       ; // x      4096  Uniform
/*2ef04030*/  HMMA.1688.F32.TF32 R20, R48, R46, R20                      ; // x      4096  tensor-core MMA
/*2ef04040*/  @P0   LDGSTS.E.BYPASS.LTC128B.128 [R85+0x6000], [R82.64+UR20] ; // x      4096  async global->shared copy (tile staging)
/*2ef04050*/  ISETP.NE.AND P0, PT, R84, 0x6, PT                          ; // x      4096  ALU
/*2ef04060*/  UMOV UR7, UR5                                              ; // x      4096  Uniform
/*2ef04070*/  SEL R80, R80, RZ, P0                                       ; // x      4096  ALU
/*2ef04080*/  HMMA.1688.F32.TF32 R24, R48, R44, R24                      ; // x      4096  tensor-core MMA
/*2ef04090*/  @P1   LDGSTS.E.BYPASS.LTC128B.128 [R81+0x6000], [R82.64+UR22] ; // x      4096  async global->shared copy (tile staging)
/*2ef040a0*/  ISETP.NE.AND P1, PT, R84, 0x6, PT                          ; // x      4096  ALU
/*2ef040b0*/  HMMA.1688.F32.TF32 R28, R48, R42, R28                      ; // x      4096  tensor-core MMA
/*2ef040c0*/  FSETP.GEU.AND P6, PT, |R56|, +INF , PT                     ; // x      4096  FMA
/*2ef040d0*/  LDGDEPBAR                                                  ; // x      4096  LSU / Load
/*2ef040e0*/  FSETP.GEU.AND P5, PT, |R57|, +INF , PT                     ; // x      4096  FMA
/*2ef040f0*/  FSETP.GEU.AND P4, PT, |R52|, +INF , PT                     ; // x      4096  FMA
/*2ef04100*/  FSETP.GEU.AND P3, PT, |R53|, +INF , PT                     ; // x      4096  FMA
/*2ef04110*/  HMMA.1688.F32.TF32 R32, R48, R40, R32                      ; // x      4096  tensor-core MMA
/*2ef04120*/  FSETP.GEU.AND P2, PT, |R54|, +INF , PT                     ; // x      4096  FMA
/*2ef04130*/  LDSM.16.M88.4 R60, [R78+UR9+0x1800]                        ; // x      4096  shared-memory matrix load (tensor-core operand)
/*2ef04140*/  UMOV UR9, UR8                                              ; // x      4096  Uniform
/*2ef04150*/  IADD3 R70, P0, R70, UR16, RZ                               ; // x      4096  ALU
/*2ef04160*/  @!P1  IMAD.MOV.U32 R73, RZ, RZ, RZ                         ; // x      4096  FMA
/*2ef04170*/  FSETP.GEU.AND P1, PT, |R55|, +INF , PT                     ; // x      4096  FMA
/*2ef04180*/  @!P6  IMAD.IADD.U32 R56, R56, 0x1, R74                     ; // x      4096  FMA
/*2ef04190*/  FSETP.GEU.AND P6, PT, |R58|, +INF , PT                     ; // x      4096  FMA
/*2ef041a0*/  @!P5  IMAD.IADD.U32 R57, R57, 0x1, R74                     ; // x      4096  FMA
/*2ef041b0*/  LDSM.16.M88.4 R64, [R79+UR10+0x1800]                       ; // x      4096  shared-memory matrix load (tensor-core operand)
/*2ef041c0*/  FSETP.GEU.AND P5, PT, |R59|, +INF , PT                     ; // x      4096  FMA
/*2ef041d0*/  @!P4  IMAD.IADD.U32 R52, R52, 0x1, R74                     ; // x      4096  FMA
/*2ef041e0*/  UMOV UR10, UR6                                             ; // x      4096  Uniform
/*2ef041f0*/  @!P3  IMAD.IADD.U32 R53, R53, 0x1, R74                     ; // x      4096  FMA
/*2ef04200*/  IADD3.X R71, R71, UR17, RZ, P0, !PT                        ; // x      4096  ALU
/*2ef04210*/  @!P2  IMAD.IADD.U32 R54, R54, 0x1, R74                     ; // x      4096  FMA
/*2ef04220*/  DEPBAR.LE SB0, 0x4                                         ; // x      4096  wait on async-copy group
/*2ef04230*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x      4096  block barrier
/*2ef04240*/  @!P1  IMAD.IADD.U32 R55, R55, 0x1, R74                     ; // x      4096  FMA
/*2ef04250*/  @!P6  IMAD.IADD.U32 R58, R58, 0x1, R74                     ; // x      4096  FMA
/*2ef04260*/  @!P5  IMAD.IADD.U32 R59, R59, 0x1, R74                     ; // x      4096  FMA
/*2ef04270*/  HMMA.1688.F32.TF32 R4, R52, R56, R4                        ; // x      4096  tensor-core MMA
/*2ef04280*/  HMMA.1688.F32.TF32 R8, R52, R58, R8                        ; // x      4096  tensor-core MMA
/*2ef04290*/  FSETP.GEU.AND P4, PT, |R60|, +INF , PT                     ; // x      4096  FMA
/*2ef042a0*/  FSETP.GEU.AND P3, PT, |R61|, +INF , PT                     ; // x      4096  FMA
/*2ef042b0*/  FSETP.GEU.AND P2, PT, |R62|, +INF , PT                     ; // x      4096  FMA
/*2ef042c0*/  LDSM.16.M88.4 R44, [R2+UR8+0x1800]                         ; // x      4096  shared-memory matrix load (tensor-core operand)
/*2ef042d0*/  FSETP.GEU.AND P1, PT, |R63|, +INF , PT                     ; // x      4096  FMA
/*2ef042e0*/  LDSM.16.M88.4 R40, [R2+UR8]                                ; // x      4096  shared-memory matrix load (tensor-core operand)
/*2ef042f0*/  @!P4  IMAD.IADD.U32 R60, R60, 0x1, R74                     ; // x      4096  FMA
/*2ef04300*/  FSETP.GEU.AND P4, PT, |R64|, +INF , PT                     ; // x      4096  FMA
/*2ef04310*/  @!P3  IMAD.IADD.U32 R61, R61, 0x1, R74                     ; // x      4096  FMA
/*2ef04320*/  FSETP.GEU.AND P3, PT, |R65|, +INF , PT                     ; // x      4096  FMA
/*2ef04330*/  @!P2  IMAD.IADD.U32 R62, R62, 0x1, R74                     ; // x      4096  FMA
/*2ef04340*/  FSETP.GEU.AND P2, PT, |R66|, +INF , PT                     ; // x      4096  FMA
/*2ef04350*/  @!P1  IMAD.IADD.U32 R63, R63, 0x1, R74                     ; // x      4096  FMA
/*2ef04360*/  LDSM.16.M88.4 R36, [R72+UR6]                               ; // x      4096  shared-memory matrix load (tensor-core operand)
/*2ef04370*/  FSETP.GEU.AND P1, PT, |R67|, +INF , PT                     ; // x      4096  FMA
/*2ef04380*/  HMMA.1688.F32.TF32 R12, R52, R60, R12                      ; // x      4096  tensor-core MMA
/*2ef04390*/  @!P4  IMAD.IADD.U32 R64, R64, 0x1, R74                     ; // x      4096  FMA
/*2ef043a0*/  LDSM.16.M88.4 R48, [R72+UR6+0x1800]                        ; // x      4096  shared-memory matrix load (tensor-core operand)
/*2ef043b0*/  @!P3  IMAD.IADD.U32 R65, R65, 0x1, R74                     ; // x      4096  FMA
/*2ef043c0*/  HMMA.1688.F32.TF32 R16, R52, R62, R16                      ; // x      4096  tensor-core MMA
/*2ef043d0*/  @!P2  IMAD.IADD.U32 R66, R66, 0x1, R74                     ; // x      4096  FMA
/*2ef043e0*/  @!P1  IMAD.IADD.U32 R67, R67, 0x1, R74                     ; // x      4096  FMA
/*2ef043f0*/  HMMA.1688.F32.TF32 R20, R64, R62, R20                      ; // x      4096  tensor-core MMA
/*2ef04400*/  FSETP.GEU.AND P2, PT, |R44|, +INF , PT                     ; // x      4096  FMA
/*2ef04410*/  FSETP.GEU.AND P1, PT, |R45|, +INF , PT                     ; // x      4096  FMA
/*2ef04420*/  HMMA.1688.F32.TF32 R24, R64, R60, R24                      ; // x      4096  tensor-core MMA
/*2ef04430*/  FSETP.GEU.AND P6, PT, |R40|, +INF , PT                     ; // x      4096  FMA
/*2ef04440*/  FSETP.GEU.AND P5, PT, |R41|, +INF , PT                     ; // x      4096  FMA
/*2ef04450*/  FSETP.GEU.AND P4, PT, |R42|, +INF , PT                     ; // x      4096  FMA
/*2ef04460*/  @!P2  IMAD.IADD.U32 R44, R44, 0x1, R74                     ; // x      4096  FMA
/*2ef04470*/  FSETP.GEU.AND P3, PT, |R43|, +INF , PT                     ; // x      4096  FMA
/*2ef04480*/  HMMA.1688.F32.TF32 R28, R64, R58, R28                      ; // x      4096  tensor-core MMA
/*2ef04490*/  FSETP.GEU.AND P2, PT, |R38|, +INF , PT                     ; // x      4096  FMA
/*2ef044a0*/  @!P1  IMAD.IADD.U32 R45, R45, 0x1, R74                     ; // x      4096  FMA
/*2ef044b0*/  FSETP.GEU.AND P1, PT, |R39|, +INF , PT                     ; // x      4096  FMA
/*2ef044c0*/  @!P6  IMAD.IADD.U32 R40, R40, 0x1, R74                     ; // x      4096  FMA
/*2ef044d0*/  FSETP.GEU.AND P6, PT, |R46|, +INF , PT                     ; // x      4096  FMA
/*2ef044e0*/  @!P5  IMAD.IADD.U32 R41, R41, 0x1, R74                     ; // x      4096  FMA
/*2ef044f0*/  FSETP.GEU.AND P5, PT, |R47|, +INF , PT                     ; // x      4096  FMA
/*2ef04500*/  HMMA.1688.F32.TF32 R32, R64, R56, R32                      ; // x      4096  tensor-core MMA
/*2ef04510*/  @!P4  IMAD.IADD.U32 R42, R42, 0x1, R74                     ; // x      4096  FMA
/*2ef04520*/  FSETP.GEU.AND P4, PT, |R36|, +INF , PT                     ; // x      4096  FMA
/*2ef04530*/  @!P3  IMAD.IADD.U32 R43, R43, 0x1, R74                     ; // x      4096  FMA
/*2ef04540*/  FSETP.GEU.AND P3, PT, |R37|, +INF , PT                     ; // x      4096  FMA
/*2ef04550*/  @!P2  IMAD.IADD.U32 R38, R38, 0x1, R74                     ; // x      4096  FMA
/*2ef04560*/  IADD3 R82, P2, R82, UR14, RZ                               ; // x      4096  ALU
/*2ef04570*/  @!P1  IMAD.IADD.U32 R39, R39, 0x1, R74                     ; // x      4096  FMA
/*2ef04580*/  IADD3 R68, P1, R68, UR16, RZ                               ; // x      4096  ALU
/*2ef04590*/  @!P6  IMAD.IADD.U32 R46, R46, 0x1, R74                     ; // x      4096  FMA
/*2ef045a0*/  IADD3.X R83, R83, UR15, RZ, P2, !PT                        ; // x      4096  ALU
/*2ef045b0*/  @!P5  IMAD.IADD.U32 R47, R47, 0x1, R74                     ; // x      4096  FMA
/*2ef045c0*/  ISETP.GT.AND P2, PT, R76, -0x4, PT                         ; // x      4096  ALU
/*2ef045d0*/  @!P4  IMAD.IADD.U32 R36, R36, 0x1, R74                     ; // x      4096  FMA
/*2ef045e0*/  FSETP.GEU.AND P6, PT, |R48|, +INF , PT                     ; // x      4096  FMA
/*2ef045f0*/  @!P3  IMAD.IADD.U32 R37, R37, 0x1, R74                     ; // x      4096  FMA
/*2ef04600*/  FSETP.GEU.AND P5, PT, |R49|, +INF , PT                     ; // x      4096  FMA
/*2ef04610*/  FSETP.GEU.AND P4, PT, |R50|, +INF , PT                     ; // x      4096  FMA
/*2ef04620*/  FSETP.GEU.AND P3, PT, |R51|, +INF , PT                     ; // x      4096  FMA
/*2ef04630*/  IADD3.X R69, R69, UR17, RZ, P1, !PT                        ; // x      4096  ALU
/*2ef04640*/  @!P6  IMAD.IADD.U32 R48, R48, 0x1, R74                     ; // x      4096  FMA
/*2ef04650*/  @!P5  IMAD.IADD.U32 R49, R49, 0x1, R74                     ; // x      4096  FMA
/*2ef04660*/  @!P4  IMAD.IADD.U32 R50, R50, 0x1, R74                     ; // x      4096  FMA
/*2ef04670*/  @!P3  IMAD.IADD.U32 R51, R51, 0x1, R74                     ; // x      4096  FMA
/*2ef04680*/  @P2   BRA 0x42ef03e70                                      ; // x      4096  branch
```
