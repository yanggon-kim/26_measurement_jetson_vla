# Hot loop — `Kernel2__cutlass_80_tensorop_s1688gemm_256x64_16x4_tn_align4` (sm87)

PC range 0x2eef9140-0x2eefa1f0; backward BRA 0x2eefa1f0 -> 0x2eef9140: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/Kernel2__cutlass_80_tensorop_s1688gemm_256x64_16x4_tn_align4_sm87.sass`; per-PC counts: `raw/Kernel2__cutlass_80_tensorop_s1688gemm_256x64_16x4_tn_align4_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*2eef9140*/  HMMA.1688.F32.TF32 R4, R132, R136, R4                      ; // x      2048  tensor-core MMA
/*2eef9150*/  LDSM.16.M88.4 R168, [R210+UR6]                             ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9160*/  UIADD3 UR5, UR6, 0x80, URZ                                 ; // x      2048  Uniform
/*2eef9170*/  UIADD3 UR4, UR7, 0x80, URZ                                 ; // x      2048  Uniform
/*2eef9180*/  R2P PR, R212, 0x1                                          ; // x      2048  ALU
/*2eef9190*/  IMAD.MOV.U32 R216, RZ, RZ, R208                            ; // x      2048  FMA
/*2eef91a0*/  IADD3 R213, R198, UR18, RZ                                 ; // x      2048  ALU
/*2eef91b0*/  HMMA.1688.F32.TF32 R8, R132, R138, R8                      ; // x      2048  tensor-core MMA
/*2eef91c0*/  LDSM.16.M88.4 R164, [R211+UR7]                             ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef91d0*/  UIADD3 UR15, UR15, 0x1, URZ                                ; // x      2048  Uniform
/*2eef91e0*/  IMAD.MOV.U32 R217, RZ, RZ, R209                            ; // x      2048  FMA
/*2eef91f0*/  UIADD3 UR16, UR16, 0x1, URZ                                ; // x      2048  Uniform
/*2eef9200*/  UISETP.NE.AND UP0, UPT, UR15, 0x4, UPT                     ; // x      2048  Uniform
/*2eef9210*/  HMMA.1688.F32.TF32 R12, R132, R140, R12                    ; // x      2048  tensor-core MMA
/*2eef9220*/  LDSM.16.M88.4 R172, [R210+UR6+0x1000]                      ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9230*/  UISETP.NE.AND UP1, UPT, UR16, 0x4, UPT                     ; // x      2048  Uniform
/*2eef9240*/  @P0   IMAD.MOV.U32 R207, RZ, RZ, R205                      ; // x      2048  FMA
/*2eef9250*/  UIADD3 UR17, UR18, 0x80, URZ                               ; // x      2048  Uniform
/*2eef9260*/  IMAD.WIDE.U32 R218, R204, 0x6, R216                        ; // x      2048  FMA
/*2eef9270*/  UIADD3 UR14, UR13, 0x80, URZ                               ; // x      2048  Uniform
/*2eef9280*/  HMMA.1688.F32.TF32 R16, R132, R142, R16                    ; // x      2048  tensor-core MMA
/*2eef9290*/  LDSM.16.M88.4 R176, [R210+UR6+0x2000]                      ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef92a0*/  @!UP0 UIADD3 UR5, UR6, -0x180, URZ                         ; // x      2048  Uniform
/*2eef92b0*/  @!UP0 UIADD3 UR4, UR7, -0x180, URZ                         ; // x      2048  Uniform
/*2eef92c0*/  IADD3 R219, R219, UR11, RZ                                 ; // x      2048  ALU
/*2eef92d0*/  @!UP1 UIADD3 UR17, UR18, -0x180, URZ                       ; // x      2048  Uniform
/*2eef92e0*/  HMMA.1688.F32.TF32 R20, R132, R144, R20                    ; // x      2048  tensor-core MMA
/*2eef92f0*/  LDSM.16.M88.4 R180, [R210+UR6+0x3000]                      ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9300*/  @!UP1 UIADD3 UR14, UR13, -0x180, URZ                       ; // x      2048  Uniform
/*2eef9310*/  @!UP1 UMOV UR16, URZ                                       ; // x      2048  Uniform
/*2eef9320*/  @!UP0 UMOV UR15, URZ                                       ; // x      2048  Uniform
/*2eef9330*/  HMMA.1688.F32.TF32 R24, R132, R146, R24                    ; // x      2048  tensor-core MMA
/*2eef9340*/  LDSM.16.M88.4 R184, [R211+UR7+0x1000]                      ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9350*/  UMOV UR6, UR5                                              ; // x      2048  Uniform
/*2eef9360*/  HMMA.1688.F32.TF32 R28, R132, R148, R28                    ; // x      2048  tensor-core MMA
/*2eef9370*/  LDSM.16.M88.4 R188, [R211+UR7+0x2000]                      ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9380*/  HMMA.1688.F32.TF32 R32, R132, R150, R32                    ; // x      2048  tensor-core MMA
/*2eef9390*/  LDSM.16.M88.4 R192, [R211+UR7+0x3000]                      ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef93a0*/  UMOV UR7, UR4                                              ; // x      2048  Uniform
/*2eef93b0*/  HMMA.1688.F32.TF32 R36, R152, R150, R36                    ; // x      2048  tensor-core MMA
/*2eef93c0*/  @P0   LDGSTS.E.BYPASS.LTC128B.128 [R201+UR13], [R206.64]   ; // x      2048  async global->shared copy (tile staging)
/*2eef93d0*/  R2P PR, R197, 0xf                                          ; // x      2048  ALU
/*2eef93e0*/  HMMA.1688.F32.TF32 R40, R152, R148, R40                    ; // x      2048  tensor-core MMA
/*2eef93f0*/  HMMA.1688.F32.TF32 R44, R152, R146, R44                    ; // x      2048  tensor-core MMA
/*2eef9400*/  @P1   LDGSTS.E.BYPASS.LTC128B.128 [R213], [R208.64+UR22]   ; // x      2048  async global->shared copy (tile staging)
/*2eef9410*/  R2P PR, R212, 0x2                                          ; // x      2048  ALU
/*2eef9420*/  IADD3 R214, P4, R206, c[0x0][0x1a0], RZ                    ; // x      2048  ALU
/*2eef9430*/  HMMA.1688.F32.TF32 R48, R152, R144, R48                    ; // x      2048  tensor-core MMA
/*2eef9440*/  @P3   LDGSTS.E.BYPASS.LTC128B.128 [R213+0x1000], [R208.64+UR24] ; // x      2048  async global->shared copy (tile staging)
/*2eef9450*/  IADD3.X R215, R205, c[0x0][0x1a4], RZ, P4, !PT             ; // x      2048  ALU
/*2eef9460*/  IMAD.WIDE.U32 R206, R204, 0x5, R216                        ; // x      2048  FMA
/*2eef9470*/  IADD3 R205, R199, UR18, RZ                                 ; // x      2048  ALU
/*2eef9480*/  HMMA.1688.F32.TF32 R52, R152, R142, R52                    ; // x      2048  tensor-core MMA
/*2eef9490*/  @P1   LDGSTS.E.BYPASS.LTC128B.128 [R200+UR13], [R214.64]   ; // x      2048  async global->shared copy (tile staging)
/*2eef94a0*/  UMOV UR13, UR14                                            ; // x      2048  Uniform
/*2eef94b0*/  IADD3 R207, R207, UR12, RZ                                 ; // x      2048  ALU
/*2eef94c0*/  HMMA.1688.F32.TF32 R56, R152, R140, R56                    ; // x      2048  tensor-core MMA
/*2eef94d0*/  @P2   LDGSTS.E.BYPASS.LTC128B.128 [R205+0x1000], [R208.64+UR20] ; // x      2048  async global->shared copy (tile staging)
/*2eef94e0*/  HMMA.1688.F32.TF32 R60, R152, R138, R60                    ; // x      2048  tensor-core MMA
/*2eef94f0*/  @P0   LDGSTS.E.BYPASS.LTC128B.128 [R199+UR18], [R216.64]   ; // x      2048  async global->shared copy (tile staging)
/*2eef9500*/  R2P PR, R197.B1, 0xf                                       ; // x      2048  ALU
/*2eef9510*/  FSETP.GEU.AND P6, PT, |R168|, +INF , PT                    ; // x      2048  FMA
/*2eef9520*/  FSETP.GEU.AND P5, PT, |R169|, +INF , PT                    ; // x      2048  FMA
/*2eef9530*/  HMMA.1688.F32.TF32 R64, R152, R136, R64                    ; // x      2048  tensor-core MMA
/*2eef9540*/  FSETP.GEU.AND P4, PT, |R164|, +INF , PT                    ; // x      2048  FMA
/*2eef9550*/  HMMA.1688.F32.TF32 R68, R156, R136, R68                    ; // x      2048  tensor-core MMA
/*2eef9560*/  @P1   LDGSTS.E.BYPASS.LTC128B.128 [R198+UR18+0x2000], [R206.64] ; // x      2048  async global->shared copy (tile staging)
/*2eef9570*/  @!P6  IMAD.IADD.U32 R168, R168, 0x1, R203                  ; // x      2048  FMA
/*2eef9580*/  FSETP.GEU.AND P6, PT, |R170|, +INF , PT                    ; // x      2048  FMA
/*2eef9590*/  @!P5  IMAD.IADD.U32 R169, R169, 0x1, R203                  ; // x      2048  FMA
/*2eef95a0*/  FSETP.GEU.AND P5, PT, |R171|, +INF , PT                    ; // x      2048  FMA
/*2eef95b0*/  HMMA.1688.F32.TF32 R72, R156, R138, R72                    ; // x      2048  tensor-core MMA
/*2eef95c0*/  @P0   LDGSTS.E.BYPASS.LTC128B.128 [R205+0x2000], [R208.64+UR26] ; // x      2048  async global->shared copy (tile staging)
/*2eef95d0*/  IMAD.WIDE.U32 R216, R204, 0x7, R216                        ; // x      2048  FMA
/*2eef95e0*/  HMMA.1688.F32.TF32 R76, R156, R140, R76                    ; // x      2048  tensor-core MMA
/*2eef95f0*/  @P2   LDGSTS.E.BYPASS.LTC128B.128 [R199+UR18+0x3000], [R218.64] ; // x      2048  async global->shared copy (tile staging)
/*2eef9600*/  IADD3 R217, R217, UR10, RZ                                 ; // x      2048  ALU
/*2eef9610*/  @!P4  IMAD.IADD.U32 R164, R164, 0x1, R203                  ; // x      2048  FMA
/*2eef9620*/  FSETP.GEU.AND P4, PT, |R172|, +INF , PT                    ; // x      2048  FMA
/*2eef9630*/  @!P6  IMAD.IADD.U32 R170, R170, 0x1, R203                  ; // x      2048  FMA
/*2eef9640*/  HMMA.1688.F32.TF32 R80, R156, R142, R80                    ; // x      2048  tensor-core MMA
/*2eef9650*/  @P3   LDGSTS.E.BYPASS.LTC128B.128 [R198+UR18+0x3000], [R216.64] ; // x      2048  async global->shared copy (tile staging)
/*2eef9660*/  UMOV UR18, UR17                                            ; // x      2048  Uniform
/*2eef9670*/  FSETP.GEU.AND P6, PT, |R176|, +INF , PT                    ; // x      2048  FMA
/*2eef9680*/  @!P5  IMAD.IADD.U32 R171, R171, 0x1, R203                  ; // x      2048  FMA
/*2eef9690*/  IADD3 R207, R202, -0x1, RZ                                 ; // x      2048  ALU
/*2eef96a0*/  IADD3 R206, P0, R214, c[0x0][0x1a8], RZ                    ; // x      2048  ALU
/*2eef96b0*/  HMMA.1688.F32.TF32 R84, R156, R144, R84                    ; // x      2048  tensor-core MMA
/*2eef96c0*/  LDGDEPBAR                                                  ; // x      2048  LSU / Load
/*2eef96d0*/  ISETP.NE.AND P3, PT, R207, RZ, PT                          ; // x      2048  ALU
/*2eef96e0*/  @!P4  IMAD.IADD.U32 R172, R172, 0x1, R203                  ; // x      2048  FMA
/*2eef96f0*/  IADD3 R208, P1, R216, c[0x0][0x1c8], RZ                    ; // x      2048  ALU
/*2eef9700*/  IADD3.X R205, R215, c[0x0][0x1ac], RZ, P0, !PT             ; // x      2048  ALU
/*2eef9710*/  HMMA.1688.F32.TF32 R88, R156, R146, R88                    ; // x      2048  tensor-core MMA
/*2eef9720*/  FSETP.GEU.AND P0, PT, |R167|, +INF , PT                    ; // x      2048  FMA
/*2eef9730*/  @!P6  IMAD.IADD.U32 R176, R176, 0x1, R203                  ; // x      2048  FMA
/*2eef9740*/  IADD3.X R209, R217, c[0x0][0x1cc], RZ, P1, !PT             ; // x      2048  ALU
/*2eef9750*/  FSETP.GEU.AND P1, PT, |R166|, +INF , PT                    ; // x      2048  FMA
/*2eef9760*/  HMMA.1688.F32.TF32 R92, R156, R148, R92                    ; // x      2048  tensor-core MMA
/*2eef9770*/  @!P3  IMAD.MOV.U32 R197, RZ, RZ, RZ                        ; // x      2048  FMA
/*2eef9780*/  FSETP.GEU.AND P3, PT, |R165|, +INF , PT                    ; // x      2048  FMA
/*2eef9790*/  FSETP.GEU.AND P5, PT, |R177|, +INF , PT                    ; // x      2048  FMA
/*2eef97a0*/  FSETP.GEU.AND P4, PT, |R178|, +INF , PT                    ; // x      2048  FMA
/*2eef97b0*/  HMMA.1688.F32.TF32 R96, R156, R150, R96                    ; // x      2048  tensor-core MMA
/*2eef97c0*/  DEPBAR.LE SB0, 0x2                                         ; // x      2048  wait on async-copy group
/*2eef97d0*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x      2048  block barrier
/*2eef97e0*/  @!P1  IMAD.IADD.U32 R166, R166, 0x1, R203                  ; // x      2048  FMA
/*2eef97f0*/  FSETP.GEU.AND P1, PT, |R174|, +INF , PT                    ; // x      2048  FMA
/*2eef9800*/  HMMA.1688.F32.TF32 R100, R160, R150, R100                  ; // x      2048  tensor-core MMA
/*2eef9810*/  @!P3  IMAD.IADD.U32 R165, R165, 0x1, R203                  ; // x      2048  FMA
/*2eef9820*/  FSETP.GEU.AND P3, PT, |R173|, +INF , PT                    ; // x      2048  FMA
/*2eef9830*/  @!P0  IMAD.IADD.U32 R167, R167, 0x1, R203                  ; // x      2048  FMA
/*2eef9840*/  HMMA.1688.F32.TF32 R104, R160, R148, R104                  ; // x      2048  tensor-core MMA
/*2eef9850*/  FSETP.GEU.AND P0, PT, |R175|, +INF , PT                    ; // x      2048  FMA
/*2eef9860*/  @!P1  IMAD.IADD.U32 R174, R174, 0x1, R203                  ; // x      2048  FMA
/*2eef9870*/  FSETP.GEU.AND P1, PT, |R180|, +INF , PT                    ; // x      2048  FMA
/*2eef9880*/  @!P5  IMAD.IADD.U32 R177, R177, 0x1, R203                  ; // x      2048  FMA
/*2eef9890*/  FSETP.GEU.AND P6, PT, |R182|, +INF , PT                    ; // x      2048  FMA
/*2eef98a0*/  HMMA.1688.F32.TF32 R108, R160, R146, R108                  ; // x      2048  tensor-core MMA
/*2eef98b0*/  FSETP.GEU.AND P5, PT, |R183|, +INF , PT                    ; // x      2048  FMA
/*2eef98c0*/  @!P3  IMAD.IADD.U32 R173, R173, 0x1, R203                  ; // x      2048  FMA
/*2eef98d0*/  FSETP.GEU.AND P3, PT, |R179|, +INF , PT                    ; // x      2048  FMA
/*2eef98e0*/  @!P4  IMAD.IADD.U32 R178, R178, 0x1, R203                  ; // x      2048  FMA
/*2eef98f0*/  FSETP.GEU.AND P4, PT, |R184|, +INF , PT                    ; // x      2048  FMA
/*2eef9900*/  HMMA.1688.F32.TF32 R112, R160, R144, R112                  ; // x      2048  tensor-core MMA
/*2eef9910*/  ISETP.NE.AND P2, PT, R207, RZ, PT                          ; // x      2048  ALU
/*2eef9920*/  @!P0  IMAD.IADD.U32 R175, R175, 0x1, R203                  ; // x      2048  FMA
/*2eef9930*/  FSETP.GEU.AND P0, PT, |R181|, +INF , PT                    ; // x      2048  FMA
/*2eef9940*/  @!P1  IMAD.IADD.U32 R180, R180, 0x1, R203                  ; // x      2048  FMA
/*2eef9950*/  FSETP.GEU.AND P1, PT, |R186|, +INF , PT                    ; // x      2048  FMA
/*2eef9960*/  HMMA.1688.F32.TF32 R116, R160, R142, R116                  ; // x      2048  tensor-core MMA
/*2eef9970*/  @!P3  IMAD.IADD.U32 R179, R179, 0x1, R203                  ; // x      2048  FMA
/*2eef9980*/  FSETP.GEU.AND P3, PT, |R185|, +INF , PT                    ; // x      2048  FMA
/*2eef9990*/  @!P6  IMAD.IADD.U32 R182, R182, 0x1, R203                  ; // x      2048  FMA
/*2eef99a0*/  FSETP.GEU.AND P6, PT, |R190|, +INF , PT                    ; // x      2048  FMA
/*2eef99b0*/  HMMA.1688.F32.TF32 R120, R160, R140, R120                  ; // x      2048  tensor-core MMA
/*2eef99c0*/  @!P0  IMAD.IADD.U32 R181, R181, 0x1, R203                  ; // x      2048  FMA
/*2eef99d0*/  FSETP.GEU.AND P0, PT, |R187|, +INF , PT                    ; // x      2048  FMA
/*2eef99e0*/  @!P5  IMAD.IADD.U32 R183, R183, 0x1, R203                  ; // x      2048  FMA
/*2eef99f0*/  FSETP.GEU.AND P5, PT, |R191|, +INF , PT                    ; // x      2048  FMA
/*2eef9a00*/  HMMA.1688.F32.TF32 R124, R160, R138, R124                  ; // x      2048  tensor-core MMA
/*2eef9a10*/  @!P4  IMAD.IADD.U32 R184, R184, 0x1, R203                  ; // x      2048  FMA
/*2eef9a20*/  FSETP.GEU.AND P4, PT, |R192|, +INF , PT                    ; // x      2048  FMA
/*2eef9a30*/  @!P3  IMAD.IADD.U32 R185, R185, 0x1, R203                  ; // x      2048  FMA
/*2eef9a40*/  FSETP.GEU.AND P3, PT, |R193|, +INF , PT                    ; // x      2048  FMA
/*2eef9a50*/  HMMA.1688.F32.TF32 R128, R160, R136, R128                  ; // x      2048  tensor-core MMA
/*2eef9a60*/  @!P1  IMAD.IADD.U32 R186, R186, 0x1, R203                  ; // x      2048  FMA
/*2eef9a70*/  FSETP.GEU.AND P1, PT, |R188|, +INF , PT                    ; // x      2048  FMA
/*2eef9a80*/  @!P0  IMAD.IADD.U32 R187, R187, 0x1, R203                  ; // x      2048  FMA
/*2eef9a90*/  FSETP.GEU.AND P0, PT, |R189|, +INF , PT                    ; // x      2048  FMA
/*2eef9aa0*/  HMMA.1688.F32.TF32 R4, R164, R168, R4                      ; // x      2048  tensor-core MMA
/*2eef9ab0*/  @!P6  IMAD.IADD.U32 R190, R190, 0x1, R203                  ; // x      2048  FMA
/*2eef9ac0*/  SEL R212, R212, RZ, P2                                     ; // x      2048  ALU
/*2eef9ad0*/  @!P5  IMAD.IADD.U32 R191, R191, 0x1, R203                  ; // x      2048  FMA
/*2eef9ae0*/  HMMA.1688.F32.TF32 R8, R164, R170, R8                      ; // x      2048  tensor-core MMA
/*2eef9af0*/  @!P1  IMAD.IADD.U32 R188, R188, 0x1, R203                  ; // x      2048  FMA
/*2eef9b00*/  FSETP.GEU.AND P1, PT, |R194|, +INF , PT                    ; // x      2048  FMA
/*2eef9b10*/  @!P0  IMAD.IADD.U32 R189, R189, 0x1, R203                  ; // x      2048  FMA
/*2eef9b20*/  FSETP.GEU.AND P0, PT, |R195|, +INF , PT                    ; // x      2048  FMA
/*2eef9b30*/  HMMA.1688.F32.TF32 R12, R164, R172, R12                    ; // x      2048  tensor-core MMA
/*2eef9b40*/  @!P4  IMAD.IADD.U32 R192, R192, 0x1, R203                  ; // x      2048  FMA
/*2eef9b50*/  @!P3  IMAD.IADD.U32 R193, R193, 0x1, R203                  ; // x      2048  FMA
/*2eef9b60*/  HMMA.1688.F32.TF32 R16, R164, R174, R16                    ; // x      2048  tensor-core MMA
/*2eef9b70*/  @!P1  IMAD.IADD.U32 R194, R194, 0x1, R203                  ; // x      2048  FMA
/*2eef9b80*/  @!P0  IMAD.IADD.U32 R195, R195, 0x1, R203                  ; // x      2048  FMA
/*2eef9b90*/  HMMA.1688.F32.TF32 R20, R164, R176, R20                    ; // x      2048  tensor-core MMA
/*2eef9ba0*/  LDSM.16.M88.4 R140, [R3+UR5+0x1000]                        ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9bb0*/  HMMA.1688.F32.TF32 R24, R164, R178, R24                    ; // x      2048  tensor-core MMA
/*2eef9bc0*/  LDSM.16.M88.4 R144, [R3+UR5+0x2000]                        ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9bd0*/  HMMA.1688.F32.TF32 R28, R164, R180, R28                    ; // x      2048  tensor-core MMA
/*2eef9be0*/  LDSM.16.M88.4 R136, [R3+UR5]                               ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9bf0*/  HMMA.1688.F32.TF32 R32, R164, R182, R32                    ; // x      2048  tensor-core MMA
/*2eef9c00*/  LDSM.16.M88.4 R132, [R196+UR4]                             ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9c10*/  HMMA.1688.F32.TF32 R36, R184, R182, R36                    ; // x      2048  tensor-core MMA
/*2eef9c20*/  LDSM.16.M88.4 R152, [R196+UR4+0x1000]                      ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9c30*/  HMMA.1688.F32.TF32 R40, R184, R180, R40                    ; // x      2048  tensor-core MMA
/*2eef9c40*/  LDSM.16.M88.4 R148, [R3+UR5+0x3000]                        ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9c50*/  HMMA.1688.F32.TF32 R44, R184, R178, R44                    ; // x      2048  tensor-core MMA
/*2eef9c60*/  LDSM.16.M88.4 R156, [R196+UR4+0x2000]                      ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9c70*/  HMMA.1688.F32.TF32 R48, R184, R176, R48                    ; // x      2048  tensor-core MMA
/*2eef9c80*/  LDSM.16.M88.4 R160, [R196+UR4+0x3000]                      ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2eef9c90*/  HMMA.1688.F32.TF32 R52, R184, R174, R52                    ; // x      2048  tensor-core MMA
/*2eef9ca0*/  HMMA.1688.F32.TF32 R56, R184, R172, R56                    ; // x      2048  tensor-core MMA
/*2eef9cb0*/  HMMA.1688.F32.TF32 R60, R184, R170, R60                    ; // x      2048  tensor-core MMA
/*2eef9cc0*/  FSETP.GEU.AND P0, PT, |R141|, +INF , PT                    ; // x      2048  FMA
/*2eef9cd0*/  FSETP.GEU.AND P1, PT, |R140|, +INF , PT                    ; // x      2048  FMA
/*2eef9ce0*/  FSETP.GEU.AND P6, PT, |R136|, +INF , PT                    ; // x      2048  FMA
/*2eef9cf0*/  HMMA.1688.F32.TF32 R64, R184, R168, R64                    ; // x      2048  tensor-core MMA
/*2eef9d00*/  FSETP.GEU.AND P5, PT, |R137|, +INF , PT                    ; // x      2048  FMA
/*2eef9d10*/  FSETP.GEU.AND P4, PT, |R138|, +INF , PT                    ; // x      2048  FMA
/*2eef9d20*/  FSETP.GEU.AND P3, PT, |R139|, +INF , PT                    ; // x      2048  FMA
/*2eef9d30*/  HMMA.1688.F32.TF32 R68, R188, R168, R68                    ; // x      2048  tensor-core MMA
/*2eef9d40*/  @!P0  IMAD.IADD.U32 R141, R141, 0x1, R203                  ; // x      2048  FMA
/*2eef9d50*/  FSETP.GEU.AND P0, PT, |R147|, +INF , PT                    ; // x      2048  FMA
/*2eef9d60*/  @!P1  IMAD.IADD.U32 R140, R140, 0x1, R203                  ; // x      2048  FMA
/*2eef9d70*/  FSETP.GEU.AND P1, PT, |R146|, +INF , PT                    ; // x      2048  FMA
/*2eef9d80*/  HMMA.1688.F32.TF32 R72, R188, R170, R72                    ; // x      2048  tensor-core MMA
/*2eef9d90*/  FSETP.GEU.AND P2, PT, |R162|, +INF , PT                    ; // x      2048  FMA
/*2eef9da0*/  @!P6  IMAD.IADD.U32 R136, R136, 0x1, R203                  ; // x      2048  FMA
/*2eef9db0*/  FSETP.GEU.AND P6, PT, |R142|, +INF , PT                    ; // x      2048  FMA
```

(first 200 of 268 instructions shown; the rest are in the full dump)
