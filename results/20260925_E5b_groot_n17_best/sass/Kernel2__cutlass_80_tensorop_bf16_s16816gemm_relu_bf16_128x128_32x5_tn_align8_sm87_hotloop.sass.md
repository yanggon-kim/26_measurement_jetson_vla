# Hot loop — `Kernel2__cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8` (sm87)

PC range 0x86217240-0x86217a80; backward BRA 0x86217a80 -> 0x86217240: of the 2 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/Kernel2__cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8_sm87.sass`; per-PC counts: `raw/Kernel2__cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_tn_align8_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*86217240*/  HMMA.16816.F32.BF16 R4, R132, R136, R4                     ; // x     49152  tensor-core MMA
/*86217250*/  LDSM.16.M88.4 R164, [R2+UR5]                               ; // x     49152  shared-memory matrix load (tensor-core operand)
/*86217260*/  UIADD3 UR8, UR5, 0x80, URZ                                 ; // x     49152  Uniform
/*86217270*/  UIADD3 UR9, UR4, 0x80, URZ                                 ; // x     49152  Uniform
/*86217280*/  R2P PR, R210, 0x3                                          ; // x     49152  ALU
/*86217290*/  UIADD3 UR18, UR18, 0x1, URZ                                ; // x     49152  Uniform
/*862172a0*/  IADD3 R205, R202, UR15, RZ                                 ; // x     49152  ALU
/*862172b0*/  HMMA.16816.F32.BF16 R8, R132, R138, R8                     ; // x     49152  tensor-core MMA
/*862172c0*/  LDSM.16.M88.4 R168, [R0+UR4]                               ; // x     49152  shared-memory matrix load (tensor-core operand)
/*862172d0*/  UISETP.NE.AND UP0, UPT, UR18, 0x5, UPT                     ; // x     49152  Uniform
/*862172e0*/  R2P PR, R210, 0xc                                          ; // x     49152  ALU
/*862172f0*/  UIADD3 UR19, UR19, 0x1, URZ                                ; // x     49152  Uniform
/*86217300*/  IADD3 R212, P4, R206, UR22, RZ                             ; // x     49152  ALU
/*86217310*/  UIADD3 UR11, UR20, 0x80, URZ                               ; // x     49152  Uniform
/*86217320*/  IADD3 R211, R203, UR15, RZ                                 ; // x     49152  ALU
/*86217330*/  HMMA.16816.F32.BF16 R12, R132, R140, R12                   ; // x     49152  tensor-core MMA
/*86217340*/  LDSM.16.M88.4 R172, [R0+UR4+0x1400]                        ; // x     49152  shared-memory matrix load (tensor-core operand)
/*86217350*/  UISETP.NE.AND UP1, UPT, UR19, 0x5, UPT                     ; // x     49152  Uniform
/*86217360*/  IADD3.X R213, R207, UR6, RZ, P4, !PT                       ; // x     49152  ALU
/*86217370*/  @!UP0 UIADD3 UR8, UR5, -0x200, URZ                         ; // x     49152  Uniform
/*86217380*/  IADD3 R217, R202, UR20, RZ                                 ; // x     49152  ALU
/*86217390*/  @!UP0 UIADD3 UR9, UR4, -0x200, URZ                         ; // x     49152  Uniform
/*862173a0*/  HMMA.16816.F32.BF16 R16, R132, R142, R16                   ; // x     49152  tensor-core MMA
/*862173b0*/  LDSM.16.M88.4 R176, [R0+UR4+0x2800]                        ; // x     49152  shared-memory matrix load (tensor-core operand)
/*862173c0*/  UIADD3 UR10, UR15, 0x80, URZ                               ; // x     49152  Uniform
/*862173d0*/  IADD3 R219, R203, UR20, RZ                                 ; // x     49152  ALU
/*862173e0*/  @!UP1 UIADD3 UR11, UR20, -0x200, URZ                       ; // x     49152  Uniform
/*862173f0*/  @!UP1 UIADD3 UR10, UR15, -0x200, URZ                       ; // x     49152  Uniform
/*86217400*/  HMMA.16816.F32.BF16 R20, R132, R144, R20                   ; // x     49152  tensor-core MMA
/*86217410*/  LDSM.16.M88.4 R180, [R0+UR4+0x3c00]                        ; // x     49152  shared-memory matrix load (tensor-core operand)
/*86217420*/  @!UP1 UMOV UR19, URZ                                       ; // x     49152  Uniform
/*86217430*/  @!UP0 UMOV UR18, URZ                                       ; // x     49152  Uniform
/*86217440*/  UMOV UR4, UR9                                              ; // x     49152  Uniform
/*86217450*/  HMMA.16816.F32.BF16 R24, R132, R146, R24                   ; // x     49152  tensor-core MMA
/*86217460*/  LDSM.16.M88.4 R184, [R2+UR5+0x1400]                        ; // x     49152  shared-memory matrix load (tensor-core operand)
/*86217470*/  HMMA.16816.F32.BF16 R28, R132, R148, R28                   ; // x     49152  tensor-core MMA
/*86217480*/  LDSM.16.M88.4 R188, [R2+UR5+0x2800]                        ; // x     49152  shared-memory matrix load (tensor-core operand)
/*86217490*/  HMMA.16816.F32.BF16 R32, R132, R150, R32                   ; // x     49152  tensor-core MMA
/*862174a0*/  LDSM.16.M88.4 R192, [R2+UR5+0x3c00]                        ; // x     49152  shared-memory matrix load (tensor-core operand)
/*862174b0*/  UMOV UR5, UR8                                              ; // x     49152  Uniform
/*862174c0*/  HMMA.16816.F32.BF16 R36, R152, R150, R36                   ; // x     49152  tensor-core MMA
/*862174d0*/  @P1   LDGSTS.E.BYPASS.LTC128B.128 [R205], [R206.64+UR26]   ; // x     49152  async global->shared copy (tile staging)
/*862174e0*/  HMMA.16816.F32.BF16 R40, R152, R148, R40                   ; // x     49152  tensor-core MMA
/*862174f0*/  @P0   LDGSTS.E.BYPASS.LTC128B.128 [R203+UR15], [R206.64]   ; // x     49152  async global->shared copy (tile staging)
/*86217500*/  R2P PR, R201, 0x3                                          ; // x     49152  ALU
/*86217510*/  HMMA.16816.F32.BF16 R44, R152, R146, R44                   ; // x     49152  tensor-core MMA
/*86217520*/  @P2   LDGSTS.E.BYPASS.LTC128B.128 [R211+0x1400], [R206.64+UR28] ; // x     49152  async global->shared copy (tile staging)
/*86217530*/  HMMA.16816.F32.BF16 R48, R152, R144, R48                   ; // x     49152  tensor-core MMA
/*86217540*/  @P3   LDGSTS.E.BYPASS.LTC128B.128 [R202+UR15+0x1400], [R212.64] ; // x     49152  async global->shared copy (tile staging)
/*86217550*/  UMOV UR15, UR10                                            ; // x     49152  Uniform
/*86217560*/  R2P PR, R201, 0xc                                          ; // x     49152  ALU
/*86217570*/  IADD3 R205, R204, -0x1, RZ                                 ; // x     49152  ALU
/*86217580*/  HMMA.16816.F32.BF16 R52, R152, R142, R52                   ; // x     49152  tensor-core MMA
/*86217590*/  @P1   LDGSTS.E.BYPASS.LTC128B.128 [R217+0xa000], [R208.64+UR24] ; // x     49152  async global->shared copy (tile staging)
/*862175a0*/  IADD3 R214, P1, R208, UR16, RZ                             ; // x     49152  ALU
/*862175b0*/  HMMA.16816.F32.BF16 R56, R152, R140, R56                   ; // x     49152  tensor-core MMA
/*862175c0*/  @P0   LDGSTS.E.BYPASS.LTC128B.128 [R203+UR20+0xa000], [R208.64] ; // x     49152  async global->shared copy (tile staging)
/*862175d0*/  IADD3.X R215, R209, UR7, RZ, P1, !PT                       ; // x     49152  ALU
/*862175e0*/  IADD3 R206, P0, R212, c[0x0][0x1a8], RZ                    ; // x     49152  ALU
/*862175f0*/  HMMA.16816.F32.BF16 R60, R152, R138, R60                   ; // x     49152  tensor-core MMA
/*86217600*/  @P2   LDGSTS.E.BYPASS.LTC128B.128 [R219+0xb400], [R208.64+UR30] ; // x     49152  async global->shared copy (tile staging)
/*86217610*/  ISETP.NE.AND P2, PT, R205, RZ, PT                          ; // x     49152  ALU
/*86217620*/  IADD3.X R207, R213, c[0x0][0x1ac], RZ, P0, !PT             ; // x     49152  ALU
/*86217630*/  ISETP.GT.AND P0, PT, R204, -0x3, PT                        ; // x     49152  ALU
/*86217640*/  HMMA.16816.F32.BF16 R64, R152, R136, R64                   ; // x     49152  tensor-core MMA
/*86217650*/  @P3   LDGSTS.E.BYPASS.LTC128B.128 [R202+UR20+0xb400], [R214.64] ; // x     49152  async global->shared copy (tile staging)
/*86217660*/  UMOV UR20, UR11                                            ; // x     49152  Uniform
/*86217670*/  ISETP.NE.AND P3, PT, R205, RZ, PT                          ; // x     49152  ALU
/*86217680*/  IMAD.MOV.U32 R204, RZ, RZ, R205                            ; // x     49152  FMA
/*86217690*/  SEL R210, R210, RZ, P2                                     ; // x     49152  ALU
/*862176a0*/  HMMA.16816.F32.BF16 R68, R156, R136, R68                   ; // x     49152  tensor-core MMA
/*862176b0*/  LDGDEPBAR                                                  ; // x     49152  LSU / Load
/*862176c0*/  HMMA.16816.F32.BF16 R72, R156, R138, R72                   ; // x     49152  tensor-core MMA
/*862176d0*/  @!P3  IMAD.MOV.U32 R201, RZ, RZ, RZ                        ; // x     49152  FMA
/*862176e0*/  IADD3 R208, P1, R214, c[0x0][0x1c8], RZ                    ; // x     49152  ALU
/*862176f0*/  HMMA.16816.F32.BF16 R76, R156, R140, R76                   ; // x     49152  tensor-core MMA
/*86217700*/  IADD3.X R209, R215, c[0x0][0x1cc], RZ, P1, !PT             ; // x     49152  ALU
/*86217710*/  HMMA.16816.F32.BF16 R80, R156, R142, R80                   ; // x     49152  tensor-core MMA
/*86217720*/  DEPBAR.LE SB0, 0x3                                         ; // x     49152  wait on async-copy group
/*86217730*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x     49152  block barrier
/*86217740*/  HMMA.16816.F32.BF16 R84, R156, R144, R84                   ; // x     49152  tensor-core MMA
/*86217750*/  HMMA.16816.F32.BF16 R88, R156, R146, R88                   ; // x     49152  tensor-core MMA
/*86217760*/  HMMA.16816.F32.BF16 R92, R156, R148, R92                   ; // x     49152  tensor-core MMA
/*86217770*/  HMMA.16816.F32.BF16 R96, R156, R150, R96                   ; // x     49152  tensor-core MMA
/*86217780*/  HMMA.16816.F32.BF16 R100, R160, R150, R100                 ; // x     49152  tensor-core MMA
/*86217790*/  HMMA.16816.F32.BF16 R104, R160, R148, R104                 ; // x     49152  tensor-core MMA
/*862177a0*/  HMMA.16816.F32.BF16 R108, R160, R146, R108                 ; // x     49152  tensor-core MMA
/*862177b0*/  HMMA.16816.F32.BF16 R112, R160, R144, R112                 ; // x     49152  tensor-core MMA
/*862177c0*/  HMMA.16816.F32.BF16 R116, R160, R142, R116                 ; // x     49152  tensor-core MMA
/*862177d0*/  HMMA.16816.F32.BF16 R120, R160, R140, R120                 ; // x     49152  tensor-core MMA
/*862177e0*/  HMMA.16816.F32.BF16 R124, R160, R138, R124                 ; // x     49152  tensor-core MMA
/*862177f0*/  HMMA.16816.F32.BF16 R128, R160, R136, R128                 ; // x     49152  tensor-core MMA
/*86217800*/  HMMA.16816.F32.BF16 R4, R164, R168, R4                     ; // x     49152  tensor-core MMA
/*86217810*/  HMMA.16816.F32.BF16 R8, R164, R170, R8                     ; // x     49152  tensor-core MMA
/*86217820*/  HMMA.16816.F32.BF16 R12, R164, R172, R12                   ; // x     49152  tensor-core MMA
/*86217830*/  HMMA.16816.F32.BF16 R16, R164, R174, R16                   ; // x     49152  tensor-core MMA
/*86217840*/  HMMA.16816.F32.BF16 R20, R164, R176, R20                   ; // x     49152  tensor-core MMA
/*86217850*/  HMMA.16816.F32.BF16 R24, R164, R178, R24                   ; // x     49152  tensor-core MMA
/*86217860*/  HMMA.16816.F32.BF16 R28, R164, R180, R28                   ; // x     49152  tensor-core MMA
/*86217870*/  HMMA.16816.F32.BF16 R32, R164, R182, R32                   ; // x     49152  tensor-core MMA
/*86217880*/  HMMA.16816.F32.BF16 R36, R184, R182, R36                   ; // x     49152  tensor-core MMA
/*86217890*/  HMMA.16816.F32.BF16 R40, R184, R180, R40                   ; // x     49152  tensor-core MMA
/*862178a0*/  HMMA.16816.F32.BF16 R44, R184, R178, R44                   ; // x     49152  tensor-core MMA
/*862178b0*/  HMMA.16816.F32.BF16 R48, R184, R176, R48                   ; // x     49152  tensor-core MMA
/*862178c0*/  HMMA.16816.F32.BF16 R52, R184, R174, R52                   ; // x     49152  tensor-core MMA
/*862178d0*/  HMMA.16816.F32.BF16 R56, R184, R172, R56                   ; // x     49152  tensor-core MMA
/*862178e0*/  HMMA.16816.F32.BF16 R60, R184, R170, R60                   ; // x     49152  tensor-core MMA
/*862178f0*/  HMMA.16816.F32.BF16 R64, R184, R168, R64                   ; // x     49152  tensor-core MMA
/*86217900*/  LDSM.16.M88.4 R132, [R196+UR8]                             ; // x     49152  shared-memory matrix load (tensor-core operand)
/*86217910*/  HMMA.16816.F32.BF16 R68, R188, R168, R68                   ; // x     49152  tensor-core MMA
/*86217920*/  LDSM.16.M88.4 R152, [R196+UR8+0x1400]                      ; // x     49152  shared-memory matrix load (tensor-core operand)
/*86217930*/  HMMA.16816.F32.BF16 R72, R188, R170, R72                   ; // x     49152  tensor-core MMA
/*86217940*/  LDSM.16.M88.4 R156, [R196+UR8+0x2800]                      ; // x     49152  shared-memory matrix load (tensor-core operand)
/*86217950*/  HMMA.16816.F32.BF16 R76, R188, R172, R76                   ; // x     49152  tensor-core MMA
/*86217960*/  LDSM.16.M88.4 R160, [R196+UR8+0x3c00]                      ; // x     49152  shared-memory matrix load (tensor-core operand)
/*86217970*/  HMMA.16816.F32.BF16 R80, R188, R174, R80                   ; // x     49152  tensor-core MMA
/*86217980*/  LDSM.16.M88.4 R136, [R3+UR9]                               ; // x     49152  shared-memory matrix load (tensor-core operand)
/*86217990*/  HMMA.16816.F32.BF16 R84, R188, R176, R84                   ; // x     49152  tensor-core MMA
/*862179a0*/  LDSM.16.M88.4 R140, [R3+UR9+0x1400]                        ; // x     49152  shared-memory matrix load (tensor-core operand)
/*862179b0*/  HMMA.16816.F32.BF16 R88, R188, R178, R88                   ; // x     49152  tensor-core MMA
/*862179c0*/  LDSM.16.M88.4 R144, [R3+UR9+0x2800]                        ; // x     49152  shared-memory matrix load (tensor-core operand)
/*862179d0*/  HMMA.16816.F32.BF16 R92, R188, R180, R92                   ; // x     49152  tensor-core MMA
/*862179e0*/  LDSM.16.M88.4 R148, [R3+UR9+0x3c00]                        ; // x     49152  shared-memory matrix load (tensor-core operand)
/*862179f0*/  HMMA.16816.F32.BF16 R96, R188, R182, R96                   ; // x     49152  tensor-core MMA
/*86217a00*/  HMMA.16816.F32.BF16 R100, R192, R182, R100                 ; // x     49152  tensor-core MMA
/*86217a10*/  HMMA.16816.F32.BF16 R104, R192, R180, R104                 ; // x     49152  tensor-core MMA
/*86217a20*/  HMMA.16816.F32.BF16 R108, R192, R178, R108                 ; // x     49152  tensor-core MMA
/*86217a30*/  HMMA.16816.F32.BF16 R112, R192, R176, R112                 ; // x     49152  tensor-core MMA
/*86217a40*/  HMMA.16816.F32.BF16 R116, R192, R174, R116                 ; // x     49152  tensor-core MMA
/*86217a50*/  HMMA.16816.F32.BF16 R120, R192, R172, R120                 ; // x     49152  tensor-core MMA
/*86217a60*/  HMMA.16816.F32.BF16 R124, R192, R170, R124                 ; // x     49152  tensor-core MMA
/*86217a70*/  HMMA.16816.F32.BF16 R128, R192, R168, R128                 ; // x     49152  tensor-core MMA
/*86217a80*/  @P0   BRA 0x386217240                                      ; // x     49152  branch
```
