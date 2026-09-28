# Hot loop — `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn` (sm87)

PC range 0x85b098f0-0x85b0a070; backward BRA 0x85b0a070 -> 0x85b098f0: of the 6 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn_sm87.sass`; per-PC counts: `raw/ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*85b098f0*/  HMMA.16816.F32.BF16 R0, R160, R192, R0                     ; // x     49152  tensor-core MMA
/*85b09900*/  LDSM.16.M88.4 R176, [R148+UR8]                             ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09910*/  R2P PR, R151.B1, 0x78                                      ; // x     49152  ALU
/*85b09920*/  HMMA.16816.F32.BF16 R4, R160, R194, R4                     ; // x     49152  tensor-core MMA
/*85b09930*/  HMMA.16816.F32.BF16 R8, R160, R196, R8                     ; // x     49152  tensor-core MMA
/*85b09940*/  LDSM.16.M88.4 R180, [R148+UR8+0x400]                       ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09950*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R146+UR6][R138.64] ; // x     49152  async global->shared copy (tile staging)
/*85b09960*/  HMMA.16816.F32.BF16 R12, R160, R198, R12                   ; // x     49152  tensor-core MMA
/*85b09970*/  HMMA.16816.F32.BF16 R16, R160, R200, R16                   ; // x     49152  tensor-core MMA
/*85b09980*/  LDSM.16.M88.4 R184, [R148+UR8+0x800]                       ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09990*/  UISETP.LT.AND UP0, UPT, UR8, 0x10000, UPT                  ; // x     49152  Uniform
/*85b099a0*/  HMMA.16816.F32.BF16 R20, R160, R202, R20                   ; // x     49152  tensor-core MMA
/*85b099b0*/  HMMA.16816.F32.BF16 R24, R160, R204, R24                   ; // x     49152  tensor-core MMA
/*85b099c0*/  LDSM.16.M88.4 R188, [R148+UR8+0xc00]                       ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b099d0*/  HMMA.16816.F32.BF16 R28, R160, R206, R28                   ; // x     49152  tensor-core MMA
/*85b099e0*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R146+UR6+0x800][R140.64] ; // x     49152  async global->shared copy (tile staging)
/*85b099f0*/  HMMA.16816.F32.BF16 R60, R164, R206, R60                   ; // x     49152  tensor-core MMA
/*85b09a00*/  LDSM.16.M88.4 R208, [R150+UR8+0x2000]                      ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09a10*/  HMMA.16816.F32.BF16 R56, R164, R204, R56                   ; // x     49152  tensor-core MMA
/*85b09a20*/  HMMA.16816.F32.BF16 R52, R164, R202, R52                   ; // x     49152  tensor-core MMA
/*85b09a30*/  LDSM.16.M88.4 R212, [R150+UR8+0x2400]                      ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09a40*/  HMMA.16816.F32.BF16 R48, R164, R200, R48                   ; // x     49152  tensor-core MMA
/*85b09a50*/  HMMA.16816.F32.BF16 R44, R164, R198, R44                   ; // x     49152  tensor-core MMA
/*85b09a60*/  LDSM.16.M88.4 R216, [R150+UR8+0x2800]                      ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09a70*/  HMMA.16816.F32.BF16 R40, R164, R196, R40                   ; // x     49152  tensor-core MMA
/*85b09a80*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R146+UR6+0x1000][R142.64] ; // x     49152  async global->shared copy (tile staging)
/*85b09a90*/  HMMA.16816.F32.BF16 R36, R164, R194, R36                   ; // x     49152  tensor-core MMA
/*85b09aa0*/  LDSM.16.M88.4 R220, [R150+UR8+0x2c00]                      ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09ab0*/  @UP0  UIADD3 UR8, UR8, 0x4000, URZ                         ; // x     49152  Uniform
/*85b09ac0*/  @!UP0 UMOV UR8, URZ                                        ; // x     49152  Uniform
/*85b09ad0*/  HMMA.16816.F32.BF16 R32, R164, R192, R32                   ; // x     49152  tensor-core MMA
/*85b09ae0*/  HMMA.16816.F32.BF16 R64, R168, R192, R64                   ; // x     49152  tensor-core MMA
/*85b09af0*/  HMMA.16816.F32.BF16 R68, R168, R194, R68                   ; // x     49152  tensor-core MMA
/*85b09b00*/  HMMA.16816.F32.BF16 R72, R168, R196, R72                   ; // x     49152  tensor-core MMA
/*85b09b10*/  HMMA.16816.F32.BF16 R76, R168, R198, R76                   ; // x     49152  tensor-core MMA
/*85b09b20*/  HMMA.16816.F32.BF16 R80, R168, R200, R80                   ; // x     49152  tensor-core MMA
/*85b09b30*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R146+UR6+0x1800][R144.64] ; // x     49152  async global->shared copy (tile staging)
/*85b09b40*/  LDGDEPBAR                                                  ; // x     49152  LSU / Load
/*85b09b50*/  @UP1  UIADD3 UR4, UR4, 0x4000, URZ                         ; // x     49152  Uniform
/*85b09b60*/  @!UP1 UMOV UR4, URZ                                        ; // x     49152  Uniform
/*85b09b70*/  HMMA.16816.F32.BF16 R84, R168, R202, R84                   ; // x     49152  tensor-core MMA
/*85b09b80*/  UISETP.LT.AND UP1, UPT, UR4, 0x10000, UPT                  ; // x     49152  Uniform
/*85b09b90*/  UMOV UR6, UR4                                              ; // x     49152  Uniform
/*85b09ba0*/  IADD3 R152, R152, -0x20, RZ                                ; // x     49152  ALU
/*85b09bb0*/  HMMA.16816.F32.BF16 R88, R168, R204, R88                   ; // x     49152  tensor-core MMA
/*85b09bc0*/  ISETP.GT.AND P0, PT, R152, RZ, PT                          ; // x     49152  ALU
/*85b09bd0*/  HMMA.16816.F32.BF16 R92, R168, R206, R92                   ; // x     49152  tensor-core MMA
/*85b09be0*/  HMMA.16816.F32.BF16 R124, R172, R206, R124                 ; // x     49152  tensor-core MMA
/*85b09bf0*/  ISETP.GT.AND P1, PT, R152, -0x80, PT                       ; // x     49152  ALU
/*85b09c00*/  IADD3 R130, P2, R130, R154, RZ                             ; // x     49152  ALU
/*85b09c10*/  HMMA.16816.F32.BF16 R120, R172, R204, R120                 ; // x     49152  tensor-core MMA
/*85b09c20*/  DEPBAR.LE SB0, 0x3                                         ; // x     49152  wait on async-copy group
/*85b09c30*/  IADD3.X R131, R131, R155, RZ, P2, !PT                      ; // x     49152  ALU
/*85b09c40*/  HMMA.16816.F32.BF16 R116, R172, R202, R116                 ; // x     49152  tensor-core MMA
/*85b09c50*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x     49152  block barrier
/*85b09c60*/  HMMA.16816.F32.BF16 R112, R172, R200, R112                 ; // x     49152  tensor-core MMA
/*85b09c70*/  IADD3 R132, P2, R132, R154, RZ                             ; // x     49152  ALU
/*85b09c80*/  IADD3.X R133, R133, R155, RZ, P2, !PT                      ; // x     49152  ALU
/*85b09c90*/  HMMA.16816.F32.BF16 R108, R172, R198, R108                 ; // x     49152  tensor-core MMA
/*85b09ca0*/  IADD3 R134, P2, R134, R154, RZ                             ; // x     49152  ALU
/*85b09cb0*/  IADD3.X R135, R135, R155, RZ, P2, !PT                      ; // x     49152  ALU
/*85b09cc0*/  HMMA.16816.F32.BF16 R104, R172, R196, R104                 ; // x     49152  tensor-core MMA
/*85b09cd0*/  IADD3 R136, P2, R136, R154, RZ                             ; // x     49152  ALU
/*85b09ce0*/  IADD3.X R137, R137, R155, RZ, P2, !PT                      ; // x     49152  ALU
/*85b09cf0*/  HMMA.16816.F32.BF16 R100, R172, R194, R100                 ; // x     49152  tensor-core MMA
/*85b09d00*/  IADD3 R138, P2, R138, R156, RZ                             ; // x     49152  ALU
/*85b09d10*/  IADD3.X R139, R139, R157, RZ, P2, !PT                      ; // x     49152  ALU
/*85b09d20*/  HMMA.16816.F32.BF16 R96, R172, R192, R96                   ; // x     49152  tensor-core MMA
/*85b09d30*/  IADD3 R140, P2, R140, R156, RZ                             ; // x     49152  ALU
/*85b09d40*/  IADD3.X R141, R141, R157, RZ, P2, !PT                      ; // x     49152  ALU
/*85b09d50*/  HMMA.16816.F32.BF16 R0, R176, R208, R0                     ; // x     49152  tensor-core MMA
/*85b09d60*/  LDSM.16.M88.4 R160, [R147+UR8]                             ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09d70*/  IADD3 R142, P2, R142, R156, RZ                             ; // x     49152  ALU
/*85b09d80*/  IADD3.X R143, R143, R157, RZ, P2, !PT                      ; // x     49152  ALU
/*85b09d90*/  HMMA.16816.F32.BF16 R4, R176, R210, R4                     ; // x     49152  tensor-core MMA
/*85b09da0*/  IADD3 R144, P2, R144, R156, RZ                             ; // x     49152  ALU
/*85b09db0*/  IADD3.X R145, R145, R157, RZ, P2, !PT                      ; // x     49152  ALU
/*85b09dc0*/  HMMA.16816.F32.BF16 R8, R176, R212, R8                     ; // x     49152  tensor-core MMA
/*85b09dd0*/  LDSM.16.M88.4 R164, [R147+UR8+0x400]                       ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09de0*/  @!P0  MOV R151, RZ                                         ; // x     49152  ALU
/*85b09df0*/  R2P PR, R151, 0x78                                         ; // x     49152  ALU
/*85b09e00*/  HMMA.16816.F32.BF16 R12, R176, R214, R12                   ; // x     49152  tensor-core MMA
/*85b09e10*/  HMMA.16816.F32.BF16 R16, R176, R216, R16                   ; // x     49152  tensor-core MMA
/*85b09e20*/  LDSM.16.M88.4 R168, [R147+UR8+0x800]                       ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09e30*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R129+UR4][R130.64] ; // x     49152  async global->shared copy (tile staging)
/*85b09e40*/  HMMA.16816.F32.BF16 R20, R176, R218, R20                   ; // x     49152  tensor-core MMA
/*85b09e50*/  HMMA.16816.F32.BF16 R24, R176, R220, R24                   ; // x     49152  tensor-core MMA
/*85b09e60*/  LDSM.16.M88.4 R172, [R147+UR8+0xc00]                       ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09e70*/  HMMA.16816.F32.BF16 R28, R176, R222, R28                   ; // x     49152  tensor-core MMA
/*85b09e80*/  HMMA.16816.F32.BF16 R60, R180, R222, R60                   ; // x     49152  tensor-core MMA
/*85b09e90*/  LDSM.16.M88.4 R192, [R149+UR8+0x2000]                      ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09ea0*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R129+UR4+0x800][R132.64] ; // x     49152  async global->shared copy (tile staging)
/*85b09eb0*/  HMMA.16816.F32.BF16 R56, R180, R220, R56                   ; // x     49152  tensor-core MMA
/*85b09ec0*/  HMMA.16816.F32.BF16 R52, R180, R218, R52                   ; // x     49152  tensor-core MMA
/*85b09ed0*/  LDSM.16.M88.4 R196, [R149+UR8+0x2400]                      ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09ee0*/  HMMA.16816.F32.BF16 R48, R180, R216, R48                   ; // x     49152  tensor-core MMA
/*85b09ef0*/  HMMA.16816.F32.BF16 R44, R180, R214, R44                   ; // x     49152  tensor-core MMA
/*85b09f00*/  LDSM.16.M88.4 R200, [R149+UR8+0x2800]                      ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09f10*/  HMMA.16816.F32.BF16 R40, R180, R212, R40                   ; // x     49152  tensor-core MMA
/*85b09f20*/  HMMA.16816.F32.BF16 R36, R180, R210, R36                   ; // x     49152  tensor-core MMA
/*85b09f30*/  LDSM.16.M88.4 R204, [R149+UR8+0x2c00]                      ; // x     49152  shared-memory matrix load (tensor-core operand)
/*85b09f40*/  HMMA.16816.F32.BF16 R32, R180, R208, R32                   ; // x     49152  tensor-core MMA
/*85b09f50*/  HMMA.16816.F32.BF16 R64, R184, R208, R64                   ; // x     49152  tensor-core MMA
/*85b09f60*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R129+UR4+0x1000][R134.64] ; // x     49152  async global->shared copy (tile staging)
/*85b09f70*/  HMMA.16816.F32.BF16 R68, R184, R210, R68                   ; // x     49152  tensor-core MMA
/*85b09f80*/  HMMA.16816.F32.BF16 R72, R184, R212, R72                   ; // x     49152  tensor-core MMA
/*85b09f90*/  HMMA.16816.F32.BF16 R76, R184, R214, R76                   ; // x     49152  tensor-core MMA
/*85b09fa0*/  HMMA.16816.F32.BF16 R80, R184, R216, R80                   ; // x     49152  tensor-core MMA
/*85b09fb0*/  HMMA.16816.F32.BF16 R84, R184, R218, R84                   ; // x     49152  tensor-core MMA
/*85b09fc0*/  HMMA.16816.F32.BF16 R88, R184, R220, R88                   ; // x     49152  tensor-core MMA
/*85b09fd0*/  HMMA.16816.F32.BF16 R92, R184, R222, R92                   ; // x     49152  tensor-core MMA
/*85b09fe0*/  HMMA.16816.F32.BF16 R124, R188, R222, R124                 ; // x     49152  tensor-core MMA
/*85b09ff0*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R129+UR4+0x1800][R136.64] ; // x     49152  async global->shared copy (tile staging)
/*85b0a000*/  HMMA.16816.F32.BF16 R120, R188, R220, R120                 ; // x     49152  tensor-core MMA
/*85b0a010*/  HMMA.16816.F32.BF16 R116, R188, R218, R116                 ; // x     49152  tensor-core MMA
/*85b0a020*/  HMMA.16816.F32.BF16 R112, R188, R216, R112                 ; // x     49152  tensor-core MMA
/*85b0a030*/  HMMA.16816.F32.BF16 R108, R188, R214, R108                 ; // x     49152  tensor-core MMA
/*85b0a040*/  HMMA.16816.F32.BF16 R104, R188, R212, R104                 ; // x     49152  tensor-core MMA
/*85b0a050*/  HMMA.16816.F32.BF16 R100, R188, R210, R100                 ; // x     49152  tensor-core MMA
/*85b0a060*/  HMMA.16816.F32.BF16 R96, R188, R208, R96                   ; // x     49152  tensor-core MMA
/*85b0a070*/  @P1   BRA.U 0x385b098f0                                    ; // x     49152  branch
```
