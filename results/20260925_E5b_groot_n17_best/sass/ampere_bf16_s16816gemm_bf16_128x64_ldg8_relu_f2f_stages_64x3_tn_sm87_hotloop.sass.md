# Hot loop — `ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x3_tn` (sm87)

PC range 0x85bbd050-0x85bbd920; backward BRA 0x85bbd920 -> 0x85bbd050: of the 6 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x3_tn_sm87.sass`; per-PC counts: `raw/ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x3_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*85bbd050*/  HMMA.16816.F32.BF16 R0, R108, R140, R0                     ; // x     10240  tensor-core MMA
/*85bbd060*/  LDSM.16.M88.4 R124, [R92+UR8]                              ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd070*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x1800][R72.64] ; // x     10240  async global->shared copy (tile staging)
/*85bbd080*/  R2P PR, R99.B1, 0x78                                       ; // x     10240  ALU
/*85bbd090*/  HMMA.16816.F32.BF16 R4, R108, R142, R4                     ; // x     10240  tensor-core MMA
/*85bbd0a0*/  HMMA.16816.F32.BF16 R8, R108, R144, R8                     ; // x     10240  tensor-core MMA
/*85bbd0b0*/  LDSM.16.M88.4 R128, [R92+UR8+0x800]                        ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd0c0*/  HMMA.16816.F32.BF16 R12, R108, R146, R12                   ; // x     10240  tensor-core MMA
/*85bbd0d0*/  HMMA.16816.F32.BF16 R28, R112, R146, R28                   ; // x     10240  tensor-core MMA
/*85bbd0e0*/  LDSM.16.M88.4 R132, [R92+UR8+0x1000]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd0f0*/  HMMA.16816.F32.BF16 R24, R112, R144, R24                   ; // x     10240  tensor-core MMA
/*85bbd100*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x2000][R74.64] ; // x     10240  async global->shared copy (tile staging)
/*85bbd110*/  HMMA.16816.F32.BF16 R20, R112, R142, R20                   ; // x     10240  tensor-core MMA
/*85bbd120*/  LDSM.16.M88.4 R136, [R92+UR8+0x1800]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd130*/  HMMA.16816.F32.BF16 R16, R112, R140, R16                   ; // x     10240  tensor-core MMA
/*85bbd140*/  HMMA.16816.F32.BF16 R32, R116, R140, R32                   ; // x     10240  tensor-core MMA
/*85bbd150*/  LDSM.16.M88.4 R148, [R96+UR8+0x4000]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd160*/  HMMA.16816.F32.BF16 R36, R116, R142, R36                   ; // x     10240  tensor-core MMA
/*85bbd170*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x2800][R76.64] ; // x     10240  async global->shared copy (tile staging)
/*85bbd180*/  HMMA.16816.F32.BF16 R40, R116, R144, R40                   ; // x     10240  tensor-core MMA
/*85bbd190*/  LDSM.16.M88.4 R152, [R96+UR8+0x4800]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd1a0*/  HMMA.16816.F32.BF16 R44, R116, R146, R44                   ; // x     10240  tensor-core MMA
/*85bbd1b0*/  HMMA.16816.F32.BF16 R60, R120, R146, R60                   ; // x     10240  tensor-core MMA
/*85bbd1c0*/  HMMA.16816.F32.BF16 R56, R120, R144, R56                   ; // x     10240  tensor-core MMA
/*85bbd1d0*/  HMMA.16816.F32.BF16 R52, R120, R142, R52                   ; // x     10240  tensor-core MMA
/*85bbd1e0*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x3000][R78.64] ; // x     10240  async global->shared copy (tile staging)
/*85bbd1f0*/  HMMA.16816.F32.BF16 R48, R120, R140, R48                   ; // x     10240  tensor-core MMA
/*85bbd200*/  HMMA.16816.F32.BF16 R0, R124, R148, R0                     ; // x     10240  tensor-core MMA
/*85bbd210*/  LDSM.16.M88.4 R108, [R93+UR8]                              ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd220*/  HMMA.16816.F32.BF16 R4, R124, R150, R4                     ; // x     10240  tensor-core MMA
/*85bbd230*/  HMMA.16816.F32.BF16 R8, R124, R152, R8                     ; // x     10240  tensor-core MMA
/*85bbd240*/  LDSM.16.M88.4 R112, [R93+UR8+0x800]                        ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd250*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x3800][R80.64] ; // x     10240  async global->shared copy (tile staging)
/*85bbd260*/  R2P PR, R99.B2, 0x78                                       ; // x     10240  ALU
/*85bbd270*/  HMMA.16816.F32.BF16 R12, R124, R154, R12                   ; // x     10240  tensor-core MMA
/*85bbd280*/  HMMA.16816.F32.BF16 R28, R128, R154, R28                   ; // x     10240  tensor-core MMA
/*85bbd290*/  LDSM.16.M88.4 R116, [R93+UR8+0x1000]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd2a0*/  HMMA.16816.F32.BF16 R24, R128, R152, R24                   ; // x     10240  tensor-core MMA
/*85bbd2b0*/  HMMA.16816.F32.BF16 R20, R128, R150, R20                   ; // x     10240  tensor-core MMA
/*85bbd2c0*/  LDSM.16.M88.4 R120, [R93+UR8+0x1800]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd2d0*/  HMMA.16816.F32.BF16 R16, R128, R148, R16                   ; // x     10240  tensor-core MMA
/*85bbd2e0*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R90+UR6][R82.64] ; // x     10240  async global->shared copy (tile staging)
/*85bbd2f0*/  HMMA.16816.F32.BF16 R32, R132, R148, R32                   ; // x     10240  tensor-core MMA
/*85bbd300*/  LDSM.16.M88.4 R140, [R97+UR8+0x4000]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd310*/  HMMA.16816.F32.BF16 R36, R132, R150, R36                   ; // x     10240  tensor-core MMA
/*85bbd320*/  HMMA.16816.F32.BF16 R40, R132, R152, R40                   ; // x     10240  tensor-core MMA
/*85bbd330*/  LDSM.16.M88.4 R144, [R97+UR8+0x4800]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd340*/  HMMA.16816.F32.BF16 R44, R132, R154, R44                   ; // x     10240  tensor-core MMA
/*85bbd350*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R90+UR6+0x800][R84.64] ; // x     10240  async global->shared copy (tile staging)
/*85bbd360*/  HMMA.16816.F32.BF16 R60, R136, R154, R60                   ; // x     10240  tensor-core MMA
/*85bbd370*/  HMMA.16816.F32.BF16 R56, R136, R152, R56                   ; // x     10240  tensor-core MMA
/*85bbd380*/  HMMA.16816.F32.BF16 R52, R136, R150, R52                   ; // x     10240  tensor-core MMA
/*85bbd390*/  HMMA.16816.F32.BF16 R48, R136, R148, R48                   ; // x     10240  tensor-core MMA
/*85bbd3a0*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R90+UR6+0x1000][R86.64] ; // x     10240  async global->shared copy (tile staging)
/*85bbd3b0*/  HMMA.16816.F32.BF16 R0, R108, R140, R0                     ; // x     10240  tensor-core MMA
/*85bbd3c0*/  LDSM.16.M88.4 R124, [R94+UR8]                              ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd3d0*/  HMMA.16816.F32.BF16 R4, R108, R142, R4                     ; // x     10240  tensor-core MMA
/*85bbd3e0*/  HMMA.16816.F32.BF16 R8, R108, R144, R8                     ; // x     10240  tensor-core MMA
/*85bbd3f0*/  LDSM.16.M88.4 R128, [R94+UR8+0x800]                        ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd400*/  HMMA.16816.F32.BF16 R12, R108, R146, R12                   ; // x     10240  tensor-core MMA
/*85bbd410*/  HMMA.16816.F32.BF16 R28, R112, R146, R28                   ; // x     10240  tensor-core MMA
/*85bbd420*/  LDSM.16.M88.4 R132, [R94+UR8+0x1000]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd430*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R90+UR6+0x1800][R88.64] ; // x     10240  async global->shared copy (tile staging)
/*85bbd440*/  UISETP.LT.AND UP0, UPT, UR8, 0xc000, UPT                   ; // x     10240  Uniform
/*85bbd450*/  LDGDEPBAR                                                  ; // x     10240  LSU / Load
/*85bbd460*/  @UP1  UIADD3 UR4, UR4, 0x6000, URZ                         ; // x     10240  Uniform
/*85bbd470*/  HMMA.16816.F32.BF16 R24, R112, R144, R24                   ; // x     10240  tensor-core MMA
/*85bbd480*/  @!UP1 UMOV UR4, URZ                                        ; // x     10240  Uniform
/*85bbd490*/  UISETP.LT.AND UP1, UPT, UR4, 0xc000, UPT                   ; // x     10240  Uniform
/*85bbd4a0*/  UMOV UR6, UR4                                              ; // x     10240  Uniform
/*85bbd4b0*/  HMMA.16816.F32.BF16 R20, R112, R142, R20                   ; // x     10240  tensor-core MMA
/*85bbd4c0*/  LDSM.16.M88.4 R136, [R94+UR8+0x1800]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd4d0*/  HMMA.16816.F32.BF16 R16, R112, R140, R16                   ; // x     10240  tensor-core MMA
/*85bbd4e0*/  HMMA.16816.F32.BF16 R32, R116, R140, R32                   ; // x     10240  tensor-core MMA
/*85bbd4f0*/  LDSM.16.M88.4 R148, [R98+UR8+0x4000]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd500*/  HMMA.16816.F32.BF16 R36, R116, R142, R36                   ; // x     10240  tensor-core MMA
/*85bbd510*/  HMMA.16816.F32.BF16 R40, R116, R144, R40                   ; // x     10240  tensor-core MMA
/*85bbd520*/  LDSM.16.M88.4 R152, [R98+UR8+0x4800]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd530*/  @UP0  UIADD3 UR8, UR8, 0x6000, URZ                         ; // x     10240  Uniform
/*85bbd540*/  @!UP0 UMOV UR8, URZ                                        ; // x     10240  Uniform
/*85bbd550*/  IADD3 R100, R100, -0x40, RZ                                ; // x     10240  ALU
/*85bbd560*/  HMMA.16816.F32.BF16 R44, R116, R146, R44                   ; // x     10240  tensor-core MMA
/*85bbd570*/  DEPBAR.LE SB0, 0x1                                         ; // x     10240  wait on async-copy group
/*85bbd580*/  ISETP.GT.AND P0, PT, R100, RZ, PT                          ; // x     10240  ALU
/*85bbd590*/  HMMA.16816.F32.BF16 R60, R120, R146, R60                   ; // x     10240  tensor-core MMA
/*85bbd5a0*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x     10240  block barrier
/*85bbd5b0*/  HMMA.16816.F32.BF16 R56, R120, R144, R56                   ; // x     10240  tensor-core MMA
/*85bbd5c0*/  HMMA.16816.F32.BF16 R52, R120, R142, R52                   ; // x     10240  tensor-core MMA
/*85bbd5d0*/  ISETP.GT.AND P1, PT, R100, -0x80, PT                       ; // x     10240  ALU
/*85bbd5e0*/  IADD3 R66, P2, R66, R102, RZ                               ; // x     10240  ALU
/*85bbd5f0*/  HMMA.16816.F32.BF16 R48, R120, R140, R48                   ; // x     10240  tensor-core MMA
/*85bbd600*/  IADD3.X R67, R67, R103, RZ, P2, !PT                        ; // x     10240  ALU
/*85bbd610*/  IADD3 R68, P2, R68, R102, RZ                               ; // x     10240  ALU
/*85bbd620*/  HMMA.16816.F32.BF16 R0, R124, R148, R0                     ; // x     10240  tensor-core MMA
/*85bbd630*/  LDSM.16.M88.4 R108, [R91+UR8]                              ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd640*/  IADD3.X R69, R69, R103, RZ, P2, !PT                        ; // x     10240  ALU
/*85bbd650*/  IADD3 R70, P2, R70, R102, RZ                               ; // x     10240  ALU
/*85bbd660*/  HMMA.16816.F32.BF16 R4, R124, R150, R4                     ; // x     10240  tensor-core MMA
/*85bbd670*/  IADD3.X R71, R71, R103, RZ, P2, !PT                        ; // x     10240  ALU
/*85bbd680*/  IADD3 R72, P2, R72, R102, RZ                               ; // x     10240  ALU
/*85bbd690*/  HMMA.16816.F32.BF16 R8, R124, R152, R8                     ; // x     10240  tensor-core MMA
/*85bbd6a0*/  LDSM.16.M88.4 R112, [R91+UR8+0x800]                        ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd6b0*/  IADD3.X R73, R73, R103, RZ, P2, !PT                        ; // x     10240  ALU
/*85bbd6c0*/  IADD3 R74, P2, R74, R102, RZ                               ; // x     10240  ALU
/*85bbd6d0*/  HMMA.16816.F32.BF16 R12, R124, R154, R12                   ; // x     10240  tensor-core MMA
/*85bbd6e0*/  IADD3.X R75, R75, R103, RZ, P2, !PT                        ; // x     10240  ALU
/*85bbd6f0*/  IADD3 R76, P2, R76, R102, RZ                               ; // x     10240  ALU
/*85bbd700*/  HMMA.16816.F32.BF16 R28, R128, R154, R28                   ; // x     10240  tensor-core MMA
/*85bbd710*/  LDSM.16.M88.4 R116, [R91+UR8+0x1000]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd720*/  IADD3.X R77, R77, R103, RZ, P2, !PT                        ; // x     10240  ALU
/*85bbd730*/  IADD3 R78, P2, R78, R102, RZ                               ; // x     10240  ALU
/*85bbd740*/  HMMA.16816.F32.BF16 R24, R128, R152, R24                   ; // x     10240  tensor-core MMA
/*85bbd750*/  IADD3.X R79, R79, R103, RZ, P2, !PT                        ; // x     10240  ALU
/*85bbd760*/  IADD3 R80, P2, R80, R102, RZ                               ; // x     10240  ALU
/*85bbd770*/  HMMA.16816.F32.BF16 R20, R128, R150, R20                   ; // x     10240  tensor-core MMA
/*85bbd780*/  LDSM.16.M88.4 R120, [R91+UR8+0x1800]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd790*/  IADD3.X R81, R81, R103, RZ, P2, !PT                        ; // x     10240  ALU
/*85bbd7a0*/  IADD3 R82, P2, R82, R104, RZ                               ; // x     10240  ALU
/*85bbd7b0*/  HMMA.16816.F32.BF16 R16, R128, R148, R16                   ; // x     10240  tensor-core MMA
/*85bbd7c0*/  IADD3.X R83, R83, R105, RZ, P2, !PT                        ; // x     10240  ALU
/*85bbd7d0*/  IADD3 R84, P2, R84, R104, RZ                               ; // x     10240  ALU
/*85bbd7e0*/  HMMA.16816.F32.BF16 R32, R132, R148, R32                   ; // x     10240  tensor-core MMA
/*85bbd7f0*/  LDSM.16.M88.4 R140, [R95+UR8+0x4000]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd800*/  IADD3.X R85, R85, R105, RZ, P2, !PT                        ; // x     10240  ALU
/*85bbd810*/  IADD3 R86, P2, R86, R104, RZ                               ; // x     10240  ALU
/*85bbd820*/  HMMA.16816.F32.BF16 R36, R132, R150, R36                   ; // x     10240  tensor-core MMA
/*85bbd830*/  IADD3.X R87, R87, R105, RZ, P2, !PT                        ; // x     10240  ALU
/*85bbd840*/  IADD3 R88, P2, R88, R104, RZ                               ; // x     10240  ALU
/*85bbd850*/  HMMA.16816.F32.BF16 R40, R132, R152, R40                   ; // x     10240  tensor-core MMA
/*85bbd860*/  LDSM.16.M88.4 R144, [R95+UR8+0x4800]                       ; // x     10240  shared-memory matrix load (tensor-core operand)
/*85bbd870*/  IADD3.X R89, R89, R105, RZ, P2, !PT                        ; // x     10240  ALU
/*85bbd880*/  @!P0  MOV R99, RZ                                          ; // x     10240  ALU
/*85bbd890*/  HMMA.16816.F32.BF16 R44, R132, R154, R44                   ; // x     10240  tensor-core MMA
/*85bbd8a0*/  R2P PR, R99, 0x78                                          ; // x     10240  ALU
/*85bbd8b0*/  HMMA.16816.F32.BF16 R60, R136, R154, R60                   ; // x     10240  tensor-core MMA
/*85bbd8c0*/  HMMA.16816.F32.BF16 R56, R136, R152, R56                   ; // x     10240  tensor-core MMA
/*85bbd8d0*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4][R66.64] ; // x     10240  async global->shared copy (tile staging)
/*85bbd8e0*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x800][R68.64] ; // x     10240  async global->shared copy (tile staging)
/*85bbd8f0*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x1000][R70.64] ; // x     10240  async global->shared copy (tile staging)
/*85bbd900*/  HMMA.16816.F32.BF16 R52, R136, R150, R52                   ; // x     10240  tensor-core MMA
/*85bbd910*/  HMMA.16816.F32.BF16 R48, R136, R148, R48                   ; // x     10240  tensor-core MMA
/*85bbd920*/  @P1   BRA.U 0x385bbd050                                    ; // x     10240  branch
```
