# Hot loop — `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn` (sm87)

PC range 0x4f145e50-0x4f146720; backward BRA 0x4f146720 -> 0x4f145e50: of the 6 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn_sm87.sass`; per-PC counts: `raw/ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*4f145e50*/  HMMA.16816.F32.BF16 R0, R108, R140, R0                     ; // x       768  tensor-core MMA
/*4f145e60*/  LDSM.16.M88.4 R124, [R92+UR8]                              ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f145e70*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x1800][R72.64] ; // x       768  async global->shared copy (tile staging)
/*4f145e80*/  R2P PR, R99.B1, 0x78                                       ; // x       768  ALU
/*4f145e90*/  HMMA.16816.F32.BF16 R4, R108, R142, R4                     ; // x       768  tensor-core MMA
/*4f145ea0*/  HMMA.16816.F32.BF16 R8, R108, R144, R8                     ; // x       768  tensor-core MMA
/*4f145eb0*/  LDSM.16.M88.4 R128, [R92+UR8+0x800]                        ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f145ec0*/  HMMA.16816.F32.BF16 R12, R108, R146, R12                   ; // x       768  tensor-core MMA
/*4f145ed0*/  HMMA.16816.F32.BF16 R28, R112, R146, R28                   ; // x       768  tensor-core MMA
/*4f145ee0*/  LDSM.16.M88.4 R132, [R92+UR8+0x1000]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f145ef0*/  HMMA.16816.F32.BF16 R24, R112, R144, R24                   ; // x       768  tensor-core MMA
/*4f145f00*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x2000][R74.64] ; // x       768  async global->shared copy (tile staging)
/*4f145f10*/  HMMA.16816.F32.BF16 R20, R112, R142, R20                   ; // x       768  tensor-core MMA
/*4f145f20*/  LDSM.16.M88.4 R136, [R92+UR8+0x1800]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f145f30*/  HMMA.16816.F32.BF16 R16, R112, R140, R16                   ; // x       768  tensor-core MMA
/*4f145f40*/  HMMA.16816.F32.BF16 R32, R116, R140, R32                   ; // x       768  tensor-core MMA
/*4f145f50*/  LDSM.16.M88.4 R148, [R96+UR8+0x4000]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f145f60*/  HMMA.16816.F32.BF16 R36, R116, R142, R36                   ; // x       768  tensor-core MMA
/*4f145f70*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x2800][R76.64] ; // x       768  async global->shared copy (tile staging)
/*4f145f80*/  HMMA.16816.F32.BF16 R40, R116, R144, R40                   ; // x       768  tensor-core MMA
/*4f145f90*/  LDSM.16.M88.4 R152, [R96+UR8+0x4800]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f145fa0*/  HMMA.16816.F32.BF16 R44, R116, R146, R44                   ; // x       768  tensor-core MMA
/*4f145fb0*/  HMMA.16816.F32.BF16 R60, R120, R146, R60                   ; // x       768  tensor-core MMA
/*4f145fc0*/  HMMA.16816.F32.BF16 R56, R120, R144, R56                   ; // x       768  tensor-core MMA
/*4f145fd0*/  HMMA.16816.F32.BF16 R52, R120, R142, R52                   ; // x       768  tensor-core MMA
/*4f145fe0*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x3000][R78.64] ; // x       768  async global->shared copy (tile staging)
/*4f145ff0*/  HMMA.16816.F32.BF16 R48, R120, R140, R48                   ; // x       768  tensor-core MMA
/*4f146000*/  HMMA.16816.F32.BF16 R0, R124, R148, R0                     ; // x       768  tensor-core MMA
/*4f146010*/  LDSM.16.M88.4 R108, [R93+UR8]                              ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146020*/  HMMA.16816.F32.BF16 R4, R124, R150, R4                     ; // x       768  tensor-core MMA
/*4f146030*/  HMMA.16816.F32.BF16 R8, R124, R152, R8                     ; // x       768  tensor-core MMA
/*4f146040*/  LDSM.16.M88.4 R112, [R93+UR8+0x800]                        ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146050*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x3800][R80.64] ; // x       768  async global->shared copy (tile staging)
/*4f146060*/  R2P PR, R99.B2, 0x78                                       ; // x       768  ALU
/*4f146070*/  HMMA.16816.F32.BF16 R12, R124, R154, R12                   ; // x       768  tensor-core MMA
/*4f146080*/  HMMA.16816.F32.BF16 R28, R128, R154, R28                   ; // x       768  tensor-core MMA
/*4f146090*/  LDSM.16.M88.4 R116, [R93+UR8+0x1000]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f1460a0*/  HMMA.16816.F32.BF16 R24, R128, R152, R24                   ; // x       768  tensor-core MMA
/*4f1460b0*/  HMMA.16816.F32.BF16 R20, R128, R150, R20                   ; // x       768  tensor-core MMA
/*4f1460c0*/  LDSM.16.M88.4 R120, [R93+UR8+0x1800]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f1460d0*/  HMMA.16816.F32.BF16 R16, R128, R148, R16                   ; // x       768  tensor-core MMA
/*4f1460e0*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R90+UR6][R82.64] ; // x       768  async global->shared copy (tile staging)
/*4f1460f0*/  HMMA.16816.F32.BF16 R32, R132, R148, R32                   ; // x       768  tensor-core MMA
/*4f146100*/  LDSM.16.M88.4 R140, [R97+UR8+0x4000]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146110*/  HMMA.16816.F32.BF16 R36, R132, R150, R36                   ; // x       768  tensor-core MMA
/*4f146120*/  HMMA.16816.F32.BF16 R40, R132, R152, R40                   ; // x       768  tensor-core MMA
/*4f146130*/  LDSM.16.M88.4 R144, [R97+UR8+0x4800]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146140*/  HMMA.16816.F32.BF16 R44, R132, R154, R44                   ; // x       768  tensor-core MMA
/*4f146150*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R90+UR6+0x800][R84.64] ; // x       768  async global->shared copy (tile staging)
/*4f146160*/  HMMA.16816.F32.BF16 R60, R136, R154, R60                   ; // x       768  tensor-core MMA
/*4f146170*/  HMMA.16816.F32.BF16 R56, R136, R152, R56                   ; // x       768  tensor-core MMA
/*4f146180*/  HMMA.16816.F32.BF16 R52, R136, R150, R52                   ; // x       768  tensor-core MMA
/*4f146190*/  HMMA.16816.F32.BF16 R48, R136, R148, R48                   ; // x       768  tensor-core MMA
/*4f1461a0*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R90+UR6+0x1000][R86.64] ; // x       768  async global->shared copy (tile staging)
/*4f1461b0*/  HMMA.16816.F32.BF16 R0, R108, R140, R0                     ; // x       768  tensor-core MMA
/*4f1461c0*/  LDSM.16.M88.4 R124, [R94+UR8]                              ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f1461d0*/  HMMA.16816.F32.BF16 R4, R108, R142, R4                     ; // x       768  tensor-core MMA
/*4f1461e0*/  HMMA.16816.F32.BF16 R8, R108, R144, R8                     ; // x       768  tensor-core MMA
/*4f1461f0*/  LDSM.16.M88.4 R128, [R94+UR8+0x800]                        ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146200*/  HMMA.16816.F32.BF16 R12, R108, R146, R12                   ; // x       768  tensor-core MMA
/*4f146210*/  HMMA.16816.F32.BF16 R28, R112, R146, R28                   ; // x       768  tensor-core MMA
/*4f146220*/  LDSM.16.M88.4 R132, [R94+UR8+0x1000]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146230*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R90+UR6+0x1800][R88.64] ; // x       768  async global->shared copy (tile staging)
/*4f146240*/  UISETP.LT.AND UP0, UPT, UR8, 0xc000, UPT                   ; // x       768  Uniform
/*4f146250*/  LDGDEPBAR                                                  ; // x       768  LSU / Load
/*4f146260*/  @UP1  UIADD3 UR4, UR4, 0x6000, URZ                         ; // x       768  Uniform
/*4f146270*/  HMMA.16816.F32.BF16 R24, R112, R144, R24                   ; // x       768  tensor-core MMA
/*4f146280*/  @!UP1 UMOV UR4, URZ                                        ; // x       768  Uniform
/*4f146290*/  UISETP.LT.AND UP1, UPT, UR4, 0xc000, UPT                   ; // x       768  Uniform
/*4f1462a0*/  UMOV UR6, UR4                                              ; // x       768  Uniform
/*4f1462b0*/  HMMA.16816.F32.BF16 R20, R112, R142, R20                   ; // x       768  tensor-core MMA
/*4f1462c0*/  LDSM.16.M88.4 R136, [R94+UR8+0x1800]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f1462d0*/  HMMA.16816.F32.BF16 R16, R112, R140, R16                   ; // x       768  tensor-core MMA
/*4f1462e0*/  HMMA.16816.F32.BF16 R32, R116, R140, R32                   ; // x       768  tensor-core MMA
/*4f1462f0*/  LDSM.16.M88.4 R148, [R98+UR8+0x4000]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146300*/  HMMA.16816.F32.BF16 R36, R116, R142, R36                   ; // x       768  tensor-core MMA
/*4f146310*/  HMMA.16816.F32.BF16 R40, R116, R144, R40                   ; // x       768  tensor-core MMA
/*4f146320*/  LDSM.16.M88.4 R152, [R98+UR8+0x4800]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146330*/  @UP0  UIADD3 UR8, UR8, 0x6000, URZ                         ; // x       768  Uniform
/*4f146340*/  @!UP0 UMOV UR8, URZ                                        ; // x       768  Uniform
/*4f146350*/  IADD3 R100, R100, -0x40, RZ                                ; // x       768  ALU
/*4f146360*/  HMMA.16816.F32.BF16 R44, R116, R146, R44                   ; // x       768  tensor-core MMA
/*4f146370*/  DEPBAR.LE SB0, 0x1                                         ; // x       768  wait on async-copy group
/*4f146380*/  ISETP.GT.AND P0, PT, R100, RZ, PT                          ; // x       768  ALU
/*4f146390*/  HMMA.16816.F32.BF16 R60, R120, R146, R60                   ; // x       768  tensor-core MMA
/*4f1463a0*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       768  block barrier
/*4f1463b0*/  HMMA.16816.F32.BF16 R56, R120, R144, R56                   ; // x       768  tensor-core MMA
/*4f1463c0*/  HMMA.16816.F32.BF16 R52, R120, R142, R52                   ; // x       768  tensor-core MMA
/*4f1463d0*/  ISETP.GT.AND P1, PT, R100, -0x80, PT                       ; // x       768  ALU
/*4f1463e0*/  IADD3 R66, P2, R66, R102, RZ                               ; // x       768  ALU
/*4f1463f0*/  HMMA.16816.F32.BF16 R48, R120, R140, R48                   ; // x       768  tensor-core MMA
/*4f146400*/  IADD3.X R67, R67, R103, RZ, P2, !PT                        ; // x       768  ALU
/*4f146410*/  IADD3 R68, P2, R68, R102, RZ                               ; // x       768  ALU
/*4f146420*/  HMMA.16816.F32.BF16 R0, R124, R148, R0                     ; // x       768  tensor-core MMA
/*4f146430*/  LDSM.16.M88.4 R108, [R91+UR8]                              ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146440*/  IADD3.X R69, R69, R103, RZ, P2, !PT                        ; // x       768  ALU
/*4f146450*/  IADD3 R70, P2, R70, R102, RZ                               ; // x       768  ALU
/*4f146460*/  HMMA.16816.F32.BF16 R4, R124, R150, R4                     ; // x       768  tensor-core MMA
/*4f146470*/  IADD3.X R71, R71, R103, RZ, P2, !PT                        ; // x       768  ALU
/*4f146480*/  IADD3 R72, P2, R72, R102, RZ                               ; // x       768  ALU
/*4f146490*/  HMMA.16816.F32.BF16 R8, R124, R152, R8                     ; // x       768  tensor-core MMA
/*4f1464a0*/  LDSM.16.M88.4 R112, [R91+UR8+0x800]                        ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f1464b0*/  IADD3.X R73, R73, R103, RZ, P2, !PT                        ; // x       768  ALU
/*4f1464c0*/  IADD3 R74, P2, R74, R102, RZ                               ; // x       768  ALU
/*4f1464d0*/  HMMA.16816.F32.BF16 R12, R124, R154, R12                   ; // x       768  tensor-core MMA
/*4f1464e0*/  IADD3.X R75, R75, R103, RZ, P2, !PT                        ; // x       768  ALU
/*4f1464f0*/  IADD3 R76, P2, R76, R102, RZ                               ; // x       768  ALU
/*4f146500*/  HMMA.16816.F32.BF16 R28, R128, R154, R28                   ; // x       768  tensor-core MMA
/*4f146510*/  LDSM.16.M88.4 R116, [R91+UR8+0x1000]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146520*/  IADD3.X R77, R77, R103, RZ, P2, !PT                        ; // x       768  ALU
/*4f146530*/  IADD3 R78, P2, R78, R102, RZ                               ; // x       768  ALU
/*4f146540*/  HMMA.16816.F32.BF16 R24, R128, R152, R24                   ; // x       768  tensor-core MMA
/*4f146550*/  IADD3.X R79, R79, R103, RZ, P2, !PT                        ; // x       768  ALU
/*4f146560*/  IADD3 R80, P2, R80, R102, RZ                               ; // x       768  ALU
/*4f146570*/  HMMA.16816.F32.BF16 R20, R128, R150, R20                   ; // x       768  tensor-core MMA
/*4f146580*/  LDSM.16.M88.4 R120, [R91+UR8+0x1800]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146590*/  IADD3.X R81, R81, R103, RZ, P2, !PT                        ; // x       768  ALU
/*4f1465a0*/  IADD3 R82, P2, R82, R104, RZ                               ; // x       768  ALU
/*4f1465b0*/  HMMA.16816.F32.BF16 R16, R128, R148, R16                   ; // x       768  tensor-core MMA
/*4f1465c0*/  IADD3.X R83, R83, R105, RZ, P2, !PT                        ; // x       768  ALU
/*4f1465d0*/  IADD3 R84, P2, R84, R104, RZ                               ; // x       768  ALU
/*4f1465e0*/  HMMA.16816.F32.BF16 R32, R132, R148, R32                   ; // x       768  tensor-core MMA
/*4f1465f0*/  LDSM.16.M88.4 R140, [R95+UR8+0x4000]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146600*/  IADD3.X R85, R85, R105, RZ, P2, !PT                        ; // x       768  ALU
/*4f146610*/  IADD3 R86, P2, R86, R104, RZ                               ; // x       768  ALU
/*4f146620*/  HMMA.16816.F32.BF16 R36, R132, R150, R36                   ; // x       768  tensor-core MMA
/*4f146630*/  IADD3.X R87, R87, R105, RZ, P2, !PT                        ; // x       768  ALU
/*4f146640*/  IADD3 R88, P2, R88, R104, RZ                               ; // x       768  ALU
/*4f146650*/  HMMA.16816.F32.BF16 R40, R132, R152, R40                   ; // x       768  tensor-core MMA
/*4f146660*/  LDSM.16.M88.4 R144, [R95+UR8+0x4800]                       ; // x       768  shared-memory matrix load (tensor-core operand)
/*4f146670*/  IADD3.X R89, R89, R105, RZ, P2, !PT                        ; // x       768  ALU
/*4f146680*/  @!P0  MOV R99, RZ                                          ; // x       768  ALU
/*4f146690*/  HMMA.16816.F32.BF16 R44, R132, R154, R44                   ; // x       768  tensor-core MMA
/*4f1466a0*/  R2P PR, R99, 0x78                                          ; // x       768  ALU
/*4f1466b0*/  HMMA.16816.F32.BF16 R60, R136, R154, R60                   ; // x       768  tensor-core MMA
/*4f1466c0*/  HMMA.16816.F32.BF16 R56, R136, R152, R56                   ; // x       768  tensor-core MMA
/*4f1466d0*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4][R66.64] ; // x       768  async global->shared copy (tile staging)
/*4f1466e0*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x800][R68.64] ; // x       768  async global->shared copy (tile staging)
/*4f1466f0*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x1000][R70.64] ; // x       768  async global->shared copy (tile staging)
/*4f146700*/  HMMA.16816.F32.BF16 R52, R136, R150, R52                   ; // x       768  tensor-core MMA
/*4f146710*/  HMMA.16816.F32.BF16 R48, R136, R148, R48                   ; // x       768  tensor-core MMA
/*4f146720*/  @P1   BRA.U 0x24f145e50                                    ; // x       768  branch
```
