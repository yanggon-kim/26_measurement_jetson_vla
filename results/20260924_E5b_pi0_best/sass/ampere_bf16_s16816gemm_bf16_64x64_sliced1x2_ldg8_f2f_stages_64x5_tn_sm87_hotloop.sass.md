# Hot loop — `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` (sm87)

PC range 0x2ee11340-0x2ee11880; backward BRA 0x2ee11880 -> 0x2ee11340: of the 6 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn_sm87.sass`; per-PC counts: `raw/ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*2ee11340*/  HMMA.16816.F32.BF16 R0, R96, R128, R0                      ; // x      2048  tensor-core MMA
/*2ee11350*/  LDSM.16.M88.4 R112, [R84+UR8]                              ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ee11360*/  R2P PR, R87.B1, 0x78                                       ; // x      2048  ALU
/*2ee11370*/  HMMA.16816.F32.BF16 R4, R96, R130, R4                      ; // x      2048  tensor-core MMA
/*2ee11380*/  HMMA.16816.F32.BF16 R8, R96, R132, R8                      ; // x      2048  tensor-core MMA
/*2ee11390*/  LDSM.16.M88.4 R116, [R84+UR8+0x800]                        ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ee113a0*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R82+UR6][R74.64] ; // x      2048  async global->shared copy (tile staging)
/*2ee113b0*/  HMMA.16816.F32.BF16 R12, R96, R134, R12                    ; // x      2048  tensor-core MMA
/*2ee113c0*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R82+UR6+0x800][R76.64] ; // x      2048  async global->shared copy (tile staging)
/*2ee113d0*/  HMMA.16816.F32.BF16 R28, R100, R134, R28                   ; // x      2048  tensor-core MMA
/*2ee113e0*/  LDSM.16.M88.4 R120, [R84+UR8+0x1000]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ee113f0*/  UISETP.LT.AND UP0, UPT, UR8, 0x10000, UPT                  ; // x      2048  Uniform
/*2ee11400*/  HMMA.16816.F32.BF16 R24, R100, R132, R24                   ; // x      2048  tensor-core MMA
/*2ee11410*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R82+UR6+0x1000][R78.64] ; // x      2048  async global->shared copy (tile staging)
/*2ee11420*/  HMMA.16816.F32.BF16 R20, R100, R130, R20                   ; // x      2048  tensor-core MMA
/*2ee11430*/  LDSM.16.M88.4 R124, [R84+UR8+0x1800]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ee11440*/  HMMA.16816.F32.BF16 R16, R100, R128, R16                   ; // x      2048  tensor-core MMA
/*2ee11450*/  HMMA.16816.F32.BF16 R32, R104, R128, R32                   ; // x      2048  tensor-core MMA
/*2ee11460*/  LDSM.16.M88.4 R136, [R86+UR8+0x2000]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ee11470*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R82+UR6+0x1800][R80.64] ; // x      2048  async global->shared copy (tile staging)
/*2ee11480*/  LDGDEPBAR                                                  ; // x      2048  LSU / Load
/*2ee11490*/  @UP1  UIADD3 UR4, UR4, 0x4000, URZ                         ; // x      2048  Uniform
/*2ee114a0*/  HMMA.16816.F32.BF16 R36, R104, R130, R36                   ; // x      2048  tensor-core MMA
/*2ee114b0*/  @!UP1 UMOV UR4, URZ                                        ; // x      2048  Uniform
/*2ee114c0*/  UISETP.LT.AND UP1, UPT, UR4, 0x10000, UPT                  ; // x      2048  Uniform
/*2ee114d0*/  UMOV UR6, UR4                                              ; // x      2048  Uniform
/*2ee114e0*/  HMMA.16816.F32.BF16 R40, R104, R132, R40                   ; // x      2048  tensor-core MMA
/*2ee114f0*/  LDSM.16.M88.4 R140, [R86+UR8+0x2800]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ee11500*/  @UP0  UIADD3 UR8, UR8, 0x4000, URZ                         ; // x      2048  Uniform
/*2ee11510*/  @!UP0 UMOV UR8, URZ                                        ; // x      2048  Uniform
/*2ee11520*/  IADD3 R88, R88, -0x40, RZ                                  ; // x      2048  ALU
/*2ee11530*/  HMMA.16816.F32.BF16 R44, R104, R134, R44                   ; // x      2048  tensor-core MMA
/*2ee11540*/  DEPBAR.LE SB0, 0x3                                         ; // x      2048  wait on async-copy group
/*2ee11550*/  ISETP.GT.AND P0, PT, R88, RZ, PT                           ; // x      2048  ALU
/*2ee11560*/  HMMA.16816.F32.BF16 R60, R108, R134, R60                   ; // x      2048  tensor-core MMA
/*2ee11570*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x      2048  block barrier
/*2ee11580*/  HMMA.16816.F32.BF16 R56, R108, R132, R56                   ; // x      2048  tensor-core MMA
/*2ee11590*/  HMMA.16816.F32.BF16 R52, R108, R130, R52                   ; // x      2048  tensor-core MMA
/*2ee115a0*/  ISETP.GT.AND P1, PT, R88, -0x100, PT                       ; // x      2048  ALU
/*2ee115b0*/  IADD3 R66, P2, R66, R90, RZ                                ; // x      2048  ALU
/*2ee115c0*/  HMMA.16816.F32.BF16 R48, R108, R128, R48                   ; // x      2048  tensor-core MMA
/*2ee115d0*/  IADD3.X R67, R67, R91, RZ, P2, !PT                         ; // x      2048  ALU
/*2ee115e0*/  IADD3 R68, P2, R68, R90, RZ                                ; // x      2048  ALU
/*2ee115f0*/  HMMA.16816.F32.BF16 R0, R112, R136, R0                     ; // x      2048  tensor-core MMA
/*2ee11600*/  LDSM.16.M88.4 R96, [R83+UR8]                               ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ee11610*/  IADD3.X R69, R69, R91, RZ, P2, !PT                         ; // x      2048  ALU
/*2ee11620*/  IADD3 R70, P2, R70, R90, RZ                                ; // x      2048  ALU
/*2ee11630*/  HMMA.16816.F32.BF16 R4, R112, R138, R4                     ; // x      2048  tensor-core MMA
/*2ee11640*/  IADD3.X R71, R71, R91, RZ, P2, !PT                         ; // x      2048  ALU
/*2ee11650*/  IADD3 R72, P2, R72, R90, RZ                                ; // x      2048  ALU
/*2ee11660*/  HMMA.16816.F32.BF16 R8, R112, R140, R8                     ; // x      2048  tensor-core MMA
/*2ee11670*/  LDSM.16.M88.4 R100, [R83+UR8+0x800]                        ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ee11680*/  IADD3.X R73, R73, R91, RZ, P2, !PT                         ; // x      2048  ALU
/*2ee11690*/  IADD3 R74, P2, R74, R92, RZ                                ; // x      2048  ALU
/*2ee116a0*/  HMMA.16816.F32.BF16 R12, R112, R142, R12                   ; // x      2048  tensor-core MMA
/*2ee116b0*/  IADD3.X R75, R75, R93, RZ, P2, !PT                         ; // x      2048  ALU
/*2ee116c0*/  IADD3 R76, P2, R76, R92, RZ                                ; // x      2048  ALU
/*2ee116d0*/  HMMA.16816.F32.BF16 R28, R116, R142, R28                   ; // x      2048  tensor-core MMA
/*2ee116e0*/  LDSM.16.M88.4 R104, [R83+UR8+0x1000]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ee116f0*/  IADD3.X R77, R77, R93, RZ, P2, !PT                         ; // x      2048  ALU
/*2ee11700*/  IADD3 R78, P2, R78, R92, RZ                                ; // x      2048  ALU
/*2ee11710*/  HMMA.16816.F32.BF16 R24, R116, R140, R24                   ; // x      2048  tensor-core MMA
/*2ee11720*/  IADD3.X R79, R79, R93, RZ, P2, !PT                         ; // x      2048  ALU
/*2ee11730*/  IADD3 R80, P2, R80, R92, RZ                                ; // x      2048  ALU
/*2ee11740*/  HMMA.16816.F32.BF16 R20, R116, R138, R20                   ; // x      2048  tensor-core MMA
/*2ee11750*/  LDSM.16.M88.4 R108, [R83+UR8+0x1800]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ee11760*/  IADD3.X R81, R81, R93, RZ, P2, !PT                         ; // x      2048  ALU
/*2ee11770*/  @!P0  MOV R87, RZ                                          ; // x      2048  ALU
/*2ee11780*/  HMMA.16816.F32.BF16 R16, R116, R136, R16                   ; // x      2048  tensor-core MMA
/*2ee11790*/  R2P PR, R87, 0x78                                          ; // x      2048  ALU
/*2ee117a0*/  HMMA.16816.F32.BF16 R32, R120, R136, R32                   ; // x      2048  tensor-core MMA
/*2ee117b0*/  LDSM.16.M88.4 R128, [R85+UR8+0x2000]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ee117c0*/  HMMA.16816.F32.BF16 R36, R120, R138, R36                   ; // x      2048  tensor-core MMA
/*2ee117d0*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4][R66.64] ; // x      2048  async global->shared copy (tile staging)
/*2ee117e0*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x800][R68.64] ; // x      2048  async global->shared copy (tile staging)
/*2ee117f0*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x1000][R70.64] ; // x      2048  async global->shared copy (tile staging)
/*2ee11800*/  HMMA.16816.F32.BF16 R40, R120, R140, R40                   ; // x      2048  tensor-core MMA
/*2ee11810*/  LDSM.16.M88.4 R132, [R85+UR8+0x2800]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ee11820*/  HMMA.16816.F32.BF16 R44, R120, R142, R44                   ; // x      2048  tensor-core MMA
/*2ee11830*/  HMMA.16816.F32.BF16 R60, R124, R142, R60                   ; // x      2048  tensor-core MMA
/*2ee11840*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x1800][R72.64] ; // x      2048  async global->shared copy (tile staging)
/*2ee11850*/  HMMA.16816.F32.BF16 R56, R124, R140, R56                   ; // x      2048  tensor-core MMA
/*2ee11860*/  HMMA.16816.F32.BF16 R52, R124, R138, R52                   ; // x      2048  tensor-core MMA
/*2ee11870*/  HMMA.16816.F32.BF16 R48, R124, R136, R48                   ; // x      2048  tensor-core MMA
/*2ee11880*/  @P1   BRA.U 0x42ee11340                                    ; // x      2048  branch
```
