# Hot loop — `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn` (sm87)

PC range 0x2ea02650-0x2ea02f20; backward BRA 0x2ea02f20 -> 0x2ea02650: of the 6 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn_sm87.sass`; per-PC counts: `raw/ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*2ea02650*/  HMMA.16816.F32.BF16 R0, R108, R140, R0                     ; // x      2048  tensor-core MMA
/*2ea02660*/  LDSM.16.M88.4 R124, [R92+UR8]                              ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02670*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x1800][R72.64] ; // x      2048  async global->shared copy (tile staging)
/*2ea02680*/  R2P PR, R99.B1, 0x78                                       ; // x      2048  ALU
/*2ea02690*/  HMMA.16816.F32.BF16 R4, R108, R142, R4                     ; // x      2048  tensor-core MMA
/*2ea026a0*/  HMMA.16816.F32.BF16 R8, R108, R144, R8                     ; // x      2048  tensor-core MMA
/*2ea026b0*/  LDSM.16.M88.4 R128, [R92+UR8+0x800]                        ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea026c0*/  HMMA.16816.F32.BF16 R12, R108, R146, R12                   ; // x      2048  tensor-core MMA
/*2ea026d0*/  HMMA.16816.F32.BF16 R28, R112, R146, R28                   ; // x      2048  tensor-core MMA
/*2ea026e0*/  LDSM.16.M88.4 R132, [R92+UR8+0x1000]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea026f0*/  HMMA.16816.F32.BF16 R24, R112, R144, R24                   ; // x      2048  tensor-core MMA
/*2ea02700*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x2000][R74.64] ; // x      2048  async global->shared copy (tile staging)
/*2ea02710*/  HMMA.16816.F32.BF16 R20, R112, R142, R20                   ; // x      2048  tensor-core MMA
/*2ea02720*/  LDSM.16.M88.4 R136, [R92+UR8+0x1800]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02730*/  HMMA.16816.F32.BF16 R16, R112, R140, R16                   ; // x      2048  tensor-core MMA
/*2ea02740*/  HMMA.16816.F32.BF16 R32, R116, R140, R32                   ; // x      2048  tensor-core MMA
/*2ea02750*/  LDSM.16.M88.4 R148, [R96+UR8+0x4000]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02760*/  HMMA.16816.F32.BF16 R36, R116, R142, R36                   ; // x      2048  tensor-core MMA
/*2ea02770*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x2800][R76.64] ; // x      2048  async global->shared copy (tile staging)
/*2ea02780*/  HMMA.16816.F32.BF16 R40, R116, R144, R40                   ; // x      2048  tensor-core MMA
/*2ea02790*/  LDSM.16.M88.4 R152, [R96+UR8+0x4800]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea027a0*/  HMMA.16816.F32.BF16 R44, R116, R146, R44                   ; // x      2048  tensor-core MMA
/*2ea027b0*/  HMMA.16816.F32.BF16 R60, R120, R146, R60                   ; // x      2048  tensor-core MMA
/*2ea027c0*/  HMMA.16816.F32.BF16 R56, R120, R144, R56                   ; // x      2048  tensor-core MMA
/*2ea027d0*/  HMMA.16816.F32.BF16 R52, R120, R142, R52                   ; // x      2048  tensor-core MMA
/*2ea027e0*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x3000][R78.64] ; // x      2048  async global->shared copy (tile staging)
/*2ea027f0*/  HMMA.16816.F32.BF16 R48, R120, R140, R48                   ; // x      2048  tensor-core MMA
/*2ea02800*/  HMMA.16816.F32.BF16 R0, R124, R148, R0                     ; // x      2048  tensor-core MMA
/*2ea02810*/  LDSM.16.M88.4 R108, [R93+UR8]                              ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02820*/  HMMA.16816.F32.BF16 R4, R124, R150, R4                     ; // x      2048  tensor-core MMA
/*2ea02830*/  HMMA.16816.F32.BF16 R8, R124, R152, R8                     ; // x      2048  tensor-core MMA
/*2ea02840*/  LDSM.16.M88.4 R112, [R93+UR8+0x800]                        ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02850*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x3800][R80.64] ; // x      2048  async global->shared copy (tile staging)
/*2ea02860*/  R2P PR, R99.B2, 0x78                                       ; // x      2048  ALU
/*2ea02870*/  HMMA.16816.F32.BF16 R12, R124, R154, R12                   ; // x      2048  tensor-core MMA
/*2ea02880*/  HMMA.16816.F32.BF16 R28, R128, R154, R28                   ; // x      2048  tensor-core MMA
/*2ea02890*/  LDSM.16.M88.4 R116, [R93+UR8+0x1000]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea028a0*/  HMMA.16816.F32.BF16 R24, R128, R152, R24                   ; // x      2048  tensor-core MMA
/*2ea028b0*/  HMMA.16816.F32.BF16 R20, R128, R150, R20                   ; // x      2048  tensor-core MMA
/*2ea028c0*/  LDSM.16.M88.4 R120, [R93+UR8+0x1800]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea028d0*/  HMMA.16816.F32.BF16 R16, R128, R148, R16                   ; // x      2048  tensor-core MMA
/*2ea028e0*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R90+UR6][R82.64] ; // x      2048  async global->shared copy (tile staging)
/*2ea028f0*/  HMMA.16816.F32.BF16 R32, R132, R148, R32                   ; // x      2048  tensor-core MMA
/*2ea02900*/  LDSM.16.M88.4 R140, [R97+UR8+0x4000]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02910*/  HMMA.16816.F32.BF16 R36, R132, R150, R36                   ; // x      2048  tensor-core MMA
/*2ea02920*/  HMMA.16816.F32.BF16 R40, R132, R152, R40                   ; // x      2048  tensor-core MMA
/*2ea02930*/  LDSM.16.M88.4 R144, [R97+UR8+0x4800]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02940*/  HMMA.16816.F32.BF16 R44, R132, R154, R44                   ; // x      2048  tensor-core MMA
/*2ea02950*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R90+UR6+0x800][R84.64] ; // x      2048  async global->shared copy (tile staging)
/*2ea02960*/  HMMA.16816.F32.BF16 R60, R136, R154, R60                   ; // x      2048  tensor-core MMA
/*2ea02970*/  HMMA.16816.F32.BF16 R56, R136, R152, R56                   ; // x      2048  tensor-core MMA
/*2ea02980*/  HMMA.16816.F32.BF16 R52, R136, R150, R52                   ; // x      2048  tensor-core MMA
/*2ea02990*/  HMMA.16816.F32.BF16 R48, R136, R148, R48                   ; // x      2048  tensor-core MMA
/*2ea029a0*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R90+UR6+0x1000][R86.64] ; // x      2048  async global->shared copy (tile staging)
/*2ea029b0*/  HMMA.16816.F32.BF16 R0, R108, R140, R0                     ; // x      2048  tensor-core MMA
/*2ea029c0*/  LDSM.16.M88.4 R124, [R94+UR8]                              ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea029d0*/  HMMA.16816.F32.BF16 R4, R108, R142, R4                     ; // x      2048  tensor-core MMA
/*2ea029e0*/  HMMA.16816.F32.BF16 R8, R108, R144, R8                     ; // x      2048  tensor-core MMA
/*2ea029f0*/  LDSM.16.M88.4 R128, [R94+UR8+0x800]                        ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02a00*/  HMMA.16816.F32.BF16 R12, R108, R146, R12                   ; // x      2048  tensor-core MMA
/*2ea02a10*/  HMMA.16816.F32.BF16 R28, R112, R146, R28                   ; // x      2048  tensor-core MMA
/*2ea02a20*/  LDSM.16.M88.4 R132, [R94+UR8+0x1000]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02a30*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R90+UR6+0x1800][R88.64] ; // x      2048  async global->shared copy (tile staging)
/*2ea02a40*/  UISETP.LT.AND UP0, UPT, UR8, 0xc000, UPT                   ; // x      2048  Uniform
/*2ea02a50*/  LDGDEPBAR                                                  ; // x      2048  LSU / Load
/*2ea02a60*/  @UP1  UIADD3 UR4, UR4, 0x6000, URZ                         ; // x      2048  Uniform
/*2ea02a70*/  HMMA.16816.F32.BF16 R24, R112, R144, R24                   ; // x      2048  tensor-core MMA
/*2ea02a80*/  @!UP1 UMOV UR4, URZ                                        ; // x      2048  Uniform
/*2ea02a90*/  UISETP.LT.AND UP1, UPT, UR4, 0xc000, UPT                   ; // x      2048  Uniform
/*2ea02aa0*/  UMOV UR6, UR4                                              ; // x      2048  Uniform
/*2ea02ab0*/  HMMA.16816.F32.BF16 R20, R112, R142, R20                   ; // x      2048  tensor-core MMA
/*2ea02ac0*/  LDSM.16.M88.4 R136, [R94+UR8+0x1800]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02ad0*/  HMMA.16816.F32.BF16 R16, R112, R140, R16                   ; // x      2048  tensor-core MMA
/*2ea02ae0*/  HMMA.16816.F32.BF16 R32, R116, R140, R32                   ; // x      2048  tensor-core MMA
/*2ea02af0*/  LDSM.16.M88.4 R148, [R98+UR8+0x4000]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02b00*/  HMMA.16816.F32.BF16 R36, R116, R142, R36                   ; // x      2048  tensor-core MMA
/*2ea02b10*/  HMMA.16816.F32.BF16 R40, R116, R144, R40                   ; // x      2048  tensor-core MMA
/*2ea02b20*/  LDSM.16.M88.4 R152, [R98+UR8+0x4800]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02b30*/  @UP0  UIADD3 UR8, UR8, 0x6000, URZ                         ; // x      2048  Uniform
/*2ea02b40*/  @!UP0 UMOV UR8, URZ                                        ; // x      2048  Uniform
/*2ea02b50*/  IADD3 R100, R100, -0x40, RZ                                ; // x      2048  ALU
/*2ea02b60*/  HMMA.16816.F32.BF16 R44, R116, R146, R44                   ; // x      2048  tensor-core MMA
/*2ea02b70*/  DEPBAR.LE SB0, 0x1                                         ; // x      2048  wait on async-copy group
/*2ea02b80*/  ISETP.GT.AND P0, PT, R100, RZ, PT                          ; // x      2048  ALU
/*2ea02b90*/  HMMA.16816.F32.BF16 R60, R120, R146, R60                   ; // x      2048  tensor-core MMA
/*2ea02ba0*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x      2048  block barrier
/*2ea02bb0*/  HMMA.16816.F32.BF16 R56, R120, R144, R56                   ; // x      2048  tensor-core MMA
/*2ea02bc0*/  HMMA.16816.F32.BF16 R52, R120, R142, R52                   ; // x      2048  tensor-core MMA
/*2ea02bd0*/  ISETP.GT.AND P1, PT, R100, -0x80, PT                       ; // x      2048  ALU
/*2ea02be0*/  IADD3 R66, P2, R66, R102, RZ                               ; // x      2048  ALU
/*2ea02bf0*/  HMMA.16816.F32.BF16 R48, R120, R140, R48                   ; // x      2048  tensor-core MMA
/*2ea02c00*/  IADD3.X R67, R67, R103, RZ, P2, !PT                        ; // x      2048  ALU
/*2ea02c10*/  IADD3 R68, P2, R68, R102, RZ                               ; // x      2048  ALU
/*2ea02c20*/  HMMA.16816.F32.BF16 R0, R124, R148, R0                     ; // x      2048  tensor-core MMA
/*2ea02c30*/  LDSM.16.M88.4 R108, [R91+UR8]                              ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02c40*/  IADD3.X R69, R69, R103, RZ, P2, !PT                        ; // x      2048  ALU
/*2ea02c50*/  IADD3 R70, P2, R70, R102, RZ                               ; // x      2048  ALU
/*2ea02c60*/  HMMA.16816.F32.BF16 R4, R124, R150, R4                     ; // x      2048  tensor-core MMA
/*2ea02c70*/  IADD3.X R71, R71, R103, RZ, P2, !PT                        ; // x      2048  ALU
/*2ea02c80*/  IADD3 R72, P2, R72, R102, RZ                               ; // x      2048  ALU
/*2ea02c90*/  HMMA.16816.F32.BF16 R8, R124, R152, R8                     ; // x      2048  tensor-core MMA
/*2ea02ca0*/  LDSM.16.M88.4 R112, [R91+UR8+0x800]                        ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02cb0*/  IADD3.X R73, R73, R103, RZ, P2, !PT                        ; // x      2048  ALU
/*2ea02cc0*/  IADD3 R74, P2, R74, R102, RZ                               ; // x      2048  ALU
/*2ea02cd0*/  HMMA.16816.F32.BF16 R12, R124, R154, R12                   ; // x      2048  tensor-core MMA
/*2ea02ce0*/  IADD3.X R75, R75, R103, RZ, P2, !PT                        ; // x      2048  ALU
/*2ea02cf0*/  IADD3 R76, P2, R76, R102, RZ                               ; // x      2048  ALU
/*2ea02d00*/  HMMA.16816.F32.BF16 R28, R128, R154, R28                   ; // x      2048  tensor-core MMA
/*2ea02d10*/  LDSM.16.M88.4 R116, [R91+UR8+0x1000]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02d20*/  IADD3.X R77, R77, R103, RZ, P2, !PT                        ; // x      2048  ALU
/*2ea02d30*/  IADD3 R78, P2, R78, R102, RZ                               ; // x      2048  ALU
/*2ea02d40*/  HMMA.16816.F32.BF16 R24, R128, R152, R24                   ; // x      2048  tensor-core MMA
/*2ea02d50*/  IADD3.X R79, R79, R103, RZ, P2, !PT                        ; // x      2048  ALU
/*2ea02d60*/  IADD3 R80, P2, R80, R102, RZ                               ; // x      2048  ALU
/*2ea02d70*/  HMMA.16816.F32.BF16 R20, R128, R150, R20                   ; // x      2048  tensor-core MMA
/*2ea02d80*/  LDSM.16.M88.4 R120, [R91+UR8+0x1800]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02d90*/  IADD3.X R81, R81, R103, RZ, P2, !PT                        ; // x      2048  ALU
/*2ea02da0*/  IADD3 R82, P2, R82, R104, RZ                               ; // x      2048  ALU
/*2ea02db0*/  HMMA.16816.F32.BF16 R16, R128, R148, R16                   ; // x      2048  tensor-core MMA
/*2ea02dc0*/  IADD3.X R83, R83, R105, RZ, P2, !PT                        ; // x      2048  ALU
/*2ea02dd0*/  IADD3 R84, P2, R84, R104, RZ                               ; // x      2048  ALU
/*2ea02de0*/  HMMA.16816.F32.BF16 R32, R132, R148, R32                   ; // x      2048  tensor-core MMA
/*2ea02df0*/  LDSM.16.M88.4 R140, [R95+UR8+0x4000]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02e00*/  IADD3.X R85, R85, R105, RZ, P2, !PT                        ; // x      2048  ALU
/*2ea02e10*/  IADD3 R86, P2, R86, R104, RZ                               ; // x      2048  ALU
/*2ea02e20*/  HMMA.16816.F32.BF16 R36, R132, R150, R36                   ; // x      2048  tensor-core MMA
/*2ea02e30*/  IADD3.X R87, R87, R105, RZ, P2, !PT                        ; // x      2048  ALU
/*2ea02e40*/  IADD3 R88, P2, R88, R104, RZ                               ; // x      2048  ALU
/*2ea02e50*/  HMMA.16816.F32.BF16 R40, R132, R152, R40                   ; // x      2048  tensor-core MMA
/*2ea02e60*/  LDSM.16.M88.4 R144, [R95+UR8+0x4800]                       ; // x      2048  shared-memory matrix load (tensor-core operand)
/*2ea02e70*/  IADD3.X R89, R89, R105, RZ, P2, !PT                        ; // x      2048  ALU
/*2ea02e80*/  @!P0  MOV R99, RZ                                          ; // x      2048  ALU
/*2ea02e90*/  HMMA.16816.F32.BF16 R44, R132, R154, R44                   ; // x      2048  tensor-core MMA
/*2ea02ea0*/  R2P PR, R99, 0x78                                          ; // x      2048  ALU
/*2ea02eb0*/  HMMA.16816.F32.BF16 R60, R136, R154, R60                   ; // x      2048  tensor-core MMA
/*2ea02ec0*/  HMMA.16816.F32.BF16 R56, R136, R152, R56                   ; // x      2048  tensor-core MMA
/*2ea02ed0*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4][R66.64] ; // x      2048  async global->shared copy (tile staging)
/*2ea02ee0*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x800][R68.64] ; // x      2048  async global->shared copy (tile staging)
/*2ea02ef0*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x1000][R70.64] ; // x      2048  async global->shared copy (tile staging)
/*2ea02f00*/  HMMA.16816.F32.BF16 R52, R136, R150, R52                   ; // x      2048  tensor-core MMA
/*2ea02f10*/  HMMA.16816.F32.BF16 R48, R136, R148, R48                   ; // x      2048  tensor-core MMA
/*2ea02f20*/  @P1   BRA.U 0x42ea02650                                    ; // x      2048  branch
```
