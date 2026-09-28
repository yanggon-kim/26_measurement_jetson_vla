# Hot loop — `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` (sm87)

PC range 0x4f552c40-0x4f553180; backward BRA 0x4f553180 -> 0x4f552c40: of the 6 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn_sm87.sass`; per-PC counts: `raw/ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*4f552c40*/  HMMA.16816.F32.BF16 R0, R96, R128, R0                      ; // x       720  tensor-core MMA
/*4f552c50*/  LDSM.16.M88.4 R112, [R84+UR8]                              ; // x       720  shared-memory matrix load (tensor-core operand)
/*4f552c60*/  R2P PR, R87.B1, 0x78                                       ; // x       720  ALU
/*4f552c70*/  HMMA.16816.F32.BF16 R4, R96, R130, R4                      ; // x       720  tensor-core MMA
/*4f552c80*/  HMMA.16816.F32.BF16 R8, R96, R132, R8                      ; // x       720  tensor-core MMA
/*4f552c90*/  LDSM.16.M88.4 R116, [R84+UR8+0x800]                        ; // x       720  shared-memory matrix load (tensor-core operand)
/*4f552ca0*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R82+UR6][R74.64] ; // x       720  async global->shared copy (tile staging)
/*4f552cb0*/  HMMA.16816.F32.BF16 R12, R96, R134, R12                    ; // x       720  tensor-core MMA
/*4f552cc0*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R82+UR6+0x800][R76.64] ; // x       720  async global->shared copy (tile staging)
/*4f552cd0*/  HMMA.16816.F32.BF16 R28, R100, R134, R28                   ; // x       720  tensor-core MMA
/*4f552ce0*/  LDSM.16.M88.4 R120, [R84+UR8+0x1000]                       ; // x       720  shared-memory matrix load (tensor-core operand)
/*4f552cf0*/  UISETP.LT.AND UP0, UPT, UR8, 0x10000, UPT                  ; // x       720  Uniform
/*4f552d00*/  HMMA.16816.F32.BF16 R24, R100, R132, R24                   ; // x       720  tensor-core MMA
/*4f552d10*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R82+UR6+0x1000][R78.64] ; // x       720  async global->shared copy (tile staging)
/*4f552d20*/  HMMA.16816.F32.BF16 R20, R100, R130, R20                   ; // x       720  tensor-core MMA
/*4f552d30*/  LDSM.16.M88.4 R124, [R84+UR8+0x1800]                       ; // x       720  shared-memory matrix load (tensor-core operand)
/*4f552d40*/  HMMA.16816.F32.BF16 R16, R100, R128, R16                   ; // x       720  tensor-core MMA
/*4f552d50*/  HMMA.16816.F32.BF16 R32, R104, R128, R32                   ; // x       720  tensor-core MMA
/*4f552d60*/  LDSM.16.M88.4 R136, [R86+UR8+0x2000]                       ; // x       720  shared-memory matrix load (tensor-core operand)
/*4f552d70*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R82+UR6+0x1800][R80.64] ; // x       720  async global->shared copy (tile staging)
/*4f552d80*/  LDGDEPBAR                                                  ; // x       720  LSU / Load
/*4f552d90*/  @UP1  UIADD3 UR4, UR4, 0x4000, URZ                         ; // x       720  Uniform
/*4f552da0*/  HMMA.16816.F32.BF16 R36, R104, R130, R36                   ; // x       720  tensor-core MMA
/*4f552db0*/  @!UP1 UMOV UR4, URZ                                        ; // x       720  Uniform
/*4f552dc0*/  UISETP.LT.AND UP1, UPT, UR4, 0x10000, UPT                  ; // x       720  Uniform
/*4f552dd0*/  UMOV UR6, UR4                                              ; // x       720  Uniform
/*4f552de0*/  HMMA.16816.F32.BF16 R40, R104, R132, R40                   ; // x       720  tensor-core MMA
/*4f552df0*/  LDSM.16.M88.4 R140, [R86+UR8+0x2800]                       ; // x       720  shared-memory matrix load (tensor-core operand)
/*4f552e00*/  @UP0  UIADD3 UR8, UR8, 0x4000, URZ                         ; // x       720  Uniform
/*4f552e10*/  @!UP0 UMOV UR8, URZ                                        ; // x       720  Uniform
/*4f552e20*/  IADD3 R88, R88, -0x40, RZ                                  ; // x       720  ALU
/*4f552e30*/  HMMA.16816.F32.BF16 R44, R104, R134, R44                   ; // x       720  tensor-core MMA
/*4f552e40*/  DEPBAR.LE SB0, 0x3                                         ; // x       720  wait on async-copy group
/*4f552e50*/  ISETP.GT.AND P0, PT, R88, RZ, PT                           ; // x       720  ALU
/*4f552e60*/  HMMA.16816.F32.BF16 R60, R108, R134, R60                   ; // x       720  tensor-core MMA
/*4f552e70*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       720  block barrier
/*4f552e80*/  HMMA.16816.F32.BF16 R56, R108, R132, R56                   ; // x       720  tensor-core MMA
/*4f552e90*/  HMMA.16816.F32.BF16 R52, R108, R130, R52                   ; // x       720  tensor-core MMA
/*4f552ea0*/  ISETP.GT.AND P1, PT, R88, -0x100, PT                       ; // x       720  ALU
/*4f552eb0*/  IADD3 R66, P2, R66, R90, RZ                                ; // x       720  ALU
/*4f552ec0*/  HMMA.16816.F32.BF16 R48, R108, R128, R48                   ; // x       720  tensor-core MMA
/*4f552ed0*/  IADD3.X R67, R67, R91, RZ, P2, !PT                         ; // x       720  ALU
/*4f552ee0*/  IADD3 R68, P2, R68, R90, RZ                                ; // x       720  ALU
/*4f552ef0*/  HMMA.16816.F32.BF16 R0, R112, R136, R0                     ; // x       720  tensor-core MMA
/*4f552f00*/  LDSM.16.M88.4 R96, [R83+UR8]                               ; // x       720  shared-memory matrix load (tensor-core operand)
/*4f552f10*/  IADD3.X R69, R69, R91, RZ, P2, !PT                         ; // x       720  ALU
/*4f552f20*/  IADD3 R70, P2, R70, R90, RZ                                ; // x       720  ALU
/*4f552f30*/  HMMA.16816.F32.BF16 R4, R112, R138, R4                     ; // x       720  tensor-core MMA
/*4f552f40*/  IADD3.X R71, R71, R91, RZ, P2, !PT                         ; // x       720  ALU
/*4f552f50*/  IADD3 R72, P2, R72, R90, RZ                                ; // x       720  ALU
/*4f552f60*/  HMMA.16816.F32.BF16 R8, R112, R140, R8                     ; // x       720  tensor-core MMA
/*4f552f70*/  LDSM.16.M88.4 R100, [R83+UR8+0x800]                        ; // x       720  shared-memory matrix load (tensor-core operand)
/*4f552f80*/  IADD3.X R73, R73, R91, RZ, P2, !PT                         ; // x       720  ALU
/*4f552f90*/  IADD3 R74, P2, R74, R92, RZ                                ; // x       720  ALU
/*4f552fa0*/  HMMA.16816.F32.BF16 R12, R112, R142, R12                   ; // x       720  tensor-core MMA
/*4f552fb0*/  IADD3.X R75, R75, R93, RZ, P2, !PT                         ; // x       720  ALU
/*4f552fc0*/  IADD3 R76, P2, R76, R92, RZ                                ; // x       720  ALU
/*4f552fd0*/  HMMA.16816.F32.BF16 R28, R116, R142, R28                   ; // x       720  tensor-core MMA
/*4f552fe0*/  LDSM.16.M88.4 R104, [R83+UR8+0x1000]                       ; // x       720  shared-memory matrix load (tensor-core operand)
/*4f552ff0*/  IADD3.X R77, R77, R93, RZ, P2, !PT                         ; // x       720  ALU
/*4f553000*/  IADD3 R78, P2, R78, R92, RZ                                ; // x       720  ALU
/*4f553010*/  HMMA.16816.F32.BF16 R24, R116, R140, R24                   ; // x       720  tensor-core MMA
/*4f553020*/  IADD3.X R79, R79, R93, RZ, P2, !PT                         ; // x       720  ALU
/*4f553030*/  IADD3 R80, P2, R80, R92, RZ                                ; // x       720  ALU
/*4f553040*/  HMMA.16816.F32.BF16 R20, R116, R138, R20                   ; // x       720  tensor-core MMA
/*4f553050*/  LDSM.16.M88.4 R108, [R83+UR8+0x1800]                       ; // x       720  shared-memory matrix load (tensor-core operand)
/*4f553060*/  IADD3.X R81, R81, R93, RZ, P2, !PT                         ; // x       720  ALU
/*4f553070*/  @!P0  MOV R87, RZ                                          ; // x       720  ALU
/*4f553080*/  HMMA.16816.F32.BF16 R16, R116, R136, R16                   ; // x       720  tensor-core MMA
/*4f553090*/  R2P PR, R87, 0x78                                          ; // x       720  ALU
/*4f5530a0*/  HMMA.16816.F32.BF16 R32, R120, R136, R32                   ; // x       720  tensor-core MMA
/*4f5530b0*/  LDSM.16.M88.4 R128, [R85+UR8+0x2000]                       ; // x       720  shared-memory matrix load (tensor-core operand)
/*4f5530c0*/  HMMA.16816.F32.BF16 R36, R120, R138, R36                   ; // x       720  tensor-core MMA
/*4f5530d0*/  @P3   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4][R66.64] ; // x       720  async global->shared copy (tile staging)
/*4f5530e0*/  @P4   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x800][R68.64] ; // x       720  async global->shared copy (tile staging)
/*4f5530f0*/  @P5   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x1000][R70.64] ; // x       720  async global->shared copy (tile staging)
/*4f553100*/  HMMA.16816.F32.BF16 R40, R120, R140, R40                   ; // x       720  tensor-core MMA
/*4f553110*/  LDSM.16.M88.4 R132, [R85+UR8+0x2800]                       ; // x       720  shared-memory matrix load (tensor-core operand)
/*4f553120*/  HMMA.16816.F32.BF16 R44, R120, R142, R44                   ; // x       720  tensor-core MMA
/*4f553130*/  HMMA.16816.F32.BF16 R60, R124, R142, R60                   ; // x       720  tensor-core MMA
/*4f553140*/  @P6   LDGSTS.E.BYPASS.LTC128B.128.CONSTANT [R65+UR4+0x1800][R72.64] ; // x       720  async global->shared copy (tile staging)
/*4f553150*/  HMMA.16816.F32.BF16 R56, R124, R140, R56                   ; // x       720  tensor-core MMA
/*4f553160*/  HMMA.16816.F32.BF16 R52, R124, R138, R52                   ; // x       720  tensor-core MMA
/*4f553170*/  HMMA.16816.F32.BF16 R48, R124, R136, R48                   ; // x       720  tensor-core MMA
/*4f553180*/  @P1   BRA.U 0x24f552c40                                    ; // x       720  branch
```
