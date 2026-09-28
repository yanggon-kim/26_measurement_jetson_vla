# Hot loop — `ampere_sgemm_128x64_tn` (sm87)

PC range 0x4cb3ebd0-0x4cb41670; backward BRA 0x4cb41670 -> 0x4cb3ebd0: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_sgemm_128x64_tn_sm87.sass`; per-PC counts: `raw/ampere_sgemm_128x64_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*4cb3ebd0*/  STS [R80], R64                                             ; // x       120  shared-memory store
/*4cb3ebe0*/  STS [R80+0x40], R65                                        ; // x       120  shared-memory store
/*4cb3ebf0*/  STS [R80+0x80], R66                                        ; // x       120  shared-memory store
/*4cb3ec00*/  STS [R80+0xc0], R67                                        ; // x       120  shared-memory store
/*4cb3ec10*/  STS [R80+0x100], R68                                       ; // x       120  shared-memory store
/*4cb3ec20*/  STS [R80+0x140], R69                                       ; // x       120  shared-memory store
/*4cb3ec30*/  STS [R80+0x180], R70                                       ; // x       120  shared-memory store
/*4cb3ec40*/  STS [R80+0x1c0], R71                                       ; // x       120  shared-memory store
/*4cb3ec50*/  STS [R81], R72                                             ; // x       120  shared-memory store
/*4cb3ec60*/  STS [R81+0x40], R73                                        ; // x       120  shared-memory store
/*4cb3ec70*/  STS [R81+0x80], R74                                        ; // x       120  shared-memory store
/*4cb3ec80*/  STS [R81+0xc0], R75                                        ; // x       120  shared-memory store
/*4cb3ec90*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       120  block barrier
/*4cb3eca0*/  IADD3 R80, R80, R87, RZ                                    ; // x       120  ALU
/*4cb3ecb0*/  IADD3 R81, R81, R87, RZ                                    ; // x       120  ALU
/*4cb3ecc0*/  LDS.128 R88, [R76]                                         ; // x       120  shared-memory load
/*4cb3ecd0*/  LDS.128 R104, [R77]                                        ; // x       120  shared-memory load
/*4cb3ece0*/  LDS.128 R92, [R76+0x80]                                    ; // x       120  shared-memory load
/*4cb3ecf0*/  LDS.128 R108, [R77+0x40]                                   ; // x       120  shared-memory load
/*4cb3ed00*/  FFMA R1, R88, R104, R1                                     ; // x      4320  FMA
/*4cb3ed10*/  ISETP.GE.AND P1, PT, R85, 0x10, PT                         ; // x      4320  ALU
/*4cb3ed20*/  FFMA R0, R88, R105, R0                                     ; // x      4320  FMA
/*4cb3ed30*/  FFMA R9, R89, R104, R9                                     ; // x      4320  FMA
/*4cb3ed40*/  IADD3 R85, R85, -0x8, RZ                                   ; // x      4320  ALU
/*4cb3ed50*/  FFMA R8, R89, R105, R8                                     ; // x      4320  FMA
/*4cb3ed60*/  FFMA R17, R90, R104, R17                                   ; // x      4320  FMA
/*4cb3ed70*/  LDS.128 R96, [R76+0x210]                                   ; // x      4320  shared-memory load
/*4cb3ed80*/  FFMA R16, R90, R105, R16                                   ; // x      4320  FMA
/*4cb3ed90*/  FFMA R25, R91, R104, R25                                   ; // x      4320  FMA
/*4cb3eda0*/  LDS.128 R112, [R77+0x110]                                  ; // x      4320  shared-memory load
/*4cb3edb0*/  FFMA R24, R91, R105, R24                                   ; // x      4320  FMA
/*4cb3edc0*/  FFMA R33, R92, R104, R33                                   ; // x      4320  FMA
/*4cb3edd0*/  LDS.128 R100, [R76+0x290]                                  ; // x      4320  shared-memory load
/*4cb3ede0*/  FFMA R32, R92, R105, R32                                   ; // x      4320  FMA
/*4cb3edf0*/  FFMA R41, R93, R104, R41                                   ; // x      4320  FMA
/*4cb3ee00*/  @!P1  R2P PR, RZ, 0x78                                     ; // x      4320  ALU
/*4cb3ee10*/  FFMA R40, R93, R105, R40                                   ; // x      4320  FMA
/*4cb3ee20*/  FFMA R49, R94, R104, R49                                   ; // x      4320  FMA
/*4cb3ee30*/  @P1   R2P PR, R86, 0x78                                    ; // x      4320  ALU
/*4cb3ee40*/  FFMA R48, R94, R105, R48                                   ; // x      4320  FMA
/*4cb3ee50*/  FFMA R57, R95, R104, R57                                   ; // x      4320  FMA
/*4cb3ee60*/  LDS.128 R116, [R77+0x150]                                  ; // x      4320  shared-memory load
/*4cb3ee70*/  FFMA R56, R95, R105, R56                                   ; // x      4320  FMA
/*4cb3ee80*/  FFMA R59, R95, R106, R59                                   ; // x      4320  FMA
/*4cb3ee90*/  FFMA R58, R95, R107, R58                                   ; // x      4320  FMA
/*4cb3eea0*/  FFMA R51, R94, R106, R51                                   ; // x      4320  FMA
/*4cb3eeb0*/  FFMA R50, R94, R107, R50                                   ; // x      4320  FMA
/*4cb3eec0*/  FFMA R43, R93, R106, R43                                   ; // x      4320  FMA
/*4cb3eed0*/  FFMA R42, R93, R107, R42                                   ; // x      4320  FMA
/*4cb3eee0*/  FFMA R35, R92, R106, R35                                   ; // x      4320  FMA
/*4cb3eef0*/  FFMA R34, R92, R107, R34                                   ; // x      4320  FMA
/*4cb3ef00*/  FFMA R27, R91, R106, R27                                   ; // x      4320  FMA
/*4cb3ef10*/  FFMA R26, R91, R107, R26                                   ; // x      4320  FMA
/*4cb3ef20*/  FFMA R19, R90, R106, R19                                   ; // x      4320  FMA
/*4cb3ef30*/  FFMA R18, R90, R107, R18                                   ; // x      4320  FMA
/*4cb3ef40*/  FFMA R11, R89, R106, R11                                   ; // x      4320  FMA
/*4cb3ef50*/  FFMA R10, R89, R107, R10                                   ; // x      4320  FMA
/*4cb3ef60*/  FFMA R3, R88, R106, R3                                     ; // x      4320  FMA
/*4cb3ef70*/  FFMA R2, R88, R107, R2                                     ; // x      4320  FMA
/*4cb3ef80*/  FFMA R5, R88, R108, R5                                     ; // x      4320  FMA
/*4cb3ef90*/  @P3   LDG.E.LTC128B.CONSTANT R64, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4cb3efa0*/  FFMA R4, R88, R109, R4                                     ; // x      4320  FMA
/*4cb3efb0*/  FFMA R13, R89, R108, R13                                   ; // x      4320  FMA
/*4cb3efc0*/  FFMA R12, R89, R109, R12                                   ; // x      4320  FMA
/*4cb3efd0*/  FFMA R21, R90, R108, R21                                   ; // x      4320  FMA
/*4cb3efe0*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4cb3eff0*/  FFMA R20, R90, R109, R20                                   ; // x      4320  FMA
/*4cb3f000*/  FFMA R29, R91, R108, R29                                   ; // x      4320  FMA
/*4cb3f010*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4cb3f020*/  FFMA R28, R91, R109, R28                                   ; // x      4320  FMA
/*4cb3f030*/  FFMA R37, R92, R108, R37                                   ; // x      4320  FMA
/*4cb3f040*/  @P4   LDG.E.LTC128B.CONSTANT R65, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4cb3f050*/  FFMA R36, R92, R109, R36                                   ; // x      4320  FMA
/*4cb3f060*/  FFMA R45, R93, R108, R45                                   ; // x      4320  FMA
/*4cb3f070*/  FFMA R44, R93, R109, R44                                   ; // x      4320  FMA
/*4cb3f080*/  FFMA R53, R94, R108, R53                                   ; // x      4320  FMA
/*4cb3f090*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4cb3f0a0*/  FFMA R52, R94, R109, R52                                   ; // x      4320  FMA
/*4cb3f0b0*/  FFMA R61, R95, R108, R61                                   ; // x      4320  FMA
/*4cb3f0c0*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4cb3f0d0*/  FFMA R60, R95, R109, R60                                   ; // x      4320  FMA
/*4cb3f0e0*/  FFMA R63, R95, R110, R63                                   ; // x      4320  FMA
/*4cb3f0f0*/  @P5   LDG.E.LTC128B.CONSTANT R66, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4cb3f100*/  FFMA R62, R95, R111, R62                                   ; // x      4320  FMA
/*4cb3f110*/  FFMA R55, R94, R110, R55                                   ; // x      4320  FMA
/*4cb3f120*/  FFMA R54, R94, R111, R54                                   ; // x      4320  FMA
/*4cb3f130*/  FFMA R47, R93, R110, R47                                   ; // x      4320  FMA
/*4cb3f140*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4cb3f150*/  FFMA R46, R93, R111, R46                                   ; // x      4320  FMA
/*4cb3f160*/  FFMA R39, R92, R110, R39                                   ; // x      4320  FMA
/*4cb3f170*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4cb3f180*/  FFMA R38, R92, R111, R38                                   ; // x      4320  FMA
/*4cb3f190*/  FFMA R31, R91, R110, R31                                   ; // x      4320  FMA
/*4cb3f1a0*/  @P6   LDG.E.LTC128B.CONSTANT R67, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4cb3f1b0*/  FFMA R30, R91, R111, R30                                   ; // x      4320  FMA
/*4cb3f1c0*/  FFMA R23, R90, R110, R23                                   ; // x      4320  FMA
/*4cb3f1d0*/  FFMA R22, R90, R111, R22                                   ; // x      4320  FMA
/*4cb3f1e0*/  FFMA R15, R89, R110, R15                                   ; // x      4320  FMA
/*4cb3f1f0*/  @P1   R2P PR, R86.B1, 0x78                                 ; // x      4320  ALU
/*4cb3f200*/  FFMA R14, R89, R111, R14                                   ; // x      4320  FMA
/*4cb3f210*/  FFMA R7, R88, R110, R7                                     ; // x      4320  FMA
/*4cb3f220*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4cb3f230*/  FFMA R6, R88, R111, R6                                     ; // x      4320  FMA
/*4cb3f240*/  FFMA R1, R96, R112, R1                                     ; // x      4320  FMA
/*4cb3f250*/  LDS.128 R88, [R76+0x420]                                   ; // x      4320  shared-memory load
/*4cb3f260*/  FFMA R0, R96, R113, R0                                     ; // x      4320  FMA
/*4cb3f270*/  FFMA R9, R97, R112, R9                                     ; // x      4320  FMA
/*4cb3f280*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4cb3f290*/  FFMA R8, R97, R113, R8                                     ; // x      4320  FMA
/*4cb3f2a0*/  FFMA R17, R98, R112, R17                                   ; // x      4320  FMA
/*4cb3f2b0*/  LDS.128 R104, [R77+0x220]                                  ; // x      4320  shared-memory load
/*4cb3f2c0*/  FFMA R16, R98, R113, R16                                   ; // x      4320  FMA
/*4cb3f2d0*/  FFMA R25, R99, R112, R25                                   ; // x      4320  FMA
/*4cb3f2e0*/  LDS.128 R92, [R76+0x4a0]                                   ; // x      4320  shared-memory load
/*4cb3f2f0*/  FFMA R24, R99, R113, R24                                   ; // x      4320  FMA
/*4cb3f300*/  FFMA R33, R100, R112, R33                                  ; // x      4320  FMA
/*4cb3f310*/  LDS.128 R108, [R77+0x260]                                  ; // x      4320  shared-memory load
/*4cb3f320*/  FFMA R32, R100, R113, R32                                  ; // x      4320  FMA
/*4cb3f330*/  FFMA R41, R101, R112, R41                                  ; // x      4320  FMA
/*4cb3f340*/  @P3   LDG.E.LTC128B.CONSTANT R68, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4cb3f350*/  FFMA R40, R101, R113, R40                                  ; // x      4320  FMA
/*4cb3f360*/  FFMA R49, R102, R112, R49                                  ; // x      4320  FMA
/*4cb3f370*/  FFMA R48, R102, R113, R48                                  ; // x      4320  FMA
/*4cb3f380*/  FFMA R57, R103, R112, R57                                  ; // x      4320  FMA
/*4cb3f390*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4cb3f3a0*/  FFMA R56, R103, R113, R56                                  ; // x      4320  FMA
/*4cb3f3b0*/  FFMA R59, R103, R114, R59                                  ; // x      4320  FMA
/*4cb3f3c0*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4cb3f3d0*/  FFMA R58, R103, R115, R58                                  ; // x      4320  FMA
/*4cb3f3e0*/  FFMA R51, R102, R114, R51                                  ; // x      4320  FMA
/*4cb3f3f0*/  @P4   LDG.E.LTC128B.CONSTANT R69, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4cb3f400*/  FFMA R50, R102, R115, R50                                  ; // x      4320  FMA
/*4cb3f410*/  FFMA R43, R101, R114, R43                                  ; // x      4320  FMA
/*4cb3f420*/  FFMA R42, R101, R115, R42                                  ; // x      4320  FMA
/*4cb3f430*/  FFMA R35, R100, R114, R35                                  ; // x      4320  FMA
/*4cb3f440*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4cb3f450*/  FFMA R34, R100, R115, R34                                  ; // x      4320  FMA
/*4cb3f460*/  FFMA R27, R99, R114, R27                                   ; // x      4320  FMA
/*4cb3f470*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4cb3f480*/  FFMA R26, R99, R115, R26                                   ; // x      4320  FMA
/*4cb3f490*/  FFMA R19, R98, R114, R19                                   ; // x      4320  FMA
/*4cb3f4a0*/  @P5   LDG.E.LTC128B.CONSTANT R70, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4cb3f4b0*/  FFMA R18, R98, R115, R18                                   ; // x      4320  FMA
/*4cb3f4c0*/  FFMA R11, R97, R114, R11                                   ; // x      4320  FMA
/*4cb3f4d0*/  FFMA R10, R97, R115, R10                                   ; // x      4320  FMA
/*4cb3f4e0*/  FFMA R3, R96, R114, R3                                     ; // x      4320  FMA
/*4cb3f4f0*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4cb3f500*/  FFMA R2, R96, R115, R2                                     ; // x      4320  FMA
/*4cb3f510*/  FFMA R5, R96, R116, R5                                     ; // x      4320  FMA
/*4cb3f520*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4cb3f530*/  FFMA R4, R96, R117, R4                                     ; // x      4320  FMA
/*4cb3f540*/  FFMA R13, R97, R116, R13                                   ; // x      4320  FMA
/*4cb3f550*/  @P6   LDG.E.LTC128B.CONSTANT R71, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4cb3f560*/  FFMA R12, R97, R117, R12                                   ; // x      4320  FMA
/*4cb3f570*/  FFMA R21, R98, R116, R21                                   ; // x      4320  FMA
/*4cb3f580*/  FFMA R20, R98, R117, R20                                   ; // x      4320  FMA
/*4cb3f590*/  FFMA R29, R99, R116, R29                                   ; // x      4320  FMA
/*4cb3f5a0*/  @P1   IADD3 R82, P2, R82, c[0x0][0x188], RZ                ; // x      4320  ALU
/*4cb3f5b0*/  FFMA R28, R99, R117, R28                                   ; // x      4320  FMA
/*4cb3f5c0*/  FFMA R37, R100, R116, R37                                  ; // x      4320  FMA
/*4cb3f5d0*/  @P1   IADD3.X R83, R83, c[0x0][0x18c], RZ, P2, !PT         ; // x      4320  ALU
/*4cb3f5e0*/  FFMA R36, R100, R117, R36                                  ; // x      4320  FMA
/*4cb3f5f0*/  FFMA R45, R101, R116, R45                                  ; // x      4320  FMA
/*4cb3f600*/  @P1   R2P PR, R86.B2, 0x78                                 ; // x      4320  ALU
/*4cb3f610*/  FFMA R44, R101, R117, R44                                  ; // x      4320  FMA
/*4cb3f620*/  FFMA R53, R102, R116, R53                                  ; // x      4320  FMA
/*4cb3f630*/  @P3   LDG.E.LTC128B.CONSTANT R72, [R78]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4cb3f640*/  FFMA R52, R102, R117, R52                                  ; // x      4320  FMA
/*4cb3f650*/  FFMA R61, R103, R116, R61                                  ; // x      4320  FMA
/*4cb3f660*/  FFMA R60, R103, R117, R60                                  ; // x      4320  FMA
/*4cb3f670*/  FFMA R63, R103, R118, R63                                  ; // x      4320  FMA
/*4cb3f680*/  @P1   IADD3 R78, P2, R78, c[0x0][0x180], RZ                ; // x      4320  ALU
/*4cb3f690*/  FFMA R62, R103, R119, R62                                  ; // x      4320  FMA
/*4cb3f6a0*/  FFMA R55, R102, R118, R55                                  ; // x      4320  FMA
/*4cb3f6b0*/  @P1   IADD3.X R79, R79, c[0x0][0x184], RZ, P2, !PT         ; // x      4320  ALU
/*4cb3f6c0*/  FFMA R54, R102, R119, R54                                  ; // x      4320  FMA
/*4cb3f6d0*/  FFMA R47, R101, R118, R47                                  ; // x      4320  FMA
/*4cb3f6e0*/  @P4   LDG.E.LTC128B.CONSTANT R73, [R78]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4cb3f6f0*/  FFMA R46, R101, R119, R46                                  ; // x      4320  FMA
/*4cb3f700*/  FFMA R39, R100, R118, R39                                  ; // x      4320  FMA
/*4cb3f710*/  FFMA R38, R100, R119, R38                                  ; // x      4320  FMA
/*4cb3f720*/  FFMA R31, R99, R118, R31                                   ; // x      4320  FMA
/*4cb3f730*/  @P1   IADD3 R78, P2, R78, c[0x0][0x180], RZ                ; // x      4320  ALU
/*4cb3f740*/  FFMA R30, R99, R119, R30                                   ; // x      4320  FMA
/*4cb3f750*/  FFMA R23, R98, R118, R23                                   ; // x      4320  FMA
/*4cb3f760*/  @P1   IADD3.X R79, R79, c[0x0][0x184], RZ, P2, !PT         ; // x      4320  ALU
/*4cb3f770*/  FFMA R22, R98, R119, R22                                   ; // x      4320  FMA
/*4cb3f780*/  FFMA R15, R97, R118, R15                                   ; // x      4320  FMA
/*4cb3f790*/  @P5   LDG.E.LTC128B.CONSTANT R74, [R78]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4cb3f7a0*/  FFMA R14, R97, R119, R14                                   ; // x      4320  FMA
/*4cb3f7b0*/  FFMA R7, R96, R118, R7                                     ; // x      4320  FMA
/*4cb3f7c0*/  FFMA R6, R96, R119, R6                                     ; // x      4320  FMA
/*4cb3f7d0*/  FFMA R1, R88, R104, R1                                     ; // x      4320  FMA
/*4cb3f7e0*/  LDS.128 R96, [R76+0x630]                                   ; // x      4320  shared-memory load
/*4cb3f7f0*/  FFMA R0, R88, R105, R0                                     ; // x      4320  FMA
/*4cb3f800*/  FFMA R9, R89, R104, R9                                     ; // x      4320  FMA
/*4cb3f810*/  @P1   IADD3 R78, P2, R78, c[0x0][0x180], RZ                ; // x      4320  ALU
/*4cb3f820*/  FFMA R8, R89, R105, R8                                     ; // x      4320  FMA
/*4cb3f830*/  FFMA R17, R90, R104, R17                                   ; // x      4320  FMA
/*4cb3f840*/  LDS.128 R112, [R77+0x330]                                  ; // x      4320  shared-memory load
```

(first 200 of 683 instructions shown; the rest are in the full dump)
