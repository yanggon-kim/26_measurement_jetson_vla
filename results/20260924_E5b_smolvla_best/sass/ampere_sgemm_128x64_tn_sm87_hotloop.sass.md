# Hot loop — `ampere_sgemm_128x64_tn` (sm87)

PC range 0x4f68b2d0-0x4f68dd70; backward BRA 0x4f68dd70 -> 0x4f68b2d0: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_sgemm_128x64_tn_sm87.sass`; per-PC counts: `raw/ampere_sgemm_128x64_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*4f68b2d0*/  STS [R80], R64                                             ; // x        96  shared-memory store
/*4f68b2e0*/  STS [R80+0x40], R65                                        ; // x        96  shared-memory store
/*4f68b2f0*/  STS [R80+0x80], R66                                        ; // x        96  shared-memory store
/*4f68b300*/  STS [R80+0xc0], R67                                        ; // x        96  shared-memory store
/*4f68b310*/  STS [R80+0x100], R68                                       ; // x        96  shared-memory store
/*4f68b320*/  STS [R80+0x140], R69                                       ; // x        96  shared-memory store
/*4f68b330*/  STS [R80+0x180], R70                                       ; // x        96  shared-memory store
/*4f68b340*/  STS [R80+0x1c0], R71                                       ; // x        96  shared-memory store
/*4f68b350*/  STS [R81], R72                                             ; // x        96  shared-memory store
/*4f68b360*/  STS [R81+0x40], R73                                        ; // x        96  shared-memory store
/*4f68b370*/  STS [R81+0x80], R74                                        ; // x        96  shared-memory store
/*4f68b380*/  STS [R81+0xc0], R75                                        ; // x        96  shared-memory store
/*4f68b390*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x        96  block barrier
/*4f68b3a0*/  IADD3 R80, R80, R87, RZ                                    ; // x        96  ALU
/*4f68b3b0*/  IADD3 R81, R81, R87, RZ                                    ; // x        96  ALU
/*4f68b3c0*/  LDS.128 R88, [R76]                                         ; // x        96  shared-memory load
/*4f68b3d0*/  LDS.128 R104, [R77]                                        ; // x        96  shared-memory load
/*4f68b3e0*/  LDS.128 R92, [R76+0x80]                                    ; // x        96  shared-memory load
/*4f68b3f0*/  LDS.128 R108, [R77+0x40]                                   ; // x        96  shared-memory load
/*4f68b400*/  FFMA R1, R88, R104, R1                                     ; // x      4320  FMA
/*4f68b410*/  ISETP.GE.AND P1, PT, R85, 0x10, PT                         ; // x      4320  ALU
/*4f68b420*/  FFMA R0, R88, R105, R0                                     ; // x      4320  FMA
/*4f68b430*/  FFMA R9, R89, R104, R9                                     ; // x      4320  FMA
/*4f68b440*/  IADD3 R85, R85, -0x8, RZ                                   ; // x      4320  ALU
/*4f68b450*/  FFMA R8, R89, R105, R8                                     ; // x      4320  FMA
/*4f68b460*/  FFMA R17, R90, R104, R17                                   ; // x      4320  FMA
/*4f68b470*/  LDS.128 R96, [R76+0x210]                                   ; // x      4320  shared-memory load
/*4f68b480*/  FFMA R16, R90, R105, R16                                   ; // x      4320  FMA
/*4f68b490*/  FFMA R25, R91, R104, R25                                   ; // x      4320  FMA
/*4f68b4a0*/  LDS.128 R112, [R77+0x110]                                  ; // x      4320  shared-memory load
/*4f68b4b0*/  FFMA R24, R91, R105, R24                                   ; // x      4320  FMA
/*4f68b4c0*/  FFMA R33, R92, R104, R33                                   ; // x      4320  FMA
/*4f68b4d0*/  LDS.128 R100, [R76+0x290]                                  ; // x      4320  shared-memory load
/*4f68b4e0*/  FFMA R32, R92, R105, R32                                   ; // x      4320  FMA
/*4f68b4f0*/  FFMA R41, R93, R104, R41                                   ; // x      4320  FMA
/*4f68b500*/  @!P1  R2P PR, RZ, 0x78                                     ; // x      4320  ALU
/*4f68b510*/  FFMA R40, R93, R105, R40                                   ; // x      4320  FMA
/*4f68b520*/  FFMA R49, R94, R104, R49                                   ; // x      4320  FMA
/*4f68b530*/  @P1   R2P PR, R86, 0x78                                    ; // x      4320  ALU
/*4f68b540*/  FFMA R48, R94, R105, R48                                   ; // x      4320  FMA
/*4f68b550*/  FFMA R57, R95, R104, R57                                   ; // x      4320  FMA
/*4f68b560*/  LDS.128 R116, [R77+0x150]                                  ; // x      4320  shared-memory load
/*4f68b570*/  FFMA R56, R95, R105, R56                                   ; // x      4320  FMA
/*4f68b580*/  FFMA R59, R95, R106, R59                                   ; // x      4320  FMA
/*4f68b590*/  FFMA R58, R95, R107, R58                                   ; // x      4320  FMA
/*4f68b5a0*/  FFMA R51, R94, R106, R51                                   ; // x      4320  FMA
/*4f68b5b0*/  FFMA R50, R94, R107, R50                                   ; // x      4320  FMA
/*4f68b5c0*/  FFMA R43, R93, R106, R43                                   ; // x      4320  FMA
/*4f68b5d0*/  FFMA R42, R93, R107, R42                                   ; // x      4320  FMA
/*4f68b5e0*/  FFMA R35, R92, R106, R35                                   ; // x      4320  FMA
/*4f68b5f0*/  FFMA R34, R92, R107, R34                                   ; // x      4320  FMA
/*4f68b600*/  FFMA R27, R91, R106, R27                                   ; // x      4320  FMA
/*4f68b610*/  FFMA R26, R91, R107, R26                                   ; // x      4320  FMA
/*4f68b620*/  FFMA R19, R90, R106, R19                                   ; // x      4320  FMA
/*4f68b630*/  FFMA R18, R90, R107, R18                                   ; // x      4320  FMA
/*4f68b640*/  FFMA R11, R89, R106, R11                                   ; // x      4320  FMA
/*4f68b650*/  FFMA R10, R89, R107, R10                                   ; // x      4320  FMA
/*4f68b660*/  FFMA R3, R88, R106, R3                                     ; // x      4320  FMA
/*4f68b670*/  FFMA R2, R88, R107, R2                                     ; // x      4320  FMA
/*4f68b680*/  FFMA R5, R88, R108, R5                                     ; // x      4320  FMA
/*4f68b690*/  @P3   LDG.E.LTC128B.CONSTANT R64, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4f68b6a0*/  FFMA R4, R88, R109, R4                                     ; // x      4320  FMA
/*4f68b6b0*/  FFMA R13, R89, R108, R13                                   ; // x      4320  FMA
/*4f68b6c0*/  FFMA R12, R89, R109, R12                                   ; // x      4320  FMA
/*4f68b6d0*/  FFMA R21, R90, R108, R21                                   ; // x      4320  FMA
/*4f68b6e0*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4f68b6f0*/  FFMA R20, R90, R109, R20                                   ; // x      4320  FMA
/*4f68b700*/  FFMA R29, R91, R108, R29                                   ; // x      4320  FMA
/*4f68b710*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4f68b720*/  FFMA R28, R91, R109, R28                                   ; // x      4320  FMA
/*4f68b730*/  FFMA R37, R92, R108, R37                                   ; // x      4320  FMA
/*4f68b740*/  @P4   LDG.E.LTC128B.CONSTANT R65, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4f68b750*/  FFMA R36, R92, R109, R36                                   ; // x      4320  FMA
/*4f68b760*/  FFMA R45, R93, R108, R45                                   ; // x      4320  FMA
/*4f68b770*/  FFMA R44, R93, R109, R44                                   ; // x      4320  FMA
/*4f68b780*/  FFMA R53, R94, R108, R53                                   ; // x      4320  FMA
/*4f68b790*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4f68b7a0*/  FFMA R52, R94, R109, R52                                   ; // x      4320  FMA
/*4f68b7b0*/  FFMA R61, R95, R108, R61                                   ; // x      4320  FMA
/*4f68b7c0*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4f68b7d0*/  FFMA R60, R95, R109, R60                                   ; // x      4320  FMA
/*4f68b7e0*/  FFMA R63, R95, R110, R63                                   ; // x      4320  FMA
/*4f68b7f0*/  @P5   LDG.E.LTC128B.CONSTANT R66, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4f68b800*/  FFMA R62, R95, R111, R62                                   ; // x      4320  FMA
/*4f68b810*/  FFMA R55, R94, R110, R55                                   ; // x      4320  FMA
/*4f68b820*/  FFMA R54, R94, R111, R54                                   ; // x      4320  FMA
/*4f68b830*/  FFMA R47, R93, R110, R47                                   ; // x      4320  FMA
/*4f68b840*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4f68b850*/  FFMA R46, R93, R111, R46                                   ; // x      4320  FMA
/*4f68b860*/  FFMA R39, R92, R110, R39                                   ; // x      4320  FMA
/*4f68b870*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4f68b880*/  FFMA R38, R92, R111, R38                                   ; // x      4320  FMA
/*4f68b890*/  FFMA R31, R91, R110, R31                                   ; // x      4320  FMA
/*4f68b8a0*/  @P6   LDG.E.LTC128B.CONSTANT R67, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4f68b8b0*/  FFMA R30, R91, R111, R30                                   ; // x      4320  FMA
/*4f68b8c0*/  FFMA R23, R90, R110, R23                                   ; // x      4320  FMA
/*4f68b8d0*/  FFMA R22, R90, R111, R22                                   ; // x      4320  FMA
/*4f68b8e0*/  FFMA R15, R89, R110, R15                                   ; // x      4320  FMA
/*4f68b8f0*/  @P1   R2P PR, R86.B1, 0x78                                 ; // x      4320  ALU
/*4f68b900*/  FFMA R14, R89, R111, R14                                   ; // x      4320  FMA
/*4f68b910*/  FFMA R7, R88, R110, R7                                     ; // x      4320  FMA
/*4f68b920*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4f68b930*/  FFMA R6, R88, R111, R6                                     ; // x      4320  FMA
/*4f68b940*/  FFMA R1, R96, R112, R1                                     ; // x      4320  FMA
/*4f68b950*/  LDS.128 R88, [R76+0x420]                                   ; // x      4320  shared-memory load
/*4f68b960*/  FFMA R0, R96, R113, R0                                     ; // x      4320  FMA
/*4f68b970*/  FFMA R9, R97, R112, R9                                     ; // x      4320  FMA
/*4f68b980*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4f68b990*/  FFMA R8, R97, R113, R8                                     ; // x      4320  FMA
/*4f68b9a0*/  FFMA R17, R98, R112, R17                                   ; // x      4320  FMA
/*4f68b9b0*/  LDS.128 R104, [R77+0x220]                                  ; // x      4320  shared-memory load
/*4f68b9c0*/  FFMA R16, R98, R113, R16                                   ; // x      4320  FMA
/*4f68b9d0*/  FFMA R25, R99, R112, R25                                   ; // x      4320  FMA
/*4f68b9e0*/  LDS.128 R92, [R76+0x4a0]                                   ; // x      4320  shared-memory load
/*4f68b9f0*/  FFMA R24, R99, R113, R24                                   ; // x      4320  FMA
/*4f68ba00*/  FFMA R33, R100, R112, R33                                  ; // x      4320  FMA
/*4f68ba10*/  LDS.128 R108, [R77+0x260]                                  ; // x      4320  shared-memory load
/*4f68ba20*/  FFMA R32, R100, R113, R32                                  ; // x      4320  FMA
/*4f68ba30*/  FFMA R41, R101, R112, R41                                  ; // x      4320  FMA
/*4f68ba40*/  @P3   LDG.E.LTC128B.CONSTANT R68, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4f68ba50*/  FFMA R40, R101, R113, R40                                  ; // x      4320  FMA
/*4f68ba60*/  FFMA R49, R102, R112, R49                                  ; // x      4320  FMA
/*4f68ba70*/  FFMA R48, R102, R113, R48                                  ; // x      4320  FMA
/*4f68ba80*/  FFMA R57, R103, R112, R57                                  ; // x      4320  FMA
/*4f68ba90*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4f68baa0*/  FFMA R56, R103, R113, R56                                  ; // x      4320  FMA
/*4f68bab0*/  FFMA R59, R103, R114, R59                                  ; // x      4320  FMA
/*4f68bac0*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4f68bad0*/  FFMA R58, R103, R115, R58                                  ; // x      4320  FMA
/*4f68bae0*/  FFMA R51, R102, R114, R51                                  ; // x      4320  FMA
/*4f68baf0*/  @P4   LDG.E.LTC128B.CONSTANT R69, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4f68bb00*/  FFMA R50, R102, R115, R50                                  ; // x      4320  FMA
/*4f68bb10*/  FFMA R43, R101, R114, R43                                  ; // x      4320  FMA
/*4f68bb20*/  FFMA R42, R101, R115, R42                                  ; // x      4320  FMA
/*4f68bb30*/  FFMA R35, R100, R114, R35                                  ; // x      4320  FMA
/*4f68bb40*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4f68bb50*/  FFMA R34, R100, R115, R34                                  ; // x      4320  FMA
/*4f68bb60*/  FFMA R27, R99, R114, R27                                   ; // x      4320  FMA
/*4f68bb70*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4f68bb80*/  FFMA R26, R99, R115, R26                                   ; // x      4320  FMA
/*4f68bb90*/  FFMA R19, R98, R114, R19                                   ; // x      4320  FMA
/*4f68bba0*/  @P5   LDG.E.LTC128B.CONSTANT R70, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4f68bbb0*/  FFMA R18, R98, R115, R18                                   ; // x      4320  FMA
/*4f68bbc0*/  FFMA R11, R97, R114, R11                                   ; // x      4320  FMA
/*4f68bbd0*/  FFMA R10, R97, R115, R10                                   ; // x      4320  FMA
/*4f68bbe0*/  FFMA R3, R96, R114, R3                                     ; // x      4320  FMA
/*4f68bbf0*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      4320  ALU
/*4f68bc00*/  FFMA R2, R96, R115, R2                                     ; // x      4320  FMA
/*4f68bc10*/  FFMA R5, R96, R116, R5                                     ; // x      4320  FMA
/*4f68bc20*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      4320  ALU
/*4f68bc30*/  FFMA R4, R96, R117, R4                                     ; // x      4320  FMA
/*4f68bc40*/  FFMA R13, R97, R116, R13                                   ; // x      4320  FMA
/*4f68bc50*/  @P6   LDG.E.LTC128B.CONSTANT R71, [R82]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4f68bc60*/  FFMA R12, R97, R117, R12                                   ; // x      4320  FMA
/*4f68bc70*/  FFMA R21, R98, R116, R21                                   ; // x      4320  FMA
/*4f68bc80*/  FFMA R20, R98, R117, R20                                   ; // x      4320  FMA
/*4f68bc90*/  FFMA R29, R99, R116, R29                                   ; // x      4320  FMA
/*4f68bca0*/  @P1   IADD3 R82, P2, R82, c[0x0][0x188], RZ                ; // x      4320  ALU
/*4f68bcb0*/  FFMA R28, R99, R117, R28                                   ; // x      4320  FMA
/*4f68bcc0*/  FFMA R37, R100, R116, R37                                  ; // x      4320  FMA
/*4f68bcd0*/  @P1   IADD3.X R83, R83, c[0x0][0x18c], RZ, P2, !PT         ; // x      4320  ALU
/*4f68bce0*/  FFMA R36, R100, R117, R36                                  ; // x      4320  FMA
/*4f68bcf0*/  FFMA R45, R101, R116, R45                                  ; // x      4320  FMA
/*4f68bd00*/  @P1   R2P PR, R86.B2, 0x78                                 ; // x      4320  ALU
/*4f68bd10*/  FFMA R44, R101, R117, R44                                  ; // x      4320  FMA
/*4f68bd20*/  FFMA R53, R102, R116, R53                                  ; // x      4320  FMA
/*4f68bd30*/  @P3   LDG.E.LTC128B.CONSTANT R72, [R78]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4f68bd40*/  FFMA R52, R102, R117, R52                                  ; // x      4320  FMA
/*4f68bd50*/  FFMA R61, R103, R116, R61                                  ; // x      4320  FMA
/*4f68bd60*/  FFMA R60, R103, R117, R60                                  ; // x      4320  FMA
/*4f68bd70*/  FFMA R63, R103, R118, R63                                  ; // x      4320  FMA
/*4f68bd80*/  @P1   IADD3 R78, P2, R78, c[0x0][0x180], RZ                ; // x      4320  ALU
/*4f68bd90*/  FFMA R62, R103, R119, R62                                  ; // x      4320  FMA
/*4f68bda0*/  FFMA R55, R102, R118, R55                                  ; // x      4320  FMA
/*4f68bdb0*/  @P1   IADD3.X R79, R79, c[0x0][0x184], RZ, P2, !PT         ; // x      4320  ALU
/*4f68bdc0*/  FFMA R54, R102, R119, R54                                  ; // x      4320  FMA
/*4f68bdd0*/  FFMA R47, R101, R118, R47                                  ; // x      4320  FMA
/*4f68bde0*/  @P4   LDG.E.LTC128B.CONSTANT R73, [R78]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4f68bdf0*/  FFMA R46, R101, R119, R46                                  ; // x      4320  FMA
/*4f68be00*/  FFMA R39, R100, R118, R39                                  ; // x      4320  FMA
/*4f68be10*/  FFMA R38, R100, R119, R38                                  ; // x      4320  FMA
/*4f68be20*/  FFMA R31, R99, R118, R31                                   ; // x      4320  FMA
/*4f68be30*/  @P1   IADD3 R78, P2, R78, c[0x0][0x180], RZ                ; // x      4320  ALU
/*4f68be40*/  FFMA R30, R99, R119, R30                                   ; // x      4320  FMA
/*4f68be50*/  FFMA R23, R98, R118, R23                                   ; // x      4320  FMA
/*4f68be60*/  @P1   IADD3.X R79, R79, c[0x0][0x184], RZ, P2, !PT         ; // x      4320  ALU
/*4f68be70*/  FFMA R22, R98, R119, R22                                   ; // x      4320  FMA
/*4f68be80*/  FFMA R15, R97, R118, R15                                   ; // x      4320  FMA
/*4f68be90*/  @P5   LDG.E.LTC128B.CONSTANT R74, [R78]                    ; // x      4320  global load (DRAM/L2: weights or activations)
/*4f68bea0*/  FFMA R14, R97, R119, R14                                   ; // x      4320  FMA
/*4f68beb0*/  FFMA R7, R96, R118, R7                                     ; // x      4320  FMA
/*4f68bec0*/  FFMA R6, R96, R119, R6                                     ; // x      4320  FMA
/*4f68bed0*/  FFMA R1, R88, R104, R1                                     ; // x      4320  FMA
/*4f68bee0*/  LDS.128 R96, [R76+0x630]                                   ; // x      4320  shared-memory load
/*4f68bef0*/  FFMA R0, R88, R105, R0                                     ; // x      4320  FMA
/*4f68bf00*/  FFMA R9, R89, R104, R9                                     ; // x      4320  FMA
/*4f68bf10*/  @P1   IADD3 R78, P2, R78, c[0x0][0x180], RZ                ; // x      4320  ALU
/*4f68bf20*/  FFMA R8, R89, R105, R8                                     ; // x      4320  FMA
/*4f68bf30*/  FFMA R17, R90, R104, R17                                   ; // x      4320  FMA
/*4f68bf40*/  LDS.128 R112, [R77+0x330]                                  ; // x      4320  shared-memory load
```

(first 200 of 683 instructions shown; the rest are in the full dump)
