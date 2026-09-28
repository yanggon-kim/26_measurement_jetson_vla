# Hot loop — `ampere_sgemm_128x64_tn` (sm87)

PC range 0x21facd0-0x21fd770; backward BRA 0x21fd770 -> 0x21facd0: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_sgemm_128x64_tn_sm87.sass`; per-PC counts: `raw/ampere_sgemm_128x64_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*021facd0*/  STS [R80], R64                                             ; // x       128  shared-memory store
/*021face0*/  STS [R80+0x40], R65                                        ; // x       128  shared-memory store
/*021facf0*/  STS [R80+0x80], R66                                        ; // x       128  shared-memory store
/*021fad00*/  STS [R80+0xc0], R67                                        ; // x       128  shared-memory store
/*021fad10*/  STS [R80+0x100], R68                                       ; // x       128  shared-memory store
/*021fad20*/  STS [R80+0x140], R69                                       ; // x       128  shared-memory store
/*021fad30*/  STS [R80+0x180], R70                                       ; // x       128  shared-memory store
/*021fad40*/  STS [R80+0x1c0], R71                                       ; // x       128  shared-memory store
/*021fad50*/  STS [R81], R72                                             ; // x       128  shared-memory store
/*021fad60*/  STS [R81+0x40], R73                                        ; // x       128  shared-memory store
/*021fad70*/  STS [R81+0x80], R74                                        ; // x       128  shared-memory store
/*021fad80*/  STS [R81+0xc0], R75                                        ; // x       128  shared-memory store
/*021fad90*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       128  block barrier
/*021fada0*/  IADD3 R80, R80, R87, RZ                                    ; // x       128  ALU
/*021fadb0*/  IADD3 R81, R81, R87, RZ                                    ; // x       128  ALU
/*021fadc0*/  LDS.128 R88, [R76]                                         ; // x       128  shared-memory load
/*021fadd0*/  LDS.128 R104, [R77]                                        ; // x       128  shared-memory load
/*021fade0*/  LDS.128 R92, [R76+0x80]                                    ; // x       128  shared-memory load
/*021fadf0*/  LDS.128 R108, [R77+0x40]                                   ; // x       128  shared-memory load
/*021fae00*/  FFMA R1, R88, R104, R1                                     ; // x      8192  FMA
/*021fae10*/  ISETP.GE.AND P1, PT, R85, 0x10, PT                         ; // x      8192  ALU
/*021fae20*/  FFMA R0, R88, R105, R0                                     ; // x      8192  FMA
/*021fae30*/  FFMA R9, R89, R104, R9                                     ; // x      8192  FMA
/*021fae40*/  IADD3 R85, R85, -0x8, RZ                                   ; // x      8192  ALU
/*021fae50*/  FFMA R8, R89, R105, R8                                     ; // x      8192  FMA
/*021fae60*/  FFMA R17, R90, R104, R17                                   ; // x      8192  FMA
/*021fae70*/  LDS.128 R96, [R76+0x210]                                   ; // x      8192  shared-memory load
/*021fae80*/  FFMA R16, R90, R105, R16                                   ; // x      8192  FMA
/*021fae90*/  FFMA R25, R91, R104, R25                                   ; // x      8192  FMA
/*021faea0*/  LDS.128 R112, [R77+0x110]                                  ; // x      8192  shared-memory load
/*021faeb0*/  FFMA R24, R91, R105, R24                                   ; // x      8192  FMA
/*021faec0*/  FFMA R33, R92, R104, R33                                   ; // x      8192  FMA
/*021faed0*/  LDS.128 R100, [R76+0x290]                                  ; // x      8192  shared-memory load
/*021faee0*/  FFMA R32, R92, R105, R32                                   ; // x      8192  FMA
/*021faef0*/  FFMA R41, R93, R104, R41                                   ; // x      8192  FMA
/*021faf00*/  @!P1  R2P PR, RZ, 0x78                                     ; // x      8192  ALU
/*021faf10*/  FFMA R40, R93, R105, R40                                   ; // x      8192  FMA
/*021faf20*/  FFMA R49, R94, R104, R49                                   ; // x      8192  FMA
/*021faf30*/  @P1   R2P PR, R86, 0x78                                    ; // x      8192  ALU
/*021faf40*/  FFMA R48, R94, R105, R48                                   ; // x      8192  FMA
/*021faf50*/  FFMA R57, R95, R104, R57                                   ; // x      8192  FMA
/*021faf60*/  LDS.128 R116, [R77+0x150]                                  ; // x      8192  shared-memory load
/*021faf70*/  FFMA R56, R95, R105, R56                                   ; // x      8192  FMA
/*021faf80*/  FFMA R59, R95, R106, R59                                   ; // x      8192  FMA
/*021faf90*/  FFMA R58, R95, R107, R58                                   ; // x      8192  FMA
/*021fafa0*/  FFMA R51, R94, R106, R51                                   ; // x      8192  FMA
/*021fafb0*/  FFMA R50, R94, R107, R50                                   ; // x      8192  FMA
/*021fafc0*/  FFMA R43, R93, R106, R43                                   ; // x      8192  FMA
/*021fafd0*/  FFMA R42, R93, R107, R42                                   ; // x      8192  FMA
/*021fafe0*/  FFMA R35, R92, R106, R35                                   ; // x      8192  FMA
/*021faff0*/  FFMA R34, R92, R107, R34                                   ; // x      8192  FMA
/*021fb000*/  FFMA R27, R91, R106, R27                                   ; // x      8192  FMA
/*021fb010*/  FFMA R26, R91, R107, R26                                   ; // x      8192  FMA
/*021fb020*/  FFMA R19, R90, R106, R19                                   ; // x      8192  FMA
/*021fb030*/  FFMA R18, R90, R107, R18                                   ; // x      8192  FMA
/*021fb040*/  FFMA R11, R89, R106, R11                                   ; // x      8192  FMA
/*021fb050*/  FFMA R10, R89, R107, R10                                   ; // x      8192  FMA
/*021fb060*/  FFMA R3, R88, R106, R3                                     ; // x      8192  FMA
/*021fb070*/  FFMA R2, R88, R107, R2                                     ; // x      8192  FMA
/*021fb080*/  FFMA R5, R88, R108, R5                                     ; // x      8192  FMA
/*021fb090*/  @P3   LDG.E.LTC128B.CONSTANT R64, [R82]                    ; // x      8192  global load (DRAM/L2: weights or activations)
/*021fb0a0*/  FFMA R4, R88, R109, R4                                     ; // x      8192  FMA
/*021fb0b0*/  FFMA R13, R89, R108, R13                                   ; // x      8192  FMA
/*021fb0c0*/  FFMA R12, R89, R109, R12                                   ; // x      8192  FMA
/*021fb0d0*/  FFMA R21, R90, R108, R21                                   ; // x      8192  FMA
/*021fb0e0*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      8192  ALU
/*021fb0f0*/  FFMA R20, R90, R109, R20                                   ; // x      8192  FMA
/*021fb100*/  FFMA R29, R91, R108, R29                                   ; // x      8192  FMA
/*021fb110*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      8192  ALU
/*021fb120*/  FFMA R28, R91, R109, R28                                   ; // x      8192  FMA
/*021fb130*/  FFMA R37, R92, R108, R37                                   ; // x      8192  FMA
/*021fb140*/  @P4   LDG.E.LTC128B.CONSTANT R65, [R82]                    ; // x      8192  global load (DRAM/L2: weights or activations)
/*021fb150*/  FFMA R36, R92, R109, R36                                   ; // x      8192  FMA
/*021fb160*/  FFMA R45, R93, R108, R45                                   ; // x      8192  FMA
/*021fb170*/  FFMA R44, R93, R109, R44                                   ; // x      8192  FMA
/*021fb180*/  FFMA R53, R94, R108, R53                                   ; // x      8192  FMA
/*021fb190*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      8192  ALU
/*021fb1a0*/  FFMA R52, R94, R109, R52                                   ; // x      8192  FMA
/*021fb1b0*/  FFMA R61, R95, R108, R61                                   ; // x      8192  FMA
/*021fb1c0*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      8192  ALU
/*021fb1d0*/  FFMA R60, R95, R109, R60                                   ; // x      8192  FMA
/*021fb1e0*/  FFMA R63, R95, R110, R63                                   ; // x      8192  FMA
/*021fb1f0*/  @P5   LDG.E.LTC128B.CONSTANT R66, [R82]                    ; // x      8192  global load (DRAM/L2: weights or activations)
/*021fb200*/  FFMA R62, R95, R111, R62                                   ; // x      8192  FMA
/*021fb210*/  FFMA R55, R94, R110, R55                                   ; // x      8192  FMA
/*021fb220*/  FFMA R54, R94, R111, R54                                   ; // x      8192  FMA
/*021fb230*/  FFMA R47, R93, R110, R47                                   ; // x      8192  FMA
/*021fb240*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      8192  ALU
/*021fb250*/  FFMA R46, R93, R111, R46                                   ; // x      8192  FMA
/*021fb260*/  FFMA R39, R92, R110, R39                                   ; // x      8192  FMA
/*021fb270*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      8192  ALU
/*021fb280*/  FFMA R38, R92, R111, R38                                   ; // x      8192  FMA
/*021fb290*/  FFMA R31, R91, R110, R31                                   ; // x      8192  FMA
/*021fb2a0*/  @P6   LDG.E.LTC128B.CONSTANT R67, [R82]                    ; // x      8192  global load (DRAM/L2: weights or activations)
/*021fb2b0*/  FFMA R30, R91, R111, R30                                   ; // x      8192  FMA
/*021fb2c0*/  FFMA R23, R90, R110, R23                                   ; // x      8192  FMA
/*021fb2d0*/  FFMA R22, R90, R111, R22                                   ; // x      8192  FMA
/*021fb2e0*/  FFMA R15, R89, R110, R15                                   ; // x      8192  FMA
/*021fb2f0*/  @P1   R2P PR, R86.B1, 0x78                                 ; // x      8192  ALU
/*021fb300*/  FFMA R14, R89, R111, R14                                   ; // x      8192  FMA
/*021fb310*/  FFMA R7, R88, R110, R7                                     ; // x      8192  FMA
/*021fb320*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      8192  ALU
/*021fb330*/  FFMA R6, R88, R111, R6                                     ; // x      8192  FMA
/*021fb340*/  FFMA R1, R96, R112, R1                                     ; // x      8192  FMA
/*021fb350*/  LDS.128 R88, [R76+0x420]                                   ; // x      8192  shared-memory load
/*021fb360*/  FFMA R0, R96, R113, R0                                     ; // x      8192  FMA
/*021fb370*/  FFMA R9, R97, R112, R9                                     ; // x      8192  FMA
/*021fb380*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      8192  ALU
/*021fb390*/  FFMA R8, R97, R113, R8                                     ; // x      8192  FMA
/*021fb3a0*/  FFMA R17, R98, R112, R17                                   ; // x      8192  FMA
/*021fb3b0*/  LDS.128 R104, [R77+0x220]                                  ; // x      8192  shared-memory load
/*021fb3c0*/  FFMA R16, R98, R113, R16                                   ; // x      8192  FMA
/*021fb3d0*/  FFMA R25, R99, R112, R25                                   ; // x      8192  FMA
/*021fb3e0*/  LDS.128 R92, [R76+0x4a0]                                   ; // x      8192  shared-memory load
/*021fb3f0*/  FFMA R24, R99, R113, R24                                   ; // x      8192  FMA
/*021fb400*/  FFMA R33, R100, R112, R33                                  ; // x      8192  FMA
/*021fb410*/  LDS.128 R108, [R77+0x260]                                  ; // x      8192  shared-memory load
/*021fb420*/  FFMA R32, R100, R113, R32                                  ; // x      8192  FMA
/*021fb430*/  FFMA R41, R101, R112, R41                                  ; // x      8192  FMA
/*021fb440*/  @P3   LDG.E.LTC128B.CONSTANT R68, [R82]                    ; // x      8192  global load (DRAM/L2: weights or activations)
/*021fb450*/  FFMA R40, R101, R113, R40                                  ; // x      8192  FMA
/*021fb460*/  FFMA R49, R102, R112, R49                                  ; // x      8192  FMA
/*021fb470*/  FFMA R48, R102, R113, R48                                  ; // x      8192  FMA
/*021fb480*/  FFMA R57, R103, R112, R57                                  ; // x      8192  FMA
/*021fb490*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      8192  ALU
/*021fb4a0*/  FFMA R56, R103, R113, R56                                  ; // x      8192  FMA
/*021fb4b0*/  FFMA R59, R103, R114, R59                                  ; // x      8192  FMA
/*021fb4c0*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      8192  ALU
/*021fb4d0*/  FFMA R58, R103, R115, R58                                  ; // x      8192  FMA
/*021fb4e0*/  FFMA R51, R102, R114, R51                                  ; // x      8192  FMA
/*021fb4f0*/  @P4   LDG.E.LTC128B.CONSTANT R69, [R82]                    ; // x      8192  global load (DRAM/L2: weights or activations)
/*021fb500*/  FFMA R50, R102, R115, R50                                  ; // x      8192  FMA
/*021fb510*/  FFMA R43, R101, R114, R43                                  ; // x      8192  FMA
/*021fb520*/  FFMA R42, R101, R115, R42                                  ; // x      8192  FMA
/*021fb530*/  FFMA R35, R100, R114, R35                                  ; // x      8192  FMA
/*021fb540*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      8192  ALU
/*021fb550*/  FFMA R34, R100, R115, R34                                  ; // x      8192  FMA
/*021fb560*/  FFMA R27, R99, R114, R27                                   ; // x      8192  FMA
/*021fb570*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      8192  ALU
/*021fb580*/  FFMA R26, R99, R115, R26                                   ; // x      8192  FMA
/*021fb590*/  FFMA R19, R98, R114, R19                                   ; // x      8192  FMA
/*021fb5a0*/  @P5   LDG.E.LTC128B.CONSTANT R70, [R82]                    ; // x      8192  global load (DRAM/L2: weights or activations)
/*021fb5b0*/  FFMA R18, R98, R115, R18                                   ; // x      8192  FMA
/*021fb5c0*/  FFMA R11, R97, R114, R11                                   ; // x      8192  FMA
/*021fb5d0*/  FFMA R10, R97, R115, R10                                   ; // x      8192  FMA
/*021fb5e0*/  FFMA R3, R96, R114, R3                                     ; // x      8192  FMA
/*021fb5f0*/  @P1   IADD3 R82, P2, R82, c[0x0][0x178], RZ                ; // x      8192  ALU
/*021fb600*/  FFMA R2, R96, R115, R2                                     ; // x      8192  FMA
/*021fb610*/  FFMA R5, R96, R116, R5                                     ; // x      8192  FMA
/*021fb620*/  @P1   IADD3.X R83, R83, c[0x0][0x17c], RZ, P2, !PT         ; // x      8192  ALU
/*021fb630*/  FFMA R4, R96, R117, R4                                     ; // x      8192  FMA
/*021fb640*/  FFMA R13, R97, R116, R13                                   ; // x      8192  FMA
/*021fb650*/  @P6   LDG.E.LTC128B.CONSTANT R71, [R82]                    ; // x      8192  global load (DRAM/L2: weights or activations)
/*021fb660*/  FFMA R12, R97, R117, R12                                   ; // x      8192  FMA
/*021fb670*/  FFMA R21, R98, R116, R21                                   ; // x      8192  FMA
/*021fb680*/  FFMA R20, R98, R117, R20                                   ; // x      8192  FMA
/*021fb690*/  FFMA R29, R99, R116, R29                                   ; // x      8192  FMA
/*021fb6a0*/  @P1   IADD3 R82, P2, R82, c[0x0][0x188], RZ                ; // x      8192  ALU
/*021fb6b0*/  FFMA R28, R99, R117, R28                                   ; // x      8192  FMA
/*021fb6c0*/  FFMA R37, R100, R116, R37                                  ; // x      8192  FMA
/*021fb6d0*/  @P1   IADD3.X R83, R83, c[0x0][0x18c], RZ, P2, !PT         ; // x      8192  ALU
/*021fb6e0*/  FFMA R36, R100, R117, R36                                  ; // x      8192  FMA
/*021fb6f0*/  FFMA R45, R101, R116, R45                                  ; // x      8192  FMA
/*021fb700*/  @P1   R2P PR, R86.B2, 0x78                                 ; // x      8192  ALU
/*021fb710*/  FFMA R44, R101, R117, R44                                  ; // x      8192  FMA
/*021fb720*/  FFMA R53, R102, R116, R53                                  ; // x      8192  FMA
/*021fb730*/  @P3   LDG.E.LTC128B.CONSTANT R72, [R78]                    ; // x      8192  global load (DRAM/L2: weights or activations)
/*021fb740*/  FFMA R52, R102, R117, R52                                  ; // x      8192  FMA
/*021fb750*/  FFMA R61, R103, R116, R61                                  ; // x      8192  FMA
/*021fb760*/  FFMA R60, R103, R117, R60                                  ; // x      8192  FMA
/*021fb770*/  FFMA R63, R103, R118, R63                                  ; // x      8192  FMA
/*021fb780*/  @P1   IADD3 R78, P2, R78, c[0x0][0x180], RZ                ; // x      8192  ALU
/*021fb790*/  FFMA R62, R103, R119, R62                                  ; // x      8192  FMA
/*021fb7a0*/  FFMA R55, R102, R118, R55                                  ; // x      8192  FMA
/*021fb7b0*/  @P1   IADD3.X R79, R79, c[0x0][0x184], RZ, P2, !PT         ; // x      8192  ALU
/*021fb7c0*/  FFMA R54, R102, R119, R54                                  ; // x      8192  FMA
/*021fb7d0*/  FFMA R47, R101, R118, R47                                  ; // x      8192  FMA
/*021fb7e0*/  @P4   LDG.E.LTC128B.CONSTANT R73, [R78]                    ; // x      8192  global load (DRAM/L2: weights or activations)
/*021fb7f0*/  FFMA R46, R101, R119, R46                                  ; // x      8192  FMA
/*021fb800*/  FFMA R39, R100, R118, R39                                  ; // x      8192  FMA
/*021fb810*/  FFMA R38, R100, R119, R38                                  ; // x      8192  FMA
/*021fb820*/  FFMA R31, R99, R118, R31                                   ; // x      8192  FMA
/*021fb830*/  @P1   IADD3 R78, P2, R78, c[0x0][0x180], RZ                ; // x      8192  ALU
/*021fb840*/  FFMA R30, R99, R119, R30                                   ; // x      8192  FMA
/*021fb850*/  FFMA R23, R98, R118, R23                                   ; // x      8192  FMA
/*021fb860*/  @P1   IADD3.X R79, R79, c[0x0][0x184], RZ, P2, !PT         ; // x      8192  ALU
/*021fb870*/  FFMA R22, R98, R119, R22                                   ; // x      8192  FMA
/*021fb880*/  FFMA R15, R97, R118, R15                                   ; // x      8192  FMA
/*021fb890*/  @P5   LDG.E.LTC128B.CONSTANT R74, [R78]                    ; // x      8192  global load (DRAM/L2: weights or activations)
/*021fb8a0*/  FFMA R14, R97, R119, R14                                   ; // x      8192  FMA
/*021fb8b0*/  FFMA R7, R96, R118, R7                                     ; // x      8192  FMA
/*021fb8c0*/  FFMA R6, R96, R119, R6                                     ; // x      8192  FMA
/*021fb8d0*/  FFMA R1, R88, R104, R1                                     ; // x      8192  FMA
/*021fb8e0*/  LDS.128 R96, [R76+0x630]                                   ; // x      8192  shared-memory load
/*021fb8f0*/  FFMA R0, R88, R105, R0                                     ; // x      8192  FMA
/*021fb900*/  FFMA R9, R89, R104, R9                                     ; // x      8192  FMA
/*021fb910*/  @P1   IADD3 R78, P2, R78, c[0x0][0x180], RZ                ; // x      8192  ALU
/*021fb920*/  FFMA R8, R89, R105, R8                                     ; // x      8192  FMA
/*021fb930*/  FFMA R17, R90, R104, R17                                   ; // x      8192  FMA
/*021fb940*/  LDS.128 R112, [R77+0x330]                                  ; // x      8192  shared-memory load
```

(first 200 of 683 instructions shown; the rest are in the full dump)
