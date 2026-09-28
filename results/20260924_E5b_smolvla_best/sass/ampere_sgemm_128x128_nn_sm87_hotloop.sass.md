# Hot loop — `ampere_sgemm_128x128_nn` (sm87)

PC range 0x4f6cfe00-0x4f6d25a0; backward BRA 0x4f6d25a0 -> 0x4f6cfe00: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_sgemm_128x128_nn_sm87.sass`; per-PC counts: `raw/ampere_sgemm_128x128_nn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*4f6cfe00*/  STS [R76], R64                                             ; // x       240  shared-memory store
/*4f6cfe10*/  STS [R76+0x80], R65                                        ; // x       240  shared-memory store
/*4f6cfe20*/  STS [R76+0x100], R66                                       ; // x       240  shared-memory store
/*4f6cfe30*/  STS [R76+0x180], R67                                       ; // x       240  shared-memory store
/*4f6cfe40*/  STS [R77], R68                                             ; // x       240  shared-memory store
/*4f6cfe50*/  STS [R77+0x80], R69                                        ; // x       240  shared-memory store
/*4f6cfe60*/  STS [R77+0x100], R70                                       ; // x       240  shared-memory store
/*4f6cfe70*/  STS [R77+0x180], R71                                       ; // x       240  shared-memory store
/*4f6cfe80*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       240  block barrier
/*4f6cfe90*/  IADD3 R76, R76, R83, RZ                                    ; // x       240  ALU
/*4f6cfea0*/  IADD3 R77, R77, R83, RZ                                    ; // x       240  ALU
/*4f6cfeb0*/  LDS.128 R84, [R72]                                         ; // x       240  shared-memory load
/*4f6cfec0*/  LDS.128 R100, [R73]                                        ; // x       240  shared-memory load
/*4f6cfed0*/  LDS.128 R88, [R72+0x80]                                    ; // x       240  shared-memory load
/*4f6cfee0*/  LDS.128 R104, [R73+0x40]                                   ; // x       240  shared-memory load
/*4f6cfef0*/  FFMA R1, R84, R100, R1                                     ; // x      2760  FMA
/*4f6cff00*/  ISETP.GE.AND P1, PT, R81, 0x10, PT                         ; // x      2760  ALU
/*4f6cff10*/  FFMA R0, R84, R101, R0                                     ; // x      2760  FMA
/*4f6cff20*/  FFMA R9, R85, R100, R9                                     ; // x      2760  FMA
/*4f6cff30*/  IADD3 R81, R81, -0x8, RZ                                   ; // x      2760  ALU
/*4f6cff40*/  FFMA R8, R85, R101, R8                                     ; // x      2760  FMA
/*4f6cff50*/  FFMA R17, R86, R100, R17                                   ; // x      2760  FMA
/*4f6cff60*/  LDS.128 R92, [R72+0x200]                                   ; // x      2760  shared-memory load
/*4f6cff70*/  FFMA R16, R86, R101, R16                                   ; // x      2760  FMA
/*4f6cff80*/  FFMA R25, R87, R100, R25                                   ; // x      2760  FMA
/*4f6cff90*/  LDS.128 R108, [R73+0x210]                                  ; // x      2760  shared-memory load
/*4f6cffa0*/  FFMA R24, R87, R101, R24                                   ; // x      2760  FMA
/*4f6cffb0*/  FFMA R33, R88, R100, R33                                   ; // x      2760  FMA
/*4f6cffc0*/  LDS.128 R96, [R72+0x280]                                   ; // x      2760  shared-memory load
/*4f6cffd0*/  FFMA R32, R88, R101, R32                                   ; // x      2760  FMA
/*4f6cffe0*/  FFMA R41, R89, R100, R41                                   ; // x      2760  FMA
/*4f6cfff0*/  @!P1  R2P PR, RZ, 0x78                                     ; // x      2760  ALU
/*4f6d0000*/  FFMA R40, R89, R101, R40                                   ; // x      2760  FMA
/*4f6d0010*/  FFMA R49, R90, R100, R49                                   ; // x      2760  FMA
/*4f6d0020*/  @P1   R2P PR, R82, 0x78                                    ; // x      2760  ALU
/*4f6d0030*/  FFMA R48, R90, R101, R48                                   ; // x      2760  FMA
/*4f6d0040*/  FFMA R57, R91, R100, R57                                   ; // x      2760  FMA
/*4f6d0050*/  LDS.128 R112, [R73+0x250]                                  ; // x      2760  shared-memory load
/*4f6d0060*/  FFMA R56, R91, R101, R56                                   ; // x      2760  FMA
/*4f6d0070*/  FFMA R59, R91, R102, R59                                   ; // x      2760  FMA
/*4f6d0080*/  FFMA R58, R91, R103, R58                                   ; // x      2760  FMA
/*4f6d0090*/  FFMA R51, R90, R102, R51                                   ; // x      2760  FMA
/*4f6d00a0*/  FFMA R50, R90, R103, R50                                   ; // x      2760  FMA
/*4f6d00b0*/  FFMA R43, R89, R102, R43                                   ; // x      2760  FMA
/*4f6d00c0*/  FFMA R42, R89, R103, R42                                   ; // x      2760  FMA
/*4f6d00d0*/  FFMA R35, R88, R102, R35                                   ; // x      2760  FMA
/*4f6d00e0*/  FFMA R34, R88, R103, R34                                   ; // x      2760  FMA
/*4f6d00f0*/  FFMA R27, R87, R102, R27                                   ; // x      2760  FMA
/*4f6d0100*/  FFMA R26, R87, R103, R26                                   ; // x      2760  FMA
/*4f6d0110*/  FFMA R19, R86, R102, R19                                   ; // x      2760  FMA
/*4f6d0120*/  FFMA R18, R86, R103, R18                                   ; // x      2760  FMA
/*4f6d0130*/  FFMA R11, R85, R102, R11                                   ; // x      2760  FMA
/*4f6d0140*/  FFMA R10, R85, R103, R10                                   ; // x      2760  FMA
/*4f6d0150*/  FFMA R3, R84, R102, R3                                     ; // x      2760  FMA
/*4f6d0160*/  FFMA R2, R84, R103, R2                                     ; // x      2760  FMA
/*4f6d0170*/  FFMA R5, R84, R104, R5                                     ; // x      2760  FMA
/*4f6d0180*/  @P3   LDG.E.LTC128B.CONSTANT R64, [R78]                    ; // x      2760  global load (DRAM/L2: weights or activations)
/*4f6d0190*/  FFMA R4, R84, R105, R4                                     ; // x      2760  FMA
/*4f6d01a0*/  FFMA R13, R85, R104, R13                                   ; // x      2760  FMA
/*4f6d01b0*/  @P4   LDG.E.LTC128B.CONSTANT R65, [R78+0x80]               ; // x      2760  global load (DRAM/L2: weights or activations)
/*4f6d01c0*/  FFMA R12, R85, R105, R12                                   ; // x      2760  FMA
/*4f6d01d0*/  FFMA R21, R86, R104, R21                                   ; // x      2760  FMA
/*4f6d01e0*/  @P5   LDG.E.LTC128B.CONSTANT R66, [R78+0x100]              ; // x      2760  global load (DRAM/L2: weights or activations)
/*4f6d01f0*/  FFMA R20, R86, R105, R20                                   ; // x      2760  FMA
/*4f6d0200*/  FFMA R29, R87, R104, R29                                   ; // x      2760  FMA
/*4f6d0210*/  @P6   LDG.E.LTC128B.CONSTANT R67, [R78+0x180]              ; // x      2760  global load (DRAM/L2: weights or activations)
/*4f6d0220*/  FFMA R28, R87, R105, R28                                   ; // x      2760  FMA
/*4f6d0230*/  FFMA R37, R88, R104, R37                                   ; // x      2760  FMA
/*4f6d0240*/  FFMA R36, R88, R105, R36                                   ; // x      2760  FMA
/*4f6d0250*/  FFMA R45, R89, R104, R45                                   ; // x      2760  FMA
/*4f6d0260*/  FFMA R44, R89, R105, R44                                   ; // x      2760  FMA
/*4f6d0270*/  FFMA R53, R90, R104, R53                                   ; // x      2760  FMA
/*4f6d0280*/  FFMA R52, R90, R105, R52                                   ; // x      2760  FMA
/*4f6d0290*/  FFMA R61, R91, R104, R61                                   ; // x      2760  FMA
/*4f6d02a0*/  FFMA R60, R91, R105, R60                                   ; // x      2760  FMA
/*4f6d02b0*/  FFMA R63, R91, R106, R63                                   ; // x      2760  FMA
/*4f6d02c0*/  FFMA R62, R91, R107, R62                                   ; // x      2760  FMA
/*4f6d02d0*/  FFMA R55, R90, R106, R55                                   ; // x      2760  FMA
/*4f6d02e0*/  FFMA R54, R90, R107, R54                                   ; // x      2760  FMA
/*4f6d02f0*/  FFMA R47, R89, R106, R47                                   ; // x      2760  FMA
/*4f6d0300*/  FFMA R46, R89, R107, R46                                   ; // x      2760  FMA
/*4f6d0310*/  FFMA R39, R88, R106, R39                                   ; // x      2760  FMA
/*4f6d0320*/  FFMA R38, R88, R107, R38                                   ; // x      2760  FMA
/*4f6d0330*/  FFMA R31, R87, R106, R31                                   ; // x      2760  FMA
/*4f6d0340*/  FFMA R30, R87, R107, R30                                   ; // x      2760  FMA
/*4f6d0350*/  FFMA R23, R86, R106, R23                                   ; // x      2760  FMA
/*4f6d0360*/  FFMA R22, R86, R107, R22                                   ; // x      2760  FMA
/*4f6d0370*/  FFMA R15, R85, R106, R15                                   ; // x      2760  FMA
/*4f6d0380*/  FFMA R14, R85, R107, R14                                   ; // x      2760  FMA
/*4f6d0390*/  FFMA R7, R84, R106, R7                                     ; // x      2760  FMA
/*4f6d03a0*/  FFMA R6, R84, R107, R6                                     ; // x      2760  FMA
/*4f6d03b0*/  FFMA R1, R92, R108, R1                                     ; // x      2760  FMA
/*4f6d03c0*/  @P1   IADD3 R78, P2, R78, c[0x0][0x188], RZ                ; // x      2760  ALU
/*4f6d03d0*/  FFMA R0, R92, R109, R0                                     ; // x      2760  FMA
/*4f6d03e0*/  FFMA R9, R93, R108, R9                                     ; // x      2760  FMA
/*4f6d03f0*/  @P1   IADD3.X R79, R79, c[0x0][0x18c], RZ, P2, !PT         ; // x      2760  ALU
/*4f6d0400*/  FFMA R8, R93, R109, R8                                     ; // x      2760  FMA
/*4f6d0410*/  FFMA R17, R94, R108, R17                                   ; // x      2760  FMA
/*4f6d0420*/  @P1   R2P PR, R82.B1, 0x78                                 ; // x      2760  ALU
/*4f6d0430*/  FFMA R16, R94, R109, R16                                   ; // x      2760  FMA
/*4f6d0440*/  FFMA R25, R95, R108, R25                                   ; // x      2760  FMA
/*4f6d0450*/  LDS.128 R84, [R72+0x400]                                   ; // x      2760  shared-memory load
/*4f6d0460*/  FFMA R24, R95, R109, R24                                   ; // x      2760  FMA
/*4f6d0470*/  FFMA R33, R96, R108, R33                                   ; // x      2760  FMA
/*4f6d0480*/  LDS.128 R100, [R73+0x420]                                  ; // x      2760  shared-memory load
/*4f6d0490*/  FFMA R32, R96, R109, R32                                   ; // x      2760  FMA
/*4f6d04a0*/  FFMA R41, R97, R108, R41                                   ; // x      2760  FMA
/*4f6d04b0*/  LDS.128 R88, [R72+0x480]                                   ; // x      2760  shared-memory load
/*4f6d04c0*/  FFMA R40, R97, R109, R40                                   ; // x      2760  FMA
/*4f6d04d0*/  FFMA R49, R98, R108, R49                                   ; // x      2760  FMA
/*4f6d04e0*/  LDS.128 R104, [R73+0x460]                                  ; // x      2760  shared-memory load
/*4f6d04f0*/  FFMA R48, R98, R109, R48                                   ; // x      2760  FMA
/*4f6d0500*/  FFMA R57, R99, R108, R57                                   ; // x      2760  FMA
/*4f6d0510*/  FFMA R56, R99, R109, R56                                   ; // x      2760  FMA
/*4f6d0520*/  FFMA R59, R99, R110, R59                                   ; // x      2760  FMA
/*4f6d0530*/  FFMA R58, R99, R111, R58                                   ; // x      2760  FMA
/*4f6d0540*/  FFMA R51, R98, R110, R51                                   ; // x      2760  FMA
/*4f6d0550*/  FFMA R50, R98, R111, R50                                   ; // x      2760  FMA
/*4f6d0560*/  FFMA R43, R97, R110, R43                                   ; // x      2760  FMA
/*4f6d0570*/  FFMA R42, R97, R111, R42                                   ; // x      2760  FMA
/*4f6d0580*/  FFMA R35, R96, R110, R35                                   ; // x      2760  FMA
/*4f6d0590*/  FFMA R34, R96, R111, R34                                   ; // x      2760  FMA
/*4f6d05a0*/  FFMA R27, R95, R110, R27                                   ; // x      2760  FMA
/*4f6d05b0*/  FFMA R26, R95, R111, R26                                   ; // x      2760  FMA
/*4f6d05c0*/  FFMA R19, R94, R110, R19                                   ; // x      2760  FMA
/*4f6d05d0*/  FFMA R18, R94, R111, R18                                   ; // x      2760  FMA
/*4f6d05e0*/  FFMA R11, R93, R110, R11                                   ; // x      2760  FMA
/*4f6d05f0*/  FFMA R10, R93, R111, R10                                   ; // x      2760  FMA
/*4f6d0600*/  FFMA R3, R92, R110, R3                                     ; // x      2760  FMA
/*4f6d0610*/  FFMA R2, R92, R111, R2                                     ; // x      2760  FMA
/*4f6d0620*/  FFMA R5, R92, R112, R5                                     ; // x      2760  FMA
/*4f6d0630*/  @P3   LDG.E.LTC128B.CONSTANT R68, [R74]                    ; // x      2760  global load (DRAM/L2: weights or activations)
/*4f6d0640*/  FFMA R4, R92, R113, R4                                     ; // x      2760  FMA
/*4f6d0650*/  FFMA R13, R93, R112, R13                                   ; // x      2760  FMA
/*4f6d0660*/  FFMA R12, R93, R113, R12                                   ; // x      2760  FMA
/*4f6d0670*/  FFMA R21, R94, R112, R21                                   ; // x      2760  FMA
/*4f6d0680*/  @P1   IADD3 R74, P2, R74, c[0x0][0x180], RZ                ; // x      2760  ALU
/*4f6d0690*/  FFMA R20, R94, R113, R20                                   ; // x      2760  FMA
/*4f6d06a0*/  FFMA R29, R95, R112, R29                                   ; // x      2760  FMA
/*4f6d06b0*/  @P1   IADD3.X R75, R75, c[0x0][0x184], RZ, P2, !PT         ; // x      2760  ALU
/*4f6d06c0*/  FFMA R28, R95, R113, R28                                   ; // x      2760  FMA
/*4f6d06d0*/  FFMA R37, R96, R112, R37                                   ; // x      2760  FMA
/*4f6d06e0*/  @P4   LDG.E.LTC128B.CONSTANT R69, [R74]                    ; // x      2760  global load (DRAM/L2: weights or activations)
/*4f6d06f0*/  FFMA R36, R96, R113, R36                                   ; // x      2760  FMA
/*4f6d0700*/  FFMA R45, R97, R112, R45                                   ; // x      2760  FMA
/*4f6d0710*/  FFMA R44, R97, R113, R44                                   ; // x      2760  FMA
/*4f6d0720*/  FFMA R53, R98, R112, R53                                   ; // x      2760  FMA
/*4f6d0730*/  @P1   IADD3 R74, P2, R74, c[0x0][0x180], RZ                ; // x      2760  ALU
/*4f6d0740*/  FFMA R52, R98, R113, R52                                   ; // x      2760  FMA
/*4f6d0750*/  FFMA R61, R99, R112, R61                                   ; // x      2760  FMA
/*4f6d0760*/  @P1   IADD3.X R75, R75, c[0x0][0x184], RZ, P2, !PT         ; // x      2760  ALU
/*4f6d0770*/  FFMA R60, R99, R113, R60                                   ; // x      2760  FMA
/*4f6d0780*/  FFMA R63, R99, R114, R63                                   ; // x      2760  FMA
/*4f6d0790*/  @P5   LDG.E.LTC128B.CONSTANT R70, [R74]                    ; // x      2760  global load (DRAM/L2: weights or activations)
/*4f6d07a0*/  FFMA R62, R99, R115, R62                                   ; // x      2760  FMA
/*4f6d07b0*/  FFMA R55, R98, R114, R55                                   ; // x      2760  FMA
/*4f6d07c0*/  FFMA R54, R98, R115, R54                                   ; // x      2760  FMA
/*4f6d07d0*/  FFMA R47, R97, R114, R47                                   ; // x      2760  FMA
/*4f6d07e0*/  @P1   IADD3 R74, P2, R74, c[0x0][0x180], RZ                ; // x      2760  ALU
/*4f6d07f0*/  FFMA R46, R97, R115, R46                                   ; // x      2760  FMA
/*4f6d0800*/  FFMA R39, R96, R114, R39                                   ; // x      2760  FMA
/*4f6d0810*/  @P1   IADD3.X R75, R75, c[0x0][0x184], RZ, P2, !PT         ; // x      2760  ALU
/*4f6d0820*/  FFMA R38, R96, R115, R38                                   ; // x      2760  FMA
/*4f6d0830*/  FFMA R31, R95, R114, R31                                   ; // x      2760  FMA
/*4f6d0840*/  @P6   LDG.E.LTC128B.CONSTANT R71, [R74]                    ; // x      2760  global load (DRAM/L2: weights or activations)
/*4f6d0850*/  FFMA R30, R95, R115, R30                                   ; // x      2760  FMA
/*4f6d0860*/  FFMA R23, R94, R114, R23                                   ; // x      2760  FMA
/*4f6d0870*/  FFMA R22, R94, R115, R22                                   ; // x      2760  FMA
/*4f6d0880*/  FFMA R15, R93, R114, R15                                   ; // x      2760  FMA
/*4f6d0890*/  FFMA R14, R93, R115, R14                                   ; // x      2760  FMA
/*4f6d08a0*/  FFMA R7, R92, R114, R7                                     ; // x      2760  FMA
/*4f6d08b0*/  FFMA R6, R92, R115, R6                                     ; // x      2760  FMA
/*4f6d08c0*/  FFMA R1, R84, R100, R1                                     ; // x      2760  FMA
/*4f6d08d0*/  @P1   IADD3 R74, P2, R74, c[0x0][0x190], RZ                ; // x      2760  ALU
/*4f6d08e0*/  FFMA R0, R84, R101, R0                                     ; // x      2760  FMA
/*4f6d08f0*/  FFMA R9, R85, R100, R9                                     ; // x      2760  FMA
/*4f6d0900*/  LDS.128 R92, [R72+0x600]                                   ; // x      2760  shared-memory load
/*4f6d0910*/  FFMA R8, R85, R101, R8                                     ; // x      2760  FMA
/*4f6d0920*/  FFMA R17, R86, R100, R17                                   ; // x      2760  FMA
/*4f6d0930*/  @P1   IADD3.X R75, R75, c[0x0][0x194], RZ, P2, !PT         ; // x      2760  ALU
/*4f6d0940*/  FFMA R16, R86, R101, R16                                   ; // x      2760  FMA
/*4f6d0950*/  FFMA R25, R87, R100, R25                                   ; // x      2760  FMA
/*4f6d0960*/  LDS.128 R108, [R73+0x630]                                  ; // x      2760  shared-memory load
/*4f6d0970*/  FFMA R24, R87, R101, R24                                   ; // x      2760  FMA
/*4f6d0980*/  FFMA R33, R88, R100, R33                                   ; // x      2760  FMA
/*4f6d0990*/  LDS.128 R96, [R72+0x680]                                   ; // x      2760  shared-memory load
/*4f6d09a0*/  FFMA R32, R88, R101, R32                                   ; // x      2760  FMA
/*4f6d09b0*/  FFMA R41, R89, R100, R41                                   ; // x      2760  FMA
/*4f6d09c0*/  LDS.128 R112, [R73+0x670]                                  ; // x      2760  shared-memory load
/*4f6d09d0*/  FFMA R40, R89, R101, R40                                   ; // x      2760  FMA
/*4f6d09e0*/  FFMA R49, R90, R100, R49                                   ; // x      2760  FMA
/*4f6d09f0*/  FFMA R48, R90, R101, R48                                   ; // x      2760  FMA
/*4f6d0a00*/  FFMA R57, R91, R100, R57                                   ; // x      2760  FMA
/*4f6d0a10*/  FFMA R56, R91, R101, R56                                   ; // x      2760  FMA
/*4f6d0a20*/  FFMA R59, R91, R102, R59                                   ; // x      2760  FMA
/*4f6d0a30*/  FFMA R58, R91, R103, R58                                   ; // x      2760  FMA
/*4f6d0a40*/  FFMA R51, R90, R102, R51                                   ; // x      2760  FMA
/*4f6d0a50*/  FFMA R50, R90, R103, R50                                   ; // x      2760  FMA
/*4f6d0a60*/  FFMA R43, R89, R102, R43                                   ; // x      2760  FMA
/*4f6d0a70*/  FFMA R42, R89, R103, R42                                   ; // x      2760  FMA
```

(first 200 of 635 instructions shown; the rest are in the full dump)
