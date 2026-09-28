# Hot loop — `ampere_sgemm_128x128_tn` (sm87)

PC range 0x4f685ae0-0x4f6882f0; backward BRA 0x4f6882f0 -> 0x4f685ae0: of the 5 backward branches, the range whose instructions executed most (ncu per-PC Instructions Executed, §3). Full dump: `raw/ampere_sgemm_128x128_tn_sm87.sass`; per-PC counts: `raw/ampere_sgemm_128x128_tn_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*4f685ae0*/  STS [R78], R64                                             ; // x       240  shared-memory store
/*4f685af0*/  STS [R78+0x40], R65                                        ; // x       240  shared-memory store
/*4f685b00*/  STS [R78+0x80], R66                                        ; // x       240  shared-memory store
/*4f685b10*/  STS [R78+0xc0], R67                                        ; // x       240  shared-memory store
/*4f685b20*/  STS [R78+0x100], R68                                       ; // x       240  shared-memory store
/*4f685b30*/  STS [R78+0x140], R69                                       ; // x       240  shared-memory store
/*4f685b40*/  STS [R78+0x180], R70                                       ; // x       240  shared-memory store
/*4f685b50*/  STS [R78+0x1c0], R71                                       ; // x       240  shared-memory store
/*4f685b60*/  BAR.SYNC.DEFER_BLOCKING 0x0                                ; // x       240  block barrier
/*4f685b70*/  IADD3 R78, R78, R83, RZ                                    ; // x       240  ALU
/*4f685b80*/  LDS.128 R84, [R72]                                         ; // x       240  shared-memory load
/*4f685b90*/  LDS.128 R100, [R73]                                        ; // x       240  shared-memory load
/*4f685ba0*/  LDS.128 R88, [R72+0x80]                                    ; // x       240  shared-memory load
/*4f685bb0*/  LDS.128 R104, [R73+0x40]                                   ; // x       240  shared-memory load
/*4f685bc0*/  FFMA R1, R84, R100, R1                                     ; // x      1920  FMA
/*4f685bd0*/  ISETP.GE.AND P1, PT, R79, 0x10, PT                         ; // x      1920  ALU
/*4f685be0*/  FFMA R0, R84, R101, R0                                     ; // x      1920  FMA
/*4f685bf0*/  FFMA R9, R85, R100, R9                                     ; // x      1920  FMA
/*4f685c00*/  IADD3 R79, R79, -0x8, RZ                                   ; // x      1920  ALU
/*4f685c10*/  FFMA R8, R85, R101, R8                                     ; // x      1920  FMA
/*4f685c20*/  FFMA R17, R86, R100, R17                                   ; // x      1920  FMA
/*4f685c30*/  LDS.128 R92, [R72+0x210]                                   ; // x      1920  shared-memory load
/*4f685c40*/  FFMA R16, R86, R101, R16                                   ; // x      1920  FMA
/*4f685c50*/  FFMA R25, R87, R100, R25                                   ; // x      1920  FMA
/*4f685c60*/  LDS.128 R108, [R73+0x210]                                  ; // x      1920  shared-memory load
/*4f685c70*/  FFMA R24, R87, R101, R24                                   ; // x      1920  FMA
/*4f685c80*/  FFMA R33, R88, R100, R33                                   ; // x      1920  FMA
/*4f685c90*/  LDS.128 R96, [R72+0x290]                                   ; // x      1920  shared-memory load
/*4f685ca0*/  FFMA R32, R88, R101, R32                                   ; // x      1920  FMA
/*4f685cb0*/  FFMA R41, R89, R100, R41                                   ; // x      1920  FMA
/*4f685cc0*/  @!P1  R2P PR, RZ, 0x78                                     ; // x      1920  ALU
/*4f685cd0*/  FFMA R40, R89, R101, R40                                   ; // x      1920  FMA
/*4f685ce0*/  FFMA R49, R90, R100, R49                                   ; // x      1920  FMA
/*4f685cf0*/  @P1   R2P PR, R82, 0x78                                    ; // x      1920  ALU
/*4f685d00*/  FFMA R48, R90, R101, R48                                   ; // x      1920  FMA
/*4f685d10*/  FFMA R57, R91, R100, R57                                   ; // x      1920  FMA
/*4f685d20*/  LDS.128 R112, [R73+0x250]                                  ; // x      1920  shared-memory load
/*4f685d30*/  FFMA R56, R91, R101, R56                                   ; // x      1920  FMA
/*4f685d40*/  FFMA R59, R91, R102, R59                                   ; // x      1920  FMA
/*4f685d50*/  FFMA R58, R91, R103, R58                                   ; // x      1920  FMA
/*4f685d60*/  FFMA R51, R90, R102, R51                                   ; // x      1920  FMA
/*4f685d70*/  FFMA R50, R90, R103, R50                                   ; // x      1920  FMA
/*4f685d80*/  FFMA R43, R89, R102, R43                                   ; // x      1920  FMA
/*4f685d90*/  FFMA R42, R89, R103, R42                                   ; // x      1920  FMA
/*4f685da0*/  FFMA R35, R88, R102, R35                                   ; // x      1920  FMA
/*4f685db0*/  FFMA R34, R88, R103, R34                                   ; // x      1920  FMA
/*4f685dc0*/  FFMA R27, R87, R102, R27                                   ; // x      1920  FMA
/*4f685dd0*/  FFMA R26, R87, R103, R26                                   ; // x      1920  FMA
/*4f685de0*/  FFMA R19, R86, R102, R19                                   ; // x      1920  FMA
/*4f685df0*/  FFMA R18, R86, R103, R18                                   ; // x      1920  FMA
/*4f685e00*/  FFMA R11, R85, R102, R11                                   ; // x      1920  FMA
/*4f685e10*/  FFMA R10, R85, R103, R10                                   ; // x      1920  FMA
/*4f685e20*/  FFMA R3, R84, R102, R3                                     ; // x      1920  FMA
/*4f685e30*/  FFMA R2, R84, R103, R2                                     ; // x      1920  FMA
/*4f685e40*/  FFMA R5, R84, R104, R5                                     ; // x      1920  FMA
/*4f685e50*/  @P3   LDG.E.LTC128B.CONSTANT R64, [R76]                    ; // x      1920  global load (DRAM/L2: weights or activations)
/*4f685e60*/  FFMA R4, R84, R105, R4                                     ; // x      1920  FMA
/*4f685e70*/  FFMA R13, R85, R104, R13                                   ; // x      1920  FMA
/*4f685e80*/  FFMA R12, R85, R105, R12                                   ; // x      1920  FMA
/*4f685e90*/  FFMA R21, R86, R104, R21                                   ; // x      1920  FMA
/*4f685ea0*/  @P1   LEA R76, P2, R74, R76, 0x6                           ; // x      1920  ALU
/*4f685eb0*/  FFMA R20, R86, R105, R20                                   ; // x      1920  FMA
/*4f685ec0*/  FFMA R29, R87, R104, R29                                   ; // x      1920  FMA
/*4f685ed0*/  @P1   LEA.HI.X R77, R74, R77, RZ, 0x6, P2                  ; // x      1920  ALU
/*4f685ee0*/  FFMA R28, R87, R105, R28                                   ; // x      1920  FMA
/*4f685ef0*/  FFMA R37, R88, R104, R37                                   ; // x      1920  FMA
/*4f685f00*/  @P4   LDG.E.LTC128B.CONSTANT R65, [R76]                    ; // x      1920  global load (DRAM/L2: weights or activations)
/*4f685f10*/  FFMA R36, R88, R105, R36                                   ; // x      1920  FMA
/*4f685f20*/  FFMA R45, R89, R104, R45                                   ; // x      1920  FMA
/*4f685f30*/  FFMA R44, R89, R105, R44                                   ; // x      1920  FMA
/*4f685f40*/  FFMA R53, R90, R104, R53                                   ; // x      1920  FMA
/*4f685f50*/  @P1   LEA R76, P2, R74, R76, 0x6                           ; // x      1920  ALU
/*4f685f60*/  FFMA R52, R90, R105, R52                                   ; // x      1920  FMA
/*4f685f70*/  FFMA R61, R91, R104, R61                                   ; // x      1920  FMA
/*4f685f80*/  @P1   LEA.HI.X R77, R74, R77, RZ, 0x6, P2                  ; // x      1920  ALU
/*4f685f90*/  FFMA R60, R91, R105, R60                                   ; // x      1920  FMA
/*4f685fa0*/  FFMA R63, R91, R106, R63                                   ; // x      1920  FMA
/*4f685fb0*/  @P5   LDG.E.LTC128B.CONSTANT R66, [R76]                    ; // x      1920  global load (DRAM/L2: weights or activations)
/*4f685fc0*/  FFMA R62, R91, R107, R62                                   ; // x      1920  FMA
/*4f685fd0*/  FFMA R55, R90, R106, R55                                   ; // x      1920  FMA
/*4f685fe0*/  FFMA R54, R90, R107, R54                                   ; // x      1920  FMA
/*4f685ff0*/  FFMA R47, R89, R106, R47                                   ; // x      1920  FMA
/*4f686000*/  @P1   LEA R76, P2, R74, R76, 0x6                           ; // x      1920  ALU
/*4f686010*/  FFMA R46, R89, R107, R46                                   ; // x      1920  FMA
/*4f686020*/  FFMA R39, R88, R106, R39                                   ; // x      1920  FMA
/*4f686030*/  @P1   LEA.HI.X R77, R74, R77, RZ, 0x6, P2                  ; // x      1920  ALU
/*4f686040*/  FFMA R38, R88, R107, R38                                   ; // x      1920  FMA
/*4f686050*/  FFMA R31, R87, R106, R31                                   ; // x      1920  FMA
/*4f686060*/  @P6   LDG.E.LTC128B.CONSTANT R67, [R76]                    ; // x      1920  global load (DRAM/L2: weights or activations)
/*4f686070*/  FFMA R30, R87, R107, R30                                   ; // x      1920  FMA
/*4f686080*/  FFMA R23, R86, R106, R23                                   ; // x      1920  FMA
/*4f686090*/  FFMA R22, R86, R107, R22                                   ; // x      1920  FMA
/*4f6860a0*/  FFMA R15, R85, R106, R15                                   ; // x      1920  FMA
/*4f6860b0*/  FFMA R14, R85, R107, R14                                   ; // x      1920  FMA
/*4f6860c0*/  FFMA R7, R84, R106, R7                                     ; // x      1920  FMA
/*4f6860d0*/  FFMA R6, R84, R107, R6                                     ; // x      1920  FMA
/*4f6860e0*/  FFMA R1, R92, R108, R1                                     ; // x      1920  FMA
/*4f6860f0*/  @P1   R2P PR, R82.B1, 0x78                                 ; // x      1920  ALU
/*4f686100*/  FFMA R0, R92, R109, R0                                     ; // x      1920  FMA
/*4f686110*/  FFMA R9, R93, R108, R9                                     ; // x      1920  FMA
/*4f686120*/  @P1   LEA R76, P2, R74, R76, 0x6                           ; // x      1920  ALU
/*4f686130*/  FFMA R8, R93, R109, R8                                     ; // x      1920  FMA
/*4f686140*/  FFMA R17, R94, R108, R17                                   ; // x      1920  FMA
/*4f686150*/  LDS.128 R84, [R72+0x420]                                   ; // x      1920  shared-memory load
/*4f686160*/  FFMA R16, R94, R109, R16                                   ; // x      1920  FMA
/*4f686170*/  FFMA R25, R95, R108, R25                                   ; // x      1920  FMA
/*4f686180*/  @P1   LEA.HI.X R77, R74, R77, RZ, 0x6, P2                  ; // x      1920  ALU
/*4f686190*/  FFMA R24, R95, R109, R24                                   ; // x      1920  FMA
/*4f6861a0*/  FFMA R33, R96, R108, R33                                   ; // x      1920  FMA
/*4f6861b0*/  LDS.128 R100, [R73+0x420]                                  ; // x      1920  shared-memory load
/*4f6861c0*/  FFMA R32, R96, R109, R32                                   ; // x      1920  FMA
/*4f6861d0*/  FFMA R41, R97, R108, R41                                   ; // x      1920  FMA
/*4f6861e0*/  LDS.128 R88, [R72+0x4a0]                                   ; // x      1920  shared-memory load
/*4f6861f0*/  FFMA R40, R97, R109, R40                                   ; // x      1920  FMA
/*4f686200*/  FFMA R49, R98, R108, R49                                   ; // x      1920  FMA
/*4f686210*/  LDS.128 R104, [R73+0x460]                                  ; // x      1920  shared-memory load
/*4f686220*/  FFMA R48, R98, R109, R48                                   ; // x      1920  FMA
/*4f686230*/  FFMA R57, R99, R108, R57                                   ; // x      1920  FMA
/*4f686240*/  FFMA R56, R99, R109, R56                                   ; // x      1920  FMA
/*4f686250*/  FFMA R59, R99, R110, R59                                   ; // x      1920  FMA
/*4f686260*/  FFMA R58, R99, R111, R58                                   ; // x      1920  FMA
/*4f686270*/  FFMA R51, R98, R110, R51                                   ; // x      1920  FMA
/*4f686280*/  FFMA R50, R98, R111, R50                                   ; // x      1920  FMA
/*4f686290*/  FFMA R43, R97, R110, R43                                   ; // x      1920  FMA
/*4f6862a0*/  FFMA R42, R97, R111, R42                                   ; // x      1920  FMA
/*4f6862b0*/  FFMA R35, R96, R110, R35                                   ; // x      1920  FMA
/*4f6862c0*/  FFMA R34, R96, R111, R34                                   ; // x      1920  FMA
/*4f6862d0*/  FFMA R27, R95, R110, R27                                   ; // x      1920  FMA
/*4f6862e0*/  FFMA R26, R95, R111, R26                                   ; // x      1920  FMA
/*4f6862f0*/  FFMA R19, R94, R110, R19                                   ; // x      1920  FMA
/*4f686300*/  FFMA R18, R94, R111, R18                                   ; // x      1920  FMA
/*4f686310*/  FFMA R11, R93, R110, R11                                   ; // x      1920  FMA
/*4f686320*/  FFMA R10, R93, R111, R10                                   ; // x      1920  FMA
/*4f686330*/  FFMA R3, R92, R110, R3                                     ; // x      1920  FMA
/*4f686340*/  FFMA R2, R92, R111, R2                                     ; // x      1920  FMA
/*4f686350*/  FFMA R5, R92, R112, R5                                     ; // x      1920  FMA
/*4f686360*/  @P3   LDG.E.LTC128B.CONSTANT R68, [R76]                    ; // x      1920  global load (DRAM/L2: weights or activations)
/*4f686370*/  FFMA R4, R92, R113, R4                                     ; // x      1920  FMA
/*4f686380*/  FFMA R13, R93, R112, R13                                   ; // x      1920  FMA
/*4f686390*/  FFMA R12, R93, R113, R12                                   ; // x      1920  FMA
/*4f6863a0*/  FFMA R21, R94, R112, R21                                   ; // x      1920  FMA
/*4f6863b0*/  @P1   LEA R76, P2, R74, R76, 0x6                           ; // x      1920  ALU
/*4f6863c0*/  FFMA R20, R94, R113, R20                                   ; // x      1920  FMA
/*4f6863d0*/  FFMA R29, R95, R112, R29                                   ; // x      1920  FMA
/*4f6863e0*/  @P1   LEA.HI.X R77, R74, R77, RZ, 0x6, P2                  ; // x      1920  ALU
/*4f6863f0*/  FFMA R28, R95, R113, R28                                   ; // x      1920  FMA
/*4f686400*/  FFMA R37, R96, R112, R37                                   ; // x      1920  FMA
/*4f686410*/  @P4   LDG.E.LTC128B.CONSTANT R69, [R76]                    ; // x      1920  global load (DRAM/L2: weights or activations)
/*4f686420*/  FFMA R36, R96, R113, R36                                   ; // x      1920  FMA
/*4f686430*/  FFMA R45, R97, R112, R45                                   ; // x      1920  FMA
/*4f686440*/  FFMA R44, R97, R113, R44                                   ; // x      1920  FMA
/*4f686450*/  FFMA R53, R98, R112, R53                                   ; // x      1920  FMA
/*4f686460*/  @P1   LEA R76, P2, R74, R76, 0x6                           ; // x      1920  ALU
/*4f686470*/  FFMA R52, R98, R113, R52                                   ; // x      1920  FMA
/*4f686480*/  FFMA R61, R99, R112, R61                                   ; // x      1920  FMA
/*4f686490*/  @P1   LEA.HI.X R77, R74, R77, RZ, 0x6, P2                  ; // x      1920  ALU
/*4f6864a0*/  FFMA R60, R99, R113, R60                                   ; // x      1920  FMA
/*4f6864b0*/  FFMA R63, R99, R114, R63                                   ; // x      1920  FMA
/*4f6864c0*/  @P5   LDG.E.LTC128B.CONSTANT R70, [R76]                    ; // x      1920  global load (DRAM/L2: weights or activations)
/*4f6864d0*/  FFMA R62, R99, R115, R62                                   ; // x      1920  FMA
/*4f6864e0*/  FFMA R55, R98, R114, R55                                   ; // x      1920  FMA
/*4f6864f0*/  FFMA R54, R98, R115, R54                                   ; // x      1920  FMA
/*4f686500*/  FFMA R47, R97, R114, R47                                   ; // x      1920  FMA
/*4f686510*/  @P1   LEA R76, P2, R74, R76, 0x6                           ; // x      1920  ALU
/*4f686520*/  FFMA R46, R97, R115, R46                                   ; // x      1920  FMA
/*4f686530*/  FFMA R39, R96, R114, R39                                   ; // x      1920  FMA
/*4f686540*/  @P1   LEA.HI.X R77, R74, R77, RZ, 0x6, P2                  ; // x      1920  ALU
/*4f686550*/  FFMA R38, R96, R115, R38                                   ; // x      1920  FMA
/*4f686560*/  FFMA R31, R95, R114, R31                                   ; // x      1920  FMA
/*4f686570*/  @P6   LDG.E.LTC128B.CONSTANT R71, [R76]                    ; // x      1920  global load (DRAM/L2: weights or activations)
/*4f686580*/  FFMA R30, R95, R115, R30                                   ; // x      1920  FMA
/*4f686590*/  FFMA R23, R94, R114, R23                                   ; // x      1920  FMA
/*4f6865a0*/  FFMA R22, R94, R115, R22                                   ; // x      1920  FMA
/*4f6865b0*/  FFMA R15, R93, R114, R15                                   ; // x      1920  FMA
/*4f6865c0*/  FFMA R14, R93, R115, R14                                   ; // x      1920  FMA
/*4f6865d0*/  FFMA R7, R92, R114, R7                                     ; // x      1920  FMA
/*4f6865e0*/  FFMA R6, R92, R115, R6                                     ; // x      1920  FMA
/*4f6865f0*/  FFMA R1, R84, R100, R1                                     ; // x      1920  FMA
/*4f686600*/  @P1   IADD3 R76, P2, R76, R80, RZ                          ; // x      1920  ALU
/*4f686610*/  FFMA R0, R84, R101, R0                                     ; // x      1920  FMA
/*4f686620*/  FFMA R9, R85, R100, R9                                     ; // x      1920  FMA
/*4f686630*/  LDS.128 R92, [R72+0x630]                                   ; // x      1920  shared-memory load
/*4f686640*/  FFMA R8, R85, R101, R8                                     ; // x      1920  FMA
/*4f686650*/  FFMA R17, R86, R100, R17                                   ; // x      1920  FMA
/*4f686660*/  @P1   IADD3.X R77, R77, R81, RZ, P2, !PT                   ; // x      1920  ALU
/*4f686670*/  FFMA R16, R86, R101, R16                                   ; // x      1920  FMA
/*4f686680*/  FFMA R25, R87, R100, R25                                   ; // x      1920  FMA
/*4f686690*/  LDS.128 R108, [R73+0x630]                                  ; // x      1920  shared-memory load
/*4f6866a0*/  FFMA R24, R87, R101, R24                                   ; // x      1920  FMA
/*4f6866b0*/  FFMA R33, R88, R100, R33                                   ; // x      1920  FMA
/*4f6866c0*/  LDS.128 R96, [R72+0x6b0]                                   ; // x      1920  shared-memory load
/*4f6866d0*/  FFMA R32, R88, R101, R32                                   ; // x      1920  FMA
/*4f6866e0*/  FFMA R41, R89, R100, R41                                   ; // x      1920  FMA
/*4f6866f0*/  LDS.128 R112, [R73+0x670]                                  ; // x      1920  shared-memory load
/*4f686700*/  FFMA R40, R89, R101, R40                                   ; // x      1920  FMA
/*4f686710*/  FFMA R49, R90, R100, R49                                   ; // x      1920  FMA
/*4f686720*/  FFMA R48, R90, R101, R48                                   ; // x      1920  FMA
/*4f686730*/  FFMA R57, R91, R100, R57                                   ; // x      1920  FMA
/*4f686740*/  FFMA R56, R91, R101, R56                                   ; // x      1920  FMA
/*4f686750*/  FFMA R59, R91, R102, R59                                   ; // x      1920  FMA
```

(first 200 of 642 instructions shown; the rest are in the full dump)
