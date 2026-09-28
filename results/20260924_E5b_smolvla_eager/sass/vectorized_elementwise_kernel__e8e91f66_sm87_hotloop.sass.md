# Hot loop — `vectorized_elementwise_kernel__e8e91f66` (sm87)

PC range 0x4cb92300-0x4cb94e70; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/vectorized_elementwise_kernel__e8e91f66_sm87.sass`; per-PC counts: `raw/vectorized_elementwise_kernel__e8e91f66_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*4cb92300*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x       144  ALU
/*4cb92310*/  @P0   BRA 0x24cb92440                                      ; // x       144  branch
/*4cb92320*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*4cb92330*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*4cb92340*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*4cb92350*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*4cb92360*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*4cb92370*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*4cb92380*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*4cb92390*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*4cb923a0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*4cb923b0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*4cb923c0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*4cb923d0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*4cb923e0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*4cb923f0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*4cb92400*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*4cb92410*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*4cb92420*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*4cb92430*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*4cb92440*/  MOV R1, c[0x0][0x28]                                       ; // x       144  ALU
/*4cb92450*/  S2R R2, SR_CTAID.X                                         ; // x       144  Branch
/*4cb92460*/  ULDC.64 UR4, c[0x0][0x118]                                 ; // x       144  Uniform
/*4cb92470*/  SHF.L.U32 R2, R2, 0xa, RZ                                  ; // x       144  ALU
/*4cb92480*/  IADD3 R3, -R2, c[0x0][0x160], RZ                           ; // x       144  ALU
/*4cb92490*/  ISETP.GE.U32.AND P0, PT, R3, 0x400, PT                     ; // x       144  ALU
/*4cb924a0*/  @!P0  BRA 0x24cb93250                                      ; // x       144  branch
/*4cb924b0*/  S2R R15, SR_TID.X                                          ; // x       140  Branch
/*4cb924c0*/  HFMA2.MMA R17, -RZ, RZ, 0, 2.384185791015625e-07           ; // x       140  FMA
/*4cb924d0*/  IMAD.WIDE R4, R2, R17, c[0x0][0x170]                       ; // x       140  FMA
/*4cb924e0*/  IMAD.WIDE.U32 R4, R15, 0x10, R4                            ; // x       140  FMA
/*4cb924f0*/  LDG.E.128 R8, [R4.64]                                      ; // x       140  global load (DRAM/L2: weights or activations)
/*4cb92500*/  LDG.E.128 R4, [R4.64+0x800]                                ; // x       140  global load (DRAM/L2: weights or activations)
/*4cb92510*/  HFMA2.MMA R18, -RZ, RZ, 3.7421875, 0                       ; // x       140  FMA
/*4cb92520*/  MOV R24, 0x3bbb989d                                        ; // x       140  ALU
/*4cb92530*/  BSSY B2, 0x24cb927f0                                       ; // x       140  Branch
/*4cb92540*/  FFMA.SAT R0, -R11, R24, 0.5                                ; // x       140  FMA
/*4cb92550*/  FFMA.SAT R25, -R8, R24, 0.5                                ; // x       140  FMA
/*4cb92560*/  FFMA.RM R3, R0, R18, 12582913                              ; // x       140  FMA
/*4cb92570*/  FFMA.RM R25, R25, R18, 12582913                            ; // x       140  FMA
/*4cb92580*/  FADD R0, R3, -12583039                                     ; // x       140  FMA
/*4cb92590*/  SHF.L.U32 R12, R3, 0x17, RZ                                ; // x       140  ALU
/*4cb925a0*/  FFMA.SAT R21, -R4, R24, 0.5                                ; // x       140  FMA
/*4cb925b0*/  FFMA.SAT R22, -R5, R24, 0.5                                ; // x       140  FMA
/*4cb925c0*/  FFMA R0, -R11, 1.4426950216293334961, -R0                  ; // x       140  FMA
/*4cb925d0*/  FFMA.SAT R23, -R6, R24, 0.5                                ; // x       140  FMA
/*4cb925e0*/  FFMA.RM R21, R21, R18, 12582913                            ; // x       140  FMA
/*4cb925f0*/  FFMA.RM R22, R22, R18, 12582913                            ; // x       140  FMA
/*4cb92600*/  FFMA R0, -R11, 1.925963033500011079e-08, R0                ; // x       140  FMA
/*4cb92610*/  FFMA.RM R23, R23, R18, 12582913                            ; // x       140  FMA
/*4cb92620*/  MUFU.EX2 R13, R0                                           ; // x       140  special-function unit
/*4cb92630*/  FFMA.SAT R0, -R10, R24, 0.5                                ; // x       140  FMA
/*4cb92640*/  FFMA.RM R0, R0, R18, 12582913                              ; // x       140  FMA
/*4cb92650*/  FFMA R16, R12, R13, 1                                      ; // x       140  FMA
/*4cb92660*/  FADD R3, R0, -12583039                                     ; // x       140  FMA
/*4cb92670*/  SHF.L.U32 R13, R0, 0x17, RZ                                ; // x       140  ALU
/*4cb92680*/  MUFU.RCP R14, R16                                          ; // x       140  special-function unit
/*4cb92690*/  FFMA R3, -R10, 1.4426950216293334961, -R3                  ; // x       140  FMA
/*4cb926a0*/  FFMA R3, -R10, 1.925963033500011079e-08, R3                ; // x       140  FMA
/*4cb926b0*/  FCHK P0, R11, R16                                          ; // x       140  FMA
/*4cb926c0*/  MUFU.EX2 R12, R3                                           ; // x       140  special-function unit
/*4cb926d0*/  FFMA R0, -R16, R14, 1                                      ; // x       140  FMA
/*4cb926e0*/  FFMA R0, R14, R0, R14                                      ; // x       140  FMA
/*4cb926f0*/  FFMA.SAT R14, -R9, R24, 0.5                                ; // x       140  FMA
/*4cb92700*/  FFMA.SAT R24, -R7, R24, 0.5                                ; // x       140  FMA
/*4cb92710*/  IMAD.WIDE.U32 R2, R2, R17, c[0x0][0x168]                   ; // x       140  FMA
/*4cb92720*/  FFMA.RM R14, R14, R18, 12582913                            ; // x       140  FMA
/*4cb92730*/  FFMA.RM R24, R24, R18, 12582913                            ; // x       140  FMA
/*4cb92740*/  IMAD.WIDE R2, R15, 0x10, R2                                ; // x       140  FMA
/*4cb92750*/  FFMA R15, R11, R0, RZ                                      ; // x       140  FMA
/*4cb92760*/  FFMA R13, R13, R12, 1                                      ; // x       140  FMA
/*4cb92770*/  FFMA R12, -R16, R15, R11                                   ; // x       140  FMA
/*4cb92780*/  FFMA R15, R0, R12, R15                                     ; // x       140  FMA
/*4cb92790*/  @!P0  BRA 0x24cb927e0                                      ; // x       140  branch
/*4cb927a0*/  IMAD.MOV.U32 R0, RZ, RZ, R16                               ; // x         0  FMA
/*4cb927b0*/  MOV R16, 0x4d0                                             ; // x         0  ALU
/*4cb927c0*/  CALL.REL.NOINC 0x24cb94750                                 ; // x         0  Branch
/*4cb927d0*/  MOV R15, R12                                               ; // x         0  ALU
/*4cb927e0*/  BSYNC B2                                                   ; // x       140  Branch
/*4cb927f0*/  MUFU.RCP R0, R13                                           ; // x       140  special-function unit
/*4cb92800*/  FADD R14, R14, -12583039                                   ; // x       140  FMA
/*4cb92810*/  HFMA2.MMA R12, -RZ, RZ, 0.96630859375, -0.0022525787353515625 ; // x       140  FMA
/*4cb92820*/  MOV R16, 0x437c0000                                        ; // x       140  ALU
/*4cb92830*/  BSSY B2, 0x24cb92980                                       ; // x       140  Branch
/*4cb92840*/  FFMA R14, -R9, 1.4426950216293334961, -R14                 ; // x       140  FMA
/*4cb92850*/  FCHK P0, R10, R13                                          ; // x       140  FMA
/*4cb92860*/  FFMA R14, -R9, 1.925963033500011079e-08, R14               ; // x       140  FMA
/*4cb92870*/  FFMA.SAT R12, -R9, R12, 0.5                                ; // x       140  FMA
/*4cb92880*/  MUFU.EX2 R26, R14                                          ; // x       140  special-function unit
/*4cb92890*/  FFMA R11, -R13, R0, 1                                      ; // x       140  FMA
/*4cb928a0*/  FFMA.RM R12, R12, R16, 12582913                            ; // x       140  FMA
/*4cb928b0*/  FFMA R11, R0, R11, R0                                      ; // x       140  FMA
/*4cb928c0*/  SHF.L.U32 R12, R12, 0x17, RZ                               ; // x       140  ALU
/*4cb928d0*/  FFMA R14, R10, R11, RZ                                     ; // x       140  FMA
/*4cb928e0*/  FFMA R0, -R13, R14, R10                                    ; // x       140  FMA
/*4cb928f0*/  FFMA R26, R26, R12, 1                                      ; // x       140  FMA
/*4cb92900*/  FFMA R14, R11, R0, R14                                     ; // x       140  FMA
/*4cb92910*/  @!P0  BRA 0x24cb92970                                      ; // x       140  branch
/*4cb92920*/  MOV R11, R10                                               ; // x         0  ALU
/*4cb92930*/  MOV R0, R13                                                ; // x         0  ALU
/*4cb92940*/  MOV R16, 0x660                                             ; // x         0  ALU
/*4cb92950*/  CALL.REL.NOINC 0x24cb94750                                 ; // x         0  Branch
/*4cb92960*/  MOV R14, R12                                               ; // x         0  ALU
/*4cb92970*/  BSYNC B2                                                   ; // x       140  Branch
/*4cb92980*/  MUFU.RCP R0, R26                                           ; // x       140  special-function unit
/*4cb92990*/  FADD R25, R25, -12583039                                   ; // x       140  FMA
/*4cb929a0*/  HFMA2.MMA R12, -RZ, RZ, 3.7421875, 0                       ; // x       140  FMA
/*4cb929b0*/  IMAD.MOV.U32 R11, RZ, RZ, 0x3bbb989d                       ; // x       140  FMA
/*4cb929c0*/  BSSY B2, 0x24cb92b10                                       ; // x       140  Branch
/*4cb929d0*/  FFMA R25, -R8, 1.4426950216293334961, -R25                 ; // x       140  FMA
/*4cb929e0*/  FFMA.SAT R11, -R8, R11, 0.5                                ; // x       140  FMA
/*4cb929f0*/  FCHK P0, R9, R26                                           ; // x       140  FMA
/*4cb92a00*/  FFMA R25, -R8, 1.925963033500011079e-08, R25               ; // x       140  FMA
/*4cb92a10*/  MUFU.EX2 R25, R25                                          ; // x       140  special-function unit
/*4cb92a20*/  FFMA R10, R0, -R26, 1                                      ; // x       140  FMA
/*4cb92a30*/  FFMA R0, R0, R10, R0                                       ; // x       140  FMA
/*4cb92a40*/  FFMA.RM R10, R11, R12, 12582913                            ; // x       140  FMA
/*4cb92a50*/  FFMA R13, R9, R0, RZ                                       ; // x       140  FMA
/*4cb92a60*/  SHF.L.U32 R10, R10, 0x17, RZ                               ; // x       140  ALU
/*4cb92a70*/  FFMA R11, R13, -R26, R9                                    ; // x       140  FMA
/*4cb92a80*/  FFMA R10, R25, R10, 1                                      ; // x       140  FMA
/*4cb92a90*/  FFMA R13, R0, R11, R13                                     ; // x       140  FMA
/*4cb92aa0*/  @!P0  BRA 0x24cb92b00                                      ; // x       140  branch
/*4cb92ab0*/  MOV R11, R9                                                ; // x         0  ALU
/*4cb92ac0*/  MOV R0, R26                                                ; // x         0  ALU
/*4cb92ad0*/  MOV R16, 0x7f0                                             ; // x         0  ALU
/*4cb92ae0*/  CALL.REL.NOINC 0x24cb94750                                 ; // x         0  Branch
/*4cb92af0*/  MOV R13, R12                                               ; // x         0  ALU
/*4cb92b00*/  BSYNC B2                                                   ; // x       140  Branch
/*4cb92b10*/  MUFU.RCP R0, R10                                           ; // x       140  special-function unit
/*4cb92b20*/  FADD R24, R24, -12583039                                   ; // x       140  FMA
/*4cb92b30*/  HFMA2.MMA R11, -RZ, RZ, 0.96630859375, -0.0022525787353515625 ; // x       140  FMA
/*4cb92b40*/  MOV R12, 0x437c0000                                        ; // x       140  ALU
/*4cb92b50*/  BSSY B2, 0x24cb92c90                                       ; // x       140  Branch
/*4cb92b60*/  FFMA R24, -R7, 1.4426950216293334961, -R24                 ; // x       140  FMA
/*4cb92b70*/  FCHK P0, R8, R10                                           ; // x       140  FMA
/*4cb92b80*/  FFMA R24, -R7, 1.925963033500011079e-08, R24               ; // x       140  FMA
/*4cb92b90*/  FFMA.SAT R11, -R7, R11, 0.5                                ; // x       140  FMA
/*4cb92ba0*/  MUFU.EX2 R24, R24                                          ; // x       140  special-function unit
/*4cb92bb0*/  FFMA R9, R0, -R10, 1                                       ; // x       140  FMA
/*4cb92bc0*/  FFMA R0, R0, R9, R0                                        ; // x       140  FMA
/*4cb92bd0*/  FFMA.RM R9, R11, R12, 12582913                             ; // x       140  FMA
/*4cb92be0*/  FFMA R12, R8, R0, RZ                                       ; // x       140  FMA
/*4cb92bf0*/  IMAD.SHL.U32 R9, R9, 0x800000, RZ                          ; // x       140  FMA
/*4cb92c00*/  FFMA R11, R12, -R10, R8                                    ; // x       140  FMA
/*4cb92c10*/  FFMA R9, R24, R9, 1                                        ; // x       140  FMA
/*4cb92c20*/  FFMA R12, R0, R11, R12                                     ; // x       140  FMA
/*4cb92c30*/  @!P0  BRA 0x24cb92c80                                      ; // x       140  branch
/*4cb92c40*/  MOV R11, R8                                                ; // x         0  ALU
/*4cb92c50*/  MOV R0, R10                                                ; // x         0  ALU
/*4cb92c60*/  MOV R16, 0x980                                             ; // x         0  ALU
/*4cb92c70*/  CALL.REL.NOINC 0x24cb94750                                 ; // x         0  Branch
/*4cb92c80*/  BSYNC B2                                                   ; // x       140  Branch
/*4cb92c90*/  MUFU.RCP R0, R9                                            ; // x       140  special-function unit
/*4cb92ca0*/  FADD R23, R23, -12583039                                   ; // x       140  FMA
/*4cb92cb0*/  HFMA2.MMA R10, -RZ, RZ, 0.96630859375, -0.0022525787353515625 ; // x       140  FMA
/*4cb92cc0*/  MOV R11, 0x437c0000                                        ; // x       140  ALU
/*4cb92cd0*/  STG.E.128 [R2.64], R12                                     ; // x       140  global store
/*4cb92ce0*/  FFMA R23, -R6, 1.4426950216293334961, -R23                 ; // x       140  FMA
/*4cb92cf0*/  BSSY B2, 0x24cb92e30                                       ; // x       140  Branch
/*4cb92d00*/  FCHK P0, R7, R9                                            ; // x       140  FMA
/*4cb92d10*/  FFMA R23, -R6, 1.925963033500011079e-08, R23               ; // x       140  FMA
/*4cb92d20*/  FFMA.SAT R10, -R6, R10, 0.5                                ; // x       140  FMA
/*4cb92d30*/  MUFU.EX2 R23, R23                                          ; // x       140  special-function unit
/*4cb92d40*/  FFMA R8, R0, -R9, 1                                        ; // x       140  FMA
/*4cb92d50*/  FFMA R0, R0, R8, R0                                        ; // x       140  FMA
/*4cb92d60*/  FFMA.RM R8, R10, R11, 12582913                             ; // x       140  FMA
/*4cb92d70*/  FFMA R15, R7, R0, RZ                                       ; // x       140  FMA
/*4cb92d80*/  SHF.L.U32 R8, R8, 0x17, RZ                                 ; // x       140  ALU
/*4cb92d90*/  FFMA R10, R15, -R9, R7                                     ; // x       140  FMA
/*4cb92da0*/  FFMA R8, R23, R8, 1                                        ; // x       140  FMA
/*4cb92db0*/  FFMA R15, R0, R10, R15                                     ; // x       140  FMA
/*4cb92dc0*/  @!P0  BRA 0x24cb92e20                                      ; // x       140  branch
/*4cb92dd0*/  MOV R11, R7                                                ; // x         0  ALU
/*4cb92de0*/  MOV R0, R9                                                 ; // x         0  ALU
/*4cb92df0*/  MOV R16, 0xb10                                             ; // x         0  ALU
/*4cb92e00*/  CALL.REL.NOINC 0x24cb94750                                 ; // x         0  Branch
/*4cb92e10*/  IMAD.MOV.U32 R15, RZ, RZ, R12                              ; // x         0  FMA
/*4cb92e20*/  BSYNC B2                                                   ; // x       140  Branch
/*4cb92e30*/  MUFU.RCP R0, R8                                            ; // x       140  special-function unit
/*4cb92e40*/  FADD R22, R22, -12583039                                   ; // x       140  FMA
/*4cb92e50*/  HFMA2.MMA R9, -RZ, RZ, 0.96630859375, -0.0022525787353515625 ; // x       140  FMA
/*4cb92e60*/  MOV R10, 0x437c0000                                        ; // x       140  ALU
/*4cb92e70*/  BSSY B2, 0x24cb92fc0                                       ; // x       140  Branch
/*4cb92e80*/  FFMA R22, -R5, 1.4426950216293334961, -R22                 ; // x       140  FMA
/*4cb92e90*/  FCHK P0, R6, R8                                            ; // x       140  FMA
/*4cb92ea0*/  FFMA R22, -R5, 1.925963033500011079e-08, R22               ; // x       140  FMA
/*4cb92eb0*/  FFMA.SAT R9, -R5, R9, 0.5                                  ; // x       140  FMA
/*4cb92ec0*/  MUFU.EX2 R22, R22                                          ; // x       140  special-function unit
/*4cb92ed0*/  FFMA R7, R0, -R8, 1                                        ; // x       140  FMA
/*4cb92ee0*/  FFMA R0, R0, R7, R0                                        ; // x       140  FMA
/*4cb92ef0*/  FFMA.RM R7, R9, R10, 12582913                              ; // x       140  FMA
/*4cb92f00*/  FFMA R14, R6, R0, RZ                                       ; // x       140  FMA
/*4cb92f10*/  SHF.L.U32 R7, R7, 0x17, RZ                                 ; // x       140  ALU
/*4cb92f20*/  FFMA R9, R14, -R8, R6                                      ; // x       140  FMA
/*4cb92f30*/  FFMA R7, R22, R7, 1                                        ; // x       140  FMA
/*4cb92f40*/  FFMA R14, R0, R9, R14                                      ; // x       140  FMA
/*4cb92f50*/  @!P0  BRA 0x24cb92fb0                                      ; // x       140  branch
/*4cb92f60*/  MOV R11, R6                                                ; // x         0  ALU
/*4cb92f70*/  MOV R0, R8                                                 ; // x         0  ALU
```

(first 200 of 696 instructions shown; the rest are in the full dump)
