# Hot loop — `elementwise_kernel__76a18c20` (sm87)

PC range 0x2dd2db00-0x2dd310f0; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/elementwise_kernel__76a18c20_sm87.sass`; per-PC counts: `raw/elementwise_kernel__76a18c20_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*2dd2db00*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x         8  ALU
/*2dd2db10*/  @P0   BRA 0x42dd2dc40                                      ; // x         8  branch
/*2dd2db20*/  BMOV.32 B0, 0xffffffff                                     ; // x         8  Other
/*2dd2db30*/  BMOV.32.CLEAR B1, B0                                       ; // x         8  Other
/*2dd2db40*/  BMOV.32.CLEAR B2, B1                                       ; // x         8  Other
/*2dd2db50*/  BMOV.32.CLEAR B3, B2                                       ; // x         8  Other
/*2dd2db60*/  BMOV.32.CLEAR B4, B3                                       ; // x         8  Other
/*2dd2db70*/  BMOV.32.CLEAR B5, B4                                       ; // x         8  Other
/*2dd2db80*/  BMOV.32.CLEAR B6, B5                                       ; // x         8  Other
/*2dd2db90*/  BMOV.32.CLEAR B7, B6                                       ; // x         8  Other
/*2dd2dba0*/  BMOV.32.CLEAR B8, B7                                       ; // x         8  Other
/*2dd2dbb0*/  BMOV.32.CLEAR B9, B8                                       ; // x         8  Other
/*2dd2dbc0*/  BMOV.32.CLEAR B10, B9                                      ; // x         8  Other
/*2dd2dbd0*/  BMOV.32.CLEAR B11, B10                                     ; // x         8  Other
/*2dd2dbe0*/  BMOV.32.CLEAR B12, B11                                     ; // x         8  Other
/*2dd2dbf0*/  BMOV.32.CLEAR B13, B12                                     ; // x         8  Other
/*2dd2dc00*/  BMOV.32.CLEAR B14, B13                                     ; // x         8  Other
/*2dd2dc10*/  BMOV.32.CLEAR B15, B14                                     ; // x         8  Other
/*2dd2dc20*/  BMOV.32 B15, 0x0                                           ; // x         8  Other
/*2dd2dc30*/  UMOV UR2, 0x1                                              ; // x         8  Uniform
/*2dd2dc40*/  MOV R1, c[0x0][0x28]                                       ; // x         8  ALU
/*2dd2dc50*/  S2R R5, SR_CTAID.X                                         ; // x         8  Branch
/*2dd2dc60*/  ULDC.64 UR4, c[0x0][0x118]                                 ; // x         8  Uniform
/*2dd2dc70*/  BSSY B0, 0x42dd2f240                                       ; // x         8  Branch
/*2dd2dc80*/  S2R R6, SR_TID.X                                           ; // x         8  Branch
/*2dd2dc90*/  IMAD R7, R5, 0x100, R6                                     ; // x         8  FMA
/*2dd2dca0*/  ISETP.GE.AND P0, PT, R7, c[0x0][0x160], PT                 ; // x         8  ALU
/*2dd2dcb0*/  @P0   BRA 0x42dd2f230                                      ; // x         8  branch
/*2dd2dcc0*/  ISETP.NE.AND P0, PT, RZ, c[0x0][0x168], PT                 ; // x         8  ALU
/*2dd2dcd0*/  HFMA2.MMA R2, -RZ, RZ, 0, 0                                ; // x         8  FMA
/*2dd2dce0*/  IMAD.MOV.U32 R0, RZ, RZ, RZ                                ; // x         8  FMA
/*2dd2dcf0*/  MOV R14, RZ                                                ; // x         8  ALU
/*2dd2dd00*/  @!P0  BRA 0x42dd2eca0                                      ; // x         8  branch
/*2dd2dd10*/  MOV R3, R7                                                 ; // x         8  ALU
/*2dd2dd20*/  IMAD.MOV.U32 R2, RZ, RZ, RZ                                ; // x         8  FMA
/*2dd2dd30*/  IMAD.HI.U32 R8, R7, c[0x0][0x170], R2                      ; // x         8  FMA
/*2dd2dd40*/  IMAD.MOV.U32 R3, RZ, RZ, c[0x0][0x168]                     ; // x         8  FMA
/*2dd2dd50*/  SHF.R.U32.HI R9, RZ, c[0x0][0x174], R8                     ; // x         8  ALU
/*2dd2dd60*/  ISETP.NE.AND P0, PT, R3, 0x1, PT                           ; // x         8  ALU
/*2dd2dd70*/  IADD3 R2, -R9, RZ, RZ                                      ; // x         8  ALU
/*2dd2dd80*/  IMAD R2, R2, c[0x0][0x16c], R7                             ; // x         8  FMA
/*2dd2dd90*/  IMAD R14, R2, c[0x0][0x298], RZ                            ; // x         8  FMA
/*2dd2dda0*/  IMAD R0, R2, c[0x0][0x29c], RZ                             ; // x         8  FMA
/*2dd2ddb0*/  IMAD R2, R2, c[0x0][0x2a0], RZ                             ; // x         8  FMA
/*2dd2ddc0*/  @!P0  BRA 0x42dd2eca0                                      ; // x         8  branch
/*2dd2ddd0*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*2dd2dde0*/  ISETP.NE.AND P0, PT, R3, 0x2, PT                           ; // x         0  ALU
/*2dd2ddf0*/  IMAD.HI.U32 R10, R9, c[0x0][0x17c], R8                     ; // x         0  FMA
/*2dd2de00*/  SHF.R.U32.HI R11, RZ, c[0x0][0x180], R10                   ; // x         0  ALU
/*2dd2de10*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*2dd2de20*/  IMAD R9, R7, c[0x0][0x178], R9                             ; // x         0  FMA
/*2dd2de30*/  IMAD R14, R9, c[0x0][0x2a4], R14                           ; // x         0  FMA
/*2dd2de40*/  IMAD R0, R9, c[0x0][0x2a8], R0                             ; // x         0  FMA
/*2dd2de50*/  IMAD R2, R9, c[0x0][0x2ac], R2                             ; // x         0  FMA
/*2dd2de60*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2de70*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*2dd2de80*/  ISETP.NE.AND P0, PT, R3, 0x3, PT                           ; // x         0  ALU
/*2dd2de90*/  IMAD.HI.U32 R8, R11, c[0x0][0x188], R10                    ; // x         0  FMA
/*2dd2dea0*/  SHF.R.U32.HI R9, RZ, c[0x0][0x18c], R8                     ; // x         0  ALU
/*2dd2deb0*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*2dd2dec0*/  IMAD R11, R7, c[0x0][0x184], R11                           ; // x         0  FMA
/*2dd2ded0*/  IMAD R14, R11, c[0x0][0x2b0], R14                          ; // x         0  FMA
/*2dd2dee0*/  IMAD R0, R11, c[0x0][0x2b4], R0                            ; // x         0  FMA
/*2dd2def0*/  IMAD R2, R11, c[0x0][0x2b8], R2                            ; // x         0  FMA
/*2dd2df00*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2df10*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*2dd2df20*/  ISETP.NE.AND P0, PT, R3, 0x4, PT                           ; // x         0  ALU
/*2dd2df30*/  IMAD.HI.U32 R10, R9, c[0x0][0x194], R8                     ; // x         0  FMA
/*2dd2df40*/  SHF.R.U32.HI R11, RZ, c[0x0][0x198], R10                   ; // x         0  ALU
/*2dd2df50*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*2dd2df60*/  IMAD R9, R7, c[0x0][0x190], R9                             ; // x         0  FMA
/*2dd2df70*/  IMAD R14, R9, c[0x0][0x2bc], R14                           ; // x         0  FMA
/*2dd2df80*/  IMAD R0, R9, c[0x0][0x2c0], R0                             ; // x         0  FMA
/*2dd2df90*/  IMAD R2, R9, c[0x0][0x2c4], R2                             ; // x         0  FMA
/*2dd2dfa0*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2dfb0*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*2dd2dfc0*/  ISETP.NE.AND P0, PT, R3, 0x5, PT                           ; // x         0  ALU
/*2dd2dfd0*/  IMAD.HI.U32 R8, R11, c[0x0][0x1a0], R10                    ; // x         0  FMA
/*2dd2dfe0*/  SHF.R.U32.HI R9, RZ, c[0x0][0x1a4], R8                     ; // x         0  ALU
/*2dd2dff0*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*2dd2e000*/  IMAD R11, R7, c[0x0][0x19c], R11                           ; // x         0  FMA
/*2dd2e010*/  IMAD R14, R11, c[0x0][0x2c8], R14                          ; // x         0  FMA
/*2dd2e020*/  IMAD R0, R11, c[0x0][0x2cc], R0                            ; // x         0  FMA
/*2dd2e030*/  IMAD R2, R11, c[0x0][0x2d0], R2                            ; // x         0  FMA
/*2dd2e040*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2e050*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*2dd2e060*/  ISETP.NE.AND P0, PT, R3, 0x6, PT                           ; // x         0  ALU
/*2dd2e070*/  IMAD.HI.U32 R10, R9, c[0x0][0x1ac], R8                     ; // x         0  FMA
/*2dd2e080*/  SHF.R.U32.HI R11, RZ, c[0x0][0x1b0], R10                   ; // x         0  ALU
/*2dd2e090*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*2dd2e0a0*/  IMAD R9, R7, c[0x0][0x1a8], R9                             ; // x         0  FMA
/*2dd2e0b0*/  IMAD R14, R9, c[0x0][0x2d4], R14                           ; // x         0  FMA
/*2dd2e0c0*/  IMAD R0, R9, c[0x0][0x2d8], R0                             ; // x         0  FMA
/*2dd2e0d0*/  IMAD R2, R9, c[0x0][0x2dc], R2                             ; // x         0  FMA
/*2dd2e0e0*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2e0f0*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*2dd2e100*/  ISETP.NE.AND P0, PT, R3, 0x7, PT                           ; // x         0  ALU
/*2dd2e110*/  IMAD.HI.U32 R8, R11, c[0x0][0x1b8], R10                    ; // x         0  FMA
/*2dd2e120*/  SHF.R.U32.HI R9, RZ, c[0x0][0x1bc], R8                     ; // x         0  ALU
/*2dd2e130*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*2dd2e140*/  IMAD R11, R7, c[0x0][0x1b4], R11                           ; // x         0  FMA
/*2dd2e150*/  IMAD R14, R11, c[0x0][0x2e0], R14                          ; // x         0  FMA
/*2dd2e160*/  IMAD R0, R11, c[0x0][0x2e4], R0                            ; // x         0  FMA
/*2dd2e170*/  IMAD R2, R11, c[0x0][0x2e8], R2                            ; // x         0  FMA
/*2dd2e180*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2e190*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*2dd2e1a0*/  ISETP.NE.AND P0, PT, R3, 0x8, PT                           ; // x         0  ALU
/*2dd2e1b0*/  IMAD.HI.U32 R10, R9, c[0x0][0x1c4], R8                     ; // x         0  FMA
/*2dd2e1c0*/  SHF.R.U32.HI R11, RZ, c[0x0][0x1c8], R10                   ; // x         0  ALU
/*2dd2e1d0*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*2dd2e1e0*/  IMAD R9, R7, c[0x0][0x1c0], R9                             ; // x         0  FMA
/*2dd2e1f0*/  IMAD R14, R9, c[0x0][0x2ec], R14                           ; // x         0  FMA
/*2dd2e200*/  IMAD R0, R9, c[0x0][0x2f0], R0                             ; // x         0  FMA
/*2dd2e210*/  IMAD R2, R9, c[0x0][0x2f4], R2                             ; // x         0  FMA
/*2dd2e220*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2e230*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*2dd2e240*/  ISETP.NE.AND P0, PT, R3, 0x9, PT                           ; // x         0  ALU
/*2dd2e250*/  IMAD.HI.U32 R8, R11, c[0x0][0x1d0], R10                    ; // x         0  FMA
/*2dd2e260*/  SHF.R.U32.HI R9, RZ, c[0x0][0x1d4], R8                     ; // x         0  ALU
/*2dd2e270*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*2dd2e280*/  IMAD R11, R7, c[0x0][0x1cc], R11                           ; // x         0  FMA
/*2dd2e290*/  IMAD R14, R11, c[0x0][0x2f8], R14                          ; // x         0  FMA
/*2dd2e2a0*/  IMAD R0, R11, c[0x0][0x2fc], R0                            ; // x         0  FMA
/*2dd2e2b0*/  IMAD R2, R11, c[0x0][0x300], R2                            ; // x         0  FMA
/*2dd2e2c0*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2e2d0*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*2dd2e2e0*/  ISETP.NE.AND P0, PT, R3, 0xa, PT                           ; // x         0  ALU
/*2dd2e2f0*/  IMAD.HI.U32 R10, R9, c[0x0][0x1dc], R8                     ; // x         0  FMA
/*2dd2e300*/  SHF.R.U32.HI R11, RZ, c[0x0][0x1e0], R10                   ; // x         0  ALU
/*2dd2e310*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*2dd2e320*/  IMAD R9, R7, c[0x0][0x1d8], R9                             ; // x         0  FMA
/*2dd2e330*/  IMAD R14, R9, c[0x0][0x304], R14                           ; // x         0  FMA
/*2dd2e340*/  IMAD R0, R9, c[0x0][0x308], R0                             ; // x         0  FMA
/*2dd2e350*/  IMAD R2, R9, c[0x0][0x30c], R2                             ; // x         0  FMA
/*2dd2e360*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2e370*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*2dd2e380*/  ISETP.NE.AND P0, PT, R3, 0xb, PT                           ; // x         0  ALU
/*2dd2e390*/  IMAD.HI.U32 R8, R11, c[0x0][0x1e8], R10                    ; // x         0  FMA
/*2dd2e3a0*/  SHF.R.U32.HI R9, RZ, c[0x0][0x1ec], R8                     ; // x         0  ALU
/*2dd2e3b0*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*2dd2e3c0*/  IMAD R11, R7, c[0x0][0x1e4], R11                           ; // x         0  FMA
/*2dd2e3d0*/  IMAD R14, R11, c[0x0][0x310], R14                          ; // x         0  FMA
/*2dd2e3e0*/  IMAD R0, R11, c[0x0][0x314], R0                            ; // x         0  FMA
/*2dd2e3f0*/  IMAD R2, R11, c[0x0][0x318], R2                            ; // x         0  FMA
/*2dd2e400*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2e410*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*2dd2e420*/  ISETP.NE.AND P0, PT, R3, 0xc, PT                           ; // x         0  ALU
/*2dd2e430*/  IMAD.HI.U32 R10, R9, c[0x0][0x1f4], R8                     ; // x         0  FMA
/*2dd2e440*/  SHF.R.U32.HI R11, RZ, c[0x0][0x1f8], R10                   ; // x         0  ALU
/*2dd2e450*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*2dd2e460*/  IMAD R9, R7, c[0x0][0x1f0], R9                             ; // x         0  FMA
/*2dd2e470*/  IMAD R14, R9, c[0x0][0x31c], R14                           ; // x         0  FMA
/*2dd2e480*/  IMAD R0, R9, c[0x0][0x320], R0                             ; // x         0  FMA
/*2dd2e490*/  IMAD R2, R9, c[0x0][0x324], R2                             ; // x         0  FMA
/*2dd2e4a0*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2e4b0*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*2dd2e4c0*/  ISETP.NE.AND P0, PT, R3, 0xd, PT                           ; // x         0  ALU
/*2dd2e4d0*/  IMAD.HI.U32 R8, R11, c[0x0][0x200], R10                    ; // x         0  FMA
/*2dd2e4e0*/  SHF.R.U32.HI R9, RZ, c[0x0][0x204], R8                     ; // x         0  ALU
/*2dd2e4f0*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*2dd2e500*/  IMAD R11, R7, c[0x0][0x1fc], R11                           ; // x         0  FMA
/*2dd2e510*/  IMAD R14, R11, c[0x0][0x328], R14                          ; // x         0  FMA
/*2dd2e520*/  IMAD R0, R11, c[0x0][0x32c], R0                            ; // x         0  FMA
/*2dd2e530*/  IMAD R2, R11, c[0x0][0x330], R2                            ; // x         0  FMA
/*2dd2e540*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2e550*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*2dd2e560*/  ISETP.NE.AND P0, PT, R3, 0xe, PT                           ; // x         0  ALU
/*2dd2e570*/  IMAD.HI.U32 R10, R9, c[0x0][0x20c], R8                     ; // x         0  FMA
/*2dd2e580*/  SHF.R.U32.HI R11, RZ, c[0x0][0x210], R10                   ; // x         0  ALU
/*2dd2e590*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*2dd2e5a0*/  IMAD R9, R7, c[0x0][0x208], R9                             ; // x         0  FMA
/*2dd2e5b0*/  IMAD R14, R9, c[0x0][0x334], R14                           ; // x         0  FMA
/*2dd2e5c0*/  IMAD R0, R9, c[0x0][0x338], R0                             ; // x         0  FMA
/*2dd2e5d0*/  IMAD R2, R9, c[0x0][0x33c], R2                             ; // x         0  FMA
/*2dd2e5e0*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2e5f0*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*2dd2e600*/  ISETP.NE.AND P0, PT, R3, 0xf, PT                           ; // x         0  ALU
/*2dd2e610*/  IMAD.HI.U32 R8, R11, c[0x0][0x218], R10                    ; // x         0  FMA
/*2dd2e620*/  SHF.R.U32.HI R9, RZ, c[0x0][0x21c], R8                     ; // x         0  ALU
/*2dd2e630*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*2dd2e640*/  IMAD R11, R7, c[0x0][0x214], R11                           ; // x         0  FMA
/*2dd2e650*/  IMAD R14, R11, c[0x0][0x340], R14                          ; // x         0  FMA
/*2dd2e660*/  IMAD R0, R11, c[0x0][0x344], R0                            ; // x         0  FMA
/*2dd2e670*/  IMAD R2, R11, c[0x0][0x348], R2                            ; // x         0  FMA
/*2dd2e680*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2e690*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*2dd2e6a0*/  ISETP.NE.AND P0, PT, R3, 0x10, PT                          ; // x         0  ALU
/*2dd2e6b0*/  IMAD.HI.U32 R10, R9, c[0x0][0x224], R8                     ; // x         0  FMA
/*2dd2e6c0*/  SHF.R.U32.HI R11, RZ, c[0x0][0x228], R10                   ; // x         0  ALU
/*2dd2e6d0*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*2dd2e6e0*/  IMAD R9, R7, c[0x0][0x220], R9                             ; // x         0  FMA
/*2dd2e6f0*/  IMAD R14, R9, c[0x0][0x34c], R14                           ; // x         0  FMA
/*2dd2e700*/  IMAD R0, R9, c[0x0][0x350], R0                             ; // x         0  FMA
/*2dd2e710*/  IMAD R2, R9, c[0x0][0x354], R2                             ; // x         0  FMA
/*2dd2e720*/  @!P0  BRA 0x42dd2eca0                                      ; // x         0  branch
/*2dd2e730*/  HFMA2.MMA R10, -RZ, RZ, 0, 0                               ; // x         0  FMA
/*2dd2e740*/  ISETP.NE.AND P0, PT, R3, 0x11, PT                          ; // x         0  ALU
/*2dd2e750*/  IMAD.HI.U32 R8, R11, c[0x0][0x230], R10                    ; // x         0  FMA
/*2dd2e760*/  SHF.R.U32.HI R9, RZ, c[0x0][0x234], R8                     ; // x         0  ALU
/*2dd2e770*/  IMAD.MOV R7, RZ, RZ, -R9                                   ; // x         0  FMA
```

(first 200 of 864 instructions shown; the rest are in the full dump)
