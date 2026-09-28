# Hot loop — `elementwise_kernel__76a18c20` (sm87)

PC range 0x229ad00-0x229e2f0; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/elementwise_kernel__76a18c20_sm87.sass`; per-PC counts: `raw/elementwise_kernel__76a18c20_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*0229ad00*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x         8  ALU
/*0229ad10*/  @P0   BRA 0x20229ae40                                      ; // x         8  branch
/*0229ad20*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*0229ad30*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*0229ad40*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*0229ad50*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*0229ad60*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*0229ad70*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*0229ad80*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*0229ad90*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*0229ada0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*0229adb0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*0229adc0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*0229add0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*0229ade0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*0229adf0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*0229ae00*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*0229ae10*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*0229ae20*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*0229ae30*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*0229ae40*/  MOV R1, c[0x0][0x28]                                       ; // x         8  ALU
/*0229ae50*/  S2R R5, SR_CTAID.X                                         ; // x         8  Branch
/*0229ae60*/  ULDC.64 UR4, c[0x0][0x118]                                 ; // x         8  Uniform
/*0229ae70*/  BSSY B0, 0x20229c440                                       ; // x         8  Branch
/*0229ae80*/  S2R R6, SR_TID.X                                           ; // x         8  Branch
/*0229ae90*/  IMAD R7, R5, 0x100, R6                                     ; // x         8  FMA
/*0229aea0*/  ISETP.GE.AND P0, PT, R7, c[0x0][0x160], PT                 ; // x         8  ALU
/*0229aeb0*/  @P0   BRA 0x20229c430                                      ; // x         8  branch
/*0229aec0*/  ISETP.NE.AND P0, PT, RZ, c[0x0][0x168], PT                 ; // x         8  ALU
/*0229aed0*/  HFMA2.MMA R2, -RZ, RZ, 0, 0                                ; // x         8  FMA
/*0229aee0*/  IMAD.MOV.U32 R0, RZ, RZ, RZ                                ; // x         8  FMA
/*0229aef0*/  MOV R14, RZ                                                ; // x         8  ALU
/*0229af00*/  @!P0  BRA 0x20229bea0                                      ; // x         8  branch
/*0229af10*/  MOV R3, R7                                                 ; // x         8  ALU
/*0229af20*/  IMAD.MOV.U32 R2, RZ, RZ, RZ                                ; // x         8  FMA
/*0229af30*/  IMAD.HI.U32 R8, R7, c[0x0][0x170], R2                      ; // x         8  FMA
/*0229af40*/  IMAD.MOV.U32 R3, RZ, RZ, c[0x0][0x168]                     ; // x         8  FMA
/*0229af50*/  SHF.R.U32.HI R9, RZ, c[0x0][0x174], R8                     ; // x         8  ALU
/*0229af60*/  ISETP.NE.AND P0, PT, R3, 0x1, PT                           ; // x         8  ALU
/*0229af70*/  IADD3 R2, -R9, RZ, RZ                                      ; // x         8  ALU
/*0229af80*/  IMAD R2, R2, c[0x0][0x16c], R7                             ; // x         8  FMA
/*0229af90*/  IMAD R14, R2, c[0x0][0x298], RZ                            ; // x         8  FMA
/*0229afa0*/  IMAD R0, R2, c[0x0][0x29c], RZ                             ; // x         8  FMA
/*0229afb0*/  IMAD R2, R2, c[0x0][0x2a0], RZ                             ; // x         8  FMA
/*0229afc0*/  @!P0  BRA 0x20229bea0                                      ; // x         8  branch
/*0229afd0*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*0229afe0*/  ISETP.NE.AND P0, PT, R3, 0x2, PT                           ; // x         0  ALU
/*0229aff0*/  IMAD.HI.U32 R10, R9, c[0x0][0x17c], R8                     ; // x         0  FMA
/*0229b000*/  SHF.R.U32.HI R11, RZ, c[0x0][0x180], R10                   ; // x         0  ALU
/*0229b010*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*0229b020*/  IMAD R9, R7, c[0x0][0x178], R9                             ; // x         0  FMA
/*0229b030*/  IMAD R14, R9, c[0x0][0x2a4], R14                           ; // x         0  FMA
/*0229b040*/  IMAD R0, R9, c[0x0][0x2a8], R0                             ; // x         0  FMA
/*0229b050*/  IMAD R2, R9, c[0x0][0x2ac], R2                             ; // x         0  FMA
/*0229b060*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b070*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*0229b080*/  ISETP.NE.AND P0, PT, R3, 0x3, PT                           ; // x         0  ALU
/*0229b090*/  IMAD.HI.U32 R8, R11, c[0x0][0x188], R10                    ; // x         0  FMA
/*0229b0a0*/  SHF.R.U32.HI R9, RZ, c[0x0][0x18c], R8                     ; // x         0  ALU
/*0229b0b0*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*0229b0c0*/  IMAD R11, R7, c[0x0][0x184], R11                           ; // x         0  FMA
/*0229b0d0*/  IMAD R14, R11, c[0x0][0x2b0], R14                          ; // x         0  FMA
/*0229b0e0*/  IMAD R0, R11, c[0x0][0x2b4], R0                            ; // x         0  FMA
/*0229b0f0*/  IMAD R2, R11, c[0x0][0x2b8], R2                            ; // x         0  FMA
/*0229b100*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b110*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*0229b120*/  ISETP.NE.AND P0, PT, R3, 0x4, PT                           ; // x         0  ALU
/*0229b130*/  IMAD.HI.U32 R10, R9, c[0x0][0x194], R8                     ; // x         0  FMA
/*0229b140*/  SHF.R.U32.HI R11, RZ, c[0x0][0x198], R10                   ; // x         0  ALU
/*0229b150*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*0229b160*/  IMAD R9, R7, c[0x0][0x190], R9                             ; // x         0  FMA
/*0229b170*/  IMAD R14, R9, c[0x0][0x2bc], R14                           ; // x         0  FMA
/*0229b180*/  IMAD R0, R9, c[0x0][0x2c0], R0                             ; // x         0  FMA
/*0229b190*/  IMAD R2, R9, c[0x0][0x2c4], R2                             ; // x         0  FMA
/*0229b1a0*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b1b0*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*0229b1c0*/  ISETP.NE.AND P0, PT, R3, 0x5, PT                           ; // x         0  ALU
/*0229b1d0*/  IMAD.HI.U32 R8, R11, c[0x0][0x1a0], R10                    ; // x         0  FMA
/*0229b1e0*/  SHF.R.U32.HI R9, RZ, c[0x0][0x1a4], R8                     ; // x         0  ALU
/*0229b1f0*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*0229b200*/  IMAD R11, R7, c[0x0][0x19c], R11                           ; // x         0  FMA
/*0229b210*/  IMAD R14, R11, c[0x0][0x2c8], R14                          ; // x         0  FMA
/*0229b220*/  IMAD R0, R11, c[0x0][0x2cc], R0                            ; // x         0  FMA
/*0229b230*/  IMAD R2, R11, c[0x0][0x2d0], R2                            ; // x         0  FMA
/*0229b240*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b250*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*0229b260*/  ISETP.NE.AND P0, PT, R3, 0x6, PT                           ; // x         0  ALU
/*0229b270*/  IMAD.HI.U32 R10, R9, c[0x0][0x1ac], R8                     ; // x         0  FMA
/*0229b280*/  SHF.R.U32.HI R11, RZ, c[0x0][0x1b0], R10                   ; // x         0  ALU
/*0229b290*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*0229b2a0*/  IMAD R9, R7, c[0x0][0x1a8], R9                             ; // x         0  FMA
/*0229b2b0*/  IMAD R14, R9, c[0x0][0x2d4], R14                           ; // x         0  FMA
/*0229b2c0*/  IMAD R0, R9, c[0x0][0x2d8], R0                             ; // x         0  FMA
/*0229b2d0*/  IMAD R2, R9, c[0x0][0x2dc], R2                             ; // x         0  FMA
/*0229b2e0*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b2f0*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*0229b300*/  ISETP.NE.AND P0, PT, R3, 0x7, PT                           ; // x         0  ALU
/*0229b310*/  IMAD.HI.U32 R8, R11, c[0x0][0x1b8], R10                    ; // x         0  FMA
/*0229b320*/  SHF.R.U32.HI R9, RZ, c[0x0][0x1bc], R8                     ; // x         0  ALU
/*0229b330*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*0229b340*/  IMAD R11, R7, c[0x0][0x1b4], R11                           ; // x         0  FMA
/*0229b350*/  IMAD R14, R11, c[0x0][0x2e0], R14                          ; // x         0  FMA
/*0229b360*/  IMAD R0, R11, c[0x0][0x2e4], R0                            ; // x         0  FMA
/*0229b370*/  IMAD R2, R11, c[0x0][0x2e8], R2                            ; // x         0  FMA
/*0229b380*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b390*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*0229b3a0*/  ISETP.NE.AND P0, PT, R3, 0x8, PT                           ; // x         0  ALU
/*0229b3b0*/  IMAD.HI.U32 R10, R9, c[0x0][0x1c4], R8                     ; // x         0  FMA
/*0229b3c0*/  SHF.R.U32.HI R11, RZ, c[0x0][0x1c8], R10                   ; // x         0  ALU
/*0229b3d0*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*0229b3e0*/  IMAD R9, R7, c[0x0][0x1c0], R9                             ; // x         0  FMA
/*0229b3f0*/  IMAD R14, R9, c[0x0][0x2ec], R14                           ; // x         0  FMA
/*0229b400*/  IMAD R0, R9, c[0x0][0x2f0], R0                             ; // x         0  FMA
/*0229b410*/  IMAD R2, R9, c[0x0][0x2f4], R2                             ; // x         0  FMA
/*0229b420*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b430*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*0229b440*/  ISETP.NE.AND P0, PT, R3, 0x9, PT                           ; // x         0  ALU
/*0229b450*/  IMAD.HI.U32 R8, R11, c[0x0][0x1d0], R10                    ; // x         0  FMA
/*0229b460*/  SHF.R.U32.HI R9, RZ, c[0x0][0x1d4], R8                     ; // x         0  ALU
/*0229b470*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*0229b480*/  IMAD R11, R7, c[0x0][0x1cc], R11                           ; // x         0  FMA
/*0229b490*/  IMAD R14, R11, c[0x0][0x2f8], R14                          ; // x         0  FMA
/*0229b4a0*/  IMAD R0, R11, c[0x0][0x2fc], R0                            ; // x         0  FMA
/*0229b4b0*/  IMAD R2, R11, c[0x0][0x300], R2                            ; // x         0  FMA
/*0229b4c0*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b4d0*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*0229b4e0*/  ISETP.NE.AND P0, PT, R3, 0xa, PT                           ; // x         0  ALU
/*0229b4f0*/  IMAD.HI.U32 R10, R9, c[0x0][0x1dc], R8                     ; // x         0  FMA
/*0229b500*/  SHF.R.U32.HI R11, RZ, c[0x0][0x1e0], R10                   ; // x         0  ALU
/*0229b510*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*0229b520*/  IMAD R9, R7, c[0x0][0x1d8], R9                             ; // x         0  FMA
/*0229b530*/  IMAD R14, R9, c[0x0][0x304], R14                           ; // x         0  FMA
/*0229b540*/  IMAD R0, R9, c[0x0][0x308], R0                             ; // x         0  FMA
/*0229b550*/  IMAD R2, R9, c[0x0][0x30c], R2                             ; // x         0  FMA
/*0229b560*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b570*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*0229b580*/  ISETP.NE.AND P0, PT, R3, 0xb, PT                           ; // x         0  ALU
/*0229b590*/  IMAD.HI.U32 R8, R11, c[0x0][0x1e8], R10                    ; // x         0  FMA
/*0229b5a0*/  SHF.R.U32.HI R9, RZ, c[0x0][0x1ec], R8                     ; // x         0  ALU
/*0229b5b0*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*0229b5c0*/  IMAD R11, R7, c[0x0][0x1e4], R11                           ; // x         0  FMA
/*0229b5d0*/  IMAD R14, R11, c[0x0][0x310], R14                          ; // x         0  FMA
/*0229b5e0*/  IMAD R0, R11, c[0x0][0x314], R0                            ; // x         0  FMA
/*0229b5f0*/  IMAD R2, R11, c[0x0][0x318], R2                            ; // x         0  FMA
/*0229b600*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b610*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*0229b620*/  ISETP.NE.AND P0, PT, R3, 0xc, PT                           ; // x         0  ALU
/*0229b630*/  IMAD.HI.U32 R10, R9, c[0x0][0x1f4], R8                     ; // x         0  FMA
/*0229b640*/  SHF.R.U32.HI R11, RZ, c[0x0][0x1f8], R10                   ; // x         0  ALU
/*0229b650*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*0229b660*/  IMAD R9, R7, c[0x0][0x1f0], R9                             ; // x         0  FMA
/*0229b670*/  IMAD R14, R9, c[0x0][0x31c], R14                           ; // x         0  FMA
/*0229b680*/  IMAD R0, R9, c[0x0][0x320], R0                             ; // x         0  FMA
/*0229b690*/  IMAD R2, R9, c[0x0][0x324], R2                             ; // x         0  FMA
/*0229b6a0*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b6b0*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*0229b6c0*/  ISETP.NE.AND P0, PT, R3, 0xd, PT                           ; // x         0  ALU
/*0229b6d0*/  IMAD.HI.U32 R8, R11, c[0x0][0x200], R10                    ; // x         0  FMA
/*0229b6e0*/  SHF.R.U32.HI R9, RZ, c[0x0][0x204], R8                     ; // x         0  ALU
/*0229b6f0*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*0229b700*/  IMAD R11, R7, c[0x0][0x1fc], R11                           ; // x         0  FMA
/*0229b710*/  IMAD R14, R11, c[0x0][0x328], R14                          ; // x         0  FMA
/*0229b720*/  IMAD R0, R11, c[0x0][0x32c], R0                            ; // x         0  FMA
/*0229b730*/  IMAD R2, R11, c[0x0][0x330], R2                            ; // x         0  FMA
/*0229b740*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b750*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*0229b760*/  ISETP.NE.AND P0, PT, R3, 0xe, PT                           ; // x         0  ALU
/*0229b770*/  IMAD.HI.U32 R10, R9, c[0x0][0x20c], R8                     ; // x         0  FMA
/*0229b780*/  SHF.R.U32.HI R11, RZ, c[0x0][0x210], R10                   ; // x         0  ALU
/*0229b790*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*0229b7a0*/  IMAD R9, R7, c[0x0][0x208], R9                             ; // x         0  FMA
/*0229b7b0*/  IMAD R14, R9, c[0x0][0x334], R14                           ; // x         0  FMA
/*0229b7c0*/  IMAD R0, R9, c[0x0][0x338], R0                             ; // x         0  FMA
/*0229b7d0*/  IMAD R2, R9, c[0x0][0x33c], R2                             ; // x         0  FMA
/*0229b7e0*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b7f0*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*0229b800*/  ISETP.NE.AND P0, PT, R3, 0xf, PT                           ; // x         0  ALU
/*0229b810*/  IMAD.HI.U32 R8, R11, c[0x0][0x218], R10                    ; // x         0  FMA
/*0229b820*/  SHF.R.U32.HI R9, RZ, c[0x0][0x21c], R8                     ; // x         0  ALU
/*0229b830*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*0229b840*/  IMAD R11, R7, c[0x0][0x214], R11                           ; // x         0  FMA
/*0229b850*/  IMAD R14, R11, c[0x0][0x340], R14                          ; // x         0  FMA
/*0229b860*/  IMAD R0, R11, c[0x0][0x344], R0                            ; // x         0  FMA
/*0229b870*/  IMAD R2, R11, c[0x0][0x348], R2                            ; // x         0  FMA
/*0229b880*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b890*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*0229b8a0*/  ISETP.NE.AND P0, PT, R3, 0x10, PT                          ; // x         0  ALU
/*0229b8b0*/  IMAD.HI.U32 R10, R9, c[0x0][0x224], R8                     ; // x         0  FMA
/*0229b8c0*/  SHF.R.U32.HI R11, RZ, c[0x0][0x228], R10                   ; // x         0  ALU
/*0229b8d0*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*0229b8e0*/  IMAD R9, R7, c[0x0][0x220], R9                             ; // x         0  FMA
/*0229b8f0*/  IMAD R14, R9, c[0x0][0x34c], R14                           ; // x         0  FMA
/*0229b900*/  IMAD R0, R9, c[0x0][0x350], R0                             ; // x         0  FMA
/*0229b910*/  IMAD R2, R9, c[0x0][0x354], R2                             ; // x         0  FMA
/*0229b920*/  @!P0  BRA 0x20229bea0                                      ; // x         0  branch
/*0229b930*/  HFMA2.MMA R10, -RZ, RZ, 0, 0                               ; // x         0  FMA
/*0229b940*/  ISETP.NE.AND P0, PT, R3, 0x11, PT                          ; // x         0  ALU
/*0229b950*/  IMAD.HI.U32 R8, R11, c[0x0][0x230], R10                    ; // x         0  FMA
/*0229b960*/  SHF.R.U32.HI R9, RZ, c[0x0][0x234], R8                     ; // x         0  ALU
/*0229b970*/  IMAD.MOV R7, RZ, RZ, -R9                                   ; // x         0  FMA
```

(first 200 of 864 instructions shown; the rest are in the full dump)
