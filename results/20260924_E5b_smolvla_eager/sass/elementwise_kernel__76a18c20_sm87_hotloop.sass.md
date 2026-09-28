# Hot loop — `elementwise_kernel__76a18c20` (sm87)

PC range 0x4cb6d200-0x4cb707f0; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/elementwise_kernel__76a18c20_sm87.sass`; per-PC counts: `raw/elementwise_kernel__76a18c20_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*4cb6d200*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x         8  ALU
/*4cb6d210*/  @P0   BRA 0x24cb6d340                                      ; // x         8  branch
/*4cb6d220*/  BMOV.32 B0, 0xffffffff                                     ; // x         8  Other
/*4cb6d230*/  BMOV.32.CLEAR B1, B0                                       ; // x         8  Other
/*4cb6d240*/  BMOV.32.CLEAR B2, B1                                       ; // x         8  Other
/*4cb6d250*/  BMOV.32.CLEAR B3, B2                                       ; // x         8  Other
/*4cb6d260*/  BMOV.32.CLEAR B4, B3                                       ; // x         8  Other
/*4cb6d270*/  BMOV.32.CLEAR B5, B4                                       ; // x         8  Other
/*4cb6d280*/  BMOV.32.CLEAR B6, B5                                       ; // x         8  Other
/*4cb6d290*/  BMOV.32.CLEAR B7, B6                                       ; // x         8  Other
/*4cb6d2a0*/  BMOV.32.CLEAR B8, B7                                       ; // x         8  Other
/*4cb6d2b0*/  BMOV.32.CLEAR B9, B8                                       ; // x         8  Other
/*4cb6d2c0*/  BMOV.32.CLEAR B10, B9                                      ; // x         8  Other
/*4cb6d2d0*/  BMOV.32.CLEAR B11, B10                                     ; // x         8  Other
/*4cb6d2e0*/  BMOV.32.CLEAR B12, B11                                     ; // x         8  Other
/*4cb6d2f0*/  BMOV.32.CLEAR B13, B12                                     ; // x         8  Other
/*4cb6d300*/  BMOV.32.CLEAR B14, B13                                     ; // x         8  Other
/*4cb6d310*/  BMOV.32.CLEAR B15, B14                                     ; // x         8  Other
/*4cb6d320*/  BMOV.32 B15, 0x0                                           ; // x         8  Other
/*4cb6d330*/  UMOV UR2, 0x1                                              ; // x         8  Uniform
/*4cb6d340*/  MOV R1, c[0x0][0x28]                                       ; // x         8  ALU
/*4cb6d350*/  S2R R5, SR_CTAID.X                                         ; // x         8  Branch
/*4cb6d360*/  ULDC.64 UR4, c[0x0][0x118]                                 ; // x         8  Uniform
/*4cb6d370*/  BSSY B0, 0x24cb6e940                                       ; // x         8  Branch
/*4cb6d380*/  S2R R6, SR_TID.X                                           ; // x         8  Branch
/*4cb6d390*/  IMAD R7, R5, 0x100, R6                                     ; // x         8  FMA
/*4cb6d3a0*/  ISETP.GE.AND P0, PT, R7, c[0x0][0x160], PT                 ; // x         8  ALU
/*4cb6d3b0*/  @P0   BRA 0x24cb6e930                                      ; // x         8  branch
/*4cb6d3c0*/  ISETP.NE.AND P0, PT, RZ, c[0x0][0x168], PT                 ; // x         8  ALU
/*4cb6d3d0*/  HFMA2.MMA R2, -RZ, RZ, 0, 0                                ; // x         8  FMA
/*4cb6d3e0*/  IMAD.MOV.U32 R0, RZ, RZ, RZ                                ; // x         8  FMA
/*4cb6d3f0*/  MOV R14, RZ                                                ; // x         8  ALU
/*4cb6d400*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         8  branch
/*4cb6d410*/  MOV R3, R7                                                 ; // x         8  ALU
/*4cb6d420*/  IMAD.MOV.U32 R2, RZ, RZ, RZ                                ; // x         8  FMA
/*4cb6d430*/  IMAD.HI.U32 R8, R7, c[0x0][0x170], R2                      ; // x         8  FMA
/*4cb6d440*/  IMAD.MOV.U32 R3, RZ, RZ, c[0x0][0x168]                     ; // x         8  FMA
/*4cb6d450*/  SHF.R.U32.HI R9, RZ, c[0x0][0x174], R8                     ; // x         8  ALU
/*4cb6d460*/  ISETP.NE.AND P0, PT, R3, 0x1, PT                           ; // x         8  ALU
/*4cb6d470*/  IADD3 R2, -R9, RZ, RZ                                      ; // x         8  ALU
/*4cb6d480*/  IMAD R2, R2, c[0x0][0x16c], R7                             ; // x         8  FMA
/*4cb6d490*/  IMAD R14, R2, c[0x0][0x298], RZ                            ; // x         8  FMA
/*4cb6d4a0*/  IMAD R0, R2, c[0x0][0x29c], RZ                             ; // x         8  FMA
/*4cb6d4b0*/  IMAD R2, R2, c[0x0][0x2a0], RZ                             ; // x         8  FMA
/*4cb6d4c0*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         8  branch
/*4cb6d4d0*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*4cb6d4e0*/  ISETP.NE.AND P0, PT, R3, 0x2, PT                           ; // x         0  ALU
/*4cb6d4f0*/  IMAD.HI.U32 R10, R9, c[0x0][0x17c], R8                     ; // x         0  FMA
/*4cb6d500*/  SHF.R.U32.HI R11, RZ, c[0x0][0x180], R10                   ; // x         0  ALU
/*4cb6d510*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*4cb6d520*/  IMAD R9, R7, c[0x0][0x178], R9                             ; // x         0  FMA
/*4cb6d530*/  IMAD R14, R9, c[0x0][0x2a4], R14                           ; // x         0  FMA
/*4cb6d540*/  IMAD R0, R9, c[0x0][0x2a8], R0                             ; // x         0  FMA
/*4cb6d550*/  IMAD R2, R9, c[0x0][0x2ac], R2                             ; // x         0  FMA
/*4cb6d560*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6d570*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*4cb6d580*/  ISETP.NE.AND P0, PT, R3, 0x3, PT                           ; // x         0  ALU
/*4cb6d590*/  IMAD.HI.U32 R8, R11, c[0x0][0x188], R10                    ; // x         0  FMA
/*4cb6d5a0*/  SHF.R.U32.HI R9, RZ, c[0x0][0x18c], R8                     ; // x         0  ALU
/*4cb6d5b0*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*4cb6d5c0*/  IMAD R11, R7, c[0x0][0x184], R11                           ; // x         0  FMA
/*4cb6d5d0*/  IMAD R14, R11, c[0x0][0x2b0], R14                          ; // x         0  FMA
/*4cb6d5e0*/  IMAD R0, R11, c[0x0][0x2b4], R0                            ; // x         0  FMA
/*4cb6d5f0*/  IMAD R2, R11, c[0x0][0x2b8], R2                            ; // x         0  FMA
/*4cb6d600*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6d610*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*4cb6d620*/  ISETP.NE.AND P0, PT, R3, 0x4, PT                           ; // x         0  ALU
/*4cb6d630*/  IMAD.HI.U32 R10, R9, c[0x0][0x194], R8                     ; // x         0  FMA
/*4cb6d640*/  SHF.R.U32.HI R11, RZ, c[0x0][0x198], R10                   ; // x         0  ALU
/*4cb6d650*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*4cb6d660*/  IMAD R9, R7, c[0x0][0x190], R9                             ; // x         0  FMA
/*4cb6d670*/  IMAD R14, R9, c[0x0][0x2bc], R14                           ; // x         0  FMA
/*4cb6d680*/  IMAD R0, R9, c[0x0][0x2c0], R0                             ; // x         0  FMA
/*4cb6d690*/  IMAD R2, R9, c[0x0][0x2c4], R2                             ; // x         0  FMA
/*4cb6d6a0*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6d6b0*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*4cb6d6c0*/  ISETP.NE.AND P0, PT, R3, 0x5, PT                           ; // x         0  ALU
/*4cb6d6d0*/  IMAD.HI.U32 R8, R11, c[0x0][0x1a0], R10                    ; // x         0  FMA
/*4cb6d6e0*/  SHF.R.U32.HI R9, RZ, c[0x0][0x1a4], R8                     ; // x         0  ALU
/*4cb6d6f0*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*4cb6d700*/  IMAD R11, R7, c[0x0][0x19c], R11                           ; // x         0  FMA
/*4cb6d710*/  IMAD R14, R11, c[0x0][0x2c8], R14                          ; // x         0  FMA
/*4cb6d720*/  IMAD R0, R11, c[0x0][0x2cc], R0                            ; // x         0  FMA
/*4cb6d730*/  IMAD R2, R11, c[0x0][0x2d0], R2                            ; // x         0  FMA
/*4cb6d740*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6d750*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*4cb6d760*/  ISETP.NE.AND P0, PT, R3, 0x6, PT                           ; // x         0  ALU
/*4cb6d770*/  IMAD.HI.U32 R10, R9, c[0x0][0x1ac], R8                     ; // x         0  FMA
/*4cb6d780*/  SHF.R.U32.HI R11, RZ, c[0x0][0x1b0], R10                   ; // x         0  ALU
/*4cb6d790*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*4cb6d7a0*/  IMAD R9, R7, c[0x0][0x1a8], R9                             ; // x         0  FMA
/*4cb6d7b0*/  IMAD R14, R9, c[0x0][0x2d4], R14                           ; // x         0  FMA
/*4cb6d7c0*/  IMAD R0, R9, c[0x0][0x2d8], R0                             ; // x         0  FMA
/*4cb6d7d0*/  IMAD R2, R9, c[0x0][0x2dc], R2                             ; // x         0  FMA
/*4cb6d7e0*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6d7f0*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*4cb6d800*/  ISETP.NE.AND P0, PT, R3, 0x7, PT                           ; // x         0  ALU
/*4cb6d810*/  IMAD.HI.U32 R8, R11, c[0x0][0x1b8], R10                    ; // x         0  FMA
/*4cb6d820*/  SHF.R.U32.HI R9, RZ, c[0x0][0x1bc], R8                     ; // x         0  ALU
/*4cb6d830*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*4cb6d840*/  IMAD R11, R7, c[0x0][0x1b4], R11                           ; // x         0  FMA
/*4cb6d850*/  IMAD R14, R11, c[0x0][0x2e0], R14                          ; // x         0  FMA
/*4cb6d860*/  IMAD R0, R11, c[0x0][0x2e4], R0                            ; // x         0  FMA
/*4cb6d870*/  IMAD R2, R11, c[0x0][0x2e8], R2                            ; // x         0  FMA
/*4cb6d880*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6d890*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*4cb6d8a0*/  ISETP.NE.AND P0, PT, R3, 0x8, PT                           ; // x         0  ALU
/*4cb6d8b0*/  IMAD.HI.U32 R10, R9, c[0x0][0x1c4], R8                     ; // x         0  FMA
/*4cb6d8c0*/  SHF.R.U32.HI R11, RZ, c[0x0][0x1c8], R10                   ; // x         0  ALU
/*4cb6d8d0*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*4cb6d8e0*/  IMAD R9, R7, c[0x0][0x1c0], R9                             ; // x         0  FMA
/*4cb6d8f0*/  IMAD R14, R9, c[0x0][0x2ec], R14                           ; // x         0  FMA
/*4cb6d900*/  IMAD R0, R9, c[0x0][0x2f0], R0                             ; // x         0  FMA
/*4cb6d910*/  IMAD R2, R9, c[0x0][0x2f4], R2                             ; // x         0  FMA
/*4cb6d920*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6d930*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*4cb6d940*/  ISETP.NE.AND P0, PT, R3, 0x9, PT                           ; // x         0  ALU
/*4cb6d950*/  IMAD.HI.U32 R8, R11, c[0x0][0x1d0], R10                    ; // x         0  FMA
/*4cb6d960*/  SHF.R.U32.HI R9, RZ, c[0x0][0x1d4], R8                     ; // x         0  ALU
/*4cb6d970*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*4cb6d980*/  IMAD R11, R7, c[0x0][0x1cc], R11                           ; // x         0  FMA
/*4cb6d990*/  IMAD R14, R11, c[0x0][0x2f8], R14                          ; // x         0  FMA
/*4cb6d9a0*/  IMAD R0, R11, c[0x0][0x2fc], R0                            ; // x         0  FMA
/*4cb6d9b0*/  IMAD R2, R11, c[0x0][0x300], R2                            ; // x         0  FMA
/*4cb6d9c0*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6d9d0*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*4cb6d9e0*/  ISETP.NE.AND P0, PT, R3, 0xa, PT                           ; // x         0  ALU
/*4cb6d9f0*/  IMAD.HI.U32 R10, R9, c[0x0][0x1dc], R8                     ; // x         0  FMA
/*4cb6da00*/  SHF.R.U32.HI R11, RZ, c[0x0][0x1e0], R10                   ; // x         0  ALU
/*4cb6da10*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*4cb6da20*/  IMAD R9, R7, c[0x0][0x1d8], R9                             ; // x         0  FMA
/*4cb6da30*/  IMAD R14, R9, c[0x0][0x304], R14                           ; // x         0  FMA
/*4cb6da40*/  IMAD R0, R9, c[0x0][0x308], R0                             ; // x         0  FMA
/*4cb6da50*/  IMAD R2, R9, c[0x0][0x30c], R2                             ; // x         0  FMA
/*4cb6da60*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6da70*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*4cb6da80*/  ISETP.NE.AND P0, PT, R3, 0xb, PT                           ; // x         0  ALU
/*4cb6da90*/  IMAD.HI.U32 R8, R11, c[0x0][0x1e8], R10                    ; // x         0  FMA
/*4cb6daa0*/  SHF.R.U32.HI R9, RZ, c[0x0][0x1ec], R8                     ; // x         0  ALU
/*4cb6dab0*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*4cb6dac0*/  IMAD R11, R7, c[0x0][0x1e4], R11                           ; // x         0  FMA
/*4cb6dad0*/  IMAD R14, R11, c[0x0][0x310], R14                          ; // x         0  FMA
/*4cb6dae0*/  IMAD R0, R11, c[0x0][0x314], R0                            ; // x         0  FMA
/*4cb6daf0*/  IMAD R2, R11, c[0x0][0x318], R2                            ; // x         0  FMA
/*4cb6db00*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6db10*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*4cb6db20*/  ISETP.NE.AND P0, PT, R3, 0xc, PT                           ; // x         0  ALU
/*4cb6db30*/  IMAD.HI.U32 R10, R9, c[0x0][0x1f4], R8                     ; // x         0  FMA
/*4cb6db40*/  SHF.R.U32.HI R11, RZ, c[0x0][0x1f8], R10                   ; // x         0  ALU
/*4cb6db50*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*4cb6db60*/  IMAD R9, R7, c[0x0][0x1f0], R9                             ; // x         0  FMA
/*4cb6db70*/  IMAD R14, R9, c[0x0][0x31c], R14                           ; // x         0  FMA
/*4cb6db80*/  IMAD R0, R9, c[0x0][0x320], R0                             ; // x         0  FMA
/*4cb6db90*/  IMAD R2, R9, c[0x0][0x324], R2                             ; // x         0  FMA
/*4cb6dba0*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6dbb0*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*4cb6dbc0*/  ISETP.NE.AND P0, PT, R3, 0xd, PT                           ; // x         0  ALU
/*4cb6dbd0*/  IMAD.HI.U32 R8, R11, c[0x0][0x200], R10                    ; // x         0  FMA
/*4cb6dbe0*/  SHF.R.U32.HI R9, RZ, c[0x0][0x204], R8                     ; // x         0  ALU
/*4cb6dbf0*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*4cb6dc00*/  IMAD R11, R7, c[0x0][0x1fc], R11                           ; // x         0  FMA
/*4cb6dc10*/  IMAD R14, R11, c[0x0][0x328], R14                          ; // x         0  FMA
/*4cb6dc20*/  IMAD R0, R11, c[0x0][0x32c], R0                            ; // x         0  FMA
/*4cb6dc30*/  IMAD R2, R11, c[0x0][0x330], R2                            ; // x         0  FMA
/*4cb6dc40*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6dc50*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*4cb6dc60*/  ISETP.NE.AND P0, PT, R3, 0xe, PT                           ; // x         0  ALU
/*4cb6dc70*/  IMAD.HI.U32 R10, R9, c[0x0][0x20c], R8                     ; // x         0  FMA
/*4cb6dc80*/  SHF.R.U32.HI R11, RZ, c[0x0][0x210], R10                   ; // x         0  ALU
/*4cb6dc90*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*4cb6dca0*/  IMAD R9, R7, c[0x0][0x208], R9                             ; // x         0  FMA
/*4cb6dcb0*/  IMAD R14, R9, c[0x0][0x334], R14                           ; // x         0  FMA
/*4cb6dcc0*/  IMAD R0, R9, c[0x0][0x338], R0                             ; // x         0  FMA
/*4cb6dcd0*/  IMAD R2, R9, c[0x0][0x33c], R2                             ; // x         0  FMA
/*4cb6dce0*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6dcf0*/  IMAD.MOV.U32 R10, RZ, RZ, RZ                               ; // x         0  FMA
/*4cb6dd00*/  ISETP.NE.AND P0, PT, R3, 0xf, PT                           ; // x         0  ALU
/*4cb6dd10*/  IMAD.HI.U32 R8, R11, c[0x0][0x218], R10                    ; // x         0  FMA
/*4cb6dd20*/  SHF.R.U32.HI R9, RZ, c[0x0][0x21c], R8                     ; // x         0  ALU
/*4cb6dd30*/  IADD3 R7, -R9, RZ, RZ                                      ; // x         0  ALU
/*4cb6dd40*/  IMAD R11, R7, c[0x0][0x214], R11                           ; // x         0  FMA
/*4cb6dd50*/  IMAD R14, R11, c[0x0][0x340], R14                          ; // x         0  FMA
/*4cb6dd60*/  IMAD R0, R11, c[0x0][0x344], R0                            ; // x         0  FMA
/*4cb6dd70*/  IMAD R2, R11, c[0x0][0x348], R2                            ; // x         0  FMA
/*4cb6dd80*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6dd90*/  IMAD.MOV.U32 R8, RZ, RZ, RZ                                ; // x         0  FMA
/*4cb6dda0*/  ISETP.NE.AND P0, PT, R3, 0x10, PT                          ; // x         0  ALU
/*4cb6ddb0*/  IMAD.HI.U32 R10, R9, c[0x0][0x224], R8                     ; // x         0  FMA
/*4cb6ddc0*/  SHF.R.U32.HI R11, RZ, c[0x0][0x228], R10                   ; // x         0  ALU
/*4cb6ddd0*/  IADD3 R7, -R11, RZ, RZ                                     ; // x         0  ALU
/*4cb6dde0*/  IMAD R9, R7, c[0x0][0x220], R9                             ; // x         0  FMA
/*4cb6ddf0*/  IMAD R14, R9, c[0x0][0x34c], R14                           ; // x         0  FMA
/*4cb6de00*/  IMAD R0, R9, c[0x0][0x350], R0                             ; // x         0  FMA
/*4cb6de10*/  IMAD R2, R9, c[0x0][0x354], R2                             ; // x         0  FMA
/*4cb6de20*/  @!P0  BRA 0x24cb6e3a0                                      ; // x         0  branch
/*4cb6de30*/  HFMA2.MMA R10, -RZ, RZ, 0, 0                               ; // x         0  FMA
/*4cb6de40*/  ISETP.NE.AND P0, PT, R3, 0x11, PT                          ; // x         0  ALU
/*4cb6de50*/  IMAD.HI.U32 R8, R11, c[0x0][0x230], R10                    ; // x         0  FMA
/*4cb6de60*/  SHF.R.U32.HI R9, RZ, c[0x0][0x234], R8                     ; // x         0  ALU
/*4cb6de70*/  IMAD.MOV R7, RZ, RZ, -R9                                   ; // x         0  FMA
```

(first 200 of 864 instructions shown; the rest are in the full dump)
