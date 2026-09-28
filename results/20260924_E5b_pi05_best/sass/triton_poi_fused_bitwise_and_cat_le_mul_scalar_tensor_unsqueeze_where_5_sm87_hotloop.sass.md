# Hot loop — `triton_poi_fused_bitwise_and_cat_le_mul_scalar_tensor_unsqueeze_where_5` (sm87)

PC range 0x227af00-0x227b4f0; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/triton_poi_fused_bitwise_and_cat_le_mul_scalar_tensor_unsqueeze_where_5_sm87.sass`; per-PC counts: `raw/triton_poi_fused_bitwise_and_cat_le_mul_scalar_tensor_unsqueeze_where_5_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*0227af00*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x       796  ALU
/*0227af10*/  @P0   BRA 0x20227b040                                      ; // x       796  branch
/*0227af20*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*0227af30*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*0227af40*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*0227af50*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*0227af60*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*0227af70*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*0227af80*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*0227af90*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*0227afa0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*0227afb0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*0227afc0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*0227afd0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*0227afe0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*0227aff0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*0227b000*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*0227b010*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*0227b020*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*0227b030*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*0227b040*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]                      ; // x       796  FMA
/*0227b050*/  S2R R0, SR_TID.X                                           ; // x       796  Branch
/*0227b060*/  IMAD.MOV.U32 R2, RZ, RZ, RZ                                ; // x       796  FMA
/*0227b070*/  PRMT R14, RZ, 0x7610, R14                                  ; // x       796  ALU
/*0227b080*/  IMAD.MOV.U32 R19, RZ, RZ, 0x4                              ; // x       796  FMA
/*0227b090*/  S2R R3, SR_CTAID.X                                         ; // x       796  Branch
/*0227b0a0*/  IMAD.SHL.U32 R0, R0, 0x2, RZ                               ; // x       796  FMA
/*0227b0b0*/  LOP3.LUT R0, R0, 0xfe, RZ, 0xc0, !PT                       ; // x       796  ALU
/*0227b0c0*/  IMAD R3, R3, 0x100, R0                                     ; // x       796  FMA
/*0227b0d0*/  IMAD.HI R0, R3, -0x7f3ede4d, R2                            ; // x       796  FMA
/*0227b0e0*/  SHF.R.U32.HI R5, RZ, 0x1f, R0                              ; // x       796  ALU
/*0227b0f0*/  LEA.HI.SX32 R2, R0, R5, 0x17                               ; // x       796  ALU
/*0227b100*/  IMAD R0, R2, -0x3fa, R3                                    ; // x       796  FMA
/*0227b110*/  ISETP.GE.AND P1, PT, R0, 0x3c8, PT                         ; // x       796  ALU
/*0227b120*/  ISETP.GT.AND P0, PT, R0, 0x3c7, PT                         ; // x       796  ALU
/*0227b130*/  ISETP.LT.AND P2, PT, R3, 0xc6d4, !P1                       ; // x       796  ALU
/*0227b140*/  ISETP.LT.AND P3, PT, R3, 0xc6d4, P0                        ; // x       796  ALU
/*0227b150*/  IADD3 R8, P4, R0, c[0x0][0x160], RZ                        ; // x       796  ALU
/*0227b160*/  LEA.HI.X.SX32 R9, R0, c[0x0][0x164], 0x1, P4               ; // x       796  ALU
/*0227b170*/  @P2   IMAD.MOV.U32 R4, RZ, RZ, 0x0                         ; // x       796  FMA
/*0227b180*/  @P2   IMAD.MOV.U32 R5, RZ, RZ, 0x14f00000                  ; // x       796  FMA
/*0227b190*/  @P2   R2UR UR4, R4                                         ; // x       796  Branch
/*0227b1a0*/  @P3   IMAD.MOV.U32 R6, RZ, RZ, 0x0                         ; // x       796  FMA
/*0227b1b0*/  @P3   IMAD.MOV.U32 R7, RZ, RZ, 0x14f00000                  ; // x       796  FMA
/*0227b1c0*/  @P3   IMAD.MOV.U32 R10, RZ, RZ, 0x0                        ; // x       796  FMA
/*0227b1d0*/  @P3   IMAD.MOV.U32 R11, RZ, RZ, 0x14f00000                 ; // x       796  FMA
/*0227b1e0*/  @P2   R2UR UR5, R5                                         ; // x       796  Branch
/*0227b1f0*/  @P3   IMAD.MOV.U32 R12, RZ, RZ, 0x0                        ; // x       796  FMA
/*0227b200*/  @P3   IMAD.MOV.U32 R13, RZ, RZ, 0x14f00000                 ; // x       796  FMA
/*0227b210*/  @P3   R2UR UR6, R6                                         ; // x       796  Branch
/*0227b220*/  @P3   R2UR UR7, R7                                         ; // x       796  Branch
/*0227b230*/  @P3   R2UR UR8, R10                                        ; // x       796  Branch
/*0227b240*/  @P2   LDG.E.EL.U16 R14desc, [UR4][R8.64]                   ; // x       796  global load (DRAM/L2: weights or activations)
/*0227b250*/  @P3   R2UR UR9, R11                                        ; // x       796  Branch
/*0227b260*/  @P3   R2UR UR10, R12                                       ; // x       796  Branch
/*0227b270*/  @P3   R2UR UR11, R13                                       ; // x       796  Branch
/*0227b280*/  IMAD.WIDE R6, R2, R19, c[0x0][0x168]                       ; // x       796  FMA
/*0227b290*/  CS2R R16, SRZ                                              ; // x       796  Branch
/*0227b2a0*/  IMAD.MOV.U32 R15, RZ, RZ, RZ                               ; // x       796  FMA
/*0227b2b0*/  IMAD.MOV.U32 R2, RZ, RZ, RZ                                ; // x       796  FMA
/*0227b2c0*/  IMAD.WIDE R4, R0, R19, c[0x0][0x168]                       ; // x       796  FMA
/*0227b2d0*/  @P3   LDG.E.EL R15desc, [UR8][R6.64]                       ; // x       796  global load (DRAM/L2: weights or activations)
/*0227b2e0*/  @P3   LDG.E.EL.64 R16desc, [UR6][R4.64+-0xf20]             ; // x       796  global load (DRAM/L2: weights or activations)
/*0227b2f0*/  @P3   LDG.E.EL R2desc, [UR10][R6.64]                       ; // x       796  global load (DRAM/L2: weights or activations)
/*0227b300*/  ISETP.GE.AND P2, PT, R3, 0xc6d4, PT                        ; // x       796  ALU
/*0227b310*/  LOP3.LUT R0, R14, 0xffff, RZ, 0xc0, !PT                    ; // x       796  ALU
/*0227b320*/  SHF.R.U32.HI R0, RZ, 0x8, R0                               ; // x       796  ALU
/*0227b330*/  PRMT R0, R0, 0x9910, RZ                                    ; // x       796  ALU
/*0227b340*/  FSETP.GTU.AND P3, PT, R16, R15, PT                         ; // x       796  FMA
/*0227b350*/  FSETP.GTU.AND P4, PT, R17, R2, PT                          ; // x       796  FMA
/*0227b360*/  IMAD.MOV.U32 R2, RZ, RZ, R0                                ; // x       796  FMA
/*0227b370*/  SEL R0, RZ, 0xff333332, !P3                                ; // x       796  ALU
/*0227b380*/  SEL R5, RZ, 0xff333332, !P4                                ; // x       796  ALU
/*0227b390*/  LOP3.LUT P3, RZ, R14, 0xff, RZ, 0xc0, !PT                  ; // x       796  ALU
/*0227b3a0*/  ISETP.NE.AND P4, PT, R2, RZ, PT                            ; // x       796  ALU
/*0227b3b0*/  SEL R4, R0, 0xff333332, P0                                 ; // x       796  ALU
/*0227b3c0*/  SEL R5, R5, 0xff333332, P0                                 ; // x       796  ALU
/*0227b3d0*/  IMAD.WIDE R2, R3, R19, c[0x0][0x170]                       ; // x       796  FMA
/*0227b3e0*/  @!P1  SEL R4, R4, RZ, !P3                                  ; // x       796  ALU
/*0227b3f0*/  @!P1  SEL R5, R5, RZ, !P4                                  ; // x       796  ALU
/*0227b400*/  @P2   EXIT                                                 ; // x       796  Branch
/*0227b410*/  ULDC.64 UR4, c[0x0][0x118]                                 ; // x       796  Uniform
/*0227b420*/  STG.E.64 [R2.64], R4                                       ; // x       796  global store
/*0227b430*/  EXIT                                                       ; // x       796  Branch
/*0227b440*/  BRA 0x20227b440                                            ; // x         0  branch
/*0227b450*/  NOP                                                        ; // x         0  Branch
/*0227b460*/  NOP                                                        ; // x         0  Branch
/*0227b470*/  NOP                                                        ; // x         0  Branch
/*0227b480*/  NOP                                                        ; // x         0  Branch
/*0227b490*/  NOP                                                        ; // x         0  Branch
/*0227b4a0*/  NOP                                                        ; // x         0  Branch
/*0227b4b0*/  NOP                                                        ; // x         0  Branch
/*0227b4c0*/  NOP                                                        ; // x         0  Branch
/*0227b4d0*/  NOP                                                        ; // x         0  Branch
/*0227b4e0*/  NOP                                                        ; // x         0  Branch
/*0227b4f0*/  NOP                                                        ; // x         0  Branch
```
