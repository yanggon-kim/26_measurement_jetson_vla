# Hot loop — `triton_poi_fused_cat_5` (sm87)

PC range 0x2293b00-0x2293e70; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/triton_poi_fused_cat_5_sm87.sass`; per-PC counts: `raw/triton_poi_fused_cat_5_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*02293b00*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x      4072  ALU
/*02293b10*/  @P0   BRA 0x202293c40                                      ; // x      4072  branch
/*02293b20*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*02293b30*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*02293b40*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*02293b50*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*02293b60*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*02293b70*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*02293b80*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*02293b90*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*02293ba0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*02293bb0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*02293bc0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*02293bd0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*02293be0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*02293bf0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*02293c00*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*02293c10*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*02293c20*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*02293c30*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*02293c40*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]                      ; // x      4072  FMA
/*02293c50*/  S2R R0, SR_CTAID.X                                         ; // x      4072  Branch
/*02293c60*/  IMAD.MOV.U32 R7, RZ, RZ, 0x4                               ; // x      4072  FMA
/*02293c70*/  CS2R R8, SRZ                                               ; // x      4072  Branch
/*02293c80*/  CS2R R10, SRZ                                              ; // x      4072  Branch
/*02293c90*/  S2R R3, SR_TID.X                                           ; // x      4072  Branch
/*02293ca0*/  ULDC.64 UR4, c[0x0][0x118]                                 ; // x      4072  Uniform
/*02293cb0*/  IMAD.SHL.U32 R0, R0, 0x200, RZ                             ; // x      4072  FMA
/*02293cc0*/  IMAD.SHL.U32 R3, R3, 0x2, RZ                               ; // x      4072  FMA
/*02293cd0*/  LOP3.LUT R6, R0, 0x1fe, R3, 0xf8, !PT                      ; // x      4072  ALU
/*02293ce0*/  IADD3 R0, R0, -0x3c800, RZ                                 ; // x      4072  ALU
/*02293cf0*/  ISETP.GE.AND P0, PT, R6, 0x3c800, PT                       ; // x      4072  ALU
/*02293d00*/  IMAD.WIDE R2, R6, R7, c[0x0][0x160]                        ; // x      4072  FMA
/*02293d10*/  ISETP.GE.U32.AND P1, PT, R0, 0x3200, PT                    ; // x      4072  ALU
/*02293d20*/  IADD3 R4, R6, -0x3c800, RZ                                 ; // x      4072  ALU
/*02293d30*/  IMAD.WIDE R4, R4, R7, c[0x0][0x168]                        ; // x      4072  FMA
/*02293d40*/  @!P0  LDG.E.64 R8, [R2.64]                                 ; // x      4072  global load (DRAM/L2: weights or activations)
/*02293d50*/  @!P1  LDG.E.64 R10, [R4.64]                                ; // x      4072  global load (DRAM/L2: weights or activations)
/*02293d60*/  ISETP.GE.AND P1, PT, R6, 0x3fa00, PT                       ; // x      4072  ALU
/*02293d70*/  IMAD.WIDE R6, R6, R7, c[0x0][0x170]                        ; // x      4072  FMA
/*02293d80*/  SEL R8, R8, R10, !P0                                       ; // x      4072  ALU
/*02293d90*/  SEL R9, R9, R11, !P0                                       ; // x      4072  ALU
/*02293da0*/  @P1   EXIT                                                 ; // x      4072  Branch
/*02293db0*/  STG.E.64 [R6.64], R8                                       ; // x      4072  global store
/*02293dc0*/  EXIT                                                       ; // x      4072  Branch
/*02293dd0*/  BRA 0x202293dd0                                            ; // x         0  branch
/*02293de0*/  NOP                                                        ; // x         0  Branch
/*02293df0*/  NOP                                                        ; // x         0  Branch
/*02293e00*/  NOP                                                        ; // x         0  Branch
/*02293e10*/  NOP                                                        ; // x         0  Branch
/*02293e20*/  NOP                                                        ; // x         0  Branch
/*02293e30*/  NOP                                                        ; // x         0  Branch
/*02293e40*/  NOP                                                        ; // x         0  Branch
/*02293e50*/  NOP                                                        ; // x         0  Branch
/*02293e60*/  NOP                                                        ; // x         0  Branch
/*02293e70*/  NOP                                                        ; // x         0  Branch
```
