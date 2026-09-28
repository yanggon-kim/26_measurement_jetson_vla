# Hot loop — `triton_poi_fused_clone_6` (sm87)

PC range 0x2294700-0x2294a70; no backward BRA: fully unrolled, the inner range is the whole kernel (§3). Full dump: `raw/triton_poi_fused_clone_6_sm87.sass`; per-PC counts: `raw/triton_poi_fused_clone_6_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*02294700*/  ISETP.NE.U32.AND P0, PT, RZ, UR2, PT                       ; // x      1600  ALU
/*02294710*/  @P0   BRA 0x202294840                                      ; // x      1600  branch
/*02294720*/  BMOV.32 B0, 0xffffffff                                     ; // x         0  Other
/*02294730*/  BMOV.32.CLEAR B1, B0                                       ; // x         0  Other
/*02294740*/  BMOV.32.CLEAR B2, B1                                       ; // x         0  Other
/*02294750*/  BMOV.32.CLEAR B3, B2                                       ; // x         0  Other
/*02294760*/  BMOV.32.CLEAR B4, B3                                       ; // x         0  Other
/*02294770*/  BMOV.32.CLEAR B5, B4                                       ; // x         0  Other
/*02294780*/  BMOV.32.CLEAR B6, B5                                       ; // x         0  Other
/*02294790*/  BMOV.32.CLEAR B7, B6                                       ; // x         0  Other
/*022947a0*/  BMOV.32.CLEAR B8, B7                                       ; // x         0  Other
/*022947b0*/  BMOV.32.CLEAR B9, B8                                       ; // x         0  Other
/*022947c0*/  BMOV.32.CLEAR B10, B9                                      ; // x         0  Other
/*022947d0*/  BMOV.32.CLEAR B11, B10                                     ; // x         0  Other
/*022947e0*/  BMOV.32.CLEAR B12, B11                                     ; // x         0  Other
/*022947f0*/  BMOV.32.CLEAR B13, B12                                     ; // x         0  Other
/*02294800*/  BMOV.32.CLEAR B14, B13                                     ; // x         0  Other
/*02294810*/  BMOV.32.CLEAR B15, B14                                     ; // x         0  Other
/*02294820*/  BMOV.32 B15, 0x0                                           ; // x         0  Other
/*02294830*/  UMOV UR2, 0x1                                              ; // x         0  Uniform
/*02294840*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]                      ; // x      1600  FMA
/*02294850*/  S2R R0, SR_TID.X                                           ; // x      1600  Branch
/*02294860*/  IMAD.MOV.U32 R7, RZ, RZ, 0x4                               ; // x      1600  FMA
/*02294870*/  ULDC.64 UR4, c[0x0][0x118]                                 ; // x      1600  Uniform
/*02294880*/  S2R R5, SR_CTAID.X                                         ; // x      1600  Branch
/*02294890*/  IMAD.SHL.U32 R0, R0, 0x2, RZ                               ; // x      1600  FMA
/*022948a0*/  SHF.R.S32.HI R3, RZ, 0x16, R5                              ; // x      1600  ALU
/*022948b0*/  LOP3.LUT R0, R0, 0x1fe, RZ, 0xc0, !PT                      ; // x      1600  ALU
/*022948c0*/  SGXT R3, R3, 0x1                                           ; // x      1600  ALU
/*022948d0*/  IMAD R0, R5, 0x200, R0                                     ; // x      1600  FMA
/*022948e0*/  LEA.HI R2, R3, R0, RZ, 0x8                                 ; // x      1600  ALU
/*022948f0*/  LEA.HI R5, R3, R0, RZ, 0xb                                 ; // x      1600  ALU
/*02294900*/  SHF.R.S32.HI R4, RZ, 0x8, R2                               ; // x      1600  ALU
/*02294910*/  LOP3.LUT R3, R2, 0xffffff00, RZ, 0xc0, !PT                 ; // x      1600  ALU
/*02294920*/  LEA.HI R2, R4, R4, RZ, 0x3                                 ; // x      1600  ALU
/*02294930*/  SHF.R.S32.HI R6, RZ, 0xb, R5                               ; // x      1600  ALU
/*02294940*/  IMAD.IADD R3, R0, 0x1, -R3                                 ; // x      1600  FMA
/*02294950*/  LOP3.LUT R5, R2, 0x7ffff8, RZ, 0xc0, !PT                   ; // x      1600  ALU
/*02294960*/  IMAD R6, R6, 0x100, R3                                     ; // x      1600  FMA
/*02294970*/  IMAD.IADD R5, R4, 0x1, -R5                                 ; // x      1600  FMA
/*02294980*/  IMAD R2, R5, 0x3200, R6                                    ; // x      1600  FMA
/*02294990*/  IMAD.WIDE R2, R2, R7, c[0x0][0x160]                        ; // x      1600  FMA
/*022949a0*/  LDG.E.64 R2, [R2.64]                                       ; // x      1600  global load (DRAM/L2: weights or activations)
/*022949b0*/  IMAD.WIDE R4, R0, R7, c[0x0][0x168]                        ; // x      1600  FMA
/*022949c0*/  STG.E.64 [R4.64], R2                                       ; // x      1600  global store
/*022949d0*/  EXIT                                                       ; // x      1600  Branch
/*022949e0*/  BRA 0x2022949e0                                            ; // x         0  branch
/*022949f0*/  NOP                                                        ; // x         0  Branch
/*02294a00*/  NOP                                                        ; // x         0  Branch
/*02294a10*/  NOP                                                        ; // x         0  Branch
/*02294a20*/  NOP                                                        ; // x         0  Branch
/*02294a30*/  NOP                                                        ; // x         0  Branch
/*02294a40*/  NOP                                                        ; // x         0  Branch
/*02294a50*/  NOP                                                        ; // x         0  Branch
/*02294a60*/  NOP                                                        ; // x         0  Branch
/*02294a70*/  NOP                                                        ; // x         0  Branch
```
