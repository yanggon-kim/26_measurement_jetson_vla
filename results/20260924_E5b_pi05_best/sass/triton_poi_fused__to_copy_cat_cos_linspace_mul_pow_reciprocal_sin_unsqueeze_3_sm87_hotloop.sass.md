# Hot loop — `triton_poi_fused__to_copy_cat_cos_linspace_mul_pow_reciprocal_sin_unsqueeze_3` (sm87)

PC range 0x2279050-0x2279210; backward BRA with the largest range (no per-PC counts available). Full dump: `raw/triton_poi_fused__to_copy_cat_cos_linspace_mul_pow_reciprocal_sin_unsqueeze_3_sm87.sass`; per-PC counts: `raw/triton_poi_fused__to_copy_cat_cos_linspace_mul_pow_reciprocal_sin_unsqueeze_3_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*02279050*/  IMAD.MOV.U32 R34, RZ, RZ, R12                              ; // x         0  FMA
/*02279060*/  ULDC.64 UR10, c[0x0][0x118]                                ; // x         0  Uniform
/*02279070*/  IMAD.MOV.U32 R35, RZ, RZ, R15                              ; // x         0  FMA
/*02279080*/  LDG.E.64.CONSTANT R34, [R34.64]                            ; // x         0  global load (DRAM/L2: weights or activations)
/*02279090*/  IMAD.WIDE.U32 R32, P5, R34, R10, R32                       ; // x         0  FMA
/*022790a0*/  IMAD R25, R34, R13, RZ                                     ; // x         0  FMA
/*022790b0*/  IMAD R26, R35, R10, RZ                                     ; // x         0  FMA
/*022790c0*/  IMAD R27, R35, R13, RZ                                     ; // x         0  FMA
/*022790d0*/  IADD3 R25, P0, R25, R33, RZ                                ; // x         0  ALU
/*022790e0*/  IADD3 R33, P4, R26, R25, RZ                                ; // x         0  ALU
/*022790f0*/  IMAD.HI.U32 R25, R34, R13, RZ                              ; // x         0  FMA
/*02279100*/  IMAD.X R25, RZ, RZ, R25, P5                                ; // x         0  FMA
/*02279110*/  IADD3 R24, P5, R24, -0x1, RZ                               ; // x         0  ALU
/*02279120*/  IMAD.HI.U32 R26, R35, R10, RZ                              ; // x         0  FMA
/*02279130*/  STL.64 [R8], R32                                           ; // x         0  LSU / Store
/*02279140*/  IADD3.X R11, R11, -0x1, RZ, P5, !PT                        ; // x         0  ALU
/*02279150*/  ISETP.NE.U32.AND P5, PT, R24, RZ, PT                       ; // x         0  ALU
/*02279160*/  IADD3.X R26, P0, R26, R25, RZ, P0, !PT                     ; // x         0  ALU
/*02279170*/  IMAD.HI.U32 R25, R35, R13, RZ                              ; // x         0  FMA
/*02279180*/  ISETP.NE.AND.EX P5, PT, R11, RZ, PT, P5                    ; // x         0  ALU
/*02279190*/  IADD3.X R26, P4, R27, R26, RZ, P4, !PT                     ; // x         0  ALU
/*022791a0*/  IMAD.X R25, RZ, RZ, R25, P0                                ; // x         0  FMA
/*022791b0*/  IADD3 R12, P0, R12, 0x8, RZ                                ; // x         0  ALU
/*022791c0*/  IADD3 R8, R8, 0x8, RZ                                      ; // x         0  ALU
/*022791d0*/  IMAD.X R27, RZ, RZ, R25, P4                                ; // x         0  FMA
/*022791e0*/  IMAD.X R15, RZ, RZ, R15, P0                                ; // x         0  FMA
/*022791f0*/  IMAD.MOV.U32 R32, RZ, RZ, R26                              ; // x         0  FMA
/*02279200*/  IMAD.MOV.U32 R33, RZ, RZ, R27                              ; // x         0  FMA
/*02279210*/  @P5   BRA 0x202279050                                      ; // x         0  branch
```
