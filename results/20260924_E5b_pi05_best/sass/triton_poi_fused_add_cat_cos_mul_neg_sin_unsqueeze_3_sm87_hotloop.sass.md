# Hot loop — `triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_3` (sm87)

PC range 0x228c7e0-0x228c8b0; backward BRA with the largest range (no per-PC counts available). Full dump: `raw/triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_3_sm87.sass`; per-PC counts: `raw/triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_3_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*0228c7e0*/  IADD3 R14, P1, R13, c[0x4][0x8], RZ                        ; // x         0  ALU
/*0228c7f0*/  IADD3.X R15, R19, c[0x4][0xc], RZ, P1, !PT                 ; // x         0  ALU
/*0228c800*/  LDG.E.CONSTANT R15, [R14.64]                               ; // x         0  global load (DRAM/L2: weights or activations)
/*0228c810*/  IADD3 R13, P2, R13, 0x4, RZ                                ; // x         0  ALU
/*0228c820*/  IMAD.MOV.U32 R10, RZ, RZ, R2                               ; // x         0  FMA
/*0228c830*/  IMAD.MOV.U32 R11, RZ, RZ, RZ                               ; // x         0  FMA
/*0228c840*/  ISETP.NE.U32.AND P1, PT, R13, 0x18, PT                     ; // x         0  ALU
/*0228c850*/  IMAD.X R19, RZ, RZ, R19, P2                                ; // x         0  FMA
/*0228c860*/  ISETP.NE.AND.EX P1, PT, R19, RZ, PT, P1                    ; // x         0  ALU
/*0228c870*/  IMAD.WIDE.U32 R10, R20, R15, R10                           ; // x         0  FMA
/*0228c880*/  STL [R3], R10                                              ; // x         0  LSU / Store
/*0228c890*/  MOV R2, R11                                                ; // x         0  ALU
/*0228c8a0*/  IADD3 R3, R3, 0x4, RZ                                      ; // x         0  ALU
/*0228c8b0*/  @P1   BRA 0x20228c7e0                                      ; // x         0  branch
```
