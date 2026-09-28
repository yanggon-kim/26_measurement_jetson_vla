# Hot loop — `triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_2` (sm87)

PC range 0x2285990-0x2285a60; backward BRA with the largest range (no per-PC counts available). Full dump: `raw/triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_2_sm87.sass`; per-PC counts: `raw/triton_poi_fused_add_cat_cos_mul_neg_sin_unsqueeze_2_sm87.pc_exec.csv`.

**Annotation is mechanical** (bench/e5b_static.py): each comment names the opcode's §4 family and, for memory ops, the memory space its opcode addresses. Which tensor a load reads (weights vs activations) is not in the SASS; see the phase notes in the pseudocode file.

```
/*02285990*/  IADD3 R6, P1, R8, c[0x4][0x8], RZ                          ; // x         0  ALU
/*022859a0*/  IADD3.X R7, R9, c[0x4][0xc], RZ, P1, !PT                   ; // x         0  ALU
/*022859b0*/  LDG.E.CONSTANT R7, [R6.64]                                 ; // x         0  global load (DRAM/L2: weights or activations)
/*022859c0*/  IADD3 R8, P1, R8, 0x4, RZ                                  ; // x         0  ALU
/*022859d0*/  IMAD.MOV.U32 R4, RZ, RZ, R2                                ; // x         0  FMA
/*022859e0*/  IMAD.MOV.U32 R5, RZ, RZ, RZ                                ; // x         0  FMA
/*022859f0*/  IMAD.X R9, RZ, RZ, R9, P1                                  ; // x         0  FMA
/*02285a00*/  ISETP.NE.U32.AND P1, PT, R8, 0x18, PT                      ; // x         0  ALU
/*02285a10*/  ISETP.NE.AND.EX P1, PT, R9, RZ, PT, P1                     ; // x         0  ALU
/*02285a20*/  IMAD.WIDE.U32 R4, R10, R7, R4                              ; // x         0  FMA
/*02285a30*/  IMAD.MOV.U32 R2, RZ, RZ, R5                                ; // x         0  FMA
/*02285a40*/  STL [R3], R4                                               ; // x         0  LSU / Store
/*02285a50*/  IADD3 R3, R3, 0x4, RZ                                      ; // x         0  ALU
/*02285a60*/  @P1   BRA 0x202285990                                      ; // x         0  branch
```
