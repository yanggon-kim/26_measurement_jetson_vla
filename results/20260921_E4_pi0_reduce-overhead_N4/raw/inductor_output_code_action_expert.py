# AOT ID: ['6_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/4a/c4a4m222tdmzr2q25jm2d3nd3m4lzbzdysm7wfr5vhdgbq2qcr6v.py
# Topologically Sorted Source Nodes: [hidden_states, float_5, square, var, add, rsqrt, normed_inputs, add_1, normed, hidden_states_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   float_5 => convert_element_type_5
#   hidden_states => convert_element_type
#   hidden_states_1 => convert_element_type_6
#   normed => mul_3
#   normed_inputs => mul_2
#   rsqrt => rsqrt
#   square => pow_1
#   var => mean
# Graph fragment:
#   %convert_element_type : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_5, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg6_1, 1.0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %add_1), kwargs = {})
#   %convert_element_type_6 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 421888}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel):
    xnumel = 51
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp13 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = 1024.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp2 * tmp11
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 1024*x0), tmp17, None)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/hc/chcueuumtbfquphalidpfa7fwbw2krzvjhdm274vfbcfwsb2f7br.py
# Topologically Sorted Source Nodes: [float_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_1, torch.float32), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/kg/ckgfhlwznkrf3lmhd4uu43fv24gbsdfpzsmt2bkas5dynnlxffsg.py
# Topologically Sorted Source Nodes: [position_ids_expanded], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   position_ids_expanded => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_2, torch.float32), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 816}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/ll/cll7zwhlst7336mwrjqxffq2mlvdb73ceirhofq7pepiywp4ltrz.py
# Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, cos_3, sin, sin_1, sin_2, sin_3, mul_6, neg_1, cat_2, mul_7, k_embed, keys], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.sin, aten.neg, aten.add]
# Source node to ATen node mapping:
#   cat_2 => cat_1
#   cos => cos
#   cos_1 => mul
#   cos_2 => convert_element_type_3
#   cos_3 => unsqueeze_4
#   emb => clone, expand_3, unsqueeze_3, view_3
#   k_embed => add_3
#   keys => cat_2
#   mul_6 => mul_6
#   mul_7 => mul_7
#   neg_1 => neg_1
#   sin => sin
#   sin_1 => mul_1
#   sin_2 => convert_element_type_4
#   sin_3 => unsqueeze_5
# Graph fragment:
#   %unsqueeze_3 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_3 : [num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_3, [1, 51, 2, 128]), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 51, 256]), kwargs = {})
#   %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_3 : [num_users=18] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %unsqueeze_4 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 1), kwargs = {})
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_4 : [num_users=18] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_4, 1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, %unsqueeze_4), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_4,), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_3], -1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_5), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%arg1_1, %add_3], -2), kwargs = {})
triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3 = async_compile.triton('triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 887808, 'x': 2663424}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 867
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 816, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + 256*(y0)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 867, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x1 + 256*((-816) + y0)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (51*((x1 % 128)) + ((-816) + y0)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl_math.cos(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 * tmp14
    tmp16 = tl.broadcast_to(x1, [YBLOCK, XBLOCK])
    tmp17 = tl.full([1, 1], 0, tl.int64)
    tmp18 = tmp16 >= tmp17
    tmp19 = tl.full([1, 1], 128, tl.int64)
    tmp20 = tmp16 < tmp19
    tmp21 = tmp20 & tmp6
    tmp22 = tl.load(in_ptr1 + (128 + 256*((-816) + y0) + (x1)), tmp21 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = -tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp21, tmp23, tmp24)
    tmp26 = tmp16 >= tmp19
    tmp27 = tl.full([1, 1], 256, tl.int64)
    tmp28 = tmp16 < tmp27
    tmp29 = tmp26 & tmp6
    tmp30 = tl.load(in_ptr1 + (256*((-816) + y0) + ((-128) + x1)), tmp29 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.where(tmp20, tmp25, tmp30)
    tmp32 = tl_math.sin(tmp10)
    tmp33 = tmp32 * tmp12
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp31 * tmp34
    tmp36 = tmp15 + tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp6, tmp36, tmp37)
    tmp39 = tl.where(tmp4, tmp5, tmp38)
    tl.store(out_ptr0 + (x1 + 256*y0), tmp39, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/6d/c6dvenyapz75wlu5d36qftxxl6hv4fnbpnruvjdfhfqao24bwhjw.py
# Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, cos_3, mul_4, neg, cat_1, sin, sin_1, sin_2, sin_3, mul_5, q_embed], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.neg, aten.sin, aten.add]
# Source node to ATen node mapping:
#   cat_1 => cat
#   cos => cos
#   cos_1 => mul
#   cos_2 => convert_element_type_3
#   cos_3 => unsqueeze_4
#   emb => clone, expand_3, unsqueeze_3, view_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   neg => neg
#   q_embed => add_2
#   sin => sin
#   sin_1 => mul_1
#   sin_2 => convert_element_type_4
#   sin_3 => unsqueeze_5
# Graph fragment:
#   %unsqueeze_3 : [num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_3 : [num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_3, [1, 51, 2, 128]), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 51, 256]), kwargs = {})
#   %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_3 : [num_users=18] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %unsqueeze_4 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_4), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_1], -1), kwargs = {})
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_4 : [num_users=18] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_4, 1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_5), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4 = async_compile.triton('triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 52224, 'x': 1044480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 51
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex
    x1 = (xindex % 256)
    x2 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x3 + 2048*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0 + 51*((x1 % 128))), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl_math.cos(tmp1)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp7 = x1
    tmp8 = tl.full([1, 1], 0, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.full([1, 1], 128, tl.int64)
    tmp11 = tmp7 < tmp10
    tmp12 = tl.load(in_ptr0 + (128 + 256*x2 + 2048*y0 + (x1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = -tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp11, tmp13, tmp14)
    tmp16 = tmp7 >= tmp10
    tmp17 = tl.full([1, 1], 256, tl.int64)
    tmp18 = tmp7 < tmp17
    tmp19 = tl.load(in_ptr0 + (256*x2 + 2048*y0 + ((-128) + x1)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp11, tmp15, tmp19)
    tmp21 = tl_math.sin(tmp1)
    tmp22 = tmp21 * tmp3
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 * tmp23
    tmp25 = tmp6 + tmp24
    tl.store(out_ptr0 + (x1 + 256*y0 + 13056*x2), tmp25, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/4s/c4sxwfm5zk7v57ujr67oz5u3d43z7d5jhzhxvix4mwxeczxlem6l.py
# Topologically Sorted Source Nodes: [attn_weights, attn_weights_1, , sub, exp, softmax, attn_weights_2], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default_17
#   attn_weights => mul_8
#   attn_weights_1 => add_4
#   attn_weights_2 => convert_element_type_15
#   exp => exp_default_17
#   softmax => div
#   sub => sub_tensor_17
# Graph fragment:
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, 0.0625), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %arg3_1), kwargs = {})
#   %prepare_softmax_online_default_17 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_4, -1), kwargs = {})
#   %sub_tensor_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_34), kwargs = {})
#   %exp_default_17 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_17,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_17, %getitem_35), kwargs = {})
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5 = async_compile.triton('triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 2299284}}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 408
    XBLOCK: tl.constexpr = 1
    r0_numel = 867
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 51)
    x1 = xindex // 51
    tmp0 = tl.load(in_ptr0 + (r0_2 + 867*x3), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_2 + 867*x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.0625
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.broadcast_to(tmp5, [R0_BLOCK])
    tmp8 = tl.broadcast_to(tmp6, [R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, float("-inf"))
    tmp11 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp10, 0))
    tmp12 = tmp6 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = tl.where(r0_mask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tmp5 - tmp11
    tmp19 = tl_math.exp(tmp18)
    tmp20 = (tmp19 / tmp17)
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr2 + (r0_2 + 867*x0 + 44224*x1), tmp21, r0_mask)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/oy/coydpugccb3w3r32rmatrcylvo6oiressoyrizflk3mzihlnzgf5.py
# Topologically Sorted Source Nodes: [values], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   values => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%arg9_1, %permute_6], -2), kwargs = {})
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1775616}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 221952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 256
    x0 = (xindex % 256)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 816, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256*(x1)), tmp4 & xmask, other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 867, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 256*((-816) + x1)), tmp6 & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/3k/c3knlu5rei3mnm4wrdj36uwkqqjqirvscn6tcgd7lsbjw62dv6nx.py
# Topologically Sorted Source Nodes: [attn_output_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_output_1 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 626688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 104448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = ((xindex // 256) % 8)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*x2 + 13056*x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/d5/cd5ku5bfa7c7jpyamgarbznhxw7lzt2engkp4jjp5jlz2l2potsp.py
# Topologically Sorted Source Nodes: [hidden_states, hidden_states_4, float_7, square_1, var_1, add_6, rsqrt_1, normed_inputs_1, add_7, normed_1, hidden_states_5], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_6 => add_6
#   add_7 => add_7
#   float_7 => convert_element_type_20
#   hidden_states => convert_element_type
#   hidden_states_4 => add_5
#   hidden_states_5 => convert_element_type_21
#   normed_1 => mul_10
#   normed_inputs_1 => mul_9
#   rsqrt_1 => rsqrt_1
#   square_1 => pow_2
#   var_1 => mean_1
# Graph fragment:
#   %convert_element_type : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, %view_23), kwargs = {})
#   %convert_element_type_20 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_20, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %rsqrt_1), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg11_1, 1.0), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %add_7), kwargs = {})
#   %convert_element_type_21 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 526336}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel):
    xnumel = 51
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [R0_BLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = 1024.0
    tmp10 = (tmp8 / tmp9)
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = tmp4 * tmp13
    tmp16 = 1.0
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 1024*x0), tmp19, None)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/mo/cmokrltegqq3ptuyi6qmkdv3knm2woka472n67rx5co5dkv4u6tz.py
# Topologically Sorted Source Nodes: [gelu, mul_11], Original ATen: [aten.gelu, aten.mul]
# Source node to ATen node mapping:
#   gelu => add_8, add_9, convert_element_type_24, convert_element_type_25, mul_11, mul_12, mul_13, mul_14, mul_15, mul_16, tanh
#   mul_11 => mul_17
# Graph fragment:
#   %convert_element_type_24 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_25, torch.float32), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, 0.5), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %convert_element_type_24), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %convert_element_type_24), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, 0.044715), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_24, %mul_13), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 0.7978845608028654), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_14,), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %add_9), kwargs = {})
#   %convert_element_type_25 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_16, torch.bfloat16), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_25, %view_27), kwargs = {})
triton_poi_fused_gelu_mul_9 = async_compile.triton('triton_poi_fused_gelu_mul_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_mul_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1671168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_mul_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 208896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp16 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = 0.044715
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = 0.7978845608028654
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/hu/chut755es4teg5xt2u6a6u77dyevjopruee5laawcgc3rv4z4guw.py
# Topologically Sorted Source Nodes: [hidden_states, hidden_states_4, hidden_states_6, float_9, square_2, var_2, add_9, rsqrt_2, normed_inputs_2, add_10, normed_2, hidden_states_7], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_10 => add_12
#   add_9 => add_11
#   float_9 => convert_element_type_30
#   hidden_states => convert_element_type
#   hidden_states_4 => add_5
#   hidden_states_6 => add_10
#   hidden_states_7 => convert_element_type_31
#   normed_2 => mul_19
#   normed_inputs_2 => mul_18
#   rsqrt_2 => rsqrt_2
#   square_2 => pow_3
#   var_2 => mean_2
# Graph fragment:
#   %convert_element_type : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, %view_23), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_29), kwargs = {})
#   %convert_element_type_30 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.float32), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_30, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %rsqrt_2), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg15_1, 1.0), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %add_12), kwargs = {})
#   %convert_element_type_31 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_19, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_10 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 630784}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel):
    xnumel = 51
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp17 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [R0_BLOCK])
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = 1024.0
    tmp12 = (tmp10 / tmp11)
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp6 * tmp15
    tmp18 = 1.0
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 1024*x0), tmp21, None)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/qk/cqk7gb6z3mnjeffzgtai4qqwgz2bfrvromymnk2kmfk3xcbkhgv7.py
# Topologically Sorted Source Nodes: [hidden_states, hidden_states_4, hidden_states_6, hidden_states_10, float_11, square_3, var_3, add_15, rsqrt_3, normed_inputs_3, add_16, normed_3, hidden_states_11], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_15 => add_17
#   add_16 => add_18
#   float_11 => convert_element_type_45
#   hidden_states => convert_element_type
#   hidden_states_10 => add_16
#   hidden_states_11 => convert_element_type_46
#   hidden_states_4 => add_5
#   hidden_states_6 => add_10
#   normed_3 => mul_26
#   normed_inputs_3 => mul_25
#   rsqrt_3 => rsqrt_3
#   square_3 => pow_4
#   var_3 => mean_3
# Graph fragment:
#   %convert_element_type : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, %view_23), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_29), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_49), kwargs = {})
#   %convert_element_type_45 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_16, torch.float32), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_45, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %rsqrt_3), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg22_1, 1.0), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %add_18), kwargs = {})
#   %convert_element_type_46 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_26, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 735232}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
    xnumel = 51
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp19 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = 1024.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp8 * tmp17
    tmp20 = 1.0
    tmp21 = tmp19 + tmp20
    tmp22 = tmp18 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 1024*x0), tmp23, None)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/ud/cudmibntnooy5ravjeclzavtsrnazpu35qtkva6ugjpeyuyjj34t.py
# Topologically Sorted Source Nodes: [hidden_states, hidden_states_4, hidden_states_6, hidden_states_10, hidden_states_12, float_13, square_4, var_4, add_18, rsqrt_4, normed_inputs_4, add_19, normed_4, hidden_states_13], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_18 => add_22
#   add_19 => add_23
#   float_13 => convert_element_type_55
#   hidden_states => convert_element_type
#   hidden_states_10 => add_16
#   hidden_states_12 => add_21
#   hidden_states_13 => convert_element_type_56
#   hidden_states_4 => add_5
#   hidden_states_6 => add_10
#   normed_4 => mul_35
#   normed_inputs_4 => mul_34
#   rsqrt_4 => rsqrt_4
#   square_4 => pow_5
#   var_4 => mean_4
# Graph fragment:
#   %convert_element_type : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, %view_23), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_29), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %view_49), kwargs = {})
#   %add_21 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_55), kwargs = {})
#   %convert_element_type_55 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_21, torch.float32), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_55, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %rsqrt_4), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg26_1, 1.0), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %add_23), kwargs = {})
#   %convert_element_type_56 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_35, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 1048576}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
    xnumel = 51
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp2 = tl.load(in_out_ptr0 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp21 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [R0_BLOCK])
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 1024.0
    tmp16 = (tmp14 / tmp15)
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp10 * tmp19
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp20 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 1024*x0), tmp9, None)
    tl.store(out_ptr1 + (r0_1 + 1024*x0), tmp25, None)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/vg/cvgky5mdjlctwpcn27midhewszohuir26naw5sfuzye4jfuc25yd.py
# Topologically Sorted Source Nodes: [hidden_states_16, float_15, square_5, var_5, add_24, rsqrt_5, normed_inputs_5, add_25, normed_5, hidden_states_17], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_24 => add_28
#   add_25 => add_29
#   float_15 => convert_element_type_70
#   hidden_states_16 => add_27
#   hidden_states_17 => convert_element_type_71
#   normed_5 => mul_42
#   normed_inputs_5 => mul_41
#   rsqrt_5 => rsqrt_5
#   square_5 => pow_6
#   var_5 => mean_5
# Graph fragment:
#   %add_27 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_75), kwargs = {})
#   %convert_element_type_70 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_27, torch.float32), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_70, 2), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_6, [-1], True), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, 1e-06), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_28,), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %rsqrt_5), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg33_1, 1.0), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %add_29), kwargs = {})
#   %convert_element_type_71 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_42, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 421888}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel):
    xnumel = 51
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = 1024.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp3 * tmp12
    tmp15 = 1.0
    tmp16 = tmp14 + tmp15
    tmp17 = tmp13 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 1024*x0), tmp18, None)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/4y/c4ypherv6kmt4dquqy7x2dsajydwqhsxshsr2kedycgpyya33hej.py
# Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_18, float_17, square_6, var_6, add_27, rsqrt_6, normed_inputs_6, add_28, normed_6, hidden_states_19], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_27 => add_33
#   add_28 => add_34
#   float_17 => convert_element_type_80
#   hidden_states_16 => add_27
#   hidden_states_18 => add_32
#   hidden_states_19 => convert_element_type_81
#   normed_6 => mul_51
#   normed_inputs_6 => mul_50
#   rsqrt_6 => rsqrt_6
#   square_6 => pow_7
#   var_6 => mean_6
# Graph fragment:
#   %add_27 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_75), kwargs = {})
#   %add_32 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %view_81), kwargs = {})
#   %convert_element_type_80 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_32, torch.float32), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_80, 2), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [-1], True), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_6, 1e-06), kwargs = {})
#   %rsqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_33,), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, %rsqrt_6), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg37_1, 1.0), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %add_34), kwargs = {})
#   %convert_element_type_81 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_51, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 526336}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel):
    xnumel = 51
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp16 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [R0_BLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = 1024.0
    tmp11 = (tmp9 / tmp10)
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp5 * tmp14
    tmp17 = 1.0
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 1024*x0), tmp20, None)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/v4/cv4vtzfmz67pl2d6ocebuk5s3kjzoxhn5f3m5nyhv6zc6b3j6nva.py
# Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_18, hidden_states_22, float_19, square_7, var_7, add_33, rsqrt_7, normed_inputs_7, add_34, normed_7, hidden_states_23], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_33 => add_39
#   add_34 => add_40
#   float_19 => convert_element_type_95
#   hidden_states_16 => add_27
#   hidden_states_18 => add_32
#   hidden_states_22 => add_38
#   hidden_states_23 => convert_element_type_96
#   normed_7 => mul_58
#   normed_inputs_7 => mul_57
#   rsqrt_7 => rsqrt_7
#   square_7 => pow_8
#   var_7 => mean_7
# Graph fragment:
#   %add_27 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_75), kwargs = {})
#   %add_32 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %view_81), kwargs = {})
#   %add_38 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_32, %view_101), kwargs = {})
#   %convert_element_type_95 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_38, torch.float32), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_95, 2), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [-1], True), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_7, 1e-06), kwargs = {})
#   %rsqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_39,), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %rsqrt_7), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg44_1, 1.0), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, %add_40), kwargs = {})
#   %convert_element_type_96 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_58, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 630784}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
    xnumel = 51
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp18 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = 1024.0
    tmp13 = (tmp11 / tmp12)
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp7 * tmp16
    tmp19 = 1.0
    tmp20 = tmp18 + tmp19
    tmp21 = tmp17 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 1024*x0), tmp22, None)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/re/cre5avjarjqtuii4azm4y54crt2ibfxrbfafeqed4oz6s6623sjz.py
# Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_18, hidden_states_22, hidden_states_24, float_21, square_8, var_8, add_36, rsqrt_8, normed_inputs_8, add_37, normed_8, hidden_states_25], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_36 => add_44
#   add_37 => add_45
#   float_21 => convert_element_type_105
#   hidden_states_16 => add_27
#   hidden_states_18 => add_32
#   hidden_states_22 => add_38
#   hidden_states_24 => add_43
#   hidden_states_25 => convert_element_type_106
#   normed_8 => mul_67
#   normed_inputs_8 => mul_66
#   rsqrt_8 => rsqrt_8
#   square_8 => pow_9
#   var_8 => mean_8
# Graph fragment:
#   %add_27 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_75), kwargs = {})
#   %add_32 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %view_81), kwargs = {})
#   %add_38 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_32, %view_101), kwargs = {})
#   %add_43 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_38, %view_107), kwargs = {})
#   %convert_element_type_105 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_43, torch.float32), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_105, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [-1], True), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_8, 1e-06), kwargs = {})
#   %rsqrt_8 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_44,), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %rsqrt_8), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg48_1, 1.0), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_66, %add_45), kwargs = {})
#   %convert_element_type_106 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_67, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_16 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 944128}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel):
    xnumel = 51
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp20 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = (tmp13 / tmp14)
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp9 * tmp18
    tmp21 = 1.0
    tmp22 = tmp20 + tmp21
    tmp23 = tmp19 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 1024*x0), tmp8, None)
    tl.store(out_ptr1 + (r0_1 + 1024*x0), tmp24, None)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/dl/cdlamdrezwvf57jtj2anducaso5tqolmndjrjf3336zmp6btgk6o.py
# Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_102, hidden_states_106, hidden_states_108, float_77, square_36, var_36], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   float_77 => convert_element_type_455
#   hidden_states_100 => add_181
#   hidden_states_102 => add_186
#   hidden_states_106 => add_192
#   hidden_states_108 => add_197
#   square_36 => pow_37
#   var_36 => mean_36
# Graph fragment:
#   %add_181 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_175, %view_439), kwargs = {})
#   %add_186 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_181, %view_445), kwargs = {})
#   %add_192 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_186, %view_465), kwargs = {})
#   %add_197 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_192, %view_471), kwargs = {})
#   %convert_element_type_455 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_197, torch.float32), kwargs = {})
#   %pow_37 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_455, 2), kwargs = {})
#   %mean_36 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_37, [-1], True), kwargs = {})
triton_per_fused__to_copy_add_mean_pow_17 = async_compile.triton('triton_per_fused__to_copy_add_mean_pow_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_pow_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 408, 'r0_': 731136}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_pow_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel):
    xnumel = 51
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(in_out_ptr0 + (r0_1 + 1024*x0), tmp8, None)
    tl.store(out_ptr0 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: /home/yanggon/00_VLA/26_measurement_jetson_vla/scratch/T003/ev_reduce-overhead_ic/44/c44gmygegwvvhvk7537rdif3qod7u7c4ucts3wzvy3gkwl6ukeg2.py
# Topologically Sorted Source Nodes: [float_77, square_36, var_36, add_162, rsqrt_36, normed_inputs_36, add_163, normed_36, hidden_states_109, suffix_out, suffix_out_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.slice]
# Source node to ATen node mapping:
#   add_162 => add_198
#   add_163 => add_199
#   float_77 => convert_element_type_455
#   hidden_states_109 => convert_element_type_456
#   normed_36 => mul_291
#   normed_inputs_36 => mul_290
#   rsqrt_36 => rsqrt_36
#   square_36 => pow_37
#   suffix_out => slice_73
#   suffix_out_1 => convert_element_type_457
#   var_36 => mean_36
# Graph fragment:
#   %convert_element_type_455 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_197, torch.float32), kwargs = {})
#   %pow_37 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_455, 2), kwargs = {})
#   %mean_36 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_37, [-1], True), kwargs = {})
#   %add_198 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_36, 1e-06), kwargs = {})
#   %rsqrt_36 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_198,), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_197, %rsqrt_36), kwargs = {})
#   %add_199 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg202_1, 1.0), kwargs = {})
#   %mul_291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_290, %add_199), kwargs = {})
#   %convert_element_type_456 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_291, torch.bfloat16), kwargs = {})
#   %slice_73 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%convert_element_type_456, 1, -50, 9223372036854775807), kwargs = {})
#   %convert_element_type_457 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_73, torch.float32), kwargs = {})
triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_slice_18 = async_compile.triton('triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_slice_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=87, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_slice_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'F217A36E92DFC30BAEA1873702836B2741EC87D10190CDC0CBD4E2B163A14577', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 516096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_slice_18(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (1024 + x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (1 + x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = 1024.0
    tmp4 = (tmp2 / tmp3)
    tmp5 = 1e-06
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.rsqrt(tmp6)
    tmp8 = tmp1 * tmp7
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 51, 1024), (52224, 1024, 1))
    assert_size_stride(arg1_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg2_1, (1, 51), (51, 1))
    assert_size_stride(arg3_1, (1, 1, 51, 867), (44217, 44217, 867, 1))
    assert_size_stride(arg4_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (1024, ), (1, ))
    assert_size_stride(arg7_1, (256, 1024), (1024, 1))
    assert_size_stride(arg8_1, (256, 1024), (1024, 1))
    assert_size_stride(arg9_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg10_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg11_1, (1024, ), (1, ))
    assert_size_stride(arg12_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg13_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg14_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg15_1, (1024, ), (1, ))
    assert_size_stride(arg16_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg17_1, (256, 1024), (1024, 1))
    assert_size_stride(arg18_1, (256, 1024), (1024, 1))
    assert_size_stride(arg19_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg20_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg21_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg22_1, (1024, ), (1, ))
    assert_size_stride(arg23_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg24_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg25_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg26_1, (1024, ), (1, ))
    assert_size_stride(arg27_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg28_1, (256, 1024), (1024, 1))
    assert_size_stride(arg29_1, (256, 1024), (1024, 1))
    assert_size_stride(arg30_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg31_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg32_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg33_1, (1024, ), (1, ))
    assert_size_stride(arg34_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg35_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg36_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg39_1, (256, 1024), (1024, 1))
    assert_size_stride(arg40_1, (256, 1024), (1024, 1))
    assert_size_stride(arg41_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg42_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg43_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg44_1, (1024, ), (1, ))
    assert_size_stride(arg45_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg46_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg47_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg48_1, (1024, ), (1, ))
    assert_size_stride(arg49_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg50_1, (256, 1024), (1024, 1))
    assert_size_stride(arg51_1, (256, 1024), (1024, 1))
    assert_size_stride(arg52_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg53_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg54_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg55_1, (1024, ), (1, ))
    assert_size_stride(arg56_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg57_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg58_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg61_1, (256, 1024), (1024, 1))
    assert_size_stride(arg62_1, (256, 1024), (1024, 1))
    assert_size_stride(arg63_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg64_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg65_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg68_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg69_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg70_1, (1024, ), (1, ))
    assert_size_stride(arg71_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg72_1, (256, 1024), (1024, 1))
    assert_size_stride(arg73_1, (256, 1024), (1024, 1))
    assert_size_stride(arg74_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg75_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg76_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg79_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg80_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg81_1, (1024, ), (1, ))
    assert_size_stride(arg82_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg83_1, (256, 1024), (1024, 1))
    assert_size_stride(arg84_1, (256, 1024), (1024, 1))
    assert_size_stride(arg85_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg86_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg87_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg88_1, (1024, ), (1, ))
    assert_size_stride(arg89_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg90_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg91_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg94_1, (256, 1024), (1024, 1))
    assert_size_stride(arg95_1, (256, 1024), (1024, 1))
    assert_size_stride(arg96_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg97_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg98_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg101_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg102_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg103_1, (1024, ), (1, ))
    assert_size_stride(arg104_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg105_1, (256, 1024), (1024, 1))
    assert_size_stride(arg106_1, (256, 1024), (1024, 1))
    assert_size_stride(arg107_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg108_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg109_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg112_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg113_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg116_1, (256, 1024), (1024, 1))
    assert_size_stride(arg117_1, (256, 1024), (1024, 1))
    assert_size_stride(arg118_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg119_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg120_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg121_1, (1024, ), (1, ))
    assert_size_stride(arg122_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg123_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg124_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg127_1, (256, 1024), (1024, 1))
    assert_size_stride(arg128_1, (256, 1024), (1024, 1))
    assert_size_stride(arg129_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg130_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg131_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg134_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg135_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg138_1, (256, 1024), (1024, 1))
    assert_size_stride(arg139_1, (256, 1024), (1024, 1))
    assert_size_stride(arg140_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg141_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg142_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg143_1, (1024, ), (1, ))
    assert_size_stride(arg144_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg145_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg146_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg147_1, (1024, ), (1, ))
    assert_size_stride(arg148_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg149_1, (256, 1024), (1024, 1))
    assert_size_stride(arg150_1, (256, 1024), (1024, 1))
    assert_size_stride(arg151_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg152_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg153_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg156_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg157_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg158_1, (1024, ), (1, ))
    assert_size_stride(arg159_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg160_1, (256, 1024), (1024, 1))
    assert_size_stride(arg161_1, (256, 1024), (1024, 1))
    assert_size_stride(arg162_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg163_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg164_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg165_1, (1024, ), (1, ))
    assert_size_stride(arg166_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg167_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg168_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg171_1, (256, 1024), (1024, 1))
    assert_size_stride(arg172_1, (256, 1024), (1024, 1))
    assert_size_stride(arg173_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg174_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg175_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg176_1, (1024, ), (1, ))
    assert_size_stride(arg177_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg178_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg179_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg180_1, (1024, ), (1, ))
    assert_size_stride(arg181_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg182_1, (256, 1024), (1024, 1))
    assert_size_stride(arg183_1, (256, 1024), (1024, 1))
    assert_size_stride(arg184_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg185_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg186_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg187_1, (1024, ), (1, ))
    assert_size_stride(arg188_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg189_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg190_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg193_1, (256, 1024), (1024, 1))
    assert_size_stride(arg194_1, (256, 1024), (1024, 1))
    assert_size_stride(arg195_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg196_1, (1, 1, 816, 256), (208896, 256, 256, 1))
    assert_size_stride(arg197_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg200_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg201_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg202_1, (1024, ), (1, ))
    assert_size_stride(arg203_1, (32, 1024), (1024, 1))
    assert_size_stride(arg204_1, (32, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 51, 1024), (52224, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states, float_5, square, var, add, rsqrt, normed_inputs, add_1, normed, hidden_states_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0.run(arg0_1, arg6_1, buf1, 51, 1024, stream=stream0)
        del arg6_1
        buf2 = empty_strided_cuda((51, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states, float_5, square, var, add, rsqrt, normed_inputs, add_1, normed, hidden_states_1, linear], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg4_1, (1024, 2048), (1, 1024), 0), out=buf2)
        del arg4_1
        buf3 = empty_strided_cuda((1, 128, 1), (128, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(arg5_1, buf3, 128, stream=stream0)
        del arg5_1
        buf4 = empty_strided_cuda((1, 1, 51), (51, 51, 1), torch.float32)
        # Topologically Sorted Source Nodes: [position_ids_expanded], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(arg2_1, buf4, 51, stream=stream0)
        del arg2_1
        buf5 = empty_strided_cuda((128, 51), (51, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_1, expand, , position_ids_expanded], Original ATen: [aten._to_copy, aten.expand, aten.bmm]
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 1), (1, 0), 0), reinterpret_tensor(buf4, (1, 51), (0, 1), 0), out=buf5)
        del buf3
        buf6 = empty_strided_cuda((51, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg7_1, (1024, 256), (1, 1024), 0), out=buf6)
        del arg7_1
        buf7 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, cos_3, sin, sin_1, sin_2, sin_3, mul_6, neg_1, cat_2, mul_7, k_embed, keys], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.sin, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg1_1, buf6, buf5, buf7, 867, 256, stream=stream0)
        del arg1_1
        buf8 = empty_strided_cuda((1, 8, 51, 256), (104448, 13056, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, cos_3, mul_4, neg, cat_1, sin, sin_1, sin_2, sin_3, mul_5, q_embed], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.neg, aten.sin, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf2, buf5, buf8, 51, 2048, stream=stream0)
        buf9 = empty_strided_cuda((8, 51, 867), (44217, 867, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, cos_3, mul_4, neg, cat_1, sin, sin_1, sin_2, sin_3, mul_5, q_embed, matmul_1, key_states_1, transpose_4], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.neg, aten.sin, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf7, (8, 256, 867), (0, 1, 256), 0), out=buf9)
        buf14 = empty_strided_cuda((1, 8, 51, 867), (353792, 44224, 867, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [attn_weights, attn_weights_1, , sub, exp, softmax, attn_weights_2], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf9, arg3_1, buf14, 408, 867, stream=stream0)
        buf12 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg8_1, (1024, 256), (1, 1024), 0), out=buf12)
        del arg8_1
        buf13 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg9_1, buf12, buf13, 221952, stream=stream0)
        del arg9_1
        buf15 = reinterpret_tensor(buf8, (8, 51, 256), (13056, 256, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [attn_weights, attn_weights_1, sub, exp, softmax, attn_weights_2, attn_output, value_states_1], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf14, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf13, (8, 867, 256), (0, 256, 1), 0), out=buf15)
        buf16 = reinterpret_tensor(buf2, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [attn_output_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf15, buf16, 104448, stream=stream0)
        buf17 = reinterpret_tensor(buf1, (51, 1024), (1024, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [attn_output_1, reshape_2, attn_output_3], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg10_1, (2048, 1024), (1, 2048), 0), out=buf17)
        del arg10_1
        buf19 = empty_strided_cuda((1, 51, 1024), (52224, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states, hidden_states_4, float_7, square_1, var_1, add_6, rsqrt_1, normed_inputs_1, add_7, normed_1, hidden_states_5], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(arg0_1, buf17, arg11_1, buf19, 51, 1024, stream=stream0)
        del arg11_1
        buf20 = empty_strided_cuda((51, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states, hidden_states_4, float_7, square_1, var_1, add_6, rsqrt_1, normed_inputs_1, add_7, normed_1, hidden_states_5, linear_4], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg12_1, (1024, 4096), (1, 1024), 0), out=buf20)
        del arg12_1
        buf21 = empty_strided_cuda((51, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg13_1, (1024, 4096), (1, 1024), 0), out=buf21)
        del arg13_1
        buf22 = reinterpret_tensor(buf20, (1, 51, 4096), (208896, 4096, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [gelu, mul_11], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf22, buf21, 208896, stream=stream0)
        buf23 = reinterpret_tensor(buf19, (51, 1024), (1024, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [gelu, mul_11, down_proj], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf22, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 1024), (1, 4096), 0), out=buf23)
        del arg14_1
        buf25 = empty_strided_cuda((1, 51, 1024), (52224, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states, hidden_states_4, hidden_states_6, float_9, square_2, var_2, add_9, rsqrt_2, normed_inputs_2, add_10, normed_2, hidden_states_7], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_10.run(arg0_1, buf17, buf23, arg15_1, buf25, 51, 1024, stream=stream0)
        del arg15_1
        buf26 = reinterpret_tensor(buf16, (51, 2048), (2048, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg16_1, (1024, 2048), (1, 1024), 0), out=buf26)
        del arg16_1
        buf27 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg17_1, (1024, 256), (1, 1024), 0), out=buf27)
        del arg17_1
        buf28 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_4, sin_4, mul_16, neg_3, cat_6, mul_17, k_embed_1, keys_1], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg19_1, buf27, buf5, buf28, 867, 256, stream=stream0)
        del arg19_1
        buf29 = reinterpret_tensor(buf15, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_4, mul_14, neg_2, cat_5, sin_4, mul_15, q_embed_1], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf26, buf5, buf29, 51, 2048, stream=stream0)
        buf30 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_4, mul_14, neg_2, cat_5, sin_4, mul_15, q_embed_1, matmul_3, key_states_3, transpose_9], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf28, (8, 256, 867), (0, 1, 256), 0), out=buf30)
        buf35 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_4, attn_weights_5, , sub, exp, softmax_1, attn_weights_6], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf30, arg3_1, buf35, 408, 867, stream=stream0)
        buf33 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg18_1, (1024, 256), (1, 1024), 0), out=buf33)
        del arg18_1
        buf34 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg20_1, buf33, buf34, 221952, stream=stream0)
        del arg20_1
        buf36 = reinterpret_tensor(buf29, (8, 51, 256), (13056, 256, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_4, attn_weights_5, sub, exp, softmax_1, attn_weights_6, attn_output_4, value_states_3], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf35, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf34, (8, 867, 256), (0, 256, 1), 0), out=buf36)
        buf37 = reinterpret_tensor(buf26, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [attn_output_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf36, buf37, 104448, stream=stream0)
        buf38 = reinterpret_tensor(buf25, (51, 1024), (1024, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [attn_output_5, reshape_5, attn_output_7], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg21_1, (2048, 1024), (1, 2048), 0), out=buf38)
        del arg21_1
        buf40 = empty_strided_cuda((1, 51, 1024), (52224, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states, hidden_states_4, hidden_states_6, hidden_states_10, float_11, square_3, var_3, add_15, rsqrt_3, normed_inputs_3, add_16, normed_3, hidden_states_11], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_11.run(arg0_1, buf17, buf23, buf38, arg22_1, buf40, 51, 1024, stream=stream0)
        del arg22_1
        buf41 = reinterpret_tensor(buf22, (51, 4096), (4096, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg23_1, (1024, 4096), (1, 1024), 0), out=buf41)
        del arg23_1
        buf42 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg24_1, (1024, 4096), (1, 1024), 0), out=buf42)
        del arg24_1
        buf43 = reinterpret_tensor(buf41, (1, 51, 4096), (208896, 4096, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [gelu_1, mul_21], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf43, buf42, 208896, stream=stream0)
        buf44 = reinterpret_tensor(buf40, (51, 1024), (1024, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [gelu_1, mul_21, down_proj_1], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg25_1, (4096, 1024), (1, 4096), 0), out=buf44)
        del arg25_1
        buf45 = reinterpret_tensor(buf17, (1, 51, 1024), (52224, 1024, 1), 0); del buf17  # reuse
        buf47 = empty_strided_cuda((1, 51, 1024), (52224, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states, hidden_states_4, hidden_states_6, hidden_states_10, hidden_states_12, float_13, square_4, var_4, add_18, rsqrt_4, normed_inputs_4, add_19, normed_4, hidden_states_13], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_12.run(buf45, arg0_1, buf23, buf38, buf44, arg26_1, buf47, 51, 1024, stream=stream0)
        del arg0_1
        del arg26_1
        buf48 = reinterpret_tensor(buf37, (51, 2048), (2048, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [float_13, square_4, var_4, add_18, rsqrt_4, normed_inputs_4, add_19, normed_4, hidden_states_13, linear_14], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg27_1, (1024, 2048), (1, 1024), 0), out=buf48)
        del arg27_1
        buf49 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg28_1, (1024, 256), (1, 1024), 0), out=buf49)
        del arg28_1
        buf50 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_5, sin_5, mul_26, neg_5, cat_10, mul_27, k_embed_2, keys_2], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg30_1, buf49, buf5, buf50, 867, 256, stream=stream0)
        del arg30_1
        buf51 = reinterpret_tensor(buf36, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_5, mul_24, neg_4, cat_9, sin_5, mul_25, q_embed_2], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf48, buf5, buf51, 51, 2048, stream=stream0)
        buf52 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_5, mul_24, neg_4, cat_9, sin_5, mul_25, q_embed_2, matmul_5, key_states_5, transpose_14], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf51, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf50, (8, 256, 867), (0, 1, 256), 0), out=buf52)
        buf57 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_8, attn_weights_9, , sub, exp, softmax_2, attn_weights_10], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf52, arg3_1, buf57, 408, 867, stream=stream0)
        buf55 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg29_1, (1024, 256), (1, 1024), 0), out=buf55)
        del arg29_1
        buf56 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg31_1, buf55, buf56, 221952, stream=stream0)
        del arg31_1
        buf58 = reinterpret_tensor(buf51, (8, 51, 256), (13056, 256, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_8, attn_weights_9, sub, exp, softmax_2, attn_weights_10, attn_output_8, value_states_5], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf57, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf56, (8, 867, 256), (0, 256, 1), 0), out=buf58)
        buf59 = reinterpret_tensor(buf48, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [attn_output_9], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf58, buf59, 104448, stream=stream0)
        buf60 = reinterpret_tensor(buf47, (51, 1024), (1024, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [attn_output_9, reshape_8, attn_output_11], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg32_1, (2048, 1024), (1, 2048), 0), out=buf60)
        del arg32_1
        buf62 = reinterpret_tensor(buf44, (1, 51, 1024), (52224, 1024, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_16, float_15, square_5, var_5, add_24, rsqrt_5, normed_inputs_5, add_25, normed_5, hidden_states_17], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf45, buf60, arg33_1, buf62, 51, 1024, stream=stream0)
        del arg33_1
        buf63 = reinterpret_tensor(buf43, (51, 4096), (4096, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_16, float_15, square_5, var_5, add_24, rsqrt_5, normed_inputs_5, add_25, normed_5, hidden_states_17, linear_18], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg34_1, (1024, 4096), (1, 1024), 0), out=buf63)
        del arg34_1
        buf64 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg35_1, (1024, 4096), (1, 1024), 0), out=buf64)
        del arg35_1
        buf65 = reinterpret_tensor(buf63, (1, 51, 4096), (208896, 4096, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [gelu_2, mul_31], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf65, buf64, 208896, stream=stream0)
        buf66 = reinterpret_tensor(buf62, (51, 1024), (1024, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [gelu_2, mul_31, down_proj_2], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf65, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg36_1, (4096, 1024), (1, 4096), 0), out=buf66)
        del arg36_1
        buf68 = reinterpret_tensor(buf38, (1, 51, 1024), (52224, 1024, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_18, float_17, square_6, var_6, add_27, rsqrt_6, normed_inputs_6, add_28, normed_6, hidden_states_19], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14.run(buf45, buf60, buf66, arg37_1, buf68, 51, 1024, stream=stream0)
        del arg37_1
        buf69 = reinterpret_tensor(buf59, (51, 2048), (2048, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg38_1, (1024, 2048), (1, 1024), 0), out=buf69)
        del arg38_1
        buf70 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg39_1, (1024, 256), (1, 1024), 0), out=buf70)
        del arg39_1
        buf71 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_6, sin_6, mul_36, neg_7, cat_14, mul_37, k_embed_3, keys_3], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg41_1, buf70, buf5, buf71, 867, 256, stream=stream0)
        del arg41_1
        buf72 = reinterpret_tensor(buf58, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_6, mul_34, neg_6, cat_13, sin_6, mul_35, q_embed_3], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf69, buf5, buf72, 51, 2048, stream=stream0)
        buf73 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_6, mul_34, neg_6, cat_13, sin_6, mul_35, q_embed_3, matmul_7, key_states_7, transpose_19], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf72, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf71, (8, 256, 867), (0, 1, 256), 0), out=buf73)
        buf78 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_12, attn_weights_13, , sub, exp, softmax_3, attn_weights_14], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf73, arg3_1, buf78, 408, 867, stream=stream0)
        buf76 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg40_1, (1024, 256), (1, 1024), 0), out=buf76)
        del arg40_1
        buf77 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg42_1, buf76, buf77, 221952, stream=stream0)
        del arg42_1
        buf79 = reinterpret_tensor(buf72, (8, 51, 256), (13056, 256, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_12, attn_weights_13, sub, exp, softmax_3, attn_weights_14, attn_output_12, value_states_7], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf78, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf77, (8, 867, 256), (0, 256, 1), 0), out=buf79)
        buf80 = reinterpret_tensor(buf69, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [attn_output_13], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf79, buf80, 104448, stream=stream0)
        buf81 = reinterpret_tensor(buf68, (51, 1024), (1024, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [attn_output_13, reshape_11, attn_output_15], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg43_1, (2048, 1024), (1, 2048), 0), out=buf81)
        del arg43_1
        buf83 = reinterpret_tensor(buf23, (1, 51, 1024), (52224, 1024, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_18, hidden_states_22, float_19, square_7, var_7, add_33, rsqrt_7, normed_inputs_7, add_34, normed_7, hidden_states_23], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15.run(buf45, buf60, buf66, buf81, arg44_1, buf83, 51, 1024, stream=stream0)
        del arg44_1
        buf84 = reinterpret_tensor(buf65, (51, 4096), (4096, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg45_1, (1024, 4096), (1, 1024), 0), out=buf84)
        del arg45_1
        buf85 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg46_1, (1024, 4096), (1, 1024), 0), out=buf85)
        del arg46_1
        buf86 = reinterpret_tensor(buf84, (1, 51, 4096), (208896, 4096, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [gelu_3, mul_41], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf86, buf85, 208896, stream=stream0)
        buf87 = reinterpret_tensor(buf83, (51, 1024), (1024, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [gelu_3, mul_41, down_proj_3], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg47_1, (4096, 1024), (1, 4096), 0), out=buf87)
        del arg47_1
        buf88 = buf45; del buf45  # reuse
        buf90 = empty_strided_cuda((1, 51, 1024), (52224, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_18, hidden_states_22, hidden_states_24, float_21, square_8, var_8, add_36, rsqrt_8, normed_inputs_8, add_37, normed_8, hidden_states_25], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_16.run(buf88, buf60, buf66, buf81, buf87, arg48_1, buf90, 51, 1024, stream=stream0)
        del arg48_1
        buf91 = reinterpret_tensor(buf80, (51, 2048), (2048, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [float_21, square_8, var_8, add_36, rsqrt_8, normed_inputs_8, add_37, normed_8, hidden_states_25, linear_28], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg49_1, (1024, 2048), (1, 1024), 0), out=buf91)
        del arg49_1
        buf92 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg50_1, (1024, 256), (1, 1024), 0), out=buf92)
        del arg50_1
        buf93 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_7, sin_7, mul_46, neg_9, cat_18, mul_47, k_embed_4, keys_4], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg52_1, buf92, buf5, buf93, 867, 256, stream=stream0)
        del arg52_1
        buf94 = reinterpret_tensor(buf79, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_7, mul_44, neg_8, cat_17, sin_7, mul_45, q_embed_4], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf91, buf5, buf94, 51, 2048, stream=stream0)
        buf95 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_7, mul_44, neg_8, cat_17, sin_7, mul_45, q_embed_4, matmul_9, key_states_9, transpose_24], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf93, (8, 256, 867), (0, 1, 256), 0), out=buf95)
        buf100 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_16, attn_weights_17, , sub, exp, softmax_4, attn_weights_18], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf95, arg3_1, buf100, 408, 867, stream=stream0)
        buf98 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf90, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg51_1, (1024, 256), (1, 1024), 0), out=buf98)
        del arg51_1
        buf99 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg53_1, buf98, buf99, 221952, stream=stream0)
        del arg53_1
        buf101 = reinterpret_tensor(buf94, (8, 51, 256), (13056, 256, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_16, attn_weights_17, sub, exp, softmax_4, attn_weights_18, attn_output_16, value_states_9], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf100, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf99, (8, 867, 256), (0, 256, 1), 0), out=buf101)
        buf102 = reinterpret_tensor(buf91, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [attn_output_17], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf101, buf102, 104448, stream=stream0)
        buf103 = reinterpret_tensor(buf90, (51, 1024), (1024, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [attn_output_17, reshape_14, attn_output_19], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg54_1, (2048, 1024), (1, 2048), 0), out=buf103)
        del arg54_1
        buf105 = reinterpret_tensor(buf87, (1, 51, 1024), (52224, 1024, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, float_23, square_9, var_9, add_42, rsqrt_9, normed_inputs_9, add_43, normed_9, hidden_states_29], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf88, buf103, arg55_1, buf105, 51, 1024, stream=stream0)
        del arg55_1
        buf106 = reinterpret_tensor(buf86, (51, 4096), (4096, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, float_23, square_9, var_9, add_42, rsqrt_9, normed_inputs_9, add_43, normed_9, hidden_states_29, linear_32], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg56_1, (1024, 4096), (1, 1024), 0), out=buf106)
        del arg56_1
        buf107 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 4096), (1, 1024), 0), out=buf107)
        del arg57_1
        buf108 = reinterpret_tensor(buf106, (1, 51, 4096), (208896, 4096, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [gelu_4, mul_51], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf108, buf107, 208896, stream=stream0)
        buf109 = reinterpret_tensor(buf105, (51, 1024), (1024, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [gelu_4, mul_51, down_proj_4], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg58_1, (4096, 1024), (1, 4096), 0), out=buf109)
        del arg58_1
        buf111 = reinterpret_tensor(buf81, (1, 51, 1024), (52224, 1024, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_30, float_25, square_10, var_10, add_45, rsqrt_10, normed_inputs_10, add_46, normed_10, hidden_states_31], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14.run(buf88, buf103, buf109, arg59_1, buf111, 51, 1024, stream=stream0)
        del arg59_1
        buf112 = reinterpret_tensor(buf102, (51, 2048), (2048, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg60_1, (1024, 2048), (1, 1024), 0), out=buf112)
        del arg60_1
        buf113 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg61_1, (1024, 256), (1, 1024), 0), out=buf113)
        del arg61_1
        buf114 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_8, sin_8, mul_56, neg_11, cat_22, mul_57, k_embed_5, keys_5], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg63_1, buf113, buf5, buf114, 867, 256, stream=stream0)
        del arg63_1
        buf115 = reinterpret_tensor(buf101, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_8, mul_54, neg_10, cat_21, sin_8, mul_55, q_embed_5], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf112, buf5, buf115, 51, 2048, stream=stream0)
        buf116 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_8, mul_54, neg_10, cat_21, sin_8, mul_55, q_embed_5, matmul_11, key_states_11, transpose_29], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf115, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf114, (8, 256, 867), (0, 1, 256), 0), out=buf116)
        buf121 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_20, attn_weights_21, , sub, exp, softmax_5, attn_weights_22], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf116, arg3_1, buf121, 408, 867, stream=stream0)
        buf119 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg62_1, (1024, 256), (1, 1024), 0), out=buf119)
        del arg62_1
        buf120 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg64_1, buf119, buf120, 221952, stream=stream0)
        del arg64_1
        buf122 = reinterpret_tensor(buf115, (8, 51, 256), (13056, 256, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_20, attn_weights_21, sub, exp, softmax_5, attn_weights_22, attn_output_20, value_states_11], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf120, (8, 867, 256), (0, 256, 1), 0), out=buf122)
        buf123 = reinterpret_tensor(buf112, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [attn_output_21], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf122, buf123, 104448, stream=stream0)
        buf124 = reinterpret_tensor(buf111, (51, 1024), (1024, 1), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [attn_output_21, reshape_17, attn_output_23], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg65_1, (2048, 1024), (1, 2048), 0), out=buf124)
        del arg65_1
        buf126 = reinterpret_tensor(buf66, (1, 51, 1024), (52224, 1024, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_30, hidden_states_34, float_27, square_11, var_11, add_51, rsqrt_11, normed_inputs_11, add_52, normed_11, hidden_states_35], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15.run(buf88, buf103, buf109, buf124, arg66_1, buf126, 51, 1024, stream=stream0)
        del arg66_1
        buf127 = reinterpret_tensor(buf108, (51, 4096), (4096, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg67_1, (1024, 4096), (1, 1024), 0), out=buf127)
        del arg67_1
        buf128 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg68_1, (1024, 4096), (1, 1024), 0), out=buf128)
        del arg68_1
        buf129 = reinterpret_tensor(buf127, (1, 51, 4096), (208896, 4096, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [gelu_5, mul_61], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf129, buf128, 208896, stream=stream0)
        buf130 = reinterpret_tensor(buf126, (51, 1024), (1024, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [gelu_5, mul_61, down_proj_5], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg69_1, (4096, 1024), (1, 4096), 0), out=buf130)
        del arg69_1
        buf131 = buf88; del buf88  # reuse
        buf133 = reinterpret_tensor(buf60, (1, 51, 1024), (52224, 1024, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_28, hidden_states_30, hidden_states_34, hidden_states_36, float_29, square_12, var_12, add_54, rsqrt_12, normed_inputs_12, add_55, normed_12, hidden_states_37], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_16.run(buf131, buf103, buf109, buf124, buf130, arg70_1, buf133, 51, 1024, stream=stream0)
        del arg70_1
        buf134 = reinterpret_tensor(buf123, (51, 2048), (2048, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [float_29, square_12, var_12, add_54, rsqrt_12, normed_inputs_12, add_55, normed_12, hidden_states_37, linear_42], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 2048), (1, 1024), 0), out=buf134)
        del arg71_1
        buf135 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg72_1, (1024, 256), (1, 1024), 0), out=buf135)
        del arg72_1
        buf136 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_9, sin_9, mul_66, neg_13, cat_26, mul_67, k_embed_6, keys_6], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg74_1, buf135, buf5, buf136, 867, 256, stream=stream0)
        del arg74_1
        buf137 = reinterpret_tensor(buf122, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_9, mul_64, neg_12, cat_25, sin_9, mul_65, q_embed_6], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf134, buf5, buf137, 51, 2048, stream=stream0)
        buf138 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_9, mul_64, neg_12, cat_25, sin_9, mul_65, q_embed_6, matmul_13, key_states_13, transpose_34], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf136, (8, 256, 867), (0, 1, 256), 0), out=buf138)
        buf143 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_24, attn_weights_25, , sub, exp, softmax_6, attn_weights_26], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf138, arg3_1, buf143, 408, 867, stream=stream0)
        buf141 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg73_1, (1024, 256), (1, 1024), 0), out=buf141)
        del arg73_1
        buf142 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg75_1, buf141, buf142, 221952, stream=stream0)
        del arg75_1
        buf144 = reinterpret_tensor(buf137, (8, 51, 256), (13056, 256, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_24, attn_weights_25, sub, exp, softmax_6, attn_weights_26, attn_output_24, value_states_13], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf143, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf142, (8, 867, 256), (0, 256, 1), 0), out=buf144)
        buf145 = reinterpret_tensor(buf134, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [attn_output_25], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf144, buf145, 104448, stream=stream0)
        buf146 = reinterpret_tensor(buf133, (51, 1024), (1024, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [attn_output_25, reshape_20, attn_output_27], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg76_1, (2048, 1024), (1, 2048), 0), out=buf146)
        del arg76_1
        buf148 = reinterpret_tensor(buf130, (1, 51, 1024), (52224, 1024, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, float_31, square_13, var_13, add_60, rsqrt_13, normed_inputs_13, add_61, normed_13, hidden_states_41], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf131, buf146, arg77_1, buf148, 51, 1024, stream=stream0)
        del arg77_1
        buf149 = reinterpret_tensor(buf129, (51, 4096), (4096, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, float_31, square_13, var_13, add_60, rsqrt_13, normed_inputs_13, add_61, normed_13, hidden_states_41, linear_46], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg78_1, (1024, 4096), (1, 1024), 0), out=buf149)
        del arg78_1
        buf150 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg79_1, (1024, 4096), (1, 1024), 0), out=buf150)
        del arg79_1
        buf151 = reinterpret_tensor(buf149, (1, 51, 4096), (208896, 4096, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [gelu_6, mul_71], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf151, buf150, 208896, stream=stream0)
        buf152 = reinterpret_tensor(buf148, (51, 1024), (1024, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [gelu_6, mul_71, down_proj_6], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf151, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg80_1, (4096, 1024), (1, 4096), 0), out=buf152)
        del arg80_1
        buf154 = reinterpret_tensor(buf124, (1, 51, 1024), (52224, 1024, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_42, float_33, square_14, var_14, add_63, rsqrt_14, normed_inputs_14, add_64, normed_14, hidden_states_43], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14.run(buf131, buf146, buf152, arg81_1, buf154, 51, 1024, stream=stream0)
        del arg81_1
        buf155 = reinterpret_tensor(buf145, (51, 2048), (2048, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg82_1, (1024, 2048), (1, 1024), 0), out=buf155)
        del arg82_1
        buf156 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg83_1, (1024, 256), (1, 1024), 0), out=buf156)
        del arg83_1
        buf157 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_10, sin_10, mul_76, neg_15, cat_30, mul_77, k_embed_7, keys_7], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg85_1, buf156, buf5, buf157, 867, 256, stream=stream0)
        del arg85_1
        buf158 = reinterpret_tensor(buf144, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_10, mul_74, neg_14, cat_29, sin_10, mul_75, q_embed_7], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf155, buf5, buf158, 51, 2048, stream=stream0)
        buf159 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_10, mul_74, neg_14, cat_29, sin_10, mul_75, q_embed_7, matmul_15, key_states_15, transpose_39], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf158, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf157, (8, 256, 867), (0, 1, 256), 0), out=buf159)
        buf164 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_28, attn_weights_29, , sub, exp, softmax_7, attn_weights_30], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf159, arg3_1, buf164, 408, 867, stream=stream0)
        buf162 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg84_1, (1024, 256), (1, 1024), 0), out=buf162)
        del arg84_1
        buf163 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg86_1, buf162, buf163, 221952, stream=stream0)
        del arg86_1
        buf165 = reinterpret_tensor(buf158, (8, 51, 256), (13056, 256, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_28, attn_weights_29, sub, exp, softmax_7, attn_weights_30, attn_output_28, value_states_15], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf164, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf163, (8, 867, 256), (0, 256, 1), 0), out=buf165)
        buf166 = reinterpret_tensor(buf155, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [attn_output_29], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf165, buf166, 104448, stream=stream0)
        buf167 = reinterpret_tensor(buf154, (51, 1024), (1024, 1), 0); del buf154  # reuse
        # Topologically Sorted Source Nodes: [attn_output_29, reshape_23, attn_output_31], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg87_1, (2048, 1024), (1, 2048), 0), out=buf167)
        del arg87_1
        buf169 = reinterpret_tensor(buf109, (1, 51, 1024), (52224, 1024, 1), 0); del buf109  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_42, hidden_states_46, float_35, square_15, var_15, add_69, rsqrt_15, normed_inputs_15, add_70, normed_15, hidden_states_47], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15.run(buf131, buf146, buf152, buf167, arg88_1, buf169, 51, 1024, stream=stream0)
        del arg88_1
        buf170 = reinterpret_tensor(buf151, (51, 4096), (4096, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg89_1, (1024, 4096), (1, 1024), 0), out=buf170)
        del arg89_1
        buf171 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg90_1, (1024, 4096), (1, 1024), 0), out=buf171)
        del arg90_1
        buf172 = reinterpret_tensor(buf170, (1, 51, 4096), (208896, 4096, 1), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [gelu_7, mul_81], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf172, buf171, 208896, stream=stream0)
        buf173 = reinterpret_tensor(buf169, (51, 1024), (1024, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [gelu_7, mul_81, down_proj_7], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg91_1, (4096, 1024), (1, 4096), 0), out=buf173)
        del arg91_1
        buf174 = buf131; del buf131  # reuse
        buf176 = reinterpret_tensor(buf103, (1, 51, 1024), (52224, 1024, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_42, hidden_states_46, hidden_states_48, float_37, square_16, var_16, add_72, rsqrt_16, normed_inputs_16, add_73, normed_16, hidden_states_49], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_16.run(buf174, buf146, buf152, buf167, buf173, arg92_1, buf176, 51, 1024, stream=stream0)
        del arg92_1
        buf177 = reinterpret_tensor(buf166, (51, 2048), (2048, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [float_37, square_16, var_16, add_72, rsqrt_16, normed_inputs_16, add_73, normed_16, hidden_states_49, linear_56], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg93_1, (1024, 2048), (1, 1024), 0), out=buf177)
        del arg93_1
        buf178 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg94_1, (1024, 256), (1, 1024), 0), out=buf178)
        del arg94_1
        buf179 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_11, sin_11, mul_86, neg_17, cat_34, mul_87, k_embed_8, keys_8], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg96_1, buf178, buf5, buf179, 867, 256, stream=stream0)
        del arg96_1
        buf180 = reinterpret_tensor(buf165, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_11, mul_84, neg_16, cat_33, sin_11, mul_85, q_embed_8], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf177, buf5, buf180, 51, 2048, stream=stream0)
        buf181 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_11, mul_84, neg_16, cat_33, sin_11, mul_85, q_embed_8, matmul_17, key_states_17, transpose_44], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf179, (8, 256, 867), (0, 1, 256), 0), out=buf181)
        buf186 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_32, attn_weights_33, , sub, exp, softmax_8, attn_weights_34], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf181, arg3_1, buf186, 408, 867, stream=stream0)
        buf184 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg95_1, (1024, 256), (1, 1024), 0), out=buf184)
        del arg95_1
        buf185 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg97_1, buf184, buf185, 221952, stream=stream0)
        del arg97_1
        buf187 = reinterpret_tensor(buf180, (8, 51, 256), (13056, 256, 1), 0); del buf180  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_32, attn_weights_33, sub, exp, softmax_8, attn_weights_34, attn_output_32, value_states_17], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf186, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf185, (8, 867, 256), (0, 256, 1), 0), out=buf187)
        buf188 = reinterpret_tensor(buf177, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [attn_output_33], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf187, buf188, 104448, stream=stream0)
        buf189 = reinterpret_tensor(buf176, (51, 1024), (1024, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [attn_output_33, reshape_26, attn_output_35], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg98_1, (2048, 1024), (1, 2048), 0), out=buf189)
        del arg98_1
        buf191 = reinterpret_tensor(buf173, (1, 51, 1024), (52224, 1024, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_52, float_39, square_17, var_17, add_78, rsqrt_17, normed_inputs_17, add_79, normed_17, hidden_states_53], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf174, buf189, arg99_1, buf191, 51, 1024, stream=stream0)
        del arg99_1
        buf192 = reinterpret_tensor(buf172, (51, 4096), (4096, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_52, float_39, square_17, var_17, add_78, rsqrt_17, normed_inputs_17, add_79, normed_17, hidden_states_53, linear_60], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg100_1, (1024, 4096), (1, 1024), 0), out=buf192)
        del arg100_1
        buf193 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg101_1, (1024, 4096), (1, 1024), 0), out=buf193)
        del arg101_1
        buf194 = reinterpret_tensor(buf192, (1, 51, 4096), (208896, 4096, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [gelu_8, mul_91], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf194, buf193, 208896, stream=stream0)
        buf195 = reinterpret_tensor(buf191, (51, 1024), (1024, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [gelu_8, mul_91, down_proj_8], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf194, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg102_1, (4096, 1024), (1, 4096), 0), out=buf195)
        del arg102_1
        buf197 = reinterpret_tensor(buf167, (1, 51, 1024), (52224, 1024, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_52, hidden_states_54, float_41, square_18, var_18, add_81, rsqrt_18, normed_inputs_18, add_82, normed_18, hidden_states_55], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14.run(buf174, buf189, buf195, arg103_1, buf197, 51, 1024, stream=stream0)
        del arg103_1
        buf198 = reinterpret_tensor(buf188, (51, 2048), (2048, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg104_1, (1024, 2048), (1, 1024), 0), out=buf198)
        del arg104_1
        buf199 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [linear_64], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg105_1, (1024, 256), (1, 1024), 0), out=buf199)
        del arg105_1
        buf200 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_12, sin_12, mul_96, neg_19, cat_38, mul_97, k_embed_9, keys_9], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg107_1, buf199, buf5, buf200, 867, 256, stream=stream0)
        del arg107_1
        buf201 = reinterpret_tensor(buf187, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_12, mul_94, neg_18, cat_37, sin_12, mul_95, q_embed_9], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf198, buf5, buf201, 51, 2048, stream=stream0)
        buf202 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_12, mul_94, neg_18, cat_37, sin_12, mul_95, q_embed_9, matmul_19, key_states_19, transpose_49], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf201, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf200, (8, 256, 867), (0, 1, 256), 0), out=buf202)
        buf207 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_36, attn_weights_37, , sub, exp, softmax_9, attn_weights_38], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf202, arg3_1, buf207, 408, 867, stream=stream0)
        buf205 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg106_1, (1024, 256), (1, 1024), 0), out=buf205)
        del arg106_1
        buf206 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg108_1, buf205, buf206, 221952, stream=stream0)
        del arg108_1
        buf208 = reinterpret_tensor(buf201, (8, 51, 256), (13056, 256, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_36, attn_weights_37, sub, exp, softmax_9, attn_weights_38, attn_output_36, value_states_19], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf207, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf206, (8, 867, 256), (0, 256, 1), 0), out=buf208)
        buf209 = reinterpret_tensor(buf198, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf198  # reuse
        # Topologically Sorted Source Nodes: [attn_output_37], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf208, buf209, 104448, stream=stream0)
        buf210 = reinterpret_tensor(buf197, (51, 1024), (1024, 1), 0); del buf197  # reuse
        # Topologically Sorted Source Nodes: [attn_output_37, reshape_29, attn_output_39], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg109_1, (2048, 1024), (1, 2048), 0), out=buf210)
        del arg109_1
        buf212 = reinterpret_tensor(buf152, (1, 51, 1024), (52224, 1024, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_52, hidden_states_54, hidden_states_58, float_43, square_19, var_19, add_87, rsqrt_19, normed_inputs_19, add_88, normed_19, hidden_states_59], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15.run(buf174, buf189, buf195, buf210, arg110_1, buf212, 51, 1024, stream=stream0)
        del arg110_1
        buf213 = reinterpret_tensor(buf194, (51, 4096), (4096, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg111_1, (1024, 4096), (1, 1024), 0), out=buf213)
        del arg111_1
        buf214 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg112_1, (1024, 4096), (1, 1024), 0), out=buf214)
        del arg112_1
        buf215 = reinterpret_tensor(buf213, (1, 51, 4096), (208896, 4096, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [gelu_9, mul_101], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf215, buf214, 208896, stream=stream0)
        buf216 = reinterpret_tensor(buf212, (51, 1024), (1024, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [gelu_9, mul_101, down_proj_9], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg113_1, (4096, 1024), (1, 4096), 0), out=buf216)
        del arg113_1
        buf217 = buf174; del buf174  # reuse
        buf219 = reinterpret_tensor(buf146, (1, 51, 1024), (52224, 1024, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_52, hidden_states_54, hidden_states_58, hidden_states_60, float_45, square_20, var_20, add_90, rsqrt_20, normed_inputs_20, add_91, normed_20, hidden_states_61], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_16.run(buf217, buf189, buf195, buf210, buf216, arg114_1, buf219, 51, 1024, stream=stream0)
        del arg114_1
        buf220 = reinterpret_tensor(buf209, (51, 2048), (2048, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [float_45, square_20, var_20, add_90, rsqrt_20, normed_inputs_20, add_91, normed_20, hidden_states_61, linear_70], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg115_1, (1024, 2048), (1, 1024), 0), out=buf220)
        del arg115_1
        buf221 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg116_1, (1024, 256), (1, 1024), 0), out=buf221)
        del arg116_1
        buf222 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_13, sin_13, mul_106, neg_21, cat_42, mul_107, k_embed_10, keys_10], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg118_1, buf221, buf5, buf222, 867, 256, stream=stream0)
        del arg118_1
        buf223 = reinterpret_tensor(buf208, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf208  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_13, mul_104, neg_20, cat_41, sin_13, mul_105, q_embed_10], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf220, buf5, buf223, 51, 2048, stream=stream0)
        buf224 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_13, mul_104, neg_20, cat_41, sin_13, mul_105, q_embed_10, matmul_21, key_states_21, transpose_54], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf223, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf222, (8, 256, 867), (0, 1, 256), 0), out=buf224)
        buf229 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_40, attn_weights_41, , sub, exp, softmax_10, attn_weights_42], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf224, arg3_1, buf229, 408, 867, stream=stream0)
        buf227 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg117_1, (1024, 256), (1, 1024), 0), out=buf227)
        del arg117_1
        buf228 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg119_1, buf227, buf228, 221952, stream=stream0)
        del arg119_1
        buf230 = reinterpret_tensor(buf223, (8, 51, 256), (13056, 256, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_40, attn_weights_41, sub, exp, softmax_10, attn_weights_42, attn_output_40, value_states_21], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf229, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf228, (8, 867, 256), (0, 256, 1), 0), out=buf230)
        buf231 = reinterpret_tensor(buf220, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [attn_output_41], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf230, buf231, 104448, stream=stream0)
        buf232 = reinterpret_tensor(buf219, (51, 1024), (1024, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [attn_output_41, reshape_32, attn_output_43], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg120_1, (2048, 1024), (1, 2048), 0), out=buf232)
        del arg120_1
        buf234 = reinterpret_tensor(buf216, (1, 51, 1024), (52224, 1024, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, float_47, square_21, var_21, add_96, rsqrt_21, normed_inputs_21, add_97, normed_21, hidden_states_65], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf217, buf232, arg121_1, buf234, 51, 1024, stream=stream0)
        del arg121_1
        buf235 = reinterpret_tensor(buf215, (51, 4096), (4096, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, float_47, square_21, var_21, add_96, rsqrt_21, normed_inputs_21, add_97, normed_21, hidden_states_65, linear_74], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg122_1, (1024, 4096), (1, 1024), 0), out=buf235)
        del arg122_1
        buf236 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [linear_75], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg123_1, (1024, 4096), (1, 1024), 0), out=buf236)
        del arg123_1
        buf237 = reinterpret_tensor(buf235, (1, 51, 4096), (208896, 4096, 1), 0); del buf235  # reuse
        # Topologically Sorted Source Nodes: [gelu_10, mul_111], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf237, buf236, 208896, stream=stream0)
        buf238 = reinterpret_tensor(buf234, (51, 1024), (1024, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [gelu_10, mul_111, down_proj_10], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg124_1, (4096, 1024), (1, 4096), 0), out=buf238)
        del arg124_1
        buf240 = reinterpret_tensor(buf210, (1, 51, 1024), (52224, 1024, 1), 0); del buf210  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_66, float_49, square_22, var_22, add_99, rsqrt_22, normed_inputs_22, add_100, normed_22, hidden_states_67], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14.run(buf217, buf232, buf238, arg125_1, buf240, 51, 1024, stream=stream0)
        del arg125_1
        buf241 = reinterpret_tensor(buf231, (51, 2048), (2048, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg126_1, (1024, 2048), (1, 1024), 0), out=buf241)
        del arg126_1
        buf242 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg127_1, (1024, 256), (1, 1024), 0), out=buf242)
        del arg127_1
        buf243 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_14, sin_14, mul_116, neg_23, cat_46, mul_117, k_embed_11, keys_11], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg129_1, buf242, buf5, buf243, 867, 256, stream=stream0)
        del arg129_1
        buf244 = reinterpret_tensor(buf230, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf230  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_14, mul_114, neg_22, cat_45, sin_14, mul_115, q_embed_11], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf241, buf5, buf244, 51, 2048, stream=stream0)
        buf245 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_14, mul_114, neg_22, cat_45, sin_14, mul_115, q_embed_11, matmul_23, key_states_23, transpose_59], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf244, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf243, (8, 256, 867), (0, 1, 256), 0), out=buf245)
        buf250 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_44, attn_weights_45, , sub, exp, softmax_11, attn_weights_46], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf245, arg3_1, buf250, 408, 867, stream=stream0)
        buf248 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg128_1, (1024, 256), (1, 1024), 0), out=buf248)
        del arg128_1
        buf249 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg130_1, buf248, buf249, 221952, stream=stream0)
        del arg130_1
        buf251 = reinterpret_tensor(buf244, (8, 51, 256), (13056, 256, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_44, attn_weights_45, sub, exp, softmax_11, attn_weights_46, attn_output_44, value_states_23], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf250, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf249, (8, 867, 256), (0, 256, 1), 0), out=buf251)
        buf252 = reinterpret_tensor(buf241, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [attn_output_45], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf251, buf252, 104448, stream=stream0)
        buf253 = reinterpret_tensor(buf240, (51, 1024), (1024, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [attn_output_45, reshape_35, attn_output_47], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg131_1, (2048, 1024), (1, 2048), 0), out=buf253)
        del arg131_1
        buf255 = reinterpret_tensor(buf195, (1, 51, 1024), (52224, 1024, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_66, hidden_states_70, float_51, square_23, var_23, add_105, rsqrt_23, normed_inputs_23, add_106, normed_23, hidden_states_71], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15.run(buf217, buf232, buf238, buf253, arg132_1, buf255, 51, 1024, stream=stream0)
        del arg132_1
        buf256 = reinterpret_tensor(buf237, (51, 4096), (4096, 1), 0); del buf237  # reuse
        # Topologically Sorted Source Nodes: [linear_81], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg133_1, (1024, 4096), (1, 1024), 0), out=buf256)
        del arg133_1
        buf257 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg134_1, (1024, 4096), (1, 1024), 0), out=buf257)
        del arg134_1
        buf258 = reinterpret_tensor(buf256, (1, 51, 4096), (208896, 4096, 1), 0); del buf256  # reuse
        # Topologically Sorted Source Nodes: [gelu_11, mul_121], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf258, buf257, 208896, stream=stream0)
        buf259 = reinterpret_tensor(buf255, (51, 1024), (1024, 1), 0); del buf255  # reuse
        # Topologically Sorted Source Nodes: [gelu_11, mul_121, down_proj_11], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg135_1, (4096, 1024), (1, 4096), 0), out=buf259)
        del arg135_1
        buf260 = buf217; del buf217  # reuse
        buf262 = reinterpret_tensor(buf189, (1, 51, 1024), (52224, 1024, 1), 0); del buf189  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_64, hidden_states_66, hidden_states_70, hidden_states_72, float_53, square_24, var_24, add_108, rsqrt_24, normed_inputs_24, add_109, normed_24, hidden_states_73], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_16.run(buf260, buf232, buf238, buf253, buf259, arg136_1, buf262, 51, 1024, stream=stream0)
        del arg136_1
        buf263 = reinterpret_tensor(buf252, (51, 2048), (2048, 1), 0); del buf252  # reuse
        # Topologically Sorted Source Nodes: [float_53, square_24, var_24, add_108, rsqrt_24, normed_inputs_24, add_109, normed_24, hidden_states_73, linear_84], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg137_1, (1024, 2048), (1, 1024), 0), out=buf263)
        del arg137_1
        buf264 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg138_1, (1024, 256), (1, 1024), 0), out=buf264)
        del arg138_1
        buf265 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_15, sin_15, mul_126, neg_25, cat_50, mul_127, k_embed_12, keys_12], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg140_1, buf264, buf5, buf265, 867, 256, stream=stream0)
        del arg140_1
        buf266 = reinterpret_tensor(buf251, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf251  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_15, mul_124, neg_24, cat_49, sin_15, mul_125, q_embed_12], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf263, buf5, buf266, 51, 2048, stream=stream0)
        buf267 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_15, mul_124, neg_24, cat_49, sin_15, mul_125, q_embed_12, matmul_25, key_states_25, transpose_64], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf266, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf265, (8, 256, 867), (0, 1, 256), 0), out=buf267)
        buf272 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_48, attn_weights_49, , sub, exp, softmax_12, attn_weights_50], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf267, arg3_1, buf272, 408, 867, stream=stream0)
        buf270 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg139_1, (1024, 256), (1, 1024), 0), out=buf270)
        del arg139_1
        buf271 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg141_1, buf270, buf271, 221952, stream=stream0)
        del arg141_1
        buf273 = reinterpret_tensor(buf266, (8, 51, 256), (13056, 256, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_48, attn_weights_49, sub, exp, softmax_12, attn_weights_50, attn_output_48, value_states_25], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf272, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf271, (8, 867, 256), (0, 256, 1), 0), out=buf273)
        buf274 = reinterpret_tensor(buf263, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [attn_output_49], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf273, buf274, 104448, stream=stream0)
        buf275 = reinterpret_tensor(buf262, (51, 1024), (1024, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [attn_output_49, reshape_38, attn_output_51], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf274, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg142_1, (2048, 1024), (1, 2048), 0), out=buf275)
        del arg142_1
        buf277 = reinterpret_tensor(buf259, (1, 51, 1024), (52224, 1024, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76, float_55, square_25, var_25, add_114, rsqrt_25, normed_inputs_25, add_115, normed_25, hidden_states_77], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf260, buf275, arg143_1, buf277, 51, 1024, stream=stream0)
        del arg143_1
        buf278 = reinterpret_tensor(buf258, (51, 4096), (4096, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76, float_55, square_25, var_25, add_114, rsqrt_25, normed_inputs_25, add_115, normed_25, hidden_states_77, linear_88], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg144_1, (1024, 4096), (1, 1024), 0), out=buf278)
        del arg144_1
        buf279 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg145_1, (1024, 4096), (1, 1024), 0), out=buf279)
        del arg145_1
        buf280 = reinterpret_tensor(buf278, (1, 51, 4096), (208896, 4096, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [gelu_12, mul_131], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf280, buf279, 208896, stream=stream0)
        buf281 = reinterpret_tensor(buf277, (51, 1024), (1024, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [gelu_12, mul_131, down_proj_12], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg146_1, (4096, 1024), (1, 4096), 0), out=buf281)
        del arg146_1
        buf283 = reinterpret_tensor(buf253, (1, 51, 1024), (52224, 1024, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76, hidden_states_78, float_57, square_26, var_26, add_117, rsqrt_26, normed_inputs_26, add_118, normed_26, hidden_states_79], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14.run(buf260, buf275, buf281, arg147_1, buf283, 51, 1024, stream=stream0)
        del arg147_1
        buf284 = reinterpret_tensor(buf274, (51, 2048), (2048, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg148_1, (1024, 2048), (1, 1024), 0), out=buf284)
        del arg148_1
        buf285 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [linear_92], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg149_1, (1024, 256), (1, 1024), 0), out=buf285)
        del arg149_1
        buf286 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_16, sin_16, mul_136, neg_27, cat_54, mul_137, k_embed_13, keys_13], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg151_1, buf285, buf5, buf286, 867, 256, stream=stream0)
        del arg151_1
        buf287 = reinterpret_tensor(buf273, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_16, mul_134, neg_26, cat_53, sin_16, mul_135, q_embed_13], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf284, buf5, buf287, 51, 2048, stream=stream0)
        buf288 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_16, mul_134, neg_26, cat_53, sin_16, mul_135, q_embed_13, matmul_27, key_states_27, transpose_69], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf287, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf286, (8, 256, 867), (0, 1, 256), 0), out=buf288)
        buf293 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_52, attn_weights_53, , sub, exp, softmax_13, attn_weights_54], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf288, arg3_1, buf293, 408, 867, stream=stream0)
        buf291 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg150_1, (1024, 256), (1, 1024), 0), out=buf291)
        del arg150_1
        buf292 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg152_1, buf291, buf292, 221952, stream=stream0)
        del arg152_1
        buf294 = reinterpret_tensor(buf287, (8, 51, 256), (13056, 256, 1), 0); del buf287  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_52, attn_weights_53, sub, exp, softmax_13, attn_weights_54, attn_output_52, value_states_27], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf293, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf292, (8, 867, 256), (0, 256, 1), 0), out=buf294)
        buf295 = reinterpret_tensor(buf284, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [attn_output_53], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf294, buf295, 104448, stream=stream0)
        buf296 = reinterpret_tensor(buf283, (51, 1024), (1024, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [attn_output_53, reshape_41, attn_output_55], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg153_1, (2048, 1024), (1, 2048), 0), out=buf296)
        del arg153_1
        buf298 = reinterpret_tensor(buf238, (1, 51, 1024), (52224, 1024, 1), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76, hidden_states_78, hidden_states_82, float_59, square_27, var_27, add_123, rsqrt_27, normed_inputs_27, add_124, normed_27, hidden_states_83], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15.run(buf260, buf275, buf281, buf296, arg154_1, buf298, 51, 1024, stream=stream0)
        del arg154_1
        buf299 = reinterpret_tensor(buf280, (51, 4096), (4096, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [linear_95], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg155_1, (1024, 4096), (1, 1024), 0), out=buf299)
        del arg155_1
        buf300 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [linear_96], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg156_1, (1024, 4096), (1, 1024), 0), out=buf300)
        del arg156_1
        buf301 = reinterpret_tensor(buf299, (1, 51, 4096), (208896, 4096, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [gelu_13, mul_141], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf301, buf300, 208896, stream=stream0)
        buf302 = reinterpret_tensor(buf298, (51, 1024), (1024, 1), 0); del buf298  # reuse
        # Topologically Sorted Source Nodes: [gelu_13, mul_141, down_proj_13], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf301, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg157_1, (4096, 1024), (1, 4096), 0), out=buf302)
        del arg157_1
        buf303 = buf260; del buf260  # reuse
        buf305 = reinterpret_tensor(buf232, (1, 51, 1024), (52224, 1024, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_76, hidden_states_78, hidden_states_82, hidden_states_84, float_61, square_28, var_28, add_126, rsqrt_28, normed_inputs_28, add_127, normed_28, hidden_states_85], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_16.run(buf303, buf275, buf281, buf296, buf302, arg158_1, buf305, 51, 1024, stream=stream0)
        del arg158_1
        buf306 = reinterpret_tensor(buf295, (51, 2048), (2048, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [float_61, square_28, var_28, add_126, rsqrt_28, normed_inputs_28, add_127, normed_28, hidden_states_85, linear_98], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg159_1, (1024, 2048), (1, 1024), 0), out=buf306)
        del arg159_1
        buf307 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [linear_99], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg160_1, (1024, 256), (1, 1024), 0), out=buf307)
        del arg160_1
        buf308 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_17, sin_17, mul_146, neg_29, cat_58, mul_147, k_embed_14, keys_14], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg162_1, buf307, buf5, buf308, 867, 256, stream=stream0)
        del arg162_1
        buf309 = reinterpret_tensor(buf294, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_17, mul_144, neg_28, cat_57, sin_17, mul_145, q_embed_14], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf306, buf5, buf309, 51, 2048, stream=stream0)
        buf310 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_17, mul_144, neg_28, cat_57, sin_17, mul_145, q_embed_14, matmul_29, key_states_29, transpose_74], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf309, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf308, (8, 256, 867), (0, 1, 256), 0), out=buf310)
        buf315 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_56, attn_weights_57, , sub, exp, softmax_14, attn_weights_58], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf310, arg3_1, buf315, 408, 867, stream=stream0)
        buf313 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [linear_100], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg161_1, (1024, 256), (1, 1024), 0), out=buf313)
        del arg161_1
        buf314 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg163_1, buf313, buf314, 221952, stream=stream0)
        del arg163_1
        buf316 = reinterpret_tensor(buf309, (8, 51, 256), (13056, 256, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_56, attn_weights_57, sub, exp, softmax_14, attn_weights_58, attn_output_56, value_states_29], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf315, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf314, (8, 867, 256), (0, 256, 1), 0), out=buf316)
        buf317 = reinterpret_tensor(buf306, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [attn_output_57], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf316, buf317, 104448, stream=stream0)
        buf318 = reinterpret_tensor(buf305, (51, 1024), (1024, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [attn_output_57, reshape_44, attn_output_59], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg164_1, (2048, 1024), (1, 2048), 0), out=buf318)
        del arg164_1
        buf320 = reinterpret_tensor(buf302, (1, 51, 1024), (52224, 1024, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, float_63, square_29, var_29, add_132, rsqrt_29, normed_inputs_29, add_133, normed_29, hidden_states_89], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf303, buf318, arg165_1, buf320, 51, 1024, stream=stream0)
        del arg165_1
        buf321 = reinterpret_tensor(buf301, (51, 4096), (4096, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, float_63, square_29, var_29, add_132, rsqrt_29, normed_inputs_29, add_133, normed_29, hidden_states_89, linear_102], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg166_1, (1024, 4096), (1, 1024), 0), out=buf321)
        del arg166_1
        buf322 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [linear_103], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg167_1, (1024, 4096), (1, 1024), 0), out=buf322)
        del arg167_1
        buf323 = reinterpret_tensor(buf321, (1, 51, 4096), (208896, 4096, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [gelu_14, mul_151], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf323, buf322, 208896, stream=stream0)
        buf324 = reinterpret_tensor(buf320, (51, 1024), (1024, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [gelu_14, mul_151, down_proj_14], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf323, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg168_1, (4096, 1024), (1, 4096), 0), out=buf324)
        del arg168_1
        buf326 = reinterpret_tensor(buf296, (1, 51, 1024), (52224, 1024, 1), 0); del buf296  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, float_65, square_30, var_30, add_135, rsqrt_30, normed_inputs_30, add_136, normed_30, hidden_states_91], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14.run(buf303, buf318, buf324, arg169_1, buf326, 51, 1024, stream=stream0)
        del arg169_1
        buf327 = reinterpret_tensor(buf317, (51, 2048), (2048, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [linear_105], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg170_1, (1024, 2048), (1, 1024), 0), out=buf327)
        del arg170_1
        buf328 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 256), (1, 1024), 0), out=buf328)
        del arg171_1
        buf329 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_18, sin_18, mul_156, neg_31, cat_62, mul_157, k_embed_15, keys_15], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg173_1, buf328, buf5, buf329, 867, 256, stream=stream0)
        del arg173_1
        buf330 = reinterpret_tensor(buf316, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_18, mul_154, neg_30, cat_61, sin_18, mul_155, q_embed_15], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf327, buf5, buf330, 51, 2048, stream=stream0)
        buf331 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_18, mul_154, neg_30, cat_61, sin_18, mul_155, q_embed_15, matmul_31, key_states_31, transpose_79], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf330, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf329, (8, 256, 867), (0, 1, 256), 0), out=buf331)
        buf336 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_60, attn_weights_61, , sub, exp, softmax_15, attn_weights_62], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf331, arg3_1, buf336, 408, 867, stream=stream0)
        buf334 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [linear_107], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg172_1, (1024, 256), (1, 1024), 0), out=buf334)
        del arg172_1
        buf335 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg174_1, buf334, buf335, 221952, stream=stream0)
        del arg174_1
        buf337 = reinterpret_tensor(buf330, (8, 51, 256), (13056, 256, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_60, attn_weights_61, sub, exp, softmax_15, attn_weights_62, attn_output_60, value_states_31], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf336, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf335, (8, 867, 256), (0, 256, 1), 0), out=buf337)
        buf338 = reinterpret_tensor(buf327, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf327  # reuse
        # Topologically Sorted Source Nodes: [attn_output_61], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf337, buf338, 104448, stream=stream0)
        buf339 = reinterpret_tensor(buf326, (51, 1024), (1024, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [attn_output_61, reshape_47, attn_output_63], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg175_1, (2048, 1024), (1, 2048), 0), out=buf339)
        del arg175_1
        buf341 = reinterpret_tensor(buf281, (1, 51, 1024), (52224, 1024, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, hidden_states_94, float_67, square_31, var_31, add_141, rsqrt_31, normed_inputs_31, add_142, normed_31, hidden_states_95], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15.run(buf303, buf318, buf324, buf339, arg176_1, buf341, 51, 1024, stream=stream0)
        del arg176_1
        buf342 = reinterpret_tensor(buf323, (51, 4096), (4096, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [linear_109], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg177_1, (1024, 4096), (1, 1024), 0), out=buf342)
        del arg177_1
        buf343 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [linear_110], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf341, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg178_1, (1024, 4096), (1, 1024), 0), out=buf343)
        del arg178_1
        buf344 = reinterpret_tensor(buf342, (1, 51, 4096), (208896, 4096, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [gelu_15, mul_161], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf344, buf343, 208896, stream=stream0)
        buf345 = reinterpret_tensor(buf341, (51, 1024), (1024, 1), 0); del buf341  # reuse
        # Topologically Sorted Source Nodes: [gelu_15, mul_161, down_proj_15], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg179_1, (4096, 1024), (1, 4096), 0), out=buf345)
        del arg179_1
        buf346 = buf303; del buf303  # reuse
        buf348 = reinterpret_tensor(buf275, (1, 51, 1024), (52224, 1024, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, hidden_states_90, hidden_states_94, hidden_states_96, float_69, square_32, var_32, add_144, rsqrt_32, normed_inputs_32, add_145, normed_32, hidden_states_97], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_16.run(buf346, buf318, buf324, buf339, buf345, arg180_1, buf348, 51, 1024, stream=stream0)
        del arg180_1
        del buf318
        buf349 = reinterpret_tensor(buf338, (51, 2048), (2048, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [float_69, square_32, var_32, add_144, rsqrt_32, normed_inputs_32, add_145, normed_32, hidden_states_97, linear_112], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg181_1, (1024, 2048), (1, 1024), 0), out=buf349)
        del arg181_1
        buf350 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [linear_113], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg182_1, (1024, 256), (1, 1024), 0), out=buf350)
        del arg182_1
        buf351 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_19, sin_19, mul_166, neg_33, cat_66, mul_167, k_embed_16, keys_16], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg184_1, buf350, buf5, buf351, 867, 256, stream=stream0)
        del arg184_1
        buf352 = reinterpret_tensor(buf337, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_19, mul_164, neg_32, cat_65, sin_19, mul_165, q_embed_16], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf349, buf5, buf352, 51, 2048, stream=stream0)
        buf353 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_19, mul_164, neg_32, cat_65, sin_19, mul_165, q_embed_16, matmul_33, key_states_33, transpose_84], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf352, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf351, (8, 256, 867), (0, 1, 256), 0), out=buf353)
        buf358 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_64, attn_weights_65, , sub, exp, softmax_16, attn_weights_66], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf353, arg3_1, buf358, 408, 867, stream=stream0)
        buf356 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [linear_114], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg183_1, (1024, 256), (1, 1024), 0), out=buf356)
        del arg183_1
        buf357 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_16], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg185_1, buf356, buf357, 221952, stream=stream0)
        del arg185_1
        buf359 = reinterpret_tensor(buf352, (8, 51, 256), (13056, 256, 1), 0); del buf352  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_64, attn_weights_65, sub, exp, softmax_16, attn_weights_66, attn_output_64, value_states_33], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf358, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf357, (8, 867, 256), (0, 256, 1), 0), out=buf359)
        buf360 = reinterpret_tensor(buf349, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf349  # reuse
        # Topologically Sorted Source Nodes: [attn_output_65], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf359, buf360, 104448, stream=stream0)
        buf361 = reinterpret_tensor(buf348, (51, 1024), (1024, 1), 0); del buf348  # reuse
        # Topologically Sorted Source Nodes: [attn_output_65, reshape_50, attn_output_67], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg186_1, (2048, 1024), (1, 2048), 0), out=buf361)
        del arg186_1
        buf363 = reinterpret_tensor(buf345, (1, 51, 1024), (52224, 1024, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, float_71, square_33, var_33, add_150, rsqrt_33, normed_inputs_33, add_151, normed_33, hidden_states_101], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_13.run(buf346, buf361, arg187_1, buf363, 51, 1024, stream=stream0)
        del arg187_1
        buf364 = reinterpret_tensor(buf344, (51, 4096), (4096, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, float_71, square_33, var_33, add_150, rsqrt_33, normed_inputs_33, add_151, normed_33, hidden_states_101, linear_116], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg188_1, (1024, 4096), (1, 1024), 0), out=buf364)
        del arg188_1
        buf365 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [linear_117], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg189_1, (1024, 4096), (1, 1024), 0), out=buf365)
        del arg189_1
        buf366 = reinterpret_tensor(buf364, (1, 51, 4096), (208896, 4096, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [gelu_16, mul_171], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf366, buf365, 208896, stream=stream0)
        buf367 = reinterpret_tensor(buf363, (51, 1024), (1024, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [gelu_16, mul_171, down_proj_16], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf366, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg190_1, (4096, 1024), (1, 4096), 0), out=buf367)
        del arg190_1
        buf369 = reinterpret_tensor(buf339, (1, 51, 1024), (52224, 1024, 1), 0); del buf339  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_102, float_73, square_34, var_34, add_153, rsqrt_34, normed_inputs_34, add_154, normed_34, hidden_states_103], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_14.run(buf346, buf361, buf367, arg191_1, buf369, 51, 1024, stream=stream0)
        del arg191_1
        buf370 = reinterpret_tensor(buf360, (51, 2048), (2048, 1), 0); del buf360  # reuse
        # Topologically Sorted Source Nodes: [linear_119], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg192_1, (1024, 2048), (1, 1024), 0), out=buf370)
        del arg192_1
        buf371 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [linear_120], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg193_1, (1024, 256), (1, 1024), 0), out=buf371)
        del arg193_1
        buf372 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_20, sin_20, mul_176, neg_35, cat_70, mul_177, k_embed_17, keys_17], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_3.run(arg195_1, buf371, buf5, buf372, 867, 256, stream=stream0)
        del arg195_1
        buf373 = reinterpret_tensor(buf359, (1, 8, 51, 256), (104448, 13056, 256, 1), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_20, mul_174, neg_34, cat_69, sin_20, mul_175, q_embed_17], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_cos_mul_neg_sin_unsqueeze_4.run(buf370, buf5, buf373, 51, 2048, stream=stream0)
        del buf5
        buf374 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_20, mul_174, neg_34, cat_69, sin_20, mul_175, q_embed_17, matmul_35, key_states_35, transpose_89], Original ATen: [aten.cat, aten.cos, aten.mul, aten._to_copy, aten.sin, aten.unsqueeze, aten.neg, aten.add, aten.view, aten.transpose, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf373, (8, 51, 256), (13056, 256, 1), 0), reinterpret_tensor(buf372, (8, 256, 867), (0, 1, 256), 0), out=buf374)
        buf379 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_68, attn_weights_69, , sub, exp, softmax_17, attn_weights_70], Original ATen: [aten.mul, aten.add, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_add_exp_mul_prepare_softmax_online_sub_5.run(buf374, arg3_1, buf379, 408, 867, stream=stream0)
        del arg3_1
        del buf374
        buf377 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [linear_121], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg194_1, (1024, 256), (1, 1024), 0), out=buf377)
        del arg194_1
        buf378 = empty_strided_cuda((1, 1, 867, 256), (221952, 221952, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [values_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(arg196_1, buf377, buf378, 221952, stream=stream0)
        del arg196_1
        del buf377
        buf380 = reinterpret_tensor(buf373, (8, 51, 256), (13056, 256, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [attn_weights_68, attn_weights_69, sub, exp, softmax_17, attn_weights_70, attn_output_68, value_states_35], Original ATen: [aten.mul, aten.add, aten.sub, aten.exp, aten._softmax, aten._to_copy, aten.view, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf379, (8, 51, 867), (44224, 867, 1), 0), reinterpret_tensor(buf378, (8, 867, 256), (0, 256, 1), 0), out=buf380)
        del buf379
        buf381 = reinterpret_tensor(buf370, (1, 51, 8, 256), (104448, 2048, 256, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [attn_output_69], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf380, buf381, 104448, stream=stream0)
        del buf380
        buf382 = reinterpret_tensor(buf369, (51, 1024), (1024, 1), 0); del buf369  # reuse
        # Topologically Sorted Source Nodes: [attn_output_69, reshape_53, attn_output_71], Original ATen: [aten.clone, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf381, (51, 2048), (2048, 1), 0), reinterpret_tensor(arg197_1, (2048, 1024), (1, 2048), 0), out=buf382)
        del arg197_1
        del buf381
        buf384 = reinterpret_tensor(buf324, (1, 51, 1024), (52224, 1024, 1), 0); del buf324  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_102, hidden_states_106, float_75, square_35, var_35, add_159, rsqrt_35, normed_inputs_35, add_160, normed_35, hidden_states_107], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_15.run(buf346, buf361, buf367, buf382, arg198_1, buf384, 51, 1024, stream=stream0)
        del arg198_1
        buf385 = reinterpret_tensor(buf366, (51, 4096), (4096, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [linear_123], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg199_1, (1024, 4096), (1, 1024), 0), out=buf385)
        del arg199_1
        buf386 = buf365; del buf365  # reuse
        # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (51, 1024), (1024, 1), 0), reinterpret_tensor(arg200_1, (1024, 4096), (1, 1024), 0), out=buf386)
        del arg200_1
        buf387 = reinterpret_tensor(buf385, (1, 51, 4096), (208896, 4096, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [gelu_17, mul_181], Original ATen: [aten.gelu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_mul_9.run(buf387, buf386, 208896, stream=stream0)
        del buf386
        buf388 = reinterpret_tensor(buf384, (51, 1024), (1024, 1), 0); del buf384  # reuse
        # Topologically Sorted Source Nodes: [gelu_17, mul_181, down_proj_17], Original ATen: [aten.gelu, aten.mul, aten.view, aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (51, 4096), (4096, 1), 0), reinterpret_tensor(arg201_1, (4096, 1024), (1, 4096), 0), out=buf388)
        del arg201_1
        del buf387
        buf389 = buf346; del buf346  # reuse
        buf390 = reinterpret_tensor(buf4, (1, 51, 1), (51, 1, 51), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_102, hidden_states_106, hidden_states_108, float_77, square_36, var_36], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_pow_17.run(buf389, buf361, buf367, buf382, buf388, buf390, 51, 1024, stream=stream0)
        del buf361
        del buf367
        del buf382
        del buf388
        buf391 = empty_strided_cuda((1, 50, 1024), (51200, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_77, square_36, var_36, add_162, rsqrt_36, normed_inputs_36, add_163, normed_36, hidden_states_109, suffix_out, suffix_out_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.slice]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mean_mul_pow_rsqrt_slice_18.run(buf389, buf390, arg202_1, buf391, 51200, stream=stream0)
        del arg202_1
        del buf389
        del buf390
        buf392 = empty_strided_cuda((50, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_77, square_36, var_36, add_162, rsqrt_36, normed_inputs_36, add_163, normed_36, hidden_states_109, suffix_out, suffix_out_1, linear_126], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.slice, aten.view, aten.addmm]
        extern_kernels.addmm(arg204_1, reinterpret_tensor(buf391, (50, 1024), (1024, 1), 0), reinterpret_tensor(arg203_1, (1024, 32), (1, 1024), 0), alpha=1, beta=1, out=buf392)
        del arg203_1
        del arg204_1
        del buf391
    return (reinterpret_tensor(buf392, (1, 50, 32), (1600, 32, 1), 0), buf13, buf7, buf34, buf28, buf56, buf50, buf77, buf71, buf99, buf93, buf120, buf114, buf142, buf136, buf163, buf157, buf185, buf179, buf206, buf200, buf228, buf222, buf249, buf243, buf271, buf265, buf292, buf286, buf314, buf308, buf335, buf329, buf357, buf351, buf378, buf372, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 51, 1024), (52224, 1024, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, 51), (51, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((1, 1, 51, 867), (44217, 44217, 867, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg98_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg101_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg102_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg105_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg106_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg109_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg112_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg113_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg114_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg116_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg117_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg118_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg119_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg120_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg121_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg123_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg124_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg125_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg127_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg128_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg129_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg130_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg131_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg134_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg135_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg138_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg139_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg140_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg141_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg142_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg143_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg145_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg146_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg147_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg149_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg150_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg151_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg152_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg153_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg156_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg157_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg158_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg160_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg161_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg162_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg163_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg164_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg165_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg167_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg168_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg171_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg172_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg173_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg174_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg175_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg176_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg178_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg179_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg180_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg182_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg183_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg184_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg185_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg186_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg187_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg189_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg190_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg191_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg193_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg194_1 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg195_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg196_1 = rand_strided((1, 1, 816, 256), (208896, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg197_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg198_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg200_1 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg201_1 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg202_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
