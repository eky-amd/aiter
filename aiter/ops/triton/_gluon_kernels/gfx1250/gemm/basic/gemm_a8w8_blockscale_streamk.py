# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils.logger import AiterTritonLogger  # debug
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_LOGGER = AiterTritonLogger()

_GLUON_REPR_KEYS = [
    "GROUP_K",
    "GROUP_N",
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "EVEN_K",
    "num_warps",
    "cache_modifier",
    "NUM_BUFFERS",
]

_gemm_a8w8_streamk_bandwidth_bound_repr = make_kernel_repr(
    "_gemm_a8w8_streamk_gfx1250_bandwidth_bound_kernel", _GLUON_REPR_KEYS
)


# ---------------------------------------------------------------------------
# MAC loop: reduce a contiguous K-tile range [local_iter, local_iter_end) of one output tile.
# ---------------------------------------------------------------------------
@gluon.jit
def _load_ab_scale(
    a_scale_ptr,
    b_scale_ptr,
    offs_a_scale,
    offs_b_scale,
    b_scale_scalar_off,
    local_iter,
    k_idx,
    iter_count,
    stride_ascale_k,
    stride_bscale_k,
    SCALAR_B_SCALE: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    k = local_iter + gl.minimum(k_idx, iter_count - 1)
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=a_scale_ptr + k * stride_ascale_k,
        offsets=offs_a_scale,
        cache=cache_modifier,
    )
    if SCALAR_B_SCALE:
        b_scale = gl.load(
            b_scale_ptr + k * stride_bscale_k + b_scale_scalar_off,
            cache_modifier=cache_modifier,
        )
    else:
        b_scale = gl.amd.cdna4.buffer_load(
            ptr=b_scale_ptr + k * stride_bscale_k,
            offsets=offs_b_scale,
            cache=cache_modifier,
        )
    return a_scale, b_scale


@gluon.jit
def _streamk_mac_range(
    a_desc,
    b_desc,
    tdm_smem_a,
    tdm_smem_b,
    a_scale_ptr,
    b_scale_ptr,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    M,
    N,
    pid_m,
    pid_n,
    local_iter,
    local_iter_end,
    GROUP_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    wmma_layout: gl.constexpr,
    warp_bases: gl.constexpr,
    SCALAR_B_SCALE: gl.constexpr,
    cache_modifier: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=8
    )

    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=wmma_layout)
    zeros = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=wmma_layout)

    offs_am = (
        pid_m * BLOCK_SIZE_M
        + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_layout))
    ) % M
    offs_bn = (
        pid_n * BLOCK_SIZE_N
        + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_layout))
    ) % N
    offs_a_scale = offs_am * stride_ascale_m
    
    offs_b_scale = (offs_bn // GROUP_N) * stride_bscale_n
    b_scale_scalar_off = ((pid_n * BLOCK_SIZE_N) // GROUP_N) * stride_bscale_n

    off_am_tdm = pid_m * BLOCK_SIZE_M
    off_bn_tdm = pid_n * BLOCK_SIZE_N

    # NOTE: This is just the number of iterations for this MAC-loop
    iter_count = local_iter_end - local_iter
    num_loads = 0
    num_computes = 0

    # -------------------- Prologue: issue NUM_BUFFERS - 1 loads --------------    
    for _ in gl.static_range(NUM_BUFFERS - 1):
        # NOTE: If the number of loads extends past the iter_count, then we just reload the last "valid" K block as dummy values
        load_idx = local_iter + gl.minimum(num_loads, iter_count - 1)
        gl.amd.gfx1250.tdm.async_load(
            a_desc,
            [off_am_tdm, load_idx * BLOCK_SIZE_K],
            tdm_smem_a.index(num_loads % NUM_BUFFERS),
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc,
            [off_bn_tdm, load_idx * BLOCK_SIZE_K],
            tdm_smem_b.index(num_loads % NUM_BUFFERS),
        )
        num_loads += 1

    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

    cur_a = tdm_smem_a.index(num_computes % NUM_BUFFERS).load(layout=dot_a_layout)
    cur_b = (
        tdm_smem_b.index(num_computes % NUM_BUFFERS)
        .permute((1, 0))
        .load(layout=dot_b_layout)
    )
        
    # -------------------- Main loop -----------------------------------------
    for _ in range(iter_count - (NUM_BUFFERS - 1)):
        # NOTE: a_scale and b_scale are globally buffer-loaded instead of from smem for each loop iteration
        a_scale, b_scale = _load_ab_scale(
            a_scale_ptr,
            b_scale_ptr,
            offs_a_scale,
            offs_b_scale,
            b_scale_scalar_off,
            local_iter,
            num_computes,
            iter_count,
            stride_ascale_k,
            stride_bscale_k,
            SCALAR_B_SCALE,
            cache_modifier,
        )
        res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
        if SCALAR_B_SCALE:
            acc += res * a_scale[:, None] * b_scale
        else:
            acc += res * a_scale[:, None] * b_scale[None, :]

        load_idx = local_iter + gl.minimum(num_loads, iter_count - 1)
        gl.amd.gfx1250.tdm.async_load(
            a_desc,
            [off_am_tdm, load_idx * BLOCK_SIZE_K],
            tdm_smem_a.index(num_loads % NUM_BUFFERS),
            pred=1,
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc,
            [off_bn_tdm, load_idx * BLOCK_SIZE_K],
            tdm_smem_b.index(num_loads % NUM_BUFFERS),
            pred=1,
        )
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
        num_loads += 1

        cur_a = tdm_smem_a.index((num_computes + 1) % NUM_BUFFERS).load(
            layout=dot_a_layout
        )
        cur_b = (
            tdm_smem_b.index((num_computes + 1) % NUM_BUFFERS)
            .permute((1, 0))
            .load(layout=dot_b_layout)
        )
        num_computes += 1

    # -------------------- Epilogue: drain the pipeline ----------------------
    for i in gl.static_range(NUM_BUFFERS - 2):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3 - i) * 2)
        next_a = tdm_smem_a.index((num_computes + 1) % NUM_BUFFERS).load(
            layout=dot_a_layout
        )
        next_b = (
            tdm_smem_b.index((num_computes + 1) % NUM_BUFFERS)
            .permute((1, 0))
            .load(layout=dot_b_layout)
        )
        a_scale, b_scale = _load_ab_scale(
            a_scale_ptr,
            b_scale_ptr,
            offs_a_scale,
            offs_b_scale,
            b_scale_scalar_off,
            local_iter,
            num_computes,
            iter_count,
            stride_ascale_k,
            stride_bscale_k,
            SCALAR_B_SCALE,
            cache_modifier,
        )
        res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
        valid = (num_computes < iter_count).to(gl.float32)
        if SCALAR_B_SCALE:
            acc += res * a_scale[:, None] * b_scale * valid
        else:
            acc += res * a_scale[:, None] * b_scale[None, :] * valid
        cur_a = next_a
        cur_b = next_b
        num_computes += 1

    # -------------------- Final tile ----------------------------------------
    a_scale, b_scale = _load_ab_scale(
        a_scale_ptr,
        b_scale_ptr,
        offs_a_scale,
        offs_b_scale,
        b_scale_scalar_off,
        local_iter,
        num_computes,
        iter_count,
        stride_ascale_k,
        stride_bscale_k,
        SCALAR_B_SCALE,
        cache_modifier,
    )
    res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
    valid = (num_computes < iter_count).to(gl.float32)
    if SCALAR_B_SCALE:
        acc += res * a_scale[:, None] * b_scale * valid
    else:
        acc += res * a_scale[:, None] * b_scale[None, :] * valid

    return acc


@gluon.jit
def _store_tile(
    acc,
    c_ptr,
    tdm_shared_c,
    tdm_smem_c,
    pid_m,
    pid_n,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
):
    tdm_smem_c.store(acc.to(c_ptr.type.element_ty))
    # Wait for every warp to finish writing its acc fragment before the store.
    gl.barrier()

    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        layout=tdm_shared_c,
    )
    gl.amd.gfx1250.tdm.async_store(
        c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], tdm_smem_c
    )
    gl.amd.gfx1250.tdm.async_wait(0)


@gluon.jit
def _atomic_add_tile(
    acc,
    c_ptr,
    pid_m,
    pid_n,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    wmma_layout: gl.constexpr,
):
    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_layout)
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_layout)
    )
    offs = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # gfx1250 drives all buffer memory ops through the cdna4 namespace (same as
    # the buffer_load/buffer_store used throughout this kernel); buffer_atomic_add
    # is its atomic sibling and carries the explicit bf16 fadd path.
    
    # TODO: Is this right? Does not seem to be a gl.amd.gfx1250.buffer_atomic_add at the moment 
    gl.atomic_add(pointer=(c_ptr + offs), val=acc.to(c_ptr.type.element_ty), mask=mask)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@gluon.jit(repr=_gemm_a8w8_streamk_bandwidth_bound_repr)
def _gemm_a8w8_streamk_bandwidth_bound_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,  # output y (zero-initialised when split tiles can occur)
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Stream-K decomposition
    num_sms,
    sk_tiles,
    iters_per_tile,
    total_tiles,
    num_pid_n,
    # Meta-parameters
    GROUP_K: gl.constexpr,
    GROUP_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    EVEN_K: gl.constexpr,
    num_warps: gl.constexpr,
    warp_bases: gl.constexpr,
    cache_modifier: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    # program setup — stream-K decomposition
    pid = gl.program_id(axis=0)
    total_sk_iters = sk_tiles * iters_per_tile
    base = total_sk_iters // num_sms
    extra = total_sk_iters % num_sms

    # acc layout
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        3, True, warp_bases, [], [16, 16, 128]
    )
    
    # TDM Shared Layouts
    tdm_shared_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_K, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_K], [1, 0]
    )
    tdm_shared_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_K, 8]], [BLOCK_SIZE_N, BLOCK_SIZE_K], [1, 0]
    )
    
    # Fast path: when a single scale group spans the whole tile in both N and K,
    # every column shares one b_scale, so load it with a single scalar global
    # load (folded into the a-scale at the multiply) instead of a BLOCK_SIZE_N
    # vector load of identical values.
    SCALAR_B_SCALE: gl.constexpr = (GROUP_N >= BLOCK_SIZE_N) and (
        GROUP_K >= BLOCK_SIZE_K
    )

    # TDM tensor descriptors — per-tile K position is supplied via the async_load offset
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        layout=tdm_shared_a,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        layout=tdm_shared_b,
    )
    
    # NUM_BUFFERS-deep A/B shared buffers feed the software-pipelined MAC loop;
    # allocated once and reused across every segment / DP tile.
    tdm_smem_a = gl.allocate_shared_memory(
        a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=tdm_shared_a
    )
    tdm_smem_b = gl.allocate_shared_memory(
        b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=tdm_shared_b
    )

    # Output staging buffer for the TDM async_store of fully-owned / DP tiles.
    tdm_shared_c: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N], [1, 0]
    )
    tdm_smem_c = gl.allocate_shared_memory(
        c_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_N], layout=tdm_shared_c
    )

    # Stream-K owns the *trailing* sk_tiles output tiles; the leading dp_tiles
    # are pure data-parallel. The flattened iteration space [0, total_sk_iters)
    # therefore indexes tiles starting at dp_tiles, not 0.
    dp_tiles = total_tiles - sk_tiles

    # Contiguous iteration range for this workgroup. The first `extra` CTAs
    # get one extra iter so the whole [0, total_sk_iters) space is covered.
    if pid < extra:
        iter = pid * base + pid
        iter_end = iter + base + 1
    else:
        iter = pid * base + extra
        iter_end = iter + base

    # -------------------- Data-parallel lead --------------------
    # The leading dp_tiles output tiles are whole waves, one workgroup each.
    tile_id = pid
    while tile_id < dp_tiles:
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        acc = _streamk_mac_range(
            a_desc,
            b_desc,
            tdm_smem_a,
            tdm_smem_b,
            a_scale_ptr,
            b_scale_ptr,
            stride_ascale_m,
            stride_ascale_k,
            stride_bscale_k,
            stride_bscale_n,
            M,
            N,
            pid_m,
            pid_n,
            0,
            iters_per_tile,
            GROUP_N,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            wmma_layout,
            warp_bases,
            SCALAR_B_SCALE,
            cache_modifier,
            NUM_BUFFERS,
        )
        _store_tile(
            acc,
            c_ptr,
            tdm_shared_c,
            tdm_smem_c,
            pid_m,
            pid_n,
            M,
            N,
            stride_cm,
            stride_cn,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )
        tile_id += num_sms

    # -------------------- Stream-K loop (trailing tiles) --------------------
    while iter < iter_end:
        # tile_idx indexes the Stream-K iteration space [0, sk_tiles); the
        # output tile it maps to is offset by the data-parallel lead.
        tile_idx = iter // iters_per_tile
        tile_iter = tile_idx * iters_per_tile
        tile_iter_end = tile_iter + iters_per_tile
        local_iter = iter - tile_iter
        local_iter_end = tile_iter_end - tile_iter
        if iter_end < tile_iter_end:
            local_iter_end = iter_end - tile_iter

        tile_id = dp_tiles + tile_idx
        
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

        acc = _streamk_mac_range(
            a_desc,
            b_desc,
            tdm_smem_a,
            tdm_smem_b,
            a_scale_ptr,
            b_scale_ptr,
            stride_ascale_m,
            stride_ascale_k,
            stride_bscale_k,
            stride_bscale_n,
            M,
            N,
            pid_m,
            pid_n,
            local_iter,
            local_iter_end,
            GROUP_N,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            wmma_layout,
            warp_bases,
            SCALAR_B_SCALE,
            cache_modifier,
            NUM_BUFFERS,
        )

        # NOTE: Equivalent to tile_started && tile_ended (i.e. iter == tile_iter && iter_end >= tile_iter_end)
        if (local_iter == 0) and (local_iter_end == iters_per_tile):
            # Solely-owned tile -> final result, write straight to y.
            _store_tile(
                acc,
                c_ptr,
                tdm_shared_c,
                tdm_smem_c,
                pid_m,
                pid_n,
                M,
                N,
                stride_cm,
                stride_cn,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )
        else:
            # Partial K-range of a split tile means atomically add into y.
            _atomic_add_tile(
                acc,
                c_ptr,
                pid_m,
                pid_n,
                M,
                N,
                stride_cm,
                stride_cn,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                wmma_layout,
            )

        iter = tile_iter_end


# Compute-bound currently aliases bandwidth-bound (single-buffered MAC loop);
# split them once the MAC loop grows a ds_read/wmma pipeline.
_KERNEL_MAP = {
    "bandwidth_bound": _gemm_a8w8_streamk_bandwidth_bound_kernel,
    "compute_bound": _gemm_a8w8_streamk_bandwidth_bound_kernel,
}
