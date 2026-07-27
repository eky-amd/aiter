# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import math

import torch
import triton

from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils._triton.arch_info import get_arch

_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)

# Stream-K work-decomposition variants; see _streamk_sk_tiles.
_STREAMK_VARIANTS = ("pure", "dp_1tile", "dp_2tile")

# Config keys carried by a tuned config that the Stream-K kernel does not accept
# (its K decomposition is Stream-K, not split-K) and that are not triton launch
# metaparams -- stripped before the kernel launch.
_STREAMK_DROP_CONFIG_KEYS = ("NUM_KSPLIT", "SPLITK_BLOCK_SIZE")

# Stream-K's own config identity, decoupled from the block-scale kernel so the
# two can be tuned independently. Backed by
# ``configs/gemm/gluon/{arch}-GEMM-A8W8_BLOCKSCALE_STREAMK.json``.
_STREAMK_CONFIG_NAME = "GEMM-A8W8_BLOCKSCALE_STREAMK"


def _is_gluon_available() -> bool:
    """Whether the Gluon Stream-K backend can run on the current GPU."""
    try:
        arch = get_arch()
        return any(s in arch for s in _GLUON_SUPPORTED_ARCHS)
    except Exception:
        return False


def _get_config(
    M: int,
    N: int,
    K: int,
    backend: Optional[str] = "gluon",
) -> tuple[dict, bool]:
    return get_gemm_config(_STREAMK_CONFIG_NAME, M, N, K, backend=backend)


def _sanitize_streamk_config(config: dict) -> dict:
    """Adapt a shared block-scale tuned config for the Stream-K kernel.

    Stream-K owns the K decomposition itself, so it ignores the split-K keys, and
    it software-pipelines the MAC loop with ``NUM_BUFFERS`` (mapped from the
    block-scale ``num_stages``). ``GROUP_K``/``GROUP_N`` are filled in from the
    scale-tensor shapes by ``_gemm_a8w8_streamk_impl``.
    """
    config = dict(config)
    # num_stages -> NUM_BUFFERS; the pipeline needs at least a double buffer. A
    # caller-supplied NUM_BUFFERS is honored when there is no num_stages to map.
    if "num_stages" in config:
        config["NUM_BUFFERS"] = max(2, int(config.pop("num_stages")))
    else:
        config["NUM_BUFFERS"] = max(2, int(config.get("NUM_BUFFERS", 2)))
    for key in _STREAMK_DROP_CONFIG_KEYS:
        config.pop(key, None)
    # The kernel passes cache_modifier straight to the load builtins; normalize a
    # missing/None value to the empty string the buffer ops expect.
    if config.get("cache_modifier") is None:
        config["cache_modifier"] = ""
    return config


# TODO: Double-check if this is the correct way to get the number of available SMs
# (Other Triton kernels write ...multi_processor_count * 2)
def _get_num_sms(device: torch.device) -> int:
    """Number of compute units to spread the Stream-K iteration space over."""
    return torch.cuda.get_device_properties(device).multi_processor_count


def _streamk_sk_tiles(variant: str, total_tiles: int, num_sms: int) -> int:
    """Number of output tiles assigned to the Stream-K region for a variant.

    This single function is the **only thing that differs between the three
    public entry points.
    
    ** The other difference is that the stream-k composition is performed before
    the data-parallel part for dp_2tile.
    """
    assert variant in _STREAMK_VARIANTS, f"unknown Stream-K variant '{variant}'"
    remainder = total_tiles % num_sms
    if variant == "pure":
        sk_tiles = total_tiles
    elif variant == "dp_1tile":
        # Only a partial-wave's worth of (trailing) tiles is load-balanced across
        # all CUs; the leading whole waves are data-parallel.
        sk_tiles = remainder
    else:  # dp_2tile
        # Partial wave + one extra full wave, so each Stream-K CTA gets a
        # meatier iteration slice (less fixup overhead per unit work). Only
        # add the wave when there is a full wave of DP work to spare.
        sk_tiles = remainder
        if total_tiles > num_sms:
            sk_tiles += num_sms
    return min(sk_tiles, total_tiles)


def _gemm_a8w8_streamk_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    variant: str,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    num_sms: Optional[int] = None,
    kernel_type: str = "bandwidth_bound",
    backend: Optional[str] = None,
):
    """Shared launch path for all three Stream-K variants.

    Args:
        x: fp8 e4m3 activations, shape (M, K).
        w: fp8 e4m3 weights, shape (N, K); internally transposed to (K, N).
        x_scale: per-block fp32 scale for x, shape (M, ceil(K / GROUP_K)).
        w_scale: per-block fp32 scale for w, shape (ceil(N / GROUP_N),
            ceil(K / GROUP_K)); internally transposed.
        variant: one of ``_STREAMK_VARIANTS``.
        num_sms: workgroups to launch (defaults to the device CU count).
        kernel_type: key into the kernel ``_KERNEL_MAP``.
        backend: only ``"gluon"`` is supported; kept for signature parity with
            the other GEMM entry points.

    Returns:
        y: output, shape (M, N).
    """
    if backend is None:
        backend = "gluon"
    backend = backend.lower()
    assert backend == "gluon", (
        f"Stream-K only supports the 'gluon' backend, got '{backend}'"
    )
    assert _is_gluon_available(), (
        f"Stream-K requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
    )
    # Imported lazily so importing this module does not require the gfx1250
    # Gluon toolchain on other arches.
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a8w8_blockscale_streamk import (
        _KERNEL_MAP,
    )

    _LOGGER.info(
        f"GEMM_A8W8_STREAMK[{variant}]: x={tuple(x.shape)} w={tuple(w.shape)} "
        f"x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    M, K = x.shape
    N, Kw = w.shape
    assert K == Kw, "Incompatible dimensions!!!"

    # Transpose w and w_scale so the kernel sees B as (K, N).
    w = w.T
    w_scale = w_scale.T

    # Load the tuned params (Stream-K's own config identity) when the caller did
    # not supply one, then adapt whichever config we have -- loaded or caller-
    # supplied -- to the keys the Stream-K kernel accepts.
    if config is None:
        config, _ = _get_config(M, N, K, backend=backend)
    config = _sanitize_streamk_config(config)

    assert kernel_type in _KERNEL_MAP, (
        f"unknown kernel_type '{kernel_type}', must be one of {list(_KERNEL_MAP)}"
    )

    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = config["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = config["BLOCK_SIZE_K"]

    # Scale block sizes, inferred from the scale tensor shapes.
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_scale.shape[0])) # scale_block_size_k = ceil(K / scale_k)
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N, w_scale.shape[1])) # scale_block_size_n = ceil(N / scale_n)
    assert config["GROUP_K"] == BLOCK_SIZE_K, (
        f"GROUP_K ({config['GROUP_K']}) must equal BLOCK_SIZE_K ({BLOCK_SIZE_K}); "
        "the block-scale post-multiply assumes one scale per K-tile."
    )

    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    iters_per_tile = triton.cdiv(K, BLOCK_SIZE_K)

    if num_sms is None:
        num_sms = _get_num_sms(x.device)

    sk_tiles = _streamk_sk_tiles(variant, total_tiles, num_sms)

    # Split tiles fold their partials into ``y`` with atomic adds, so ``y`` must
    # start at zero whenever the Stream-K region is non-empty; the solely-owned
    # and data-parallel tiles then overwrite their zeros with a direct store.
    # Pure-DP (sk_tiles == 0) has a single writer per tile, so no zero-init.
    if sk_tiles > 0:
        if y is None:
            y = torch.zeros((M, N), dtype=dtype, device=x.device)
        else:
            y.zero_()
    elif y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    # WMMA warp layout bases, matching the block-scale gfx1250 convention.
    warp_bases = [(0, 1)]
    for i in range(int(math.log2(config["num_warps"] // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)

    grid = (num_sms,)
    _KERNEL_MAP[kernel_type][grid](
        x,
        w,
        y,
        x_scale,
        w_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        num_sms,
        sk_tiles,
        iters_per_tile,
        total_tiles,
        num_pid_n,
        warp_bases=warp_bases,
        **config,
    )

    return y


def gemm_a8w8_streamk_pure(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    num_sms: Optional[int] = None,
    kernel_type: str = "bandwidth_bound",
    backend: Optional[str] = None,
):
    """Pure Stream-K: every output tile's K-reduction is spread across all
    ``num_sms`` workgroups (``sk_tiles = total_tiles``, no data-parallel tail).
    Best load balance; highest reduction/fixup overhead."""
    return _gemm_a8w8_streamk_impl(
        x, w, x_scale, w_scale, "pure", dtype, y, config, num_sms, kernel_type, backend
    )


def gemm_a8w8_streamk_dp_1tile(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    num_sms: Optional[int] = None,
    kernel_type: str = "bandwidth_bound",
    backend: Optional[str] = None,
):
    """One-tile hybrid: only a partial wave's worth of trailing tiles
    (``sk_tiles = total_tiles % num_sms``) is Stream-K'd; the leading full
    waves run data-parallel. Minimal fixup, removes the partial-wave tail."""
    return _gemm_a8w8_streamk_impl(
        x, w, x_scale, w_scale, "dp_1tile", dtype, y, config, num_sms, kernel_type, backend
    )


def gemm_a8w8_streamk_dp_2tile(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    num_sms: Optional[int] = None,
    kernel_type: str = "bandwidth_bound",
    backend: Optional[str] = None,
):
    """Two-tile hybrid: partial wave plus one extra full wave
    (``sk_tiles = total_tiles % num_sms + num_sms``) go to the Stream-K region.
    Each Stream-K CTA gets a larger, more even slice, cutting per-CTA fixup
    overhead vs. the one-tile variant on short-K / many-tile shapes."""
    return _gemm_a8w8_streamk_impl(
        x, w, x_scale, w_scale, "dp_2tile", dtype, y, config, num_sms, kernel_type, backend
    )
