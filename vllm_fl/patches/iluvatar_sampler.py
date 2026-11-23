# SPDX-License-Identifier: Apache-2.0
# Launch and cache handling follows vLLM's existing topk_topp_triton implementation.
# Copyright contributors to the vLLM project; row launch policy by eggking.
"""Expose independent row work instead of one persistent program per SM.

The upstream Triton kernel, tile sizes, warp count, RNG and sampling thresholds
are unchanged. Limit the policy to the validated model layout on Iluvatar.
"""

from functools import lru_cache
import torch
from vllm.v1.sample.ops import topk_topp_triton as kernel_module

_original = kernel_module.apply_top_k_top_p_triton


@lru_cache(maxsize=16)
def _launch_limit(index):
    if 'iluvatar' not in torch.cuda.get_device_name(index).lower():
        return 0
    return 4 * kernel_module.num_compute_units(index)


def apply_top_k_top_p_row_parallel(logits, k, p, mask_value=float('-inf')):
    if (logits.device.type != 'cuda' or logits.dtype != torch.float32 or
            logits.ndim != 2 or logits.shape[1] != 130560 or
            not 32 <= logits.shape[0] <= 256 or logits.stride(1) != 1 or
            k is not None or p is None):
        return _original(logits, k, p, mask_value)
    index = logits.device.index
    limit = _launch_limit(index if index is not None else torch.cuda.current_device())
    if not limit:
        return _original(logits, k, p, mask_value)
    batch, vocab = logits.shape
    assert p.ndim == 1 and p.shape[0] == batch
    programs = min(batch, limit)
    key = (logits.device, logits.dtype, vocab)
    scratch = kernel_module._TRITON_BUFFER_CACHE.get(key)
    if scratch is None or scratch.shape[0] < programs:
        size = min(kernel_module.next_power_of_2(programs), limit)
        scratch = logits.new_empty((size, vocab))
        kernel_module._TRITON_BUFFER_CACHE[key] = scratch
    if scratch.shape[0] > programs:
        scratch = scratch[:programs]
    tables = kernel_module._TRITON_TABLE_CACHE.get(logits.device)
    if tables is None:
        normal = logits.new_tensor(kernel_module._NORMAL_CDF_TO_SIGMA_TABLE)
        percentile = logits.new_tensor(kernel_module._PERCENTILE_TO_STD_TABLE)
        kernel_module._TRITON_TABLE_CACHE[logits.device] = (normal, percentile)
    else:
        normal, percentile = tables
    kernel_module._topk_topp_kernel[(programs,)](
        logits, logits.stride(0), scratch, percentile, normal, logits, p.to(torch.float32),
        BATCH_SIZE=batch, VOCAB_SIZE=vocab, MASK_VALUE=mask_value,
        BLOCK_SIZE=8192, BLOCK_SIZE_TRUNC=4096,
        TOPK_ENABLED=False, TOPP_ENABLED=True, num_warps=4)
    return logits


apply_top_k_top_p_row_parallel._eggking_row_parallel = True


def patch_iluvatar_sampler():
    """Install idempotently in plugin workers; other devices/layouts use stock."""
    if getattr(kernel_module.apply_top_k_top_p_triton, '_eggking_row_parallel', False):
        return False
    from vllm.v1.sample.ops import topk_topp_sampler
    kernel_module.apply_top_k_top_p_triton = apply_top_k_top_p_row_parallel
    topk_topp_sampler.apply_top_k_top_p_triton = apply_top_k_top_p_row_parallel
    return True
