# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Narrow attention launch policies by eggking; pinned kernel math is unchanged.
"""Reuse KV tiles in prefill and avoid excess decode launch resources.

The default GQA=8 launch groups only two query tokens in each program.  For
long prefill, BLOCK_M=128 groups sixteen, reducing repeated KV loads and loop
work. Decode and all unvalidated feature/layout combinations use the original
wrapper. For ordinary BF16 3D decode, use one warp and one pipeline stage.
The decode launch direction and proxy structure are adapted from xy200303's
Apache-2.0 A7 implementation; eggking adds the narrow feature/shape guard and
validates this policy against its own frozen stack. Source:
https://github.com/xy200303/vllm-plugin-FL/blob/0207b7e1/vllm_fl/dispatch/backends/vendor/iluvatar/impl/decode_attn_pin.py
No attention math, KV representation or serving settings change.
"""

from functools import lru_cache

import torch
from vllm.logger import init_logger
from vllm.v1.attention.ops import triton_unified_attention as native
from vllm.v1.kv_cache_interface import KVQuantMode

_original = native.unified_attention
logger = init_logger(__name__)
_announced = False


class _DecodeLaunchPolicy:
    """Retain the native kernel; specialize only the validated 3D shape."""
    _eggking_decode_launch = True

    def __init__(self, kernel):
        self.kernel = kernel

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    @staticmethod
    def eligible(kw):
        q, k, v = kw.get('query_ptr'), kw.get('key_cache_ptr'), kw.get('value_cache_ptr')
        if not all(isinstance(t, torch.Tensor) for t in (q, k, v)):
            return False
        if not (
            kw.get('IS_3D') is True and q.device.type == 'cuda'
            and q.dtype == k.dtype == v.dtype == torch.bfloat16
            and q.ndim == 3 and 1 <= q.shape[0] <= 64 and q.shape[1:] == (16, 128)
            and k.ndim == v.ndim == 4 and k.shape[1:] == v.shape[1:] == (16, 2, 128)
            and all(t.stride(-1) == 1 for t in (q, k, v))
            and kw.get('BLOCK_M') == 16 and kw.get('BLOCK_Q') == 2
            and kw.get('TILE_SIZE') == 16 and kw.get('NUM_SEGMENTS_PER_SEQ') == 16
            and kw.get('KV_QUANT_MODE') == KVQuantMode.NONE
            and kw.get('USE_CAUSAL') is True and kw.get('SLIDING_WINDOW') == 0
            and kw.get('CHUNK_LOOKBACK') == -1
            and not any(kw.get(key, False) for key in (
                'USE_ALIBI_SLOPES', 'USE_ALIBI_SQRT', 'USE_QQ_BIAS', 'USE_SOFTCAP',
                'USE_SINKS', 'USE_PER_SEQ_CAUSAL', 'USE_MM_PREFIX', 'USE_FP8',
                'Q_IS_FP8', 'USE_TD', 'USE_TD_QO'))
        ):
            return False
        index = q.device.index
        return _is_iluvatar(index if index is not None else torch.cuda.current_device())

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(**kw):
            if self.eligible(kw):
                kw['num_warps'], kw['num_stages'] = 1, 1
            return launch(**kw)
        return run


@lru_cache(maxsize=16)
def _is_iluvatar(index):
    return 'iluvatar' in torch.cuda.get_device_name(index).lower()


def unified_attention_prefill(
    q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
    softmax_scale, causal, window_size, block_table, softcap,
    q_descale, k_descale, v_descale, seq_threshold_3D=None,
    num_par_softmax_segments=None, softmax_segm_output=None,
    softmax_segm_max=None, softmax_segm_expsum=None, alibi_slopes=None,
    output_scale=None, qq_bias=None, sinks=None, mm_prefix_range=None,
    use_alibi_sqrt=False, kv_quant_mode=KVQuantMode.NONE,
    k_scale_cache=None, v_scale_cache=None, chunk_lookback=-1, use_td=False,
):
    eligible = (
        q.device.type == 'cuda' and q.dtype == torch.bfloat16
        and k.dtype == v.dtype == out.dtype == q.dtype
        and q.ndim == out.ndim == 3 and k.ndim == v.ndim == 4
        and q.shape[1:] == (16, 128) and out.shape == q.shape
        and k.shape[1:] == v.shape[1:] == (16, 2, 128)
        and all(t.stride(-1) == 1 for t in (q, k, v, out))
        and max_seqlen_q >= 128 and max_seqlen_k >= 512
        and causal is True and tuple(window_size) == (-1, -1) and softcap == 0
        and alibi_slopes is None and output_scale is None and qq_bias is None
        and sinks is None and mm_prefix_range is None and not use_alibi_sqrt
        and kv_quant_mode == KVQuantMode.NONE
        and k_scale_cache is None and v_scale_cache is None
        and chunk_lookback == -1 and not use_td and not native.is_batch_invariant
    )
    if eligible:
        index = q.device.index
        eligible = _is_iluvatar(index if index is not None else torch.cuda.current_device())
    if not eligible:
        return _original(
            q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
            softmax_scale, causal, window_size, block_table, softcap,
            q_descale, k_descale, v_descale, seq_threshold_3D,
            num_par_softmax_segments, softmax_segm_output, softmax_segm_max,
            softmax_segm_expsum, alibi_slopes, output_scale, qq_bias, sinks,
            mm_prefix_range, use_alibi_sqrt, kv_quant_mode, k_scale_cache,
            v_scale_cache, chunk_lookback, use_td,
        )

    # Same upper-bound grid construction and kernel arguments as the pinned
    # native 2D path. Only BLOCK_M/BLOCK_Q and the dependent grid change.
    global _announced
    if not _announced:
        logger.info('eggking Iluvatar prefill: BLOCK_M=128, q=%s, max_q=%s, max_k=%s',
                    tuple(q.shape), max_seqlen_q, max_seqlen_k)
        _announced = True
    block_m, block_q = 128, 16
    num_seqs = len(seqused_k)
    grid = (q.shape[0] // block_q + num_seqs, 2)
    native.kernel_unified_attention[grid](
        output_ptr=out, segm_output_ptr=out, segm_max_ptr=out, segm_expsum_ptr=out,
        query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, sink_ptr=None,
        block_tables_ptr=block_table, seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=None, qq_bias_ptr=None,
        k_scale_cache_ptr=k, v_scale_cache_ptr=v,
        scale=softmax_scale, q_scale=q_descale, k_scale=k_descale, v_scale=v_descale,
        out_scale=1.0, softcap=softcap, num_query_heads=16, num_queries_per_kv=8,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0), query_stride_1=q.stride(1),
        output_stride_0=out.stride(0), output_stride_1=out.stride(1), qq_bias_stride_0=0,
        BLOCK_SIZE=16, TILE_SIZE=native._get_tile_size(128, 0, q.element_size(), is_prefill=True),
        HEAD_SIZE=128, HEAD_SIZE_PADDED=128, USE_ALIBI_SLOPES=False,
        USE_ALIBI_SQRT=False, USE_QQ_BIAS=False, USE_SOFTCAP=False, USE_SINKS=False,
        SLIDING_WINDOW=0, USE_CAUSAL=True, USE_PER_SEQ_CAUSAL=False,
        per_seq_causal_ptr=None, USE_MM_PREFIX=False, MAX_MM_RANGES=0,
        mm_prefix_range_ptr=None,
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        stride_ks_blk=0, stride_ks_slot=0, stride_ks_head=0,
        stride_vs_blk=0, stride_vs_slot=0, stride_vs_head=0,
        query_start_len_ptr=cu_seqlens_q, BLOCK_Q=block_q, num_seqs=num_seqs,
        BLOCK_M=block_m, NUM_SEGMENTS_PER_SEQ=1, USE_FP8=False, IS_3D=False,
        KV_QUANT_MODE=kv_quant_mode, Q_IS_FP8=False, CHUNK_LOOKBACK=-1,
        CHUNK_SIZE=-1, USE_TD=False, USE_TD_QO=False,
    )


unified_attention_prefill._eggking_prefill_tiling = True


def patch_iluvatar_attention():
    """Install once in the native module and its backend consumer alias."""
    changed = False
    if not getattr(native.kernel_unified_attention, '_eggking_decode_launch', False):
        native.kernel_unified_attention = _DecodeLaunchPolicy(native.kernel_unified_attention)
        logger.info('eggking Iluvatar decode: guarded BF16 3D launch uses one warp/stage')
        changed = True
    if not getattr(native.unified_attention, '_eggking_prefill_tiling', False):
        from vllm.v1.attention.backends import triton_attn
        native.unified_attention = unified_attention_prefill
        triton_attn.unified_attention = unified_attention_prefill
        changed = True
    from vllm_fl.patches.iluvatar_prefill import patch_paged_prefill
    return patch_paged_prefill() or changed
