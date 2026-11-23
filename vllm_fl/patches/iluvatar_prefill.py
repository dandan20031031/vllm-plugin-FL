# SPDX-License-Identifier: Apache-2.0
"""Strided fused KV gathering and per-step metadata reuse for BI-V150.

Uses the existing CoreX varlen primitive with bottom-right causal masking.
The packed-prefill / CPU-builder direction follows xy200303's Apache-2.0
vendor_flash_attn.py at plugin commit 0207b7e1. This adapter implements a
one/two-prefill layout, accepts independent strided K/V cache views,
keeps device decode lengths, and preserves the prior wrapper for all fallbacks.
https://github.com/xy200303/vllm-plugin-FL/blob/0207b7e1/vllm_fl/dispatch/backends/vendor/iluvatar/impl/vendor_flash_attn.py
"""

from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl
from vllm.logger import init_logger
from vllm.v1.attention.ops import triton_unified_attention as native
from vllm.v1.kv_cache_interface import KVQuantMode

try:
    import ixformer
except (ImportError, OSError):
    ixformer = None

logger = init_logger(__name__)
_descriptors = OrderedDict()
_pools = {}
_stats = Counter()
_fallback = None
_original_build = None


@triton.jit
def _gather_kv(K, V, P, CU_K, KOUT, VOUT,
               PS: tl.constexpr,
               KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
               VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
               BLOCK: tl.constexpr):
    # Dense sequence starts are token offsets, not page offsets. In particular,
    # a non-page-aligned first sequence must not add padding before sequence 2.
    sequence = tl.program_id(1)
    begin = tl.load(CU_K + sequence)
    end = tl.load(CU_K + sequence + 1)
    length = end - begin
    page_index = tl.program_id(0)
    page = tl.load(P + sequence * PS + page_index,
                   page_index * 16 < length, other=0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    row = offsets // 256
    h = (offsets % 256) // 128
    d = offsets % 128
    valid = page_index * 16 + row < length
    kval = tl.load(K + page * KS0 + row * KS1 + h * KS2 + d, valid, other=0)
    vval = tl.load(V + page * VS0 + row * VS1 + h * VS2 + d, valid, other=0)
    dst = begin * 256 + page_index * 4096 + offsets
    tl.store(KOUT + dst, kval, valid)
    tl.store(VOUT + dst, vval, valid)


@dataclass(frozen=True)
class _Step:
    n_decode: int
    q_len: int
    k_len: int
    n_seqs: int
    q_lengths: tuple
    k_lengths: tuple
    # Retain original tensors so allocator address reuse cannot hit stale keys.
    query_start: torch.Tensor
    seq_lens: torch.Tensor
    table: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor


def _key(cu, seq, table):
    return (cu.data_ptr(), seq.data_ptr(), table.data_ptr())


@lru_cache(maxsize=8)
def _device_ok(index):
    return 'iluvatar' in torch.cuda.get_device_name(index).lower()


def _decline(reason, lengths):
    name = 'metadata_fallback_' + reason
    _stats[name] += 1
    if _stats[name] == 1:
        logger.warning('eggking paged prefill metadata fallback: reason=%s query_lengths=%s',
                       reason, tuple(lengths))


def _attach_step(builder, common, result):
    key = _key(result.query_start_loc, result.seq_lens, result.block_table)
    # Always invalidate, including an unsupported step reusing the same buffers.
    _descriptors.pop(key, None)
    if ixformer is None or result.max_query_len < 128 or result.use_cascade:
        return
    if (builder.num_heads_q, builder.num_heads_kv, builder.headdim) != (16, 2, 128):
        return
    if builder.device.type != 'cuda' or not _device_ok(builder.device.index):
        return
    if getattr(builder.vllm_config, 'speculative_config', None) is not None:
        return
    if result.causal is not True or result.mm_prefix_range_tensor is not None:
        return
    cu_cpu = common.query_start_loc_cpu
    upper = common.seq_lens_cpu_upper_bound
    if upper is None or cu_cpu.device.type != 'cpu' or upper.device.type != 'cpu':
        return
    n = common.num_reqs
    if not 1 <= n <= 64 or cu_cpu.numel() < n + 1 or upper.numel() < n:
        return
    offsets = cu_cpu[:n + 1].tolist()
    lengths = [offsets[i + 1] - offsets[i] for i in range(n)]
    prefill = [i for i, length in enumerate(lengths) if length > 1]
    if not 1 <= len(prefill) <= 2:
        _decline('prefill_count', lengths)
        return
    nd = prefill[0]
    if prefill != list(range(nd, n)) or any(length != 1 for length in lengths[:nd]):
        _decline('interleaved_layout', lengths)
        return
    q_lengths = tuple(lengths[nd:])
    k_lengths = tuple(int(length) for length in upper[nd:n].tolist())
    nq, nk = sum(q_lengths), sum(k_lengths)
    if not (128 <= nq <= 2048 and nk <= 65536
            and all(qi <= ki <= 32768 for qi, ki in zip(q_lengths, k_lengths))):
        _decline('length_limits', lengths)
        return
    if offsets[0] != 0 or offsets[-1] != result.num_actual_tokens:
        return
    if len(result.seq_lens) != n:
        return
    # The CPU upper bound is exact on prefill only. Decode uses the original
    # device seq_lens below, never these CPU bounds.
    q_offsets, k_offsets = [0], [0]
    for qi, ki in zip(q_lengths, k_lengths):
        q_offsets.append(q_offsets[-1] + qi)
        k_offsets.append(k_offsets[-1] + ki)
    cu_q = torch.tensor(q_offsets, dtype=torch.int32, device=builder.device)
    cu_k = torch.tensor(k_offsets, dtype=torch.int32, device=builder.device)
    _descriptors[key] = _Step(nd, nq, nk, n, q_lengths, k_lengths, result.query_start_loc,
                              result.seq_lens, result.block_table, cu_q, cu_k)
    while len(_descriptors) > 8:
        _descriptors.popitem(last=False)
    _stats['metadata_steps'] += 1


def _build(self, common_prefix_len, common_attn_metadata, fast_build=False):
    result = _original_build(self, common_prefix_len, common_attn_metadata, fast_build)
    _attach_step(self, common_attn_metadata, result)
    return result


def _pool(q, length):
    stream = torch.cuda.current_stream(q.device).cuda_stream
    key = (q.device.index, stream)
    pair = _pools.get(key)
    if pair is None or pair[0].shape[0] < length:
        capacity = max(512, triton.next_power_of_2(length))
        pair = (torch.empty((capacity, 2, 128), dtype=q.dtype, device=q.device),
                torch.empty((capacity, 2, 128), dtype=q.dtype, device=q.device))
        _pools[key] = pair
        _stats['pool_allocations'] += 1
    return pair[0][:length], pair[1][:length]


def mixed_attention(
    q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
    softmax_scale, causal, window_size, block_table, softcap,
    q_descale, k_descale, v_descale, seq_threshold_3D=None,
    num_par_softmax_segments=None, softmax_segm_output=None,
    softmax_segm_max=None, softmax_segm_expsum=None, alibi_slopes=None,
    output_scale=None, qq_bias=None, sinks=None, mm_prefix_range=None,
    use_alibi_sqrt=False, kv_quant_mode=KVQuantMode.NONE,
    k_scale_cache=None, v_scale_cache=None, chunk_lookback=-1, use_td=False,
):
    def prior():
        return _fallback(
            q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
            softmax_scale, causal, window_size, block_table, softcap,
            q_descale, k_descale, v_descale, seq_threshold_3D,
            num_par_softmax_segments, softmax_segm_output, softmax_segm_max,
            softmax_segm_expsum, alibi_slopes, output_scale, qq_bias, sinks,
            mm_prefix_range, use_alibi_sqrt, kv_quant_mode, k_scale_cache,
            v_scale_cache, chunk_lookback, use_td)

    if max_seqlen_q < 128:
        return prior()
    eligible = (
        q.device.type == 'cuda' and q.dtype == k.dtype == v.dtype == out.dtype == torch.bfloat16
        and q.ndim == out.ndim == 3 and q.shape[1:] == (16, 128) and out.shape == q.shape
        and q.stride()[-2:] == out.stride()[-2:] == (128, 1)
        and q.stride(0) == out.stride(0) == 2048
        and k.ndim == v.ndim == 4 and k.shape[1:] == v.shape[1:] == (16, 2, 128)
        and k.stride(-1) == v.stride(-1) == block_table.stride(-1) == 1
        and causal is True and tuple(window_size) == (-1, -1) and softcap == 0
        and alibi_slopes is None and output_scale is None and qq_bias is None
        and sinks is None and mm_prefix_range is None and not use_alibi_sqrt
        and kv_quant_mode == KVQuantMode.NONE and k_scale_cache is None and v_scale_cache is None
        and chunk_lookback == -1 and not use_td and not native.is_batch_invariant
        and not any(t.requires_grad for t in (q, k, v, out))
    )
    if not eligible or not _device_ok(q.device.index):
        return prior()
    step = _descriptors.get(_key(cu_seqlens_q, seqused_k, block_table))
    if step is None or q.shape[0] != step.n_decode + step.q_len:
        _stats['descriptor_fallback'] += 1
        return prior()
    if torch.cuda.is_current_stream_capturing():
        return prior()
    nd = step.n_decode
    if nd and (softmax_segm_output is None or seq_threshold_3D is None
               or num_par_softmax_segments is None):
        return prior()
    try:
        # Preallocate before writing output. Only basic views, no per-layer
        # D2H, advanced indexing, .long() casts or data-dependent allocations.
        packed_k, packed_v = _pool(q, step.k_len)
        pages = block_table[nd:]
        if nd:
            _fallback(
                q[:nd], k, v, out[:nd], cu_seqlens_q[:nd + 1], 1,
                seqused_k[:nd], max_seqlen_k, softmax_scale, True, (-1, -1),
                block_table[:nd], 0, q_descale, k_descale, v_descale,
                seq_threshold_3D, num_par_softmax_segments,
                softmax_segm_output, softmax_segm_max, softmax_segm_expsum)
        _gather_kv[((max(step.k_lengths) + 15) // 16, len(step.k_lengths))](
            k, v, pages, step.cu_k, packed_k, packed_v, pages.stride(0),
            *k.stride()[:3], *v.stride()[:3], BLOCK=4096, num_warps=4)
        ixformer.flash_attn_varlen_func(
            q[nd:], packed_k, packed_v, step.cu_q, step.cu_k,
            max(step.q_lengths), max(step.k_lengths), dropout_p=0.0, softmax_scale=softmax_scale,
            causal=True, out=out[nd:])
        _stats['prefill_dispatched'] += 1
        _stats['decode_rows'] += nd
        if _stats['prefill_dispatched'] == 1:
            logger.warning('eggking paged prefill dispatched: fused strided KV gather, q=%d k=%d decode=%d',
                           step.q_len, step.k_len, nd)
        if len(step.q_lengths) == 2:
            _stats['multi_prefill_dispatched'] += 1
            if _stats['multi_prefill_dispatched'] == 1:
                logger.warning('eggking multi-prefill dispatched: query_lengths=%s key_lengths=%s decode=%d',
                               step.q_lengths, step.k_lengths, nd)
            if max(step.q_lengths) >= 1024:
                _stats['long_multi_prefill_dispatched'] += 1
                if _stats['long_multi_prefill_dispatched'] == 1:
                    logger.warning('eggking long multi-prefill dispatched: query_lengths=%s key_lengths=%s decode=%d',
                                   step.q_lengths, step.k_lengths, nd)
        return
    except Exception:
        _stats['exceptions'] += 1
        if _stats['exceptions'] == 1:
            logger.exception('eggking paged prefill failed; retrying the prior full wrapper')
        return prior()


mixed_attention._eggking_prefill_tiling = True
mixed_attention._eggking_paged_prefill = True
mixed_attention._eggking_multi_prefill = True


def patch_paged_prefill():
    global _fallback, _original_build
    if ixformer is None or getattr(native.unified_attention, '_eggking_paged_prefill', False):
        return False
    from vllm.v1.attention.backends import triton_attn
    _fallback = native.unified_attention
    _original_build = triton_attn.TritonAttentionMetadataBuilder.build
    triton_attn.TritonAttentionMetadataBuilder.build = _build
    native.unified_attention = mixed_attention
    triton_attn.unified_attention = mixed_attention
    logger.warning('eggking paged prefill registered: one/two-prefill step metadata and fused KV gather')
    return True
