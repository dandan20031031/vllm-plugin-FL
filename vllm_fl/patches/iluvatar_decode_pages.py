# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from xy200303/FlagGems commit 9c06143c, blob
# 21c41bc3244e84b8256c547f49746f5b25a53872 (Apache-2.0).
# eggking changes: device query-row mapping and empty-row safety; bounded
# metadata qualification, stream-owned preallocated slabs, and integration
# with the frozen b332 fallback. The public attention core is attributed above.
# Capture preparation below preserves the platform's selected CUDA stream and
# original GraphWrapper bytecode; it reserves scratch before capture begins.
"""Experimental page decode adapter; no package installation by importing it."""
from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import lru_cache, update_wrapper
from types import FunctionType
from pathlib import Path
import hashlib
import logging
import torch
import triton
import triton.language as tl
from vllm.v1.kv_cache_interface import KVQuantMode

LOG2E = 1.4426950408889634
NSEG = 32
logger = logging.getLogger(__name__)

@triton.jit
def _segments(
    Q, K, V, SEG_ACC, SEG_MAX, SEG_SUM, BT, SEQ_LEN, CU_Q, Q_ROWS,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_btb, stride_btp,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    GQA: tl.constexpr, NSEG: tl.constexpr, scale,
):
    """Same math as ``_segment_kernel`` with BLOCK_N == BS (one page/iter)."""
    pid_g = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, GQA)
    qrow = tl.load(CU_Q + pid_s)
    qlen = tl.load(CU_Q + pid_s + 1) - qrow
    length = tl.load(SEQ_LEN + pid_s)
    active = (qlen == 1) & (qrow >= 0) & (qrow < Q_ROWS) & (length > 0)
    kvlen = tl.where(active, length, 0)
    units = (kvlen + NSEG * BS - 1) // (NSEG * BS)
    start = pid_g * units * BS
    end0 = start + units * BS
    # clamp: a segment entirely past kvlen must not iterate, otherwise m stays
    # -inf and exp2(-inf - (-inf)) would turn the partial into NaN
    end = tl.minimum(end0, kvlen)

    m_i = tl.full((GQA,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((GQA,), dtype=tl.float32)
    acc = tl.zeros((GQA, D), dtype=tl.float32)

    qv = tl.load(Q + qrow * stride_qb + (pid_h * GQA + offs_g)[:, None] * stride_qh
                 + offs_d[None, :] * stride_qd, mask=active, other=0.0)
    pos = tl.arange(0, BS)

    for t0 in range(start, end, BS):
        page = tl.load(BT + pid_s * stride_btb + (t0 // BS) * stride_btp)
        if t0 + BS <= kvlen:
            kt = tl.load(K + page * stride_kb + pos[None, :] * stride_ks
                         + pid_h * stride_kh + offs_d[:, None] * stride_kd)
            vt = tl.load(V + page * stride_vb + pos[:, None] * stride_vs
                         + pid_h * stride_vh + offs_d[None, :] * stride_vd)
            s = tl.dot(qv, kt) * scale
        else:
            mask_t = t0 + pos < kvlen
            kt = tl.load(K + page * stride_kb + pos[None, :] * stride_ks
                         + pid_h * stride_kh + offs_d[:, None] * stride_kd,
                         mask=mask_t[None, :], other=0.0)
            vt = tl.load(V + page * stride_vb + pos[:, None] * stride_vs
                         + pid_h * stride_vh + offs_d[None, :] * stride_vd,
                         mask=mask_t[:, None], other=0.0)
            s = tl.where(mask_t[None, :], tl.dot(qv, kt) * scale,
                         float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new

    base = (pid_s * HKV + pid_h) * NSEG + pid_g
    tl.store(SEG_MAX + base * GQA + offs_g, m_i)
    tl.store(SEG_SUM + base * GQA + offs_g, l_i)
    tl.store(SEG_ACC + (base * GQA + offs_g)[:, None] * D + offs_d[None, :], acc)

@triton.jit
def _combine(
    SEG_ACC, SEG_MAX, SEG_SUM, OUT, CU_Q, Q_ROWS,
    stride_ob, stride_oh, stride_od,
    HQ: tl.constexpr, HKV: tl.constexpr, GQA: tl.constexpr, D: tl.constexpr,
    NSEG: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    qrow = tl.load(CU_Q + pid_s)
    qlen = tl.load(CU_Q + pid_s + 1) - qrow
    if qlen != 1 or qrow < 0 or qrow >= Q_ROWS:
        return
    offs_g = tl.arange(0, GQA)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, NSEG)

    base = (pid_s * HKV + pid_h) * NSEG
    mx = tl.load(SEG_MAX + (base + offs_n)[None, :] * GQA + offs_g[:, None])
    sm = tl.load(SEG_SUM + (base + offs_n)[None, :] * GQA + offs_g[:, None])
    m = tl.max(mx, 1)
    m = tl.where(m > float("-inf"), m, 0.0)
    w = tl.exp2(mx - m[:, None])
    denom = tl.sum(w * sm, 1)
    accs = tl.load(SEG_ACC + ((base + offs_n)[None, :, None] * GQA + offs_g[:, None, None]) * D
                   + offs_d[None, None, :])
    out = tl.sum(accs * w[:, :, None], 1) / tl.where(denom > 0, denom, 1.0)[:, None]
    tl.store(OUT + qrow * stride_ob + (pid_h * GQA + offs_g)[:, None] * stride_oh
             + offs_d[None, :] * stride_od, out.to(OUT.dtype.element_ty))


@dataclass(frozen=True)
class _Step:
    nseq: int
    query_tokens: int
    pure_decode: bool
    leading_decode: int
    # Strong references prevent allocator-address reuse from matching a stale key.
    cu: torch.Tensor
    seq: torch.Tensor
    table: torch.Tensor


_steps = OrderedDict()
_pools = {}
_stats = Counter()
_fallback = None
_previous_build = None
_GRAPH_SHA = 'ba78ed0a736bb21a4f2c307fc9775f15e3742b94533eff333b0c364eb8213fc2'
_TORCH_GRAPH_SHA = '61719c27da56cf9acc935b23fb89c8cc5e2a6ceb3b9ad8c8c4b726b32e001f87'


class _DeviceView:
    def __init__(self, original, prepare):
        self.original, self.prepare = original, prepare

    def __getattr__(self, name):
        return getattr(self.original, name)

    def graph(self, *args, **kwargs):
        context = self.original.graph(*args, **kwargs)
        self.prepare(context)
        return context


class _PlatformView:
    def __init__(self, original, prepare):
        self.original = original
        self.torch_device_fn = _DeviceView(original.torch_device_fn, prepare)

    def __getattr__(self, name):
        return getattr(self.original, name)


def _prepared_capture_call(original, prepare):
    """Private globals view; no global CUDA/platform or stream-policy patch."""
    namespace = dict(original.__globals__)
    namespace['current_platform'] = _PlatformView(namespace['current_platform'], prepare)
    result = FunctionType(original.__code__, namespace, original.__name__,
                          original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    update_wrapper(result, original)
    result._eggking_prepared_capture = True
    return result


def _prepare_capture(context, forward_context=None):
    """Reserve only qualified decode scratch on the original capture stream.

    All allocations finish before context.__enter__. The graph context and its
    stream remain unchanged, and unknown metadata/context types keep fallback.
    Only the existing single-worker serialized graph interface is supported.
    """
    if torch.cuda.is_current_stream_capturing() or not hasattr(context, 'capture_stream'):
        _stats['prepare_skipped_context'] += 1
        return False
    if forward_context is None:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if not is_forward_context_available():
            return False
        forward_context = get_forward_context()
    from vllm.config import CUDAGraphMode
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        return False
    metadata = getattr(forward_context, 'attn_metadata', None)
    if not isinstance(metadata, dict):
        _stats['prepare_skipped_metadata'] += 1
        return False
    steps = []
    for item in metadata.values():
        if not all(hasattr(item, attr) for attr in ('query_start_loc', 'seq_lens', 'block_table')):
            continue
        step = _steps.get(_key(item.query_start_loc, item.seq_lens, item.block_table))
        if step is not None and step.pure_decode and step.query_tokens > 0:
            steps.append(step)
    if not steps:
        _stats['prepare_skipped_descriptor'] += 1
        return False
    device = steps[0].cu.device
    if any(step.cu.device != device for step in steps) or not _device_ok(device.index):
        return False
    stream = context.capture_stream
    if stream.device != device:
        return False
    with torch.cuda.stream(stream):
        pool = _pool(device)
    if pool is None:
        _stats['prepare_skipped_capacity'] += 1
        return False
    _stats['capture_prepared'] += 1
    if _stats['capture_prepared'] == 1:
        logger.warning('eggking page decode capture prepared: sequences=%d capture_stream=%d',
                       steps[0].nseq, stream.cuda_stream)
    return True


def _key(cu, seq, table):
    return (cu.device.index, cu.data_ptr(), seq.data_ptr(), table.data_ptr())


@lru_cache(maxsize=8)
def _device_ok(index):
    return 'iluvatar' in torch.cuda.get_device_name(index).lower()


def _pool(device):
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    pool = _pools.get(key)
    if pool is None:
        # Never allocate/resize during capture, nor evict a pool referenced by
        # an already captured graph. An unknown stream falls back safely.
        if torch.cuda.is_current_stream_capturing() or len(_pools) >= 8:
            return None
        pool = (
            torch.empty((64, 2, NSEG, 8, 128), dtype=torch.float32, device=device),
            torch.empty((64, 2, NSEG, 8), dtype=torch.float32, device=device),
            torch.empty((64, 2, NSEG, 8), dtype=torch.float32, device=device),
        )
        _pools[key] = pool
        _stats['pool_allocations'] += 1
    return pool


def _attach_step(builder, common, result, common_prefix_len):
    key = _key(result.query_start_loc, result.seq_lens, result.block_table)
    _steps.pop(key, None)
    if (common_prefix_len or result.use_cascade or result.causal is not True
            or result.mm_prefix_range_tensor is not None
            or (builder.num_heads_q, builder.num_heads_kv, builder.headdim) != (16, 2, 128)
            or builder.device.type != 'cuda' or not _device_ok(builder.device.index)
            or getattr(builder.vllm_config, 'speculative_config', None) is not None):
        return
    n = common.num_reqs
    cpu = common.query_start_loc_cpu
    if not 1 <= n <= 64 or cpu.device.type != 'cpu' or cpu.numel() < n + 1:
        return
    offsets = cpu[:n + 1].tolist()
    lengths = tuple(offsets[i + 1] - offsets[i] for i in range(n))
    if offsets[0] != 0 or offsets[-1] != common.num_actual_tokens or min(lengths) < 0:
        return
    pure = all(x <= 1 for x in lengths)
    prefix = 0
    for length in lengths:
        if length != 1:
            break
        prefix += 1
    if (not pure and prefix == 0) or len(result.seq_lens) != n:
        return
    _steps[key] = _Step(n, offsets[-1], pure, prefix,
                        result.query_start_loc, result.seq_lens, result.block_table)
    while len(_steps) > 16:
        _steps.popitem(last=False)
    _stats['metadata_steps'] += 1
    # No decode seq_lens CPU upper bound is read here. The kernels load the
    # precise device lengths, including graph replay updates.


def _build(self, common_prefix_len, common_attn_metadata, fast_build=False):
    result = _previous_build(self, common_prefix_len, common_attn_metadata, fast_build)
    _attach_step(self, common_attn_metadata, result, common_prefix_len)
    return result


def unified_decode_pages(
    q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
    softmax_scale, causal, window_size, block_table, softcap,
    q_descale, k_descale, v_descale, seq_threshold_3D=None,
    num_par_softmax_segments=None, softmax_segm_output=None,
    softmax_segm_max=None, softmax_segm_expsum=None, alibi_slopes=None,
    output_scale=None, qq_bias=None, sinks=None, mm_prefix_range=None,
    use_alibi_sqrt=False, kv_quant_mode=KVQuantMode.NONE,
    k_scale_cache=None, v_scale_cache=None, chunk_lookback=-1, use_td=False,
):
    def prior(reason):
        _stats['fallback_' + reason] += 1
        return _fallback(
            q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
            softmax_scale, causal, window_size, block_table, softcap,
            q_descale, k_descale, v_descale, seq_threshold_3D,
            num_par_softmax_segments, softmax_segm_output, softmax_segm_max,
            softmax_segm_expsum, alibi_slopes, output_scale, qq_bias, sinks,
            mm_prefix_range, use_alibi_sqrt, kv_quant_mode, k_scale_cache,
            v_scale_cache, chunk_lookback, use_td)

    from vllm.v1.attention.ops import triton_unified_attention as native
    if max_seqlen_q != 1:
        return prior('query_length')
    eligible = (
        q.device.type == 'cuda' and all(t.device == q.device for t in
            (k, v, out, cu_seqlens_q, seqused_k, block_table))
        and q.dtype == k.dtype == v.dtype == out.dtype == torch.bfloat16
        and q.ndim == out.ndim == 3 and q.shape[1:] == (16, 128)
        and q.shape == out.shape and 1 <= q.shape[0] <= 64
        and q.stride()[-2:] == (128, 1) and q.stride(0) >= 2048
        and out.is_contiguous() and out.data_ptr() != q.data_ptr()
        and k.ndim == v.ndim == 4 and k.shape == v.shape
        and k.shape[1:] == (16, 2, 128)
        and k.stride(-1) == v.stride(-1) == 1
        and min(k.stride()) > 0 and min(v.stride()) > 0
        and cu_seqlens_q.dtype == seqused_k.dtype == block_table.dtype == torch.int32
        and cu_seqlens_q.ndim == seqused_k.ndim == 1 and block_table.ndim == 2
        and cu_seqlens_q.is_contiguous() and seqused_k.is_contiguous()
        and block_table.stride(-1) == 1 and 1 <= len(seqused_k) <= 64
        and len(cu_seqlens_q) == len(seqused_k) + 1 and len(block_table) == len(seqused_k)
        and causal is True and tuple(window_size) == (-1, -1) and softcap == 0
        and alibi_slopes is None and output_scale is None and qq_bias is None
        and sinks is None and mm_prefix_range is None and not use_alibi_sqrt
        and kv_quant_mode == KVQuantMode.NONE and k_scale_cache is None and v_scale_cache is None
        and chunk_lookback == -1 and not use_td and not native.is_batch_invariant
        and not any(t.requires_grad for t in (q, k, v, out))
    )
    if not eligible or not _device_ok(q.device.index):
        return prior('features')
    step = _steps.get(_key(cu_seqlens_q, seqused_k, block_table))
    n = len(seqused_k)
    # b332's mixed path slices only its initial contiguous decode prefix.
    # Basic prefix views retain the same pointers; all other slices fall back.
    if step is None or not (
        (step.pure_decode and n == step.nseq and step.query_tokens <= q.shape[0])
        or (not step.pure_decode and n <= step.leading_decode and q.shape[0] == n)
    ):
        return prior('descriptor')
    pool = _pool(q.device)
    if pool is None:
        return prior('capture_pool')
    # Zero padded token slots too. qlen=0 rows never dereference pages/query;
    # qlen=1, seq_len=0 produces an explicit finite zero in the combine.
    out.zero_()
    _segments[(NSEG, 2, n)](
        q, k, v, *pool, block_table, seqused_k, cu_seqlens_q, q.shape[0],
        *q.stride(), *k.stride(), *v.stride(), *block_table.stride(),
        HQ=16, HKV=2, D=128, BS=16, GQA=8, NSEG=NSEG,
        scale=softmax_scale * LOG2E, num_warps=1, num_stages=1)
    _combine[(n, 2)](
        *pool, out, cu_seqlens_q, q.shape[0], *out.stride(),
        HQ=16, HKV=2, GQA=8, D=128, NSEG=NSEG, num_warps=4)
    _stats['dispatched'] += 1
    if torch.cuda.is_current_stream_capturing():
        _stats['capture_dispatched'] += 1
        if _stats['capture_dispatched'] == 1:
            logger.warning('eggking page decode captured: sequences=%d stream=%d',
                           n, torch.cuda.current_stream(q.device).cuda_stream)
    if _stats['dispatched'] == 1:
        logger.warning('eggking page decode dispatched: sequences=%d query_rows=%d segments=32', n, q.shape[0])


unified_decode_pages._eggking_decode_pages = True
_build._eggking_decode_pages = True


def patch_decode_pages():
    """Install process-local hooks after b332's attention/prefill registration."""
    global _fallback, _previous_build
    from vllm_fl.patches import iluvatar_attention as adapter
    from vllm.v1.attention.backends import triton_attn
    from vllm_fl.compilation import graph as graph_module
    import importlib
    torch_graphs = importlib.import_module('torch.cuda.graphs')
    if getattr(adapter._original, '_eggking_decode_pages', False):
        return False
    if adapter._original is None or not getattr(triton_attn.unified_attention, '_eggking_paged_prefill', False):
        raise RuntimeError('Page decode requires the qualified b332 attention/prefill stack')
    # Refuse unknown wrapper implementations before modifying any hook.
    method = graph_module.GraphWrapper.__call__
    if (hashlib.sha256(Path(graph_module.__file__).read_bytes()).hexdigest() != _GRAPH_SHA
            or hashlib.sha256(Path(torch_graphs.__file__).read_bytes()).hexdigest() != _TORCH_GRAPH_SHA
            or method.__code__.co_filename != graph_module.__file__
            or getattr(method, '_eggking_prepared_capture', False)):
        logger.warning('eggking page decode skipped: unsupported graph wrapper owner/source')
        return False
    prepared = _prepared_capture_call(method, _prepare_capture)
    _fallback = adapter._original
    _previous_build = triton_attn.TritonAttentionMetadataBuilder.build
    adapter._original = unified_decode_pages
    triton_attn.TritonAttentionMetadataBuilder.build = _build
    graph_module.GraphWrapper.__call__ = prepared
    logger.warning('eggking page decode registered: attributed core, step metadata, stream-owned slabs')
    return True
