# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 eggking
# Arithmetic/masks follow FlagGems ops/index.py, fixed a7620cc (Apache-2.0).
# Unlike its shape-generated source, N and M remain runtime, non-specialized.
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N', 'M'])
def _index1d(Input, Index, Output, N, M, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw = tl.load(Index + i, i < M, other=0)
    index = tl.where(raw < 0, raw + N, raw)
    valid = (i < M) & (index >= 0) & (index < N)
    value = tl.load(Input + index, valid, other=0)
    # Invalid indices leave output untouched, exactly the selected native mask.
    # This is not a claim to implement PyTorch's out-of-range exception contract.
    tl.store(Output + i, value, valid)


class SharedIndex1D:
    def __init__(self, original):
        self.original = original
        self.enabled = 'bi-v150' in torch.cuda.get_device_name().lower()
        self.hits = 0

    def supported(self, x, indices):
        return (self.enabled and isinstance(indices, (list, tuple)) and len(indices) == 1
                and isinstance(indices[0], torch.Tensor) and x.device.type == 'cuda'
                and x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous()
                and 1 <= x.numel() <= 256 and not x.requires_grad
                and indices[0].device == x.device and indices[0].dtype == torch.int64
                and indices[0].ndim == 1 and indices[0].is_contiguous()
                and 1 <= indices[0].numel() <= 256 and not indices[0].requires_grad)

    def launch(self, x, index, out):
        _index1d[(triton.cdiv(index.numel(), 32),)](
            x, index, out, x.numel(), index.numel(), BLOCK=32,
            num_warps=4, num_stages=2)

    def __call__(self, x, indices):
        if not self.supported(x, indices):
            return self.original(x, indices)
        out = torch.empty(indices[0].shape, dtype=x.dtype, device=x.device)
        self.launch(x, indices[0], out)
        self.hits += 1
        return out


# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 eggking
"""Patch the pinned helper's globals so existing registered aliases still dispatch.

No torch/GPU import or hardware initialization. Adapter creation is deferred
until an eager eligible-device invocation; unknown capture state uses native.
"""
from functools import wraps
import hashlib
from pathlib import Path
from types import CodeType

INDEX_SHA='47a7108bcf8424a96651021703f8731f1866a0e2eff88a1ac647e574ccf7e61d'
HELPER='_run_linearized_adjacent_index'


class Handle:
    def __init__(self, namespace, original, wrapper, state):
        self.namespace,self.original,self.wrapper,self.state=namespace,original,wrapper,state

    def restore(self):
        if self.namespace.get(HELPER) is not self.wrapper:
            raise RuntimeError('Another owner changed the helper; refuse to overwrite')
        self.namespace[HELPER]=self.original


def install(index_function, adapter_factory):
    source=Path(index_function.__code__.co_filename)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=INDEX_SHA:
        raise RuntimeError('Unknown native index source')
    namespace=index_function.__globals__
    helper=namespace.get(HELPER)
    if getattr(helper,'_eggking_shared_index1d',False):
        if (getattr(helper,'_eggking_index_origin',None) is index_function
                and getattr(helper,'_eggking_registration_sha',None)==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            return None
        raise RuntimeError('Unknown existing index adapter owner')
    if (not callable(helper) or helper.__name__!=HELPER
            or Path(helper.__code__.co_filename)!=source):
        raise RuntimeError('Unknown native helper owner')
    # Compile the complete module without executing it. CPython 3.12 selects
    # different LOAD_GLOBAL/LOAD_ATTR call encodings when imports are omitted
    # from an extracted AST. Keep that import-aware symbol table intact.
    compiled=compile(source.read_text(),str(source),'exec',dont_inherit=True)
    expected={code.co_name:code for code in compiled.co_consts
              if isinstance(code,CodeType) and code.co_name in ('index',HELPER)}
    if len(expected)!=2:
        raise RuntimeError('Unexpected native definition contract')
    for actual in (index_function,helper):
        native=expected[actual.__name__]
        code=actual.__code__
        if (code.co_code,code.co_consts,code.co_names,code.co_varnames,code.co_argcount)!=(
                native.co_code,native.co_consts,native.co_names,native.co_varnames,native.co_argcount):
            raise RuntimeError('Native function code was replaced at runtime')
    state=dict(adapters={},native_fallback_calls=0,capture_without_adapter_fallbacks=0)

    @wraps(helper)
    def wrapper(x,indices):
        device=x.device
        if device.type!='cuda':
            state['native_fallback_calls']+=1
            return helper(x,indices)
        key=device.index
        adapter=state['adapters'].get(key)
        if adapter is None:
            if namespace['torch'].cuda.is_current_stream_capturing():
                state['capture_without_adapter_fallbacks']+=1
                return helper(x,indices)
            adapter=adapter_factory(helper,key)
            state['adapters'][key]=adapter
        return adapter(x,indices)
    wrapper._eggking_shared_index1d=True
    wrapper._eggking_index_origin=index_function
    wrapper._eggking_registration_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    namespace[HELPER]=wrapper
    return Handle(namespace,helper,wrapper,state)

import json


import logging
import traceback
from datetime import datetime, timezone

_LOGGER=logging.getLogger(__name__)
_registration=None


class DeviceIndex(SharedIndex1D):
    def __init__(self, original, device_index):
        self.original=original
        self.device_index=device_index
        self.enabled='bi-v150' in torch.cuda.get_device_name(device_index).lower()
        self.hits=0
        self.logged_lengths=set()

    def supported(self, x, indices):
        return (super().supported(x,indices) and x.device.index==self.device_index
                and self.device_index==torch.cuda.current_device())

    def __call__(self, x, indices):
        previous=self.hits
        result=super().__call__(x,indices)
        n=x.numel()
        if self.hits>previous and n<=64 and n not in self.logged_lengths and len(self.logged_lengths)<16:
            self.logged_lengths.add(n)
            payload=dict(utc=datetime.now(timezone.utc).isoformat(),n=n,m=indices[0].numel(),device=self.device_index,
                caller=[dict(file=f.filename,line=f.lineno,function=f.name) for f in traceback.extract_stack(limit=10)[:-1]])
            _LOGGER.warning('eggking shared index dispatched: %s',json.dumps(payload,sort_keys=True))
        return result


def patch_iluvatar_index1d():
    global _registration
    import flag_gems
    handle=install(flag_gems.index,lambda original,index:DeviceIndex(original,index))
    if handle is None:
        return False
    _registration=handle
    payload=dict(module_path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 native_index_sha256=INDEX_SHA,hardware_initialized_by_registration=False)
    _LOGGER.warning('eggking shared index registered: %s',json.dumps(payload,sort_keys=True))
    return True
