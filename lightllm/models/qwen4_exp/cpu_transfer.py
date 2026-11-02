import torch
import triton

from lightllm.utils.dist_utils import get_current_rank_in_dp
from .cache_layout import Qwen4ExpPageLayout
from .prefix_cache import _copy_rows
from .triton_kernel.cache_pages import copy_slot_pages, copy_state_pages


def transfer_extra_pages(mem_manager, mem_indices, state_indices, page_indices, cpu_pages, readies=None):
    offload = readies is not None

    if offload and get_current_rank_in_dp() != 0:
        return
    config = mem_manager.linear_config
    views = Qwen4ExpPageLayout(config).views(cpu_pages)
    pages = page_indices.numel()
    if not pages:
        return
    for name, (slots, axis) in mem_manager._slot_buffers().items():
        if name == "kv_buffer":
            continue
        cached = views[name]
        tokens, layers, dim = cached.shape[1:]
        if not dim:
            continue
        slot_stride = slots.stride(axis)
        layer_stride = slots.stride(0) if axis == 1 else 0
        copy_slot_pages[(pages, triton.cdiv(tokens * layers * dim, 256))](
            slots,
            cached,
            mem_indices,
            page_indices,
            readies,
            slot_stride,
            layer_stride,
            cached.stride(0),
            tokens,
            layers,
            dim,
            offload,
            256,
        )
    histories = mem_manager.linear_att_big_page_buffers.extra_state_cache.histories
    for layer, history in histories.items():
        for kind in ("ngram", "convolution"):
            live, cached = getattr(history, kind), views[f"lexical.{layer}.{kind}"]
            width = live[0].numel()
            if width:
                copy_state_pages[(pages, triton.cdiv(width, 256))](
                    live, cached, state_indices, page_indices, readies, width, cached.stride(0), offload, 256
                )


def copy_prefix_extension(source_pool, source_index, destination_pool, destination_index, device):
    source, target = source_pool.extra_state_cache, destination_pool.extra_state_cache
    if set(source.histories) != set(target.histories):
        raise ValueError("Prefix transfer requires matching lexical layers")
    source_rows = torch.tensor([source_index], dtype=torch.int32, device=device)
    destination_rows = torch.tensor([destination_index], dtype=torch.int32, device=device)
    for layer, history in source.histories.items():
        for kind in ("ngram", "convolution"):
            live, cached = getattr(history, kind), getattr(target.histories[layer], kind)
            if live.shape[1:] != cached.shape[1:] or live.dtype != cached.dtype:
                raise ValueError("Prefix transfer requires matching lexical history geometry")
            width = live[0].numel()
            if width:
                _copy_rows[(1, triton.cdiv(width, 256))](live, cached, source_rows, destination_rows, width, 256)
