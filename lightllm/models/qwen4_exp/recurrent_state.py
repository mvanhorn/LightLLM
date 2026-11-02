import torch
import triton
import triton.language as tl

from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextLinearAttPageHelper


def state_span(owner):
    return owner.req_to_ssm_state.buffer.shape[1] // owner.req_to_conv_state.buffer.shape[1]


def accepted_state(owner, request):
    conv, ssm = owner.req_to_conv_state.buffer, owner.req_to_ssm_state.buffer
    context = owner.linear_config.conv_kernel_size - 1
    indices = getattr(owner, "req_to_mtp_state_index", None)
    if indices is None:
        return conv[:, request, ..., :context], ssm[:, request * state_span(owner)]
    offset = indices[request : request + 1].long()
    history = torch.arange(context, device=conv.device) + offset
    return conv[:, request].index_select(-1, history), ssm.index_select(
        1, offset + request * state_span(owner)
    ).squeeze(1)


class Qwen4ExpRecurrentPageHelper(Qwen3NextLinearAttPageHelper):
    def _write_one_rank(self, mem, tp_index, req_idx, conv_page, ssm_page):
        conv, ssm = accepted_state(mem, req_idx)
        self._copy_conv_state_to_page(conv, conv_page, mem, tp_index)
        self._copy_ssm_state_to_page(ssm, ssm_page, mem, tp_index)

    def _read_one_rank(self, mem, tp_index, req_idx, conv_page, ssm_page):
        conv = mem.req_to_conv_state.buffer[:, req_idx, ..., : self.conv_shape[-1]]
        ssm = mem.req_to_ssm_state.buffer[:, req_idx * state_span(mem)]
        self._copy_page_to_conv_state(conv_page, conv, mem, tp_index)
        self._copy_page_to_ssm_state(ssm_page, ssm, mem, tp_index)
        indices = getattr(mem, "req_to_mtp_state_index", None)
        if indices is not None:
            indices[req_idx] = 0


@triton.jit
def _copy_prefix(
    Source,
    Target,
    Requests,
    Destinations,
    StateIndices,
    SOURCE_LAYER: tl.constexpr,
    SOURCE_REQUEST: tl.constexpr,
    SOURCE_OFFSET: tl.constexpr,
    TARGET_PAGE: tl.constexpr,
    TARGET_LAYER: tl.constexpr,
    CONTEXT: tl.constexpr,
    SOURCE_CONTEXT: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, layer = tl.program_id(0), tl.program_id(1).to(tl.int64)
    destination = tl.load(Destinations + row).to(tl.int64)
    if destination >= 0:
        request = tl.load(Requests + row).to(tl.int64)
        offset = tl.load(StateIndices + request).to(tl.int64)
        column = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        source_column = column // CONTEXT * SOURCE_CONTEXT + column % CONTEXT
        values = tl.load(
            Source + layer * SOURCE_LAYER + request * SOURCE_REQUEST + offset * SOURCE_OFFSET + source_column,
            column < WIDTH,
            other=0,
        )
        tl.store(Target + destination * TARGET_PAGE + layer * TARGET_LAYER + column, values, column < WIDTH)


def copy_accepted_to_prefix(owner, requests, destinations, pool):
    if not requests.numel():
        return
    conv, ssm = owner.req_to_conv_state.buffer, owner.req_to_ssm_state.buffer
    context = owner.linear_config.conv_kernel_size - 1
    fields = (
        (conv, pool.conv_state_cache.buffer, conv.stride(1), 1, context, conv.shape[-1]),
        (ssm, pool.ssm_state_cache.buffer, ssm.stride(1) * state_span(owner), ssm.stride(1), 1, 1),
    )
    for source, target, request_stride, offset_stride, history, source_history in fields:
        width = target[0, 0].numel()
        if width:
            _copy_prefix[(requests.numel(), source.shape[0], triton.cdiv(width, 256))](
                source,
                target,
                requests,
                destinations,
                owner.req_to_mtp_state_index,
                source.stride(0),
                request_stride,
                offset_stride,
                target.stride(0),
                target.stride(1),
                history,
                source_history,
                width,
                256,
            )
