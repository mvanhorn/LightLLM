from types import SimpleNamespace

import torch

from lightllm.common.linear_att_cache_manager import LinearAttCacheManager
from lightllm.server.router.model_infer.infer_batch import InferenceContext
from lightllm.common.basemodel.hidden_collector import NoopHiddenCollector
from .runtime import state_for
from .reference import reference_model


def prefix_context(runtime):
    context = InferenceContext()
    context.is_linear_att_mixed_model = True
    context.req_manager = runtime.req
    context.args = SimpleNamespace(
        linear_att_hash_page_size=2,
        linear_att_page_block_num=2,
        mtp_step=runtime.req.mtp_step,
        disable_chunked_prefill=False,
    )
    runtime.mem.linear_att_big_page_buffers = LinearAttCacheManager(6, runtime.req.linear_config, keep_num=2)
    runtime.mem.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = 4
    runtime.mem.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = 5
    context.radix_cache = SimpleNamespace(
        linear_att_big_page_buffers=runtime.mem.linear_att_big_page_buffers,
        linear_att_small_page_buffers=LinearAttCacheManager(3, runtime.req.linear_config),
        free_one_small_page_linear_att_buffer=lambda: None,
    )
    return context


def prefix_request(request, length):
    return SimpleNamespace(
        req_idx=request,
        linear_att_cache_len=length,
        linear_att_len_to_big_page_id={},
        tail_linear_att_small_page_buffer_id=None,
        get_chuncked_input_token_len=lambda: length,
    )


def run_prefill(runtime, model, requests, lengths, tokens, ready):
    state = state_for(runtime, requests, lengths, tokens, ready=ready)
    state.hidden_collector = NoopHiddenCollector()
    return state, model._context_forward(state)


def run_decode(runtime, model, request, length, token):
    state = state_for(runtime, [request], [length], [token])
    state.hidden_collector = NoopHiddenCollector()
    return state, model._token_forward(state)


def assert_logits(runtime, source, config, tokens, output):
    tokens = torch.tensor(tokens, device="cuda", dtype=torch.int64)
    positions = torch.arange(tokens.numel(), device="cuda")
    cos = runtime.model._cos_cached[positions].repeat(1, 2)
    sin = runtime.model._sin_cached[positions].repeat(1, 2)
    expected = reference_model(tokens, cos, sin, [tokens.numel()], source, config)[-1:]
    torch.testing.assert_close(output.logits, expected, atol=0.025, rtol=0.025)
