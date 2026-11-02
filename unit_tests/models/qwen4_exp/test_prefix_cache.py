import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.linear_att_cache_manager import LinearAttCacheManager, LinearAttCacheConfig
from lightllm.common.req_manager import ReqManagerForMamba
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.models.qwen4_exp.configuration import Qwen4ExpCacheConfig
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from .runtime import assemble_model
from .prefix import prefix_context, prefix_request, run_prefill, run_decode, assert_logits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_scheduler_big_and_small_prefix_snapshots_restore_all_request_state(runtime):
    context = prefix_context(runtime)
    req, lexical = runtime.req, runtime.req.lexical_caches[1]
    conv, ssm = req.req_to_conv_state.buffer, req.req_to_ssm_state.buffer
    for request in (1, 3):
        lexical.ngram[request].copy_(torch.tensor([request + 7, request + 11], device="cuda"))
        lexical.convolution[request].normal_()
        conv[:, request].normal_()
        ssm[:, request].normal_()
    lexical_before = {request: req.snapshot_lexical(request) for request in (1, 3)}
    conv_before, ssm_before = conv.clone(), ssm.clone()
    big, small = prefix_request(1, 4), prefix_request(3, 6)
    context.copy_linear_att_state_to_cache_buffer(torch.tensor([1, 3], device="cuda", dtype=torch.int32), [big, small])
    big_index, small_index = big.linear_att_len_to_big_page_id[4], small.tail_linear_att_small_page_buffer_id
    torch.cuda.synchronize()
    assert context.radix_cache.linear_att_big_page_buffers.extra_state_cache.histories[1].ngram.is_pinned()
    req.init_linear_att_state(SimpleNamespace(req_idx=1))
    req.init_linear_att_state(SimpleNamespace(req_idx=3))
    req.copy_big_page_buffer_to_linear_att_state(big_index, SimpleNamespace(req_idx=2))
    req.copy_small_page_buffer_to_linear_att_state(
        SimpleNamespace(req_idx=0, shared_kv_node=SimpleNamespace(small_page_buffer_idx=small_index)),
        context.radix_cache.linear_att_small_page_buffers,
    )
    for destination, source in ((2, 1), (0, 3)):
        torch.testing.assert_close(conv[:, destination], conv_before[:, source], atol=0, rtol=0)
        torch.testing.assert_close(ssm[:, destination], ssm_before[:, source], atol=0, rtol=0)
        torch.testing.assert_close(lexical.ngram[destination], lexical_before[source][1].ngram, atol=0, rtol=0)
        torch.testing.assert_close(
            lexical.convolution[destination], lexical_before[source][1].convolution, atol=0, rtol=0
        )

    lexical.ngram[2].fill_(31)
    lexical.convolution[2].normal_()
    req.copy_big_page_buffer_to_linear_att_state(big_index, SimpleNamespace(req_idx=0))
    torch.testing.assert_close(lexical.ngram[0], lexical_before[1][1].ngram, atol=0, rtol=0)
    req.copy_extra_state_to_prefix_cache([2], [big_index], context.radix_cache.linear_att_big_page_buffers)
    req.copy_big_page_buffer_to_linear_att_state(big_index, SimpleNamespace(req_idx=0))
    assert torch.all(lexical.ngram[0] == 31)
    context.radix_cache.linear_att_big_page_buffers.clear_to_init_state()
    assert torch.all(context.radix_cache.linear_att_big_page_buffers.extra_state_cache.histories[1].ngram == 63)


def test_prefix_extension_keeps_legacy_pool_layout_and_dispatches_qwen_config(runtime, config, tmp_path):
    current = runtime.req.linear_config
    legacy = LinearAttCacheConfig(
        **{field.name: getattr(current, field.name) for field in dataclasses.fields(LinearAttCacheConfig)}
    )
    pool = LinearAttCacheManager(2, legacy)
    assert pool.extra_state_cache is None
    assert pool.get_state_cache(0)[0].shape == (3, 512, 3)
    assert pool.get_state_cache(0)[1].shape == (3, 2, 128, 128)
    ReqManagerForMamba.copy_extra_state_to_prefix_cache(runtime.req, None, None, pool)
    with pytest.raises(ValueError, match="missing its lexical state"):
        runtime.req._prefix_extension(pool)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": config}))
    loaded = LinearAttCacheConfig.load_from_args()
    assert isinstance(loaded, Qwen4ExpCacheConfig)
    assert loaded.lexical_layers == (1,)
    assert loaded.get_cpu_cache_big_page_bytes() > legacy.get_cpu_cache_big_page_bytes()


def test_complete_native_model_prefix_forks_and_partial_slot_remap(runtime, config):
    Qwen3NextTpPartModel._init_triton(None)
    model, source = assemble_model(runtime, config)
    context = prefix_context(runtime)
    req = runtime.req
    req.init_linear_att_state(SimpleNamespace(req_idx=1))
    prefix, output = run_prefill(runtime, model, [1], [4], [7, 63, 11, 13], [0])
    assert_logits(runtime, source, config, [7, 63, 11, 13], output)
    saved = prefix_request(1, 4)
    context.copy_linear_att_state_to_cache_buffer(prefix.b_req_idx, [saved])
    original = req.snapshot_lexical(1)
    original_conv, original_ssm = req.req_to_conv_state.buffer[:, 1].clone(), req.req_to_ssm_state.buffer[:, 1].clone()

    req.req_to_token_indexs[3, :4].copy_(req.req_to_token_indexs[1, :4])
    req.copy_big_page_buffer_to_linear_att_state(saved.linear_att_len_to_big_page_id[4], SimpleNamespace(req_idx=3))
    _, output = run_decode(runtime, model, 3, 5, 17)
    assert_logits(runtime, source, config, [7, 63, 11, 13, 17], output)
    _, output = run_decode(runtime, model, 3, 6, 19)
    assert_logits(runtime, source, config, [7, 63, 11, 13, 17, 19], output)
    tail = prefix_request(3, 6)
    context.copy_linear_att_state_to_cache_buffer(torch.tensor([3], device="cuda", dtype=torch.int32), [tail])
    tail_slots = runtime.mem.alloc(2).cuda()
    runtime.mem.operator.copy_mem_to_mem(req.req_to_token_indexs[3, 4:6], tail_slots)
    req.req_to_token_indexs[2, :4].copy_(req.req_to_token_indexs[1, :4])
    req.req_to_token_indexs[2, 4:6].copy_(tail_slots)
    req.copy_small_page_buffer_to_linear_att_state(
        SimpleNamespace(
            req_idx=2, shared_kv_node=SimpleNamespace(small_page_buffer_idx=tail.tail_linear_att_small_page_buffer_id)
        ),
        context.radix_cache.linear_att_small_page_buffers,
    )

    _, output = run_prefill(runtime, model, [2], [8], [23, 29], [6])
    assert_logits(runtime, source, config, [7, 63, 11, 13, 17, 19, 23, 29], output)
    torch.testing.assert_close(req.lexical_caches[1].ngram[1], original[1].ngram, atol=0, rtol=0)
    torch.testing.assert_close(req.lexical_caches[1].convolution[1], original[1].convolution, atol=0, rtol=0)
    torch.testing.assert_close(req.req_to_conv_state.buffer[:, 1], original_conv, atol=0, rtol=0)
    torch.testing.assert_close(req.req_to_ssm_state.buffer[:, 1], original_ssm, atol=0, rtol=0)


@pytest.mark.parametrize("prefix_length", [6, 8])
@pytest.mark.parametrize(
    "runtime",
    [{"linear_att_hash_page_size": 2, "linear_att_page_block_num": 2, "max_req_total_len": 64}],
    indirect=True,
)
def test_native_cpu_pages_restore_complete_state_and_resumed_logits(runtime, config, monkeypatch, prefix_length):
    Qwen3NextTpPartModel._init_triton(None)
    model, source = assemble_model(runtime, config)
    context = prefix_context(runtime)
    context.radix_cache.get_big_page_ids_by_node = lambda node: []
    monkeypatch.setattr(g_infer_context, "req_manager", runtime.req)
    monkeypatch.setattr(g_infer_context, "radix_cache", context.radix_cache)
    for tensor, _ in runtime.mem._slot_buffers().values():
        tensor.zero_()
    runtime.req.init_linear_att_state(SimpleNamespace(req_idx=1))
    first, _ = run_prefill(runtime, model, [1], [4], [7, 63, 11, 13], [0])
    initial = prefix_request(1, 4)
    context.copy_linear_att_state_to_cache_buffer(first.b_req_idx, [initial])
    tokens = [7, 63, 11, 13, 17, 19, 23, 29][:prefix_length]
    final, _ = run_prefill(runtime, model, [1], [prefix_length], tokens[4:], [4])
    saved = prefix_request(1, prefix_length)
    saved.linear_att_len_to_big_page_id = initial.linear_att_len_to_big_page_id
    context.copy_linear_att_state_to_cache_buffer(final.b_req_idx, [saved])
    saved.cur_kv_len, saved.shared_kv_node = prefix_length, None
    slots = runtime.req.req_to_token_indexs[1, :prefix_length].clone()
    snapshot = runtime.mem.get_index_kv_buffer(slots)
    lexical = runtime.req.snapshot_lexical(1)
    conv = runtime.req.req_to_conv_state.buffer[:, 1].clone()
    ssm = runtime.req.req_to_ssm_state.buffer[:, 1].clone()
    pages = torch.full(
        (2, 1, 1, 1, runtime.req.linear_config.get_cpu_cache_big_page_bytes()),
        0x5A,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    )
    client = SimpleNamespace(cpu_kv_cache_tensor=pages)
    indices = torch.tensor([1, 0], device="cuda", dtype=torch.int64)
    ready = torch.tensor([True, False], device="cuda")

    runtime.mem.operator.offload_gpu_kv_to_cpu_cache(slots, indices, ready, client, saved)
    torch.cuda.synchronize()
    assert torch.all(pages[1] == 0x5A)
    ready.zero_()
    runtime.mem.operator.offload_gpu_kv_to_cpu_cache(slots, indices, ready, client, saved)
    destination = runtime.mem.alloc(prefix_length).cuda()
    restored = prefix_request(2, prefix_length)
    restored.cur_kv_len = prefix_length
    runtime.req.init_linear_att_state(SimpleNamespace(req_idx=2))
    runtime.mem.operator.load_cpu_cache_to_gpu(destination, indices, client, restored)
    runtime.req.req_to_token_indexs[2, :prefix_length].copy_(destination)
    for name, values in runtime.mem.get_index_kv_buffer(destination).items():
        torch.testing.assert_close(values, snapshot[name], atol=0, rtol=0)
    torch.testing.assert_close(runtime.req.req_to_conv_state.buffer[:, 2], conv, atol=0, rtol=0)
    torch.testing.assert_close(runtime.req.req_to_ssm_state.buffer[:, 2], ssm, atol=0, rtol=0)
    torch.testing.assert_close(runtime.req.lexical_caches[1].ngram[2], lexical[1].ngram, atol=0, rtol=0)
    torch.testing.assert_close(runtime.req.lexical_caches[1].convolution[2], lexical[1].convolution, atol=0, rtol=0)
    _, output = run_decode(runtime, model, 2, prefix_length + 1, 31)
    assert_logits(runtime, source, config, tokens + [31], output)


@pytest.mark.parametrize("runtime", [{"linear_att_hash_page_size": 2, "linear_att_page_block_num": 2}], indirect=True)
def test_cpu_extension_uses_64_bit_layer_offsets_at_two_million_slots(runtime, config):
    from lightllm.models.qwen4_exp.configuration import linear_cache_config
    from lightllm.models.qwen4_exp.cpu_transfer import transfer_extra_pages

    slots, layers, dim = 2_000_001, 12, 128
    required = 2 * slots * layers * dim * torch.bfloat16.itemsize
    if torch.cuda.mem_get_info()[0] < required + 2 * 1024**3:
        pytest.skip("Large-address QSA regression requires 15 GB free GPU memory")
    geometry = dict(
        config,
        n_layer=48,
        num_hidden_layers=48,
        indexer_head_dim=dim,
        layer_types=["full_attention" if (layer + 1) % 4 == 0 else "linear_attention" for layer in range(48)],
    )
    cache_config = linear_cache_config(geometry, 1, torch.bfloat16, torch.float32)
    raw = torch.empty(layers, slots, dim, device="cuda", dtype=torch.bfloat16)
    pooled = torch.empty_like(raw)
    cos = torch.empty(slots, 4, device="cuda", dtype=torch.bfloat16)
    sin = torch.empty_like(cos)
    tensors = {
        "qsa_raw_keys": (raw, 1),
        "qsa_pooled_keys": (pooled, 1),
        "qsa_key_cos": (cos, 0),
        "qsa_key_sin": (sin, 0),
    }
    owner = SimpleNamespace(
        linear_config=cache_config,
        linear_att_big_page_buffers=LinearAttCacheManager(1, cache_config),
        _slot_buffers=lambda: tensors,
    )
    source = torch.arange(slots - 4, slots, device="cuda", dtype=torch.int64)
    destination = torch.tensor([1, 3, 5, 7], device="cuda", dtype=torch.int64)
    expected = {}
    for name, (tensor, axis) in tensors.items():
        shape = list(tensor.shape)
        shape[axis] = 4
        expected[name] = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        tensor.index_copy_(axis, source, expected[name])

    assert raw.stride(0) * (layers - 1) > 2**31
    pages = torch.empty(
        (1, cache_config.get_cpu_cache_big_page_bytes()), dtype=torch.uint8, device="cpu", pin_memory=True
    )
    page = torch.zeros(1, device="cuda", dtype=torch.int64)
    transfer_extra_pages(owner, source.int(), page, page, pages, torch.zeros(1, device="cuda", dtype=torch.bool))
    transfer_extra_pages(owner, destination.int(), page, page, pages)
    for name, (tensor, axis) in tensors.items():
        torch.testing.assert_close(tensor.index_select(axis, destination), expected[name], atol=0, rtol=0)
