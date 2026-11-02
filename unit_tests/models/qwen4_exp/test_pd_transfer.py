from types import SimpleNamespace

import pytest
import torch

from lightllm.models.qwen4_exp.mem_manager import Qwen4ExpMemManager
from lightllm.models.qwen4_exp.configuration import linear_cache_config
from lightllm.models.qwen4_exp.lexical_cache import LexicalSnapshot
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from .runtime import assemble_model
from .prefix import run_prefill, run_decode, assert_logits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_complete_model_pd_token_and_recurrent_pages_resume_logits(runtime, config):
    Qwen3NextTpPartModel._init_triton(None)
    model, source = assemble_model(runtime, config)
    mem, req = runtime.mem, runtime.req
    req.init_linear_att_state(SimpleNamespace(req_idx=1))
    for tensor, _ in mem._slot_buffers().values():
        tensor.zero_()
    _, output = run_prefill(runtime, model, [1], [4], [7, 63, 11, 13], [0])
    assert_logits(runtime, source, config, [7, 63, 11, 13], output)
    source_slots = req.req_to_token_indexs[1, :4].tolist()
    slots_before = mem.get_index_kv_buffer(source_slots)
    lexical = req.snapshot_lexical(1)
    conv, ssm = req.req_to_conv_state.buffer[:, 1].clone(), req.req_to_ssm_state.buffer[:, 1].clone()
    buffer = mem.alloc_paged_kv_move_buffer(2, 4)
    assert buffer.ndim == 5 and buffer.dtype == torch.uint8
    assert buffer[0].numel() >= mem.pd_layout.kv_bytes
    assert buffer[1].numel() >= mem.pd_layout.state_bytes
    mem.write_mem_to_page_kv_move_buffer(source_slots, 0, 0, [mem], 1)
    mem.write_mem_to_page_kv_move_buffer([], 1, 0, [mem], 1, page_kind="linear_att_state", req_idx=1)
    wire = buffer.clone()
    buffer.zero_()
    buffer.copy_(wire)
    destination = mem.alloc(4).tolist()
    req.init_linear_att_state(SimpleNamespace(req_idx=3))
    mem.read_page_kv_move_buffer_to_mem(destination, 0, 0, [mem], 1)
    mem.read_page_kv_move_buffer_to_mem([], 1, 0, [mem], 1, page_kind="linear_att_state", req_idx=3)
    req.req_to_token_indexs[3, :4].copy_(torch.tensor(destination, device="cuda", dtype=torch.int32))
    for name, value in mem.get_index_kv_buffer(destination).items():
        torch.testing.assert_close(value, slots_before[name], atol=0, rtol=0)
    torch.testing.assert_close(req.req_to_conv_state.buffer[:, 3], conv, atol=0, rtol=0)
    torch.testing.assert_close(req.req_to_ssm_state.buffer[:, 3], ssm, atol=0, rtol=0)
    torch.testing.assert_close(req.lexical_caches[1].ngram[3], lexical[1].ngram, atol=0, rtol=0)
    torch.testing.assert_close(req.lexical_caches[1].convolution[3], lexical[1].convolution, atol=0, rtol=0)
    _, output = run_decode(runtime, model, 3, 5, 17)
    assert_logits(runtime, source, config, [7, 63, 11, 13, 17], output)
    torch.testing.assert_close(req.lexical_caches[1].ngram[1], lexical[1].ngram, atol=0, rtol=0)


def shard_managers(config, world, global_kv, global_conv, global_ssm, raw, pooled, cos, sin, lexical):
    cache = linear_cache_config(config, world, torch.bfloat16, torch.float32)
    managers = []
    key_dim, value_dim = cache.head_linear_k_dim, cache.head_linear_v_dim
    key_heads, value_heads = cache.num_linear_k_heads, cache.num_linear_v_heads
    q_width = cache.global_linear_k_heads * key_dim
    v_width = cache.global_linear_v_heads * value_dim
    local_heads = cache.full_att_num_kv_heads
    for rank in range(world):
        mem = Qwen4ExpMemManager.__new__(Qwen4ExpMemManager)
        mem.linear_config, mem.dtype = cache, torch.bfloat16
        mem.head_num, mem.head_dim, mem.layer_num = local_heads, cache.full_att_head_dim, 1
        group = (
            rank * local_heads
            if world <= cache.full_att_all_num_kv_heads
            else rank // (world // cache.full_att_all_num_kv_heads)
        )
        mem.kv_buffer = torch.cat(
            [
                global_kv[:, :, group : group + local_heads],
                global_kv[
                    :,
                    :,
                    cache.full_att_all_num_kv_heads + group : cache.full_att_all_num_kv_heads + group + local_heads,
                ],
            ],
            2,
        ).contiguous()
        q_start, v_start = rank * key_heads * key_dim, rank * value_heads * value_dim
        conv = torch.cat(
            [
                global_conv[:, :, q_start : q_start + key_heads * key_dim],
                global_conv[:, :, q_width + q_start : q_width + q_start + key_heads * key_dim],
                global_conv[:, :, 2 * q_width + v_start : 2 * q_width + v_start + value_heads * value_dim],
            ],
            2,
        ).contiguous()
        mem.req_to_conv_state = SimpleNamespace(buffer=conv)
        mem.req_to_ssm_state = SimpleNamespace(
            buffer=global_ssm[:, :, rank * value_heads : (rank + 1) * value_heads].contiguous()
        )
        mem.req_to_lexical_state = {1: LexicalSnapshot(lexical.ngram.clone(), lexical.convolution.clone())}
        mem.qsa_raw_keys, mem.qsa_pooled_keys = raw.clone(), pooled.clone()
        mem.qsa_key_cos, mem.qsa_key_sin = cos.clone(), sin.clone()
        managers.append(mem)
    assert v_width == config["linear_num_value_heads"] * value_dim
    return managers


def test_pd_layout_changes_tp_and_preserves_replicated_kv_and_sharded_gdn(config):
    torch.manual_seed(1607)
    config = dict(config, linear_num_key_heads=4, linear_num_value_heads=4)
    kv = torch.randn(1, 16, 4, 8, device="cuda", dtype=torch.bfloat16)
    conv = torch.randn(3, 5, 1536, 3, device="cuda", dtype=torch.bfloat16)
    ssm = torch.randn(3, 5, 4, 128, 128, device="cuda", dtype=torch.float32)
    raw = torch.randn(1, 16, 16, device="cuda", dtype=torch.bfloat16)
    pooled = torch.randn_like(raw)
    cos, sin = torch.randn(16, 4, device="cuda", dtype=torch.bfloat16), torch.randn(
        16, 4, device="cuda", dtype=torch.bfloat16
    )
    lexical = LexicalSnapshot(
        torch.arange(10, device="cuda", dtype=torch.int64).view(5, 2),
        torch.randn(5, 9, 64, device="cuda", dtype=torch.bfloat16),
    )
    source = shard_managers(config, 4, kv, conv, ssm, raw, pooled, cos, sin, lexical)
    destination = shard_managers(
        config,
        2,
        kv * 0,
        conv * 0,
        ssm * 0,
        raw * 0,
        pooled * 0,
        cos * 0,
        sin * 0,
        LexicalSnapshot(lexical.ngram * 0, lexical.convolution * 0),
    )
    outgoing = source[0].alloc_paged_kv_move_buffer(2, 4)
    incoming = destination[0].alloc_paged_kv_move_buffer(2, 4)
    assert outgoing.shape == incoming.shape
    source_slots, destination_slots = [10, 2, 7], [1, 5, 11]
    source[0].write_mem_to_page_kv_move_buffer(source_slots, 0, 0, source, 4)
    source[0].write_mem_to_page_kv_move_buffer([], 1, 0, source, 4, "linear_att_state", 1)
    incoming.copy_(outgoing)
    destination[0].read_page_kv_move_buffer_to_mem(destination_slots, 0, 0, destination, 2)
    destination[0].read_page_kv_move_buffer_to_mem([], 1, 0, destination, 2, "linear_att_state", 3)
    for rank, mem in enumerate(destination):
        expected = torch.cat([kv[:, source_slots, rank : rank + 1], kv[:, source_slots, 2 + rank : 3 + rank]], 2)
        torch.testing.assert_close(mem.kv_buffer[:, destination_slots], expected, atol=0, rtol=0)
        torch.testing.assert_close(mem.qsa_raw_keys[:, destination_slots], raw[:, source_slots], atol=0, rtol=0)
        torch.testing.assert_close(mem.qsa_pooled_keys[:, destination_slots], pooled[:, source_slots], atol=0, rtol=0)
        torch.testing.assert_close(mem.qsa_key_cos[destination_slots], cos[source_slots], atol=0, rtol=0)
        torch.testing.assert_close(mem.qsa_key_sin[destination_slots], sin[source_slots], atol=0, rtol=0)
        expected_conv = torch.cat(
            [
                conv[:, 1, rank * 256 : (rank + 1) * 256],
                conv[:, 1, 512 + rank * 256 : 512 + (rank + 1) * 256],
                conv[:, 1, 1024 + rank * 256 : 1024 + (rank + 1) * 256],
            ],
            1,
        )
        torch.testing.assert_close(mem.req_to_conv_state.buffer[:, 3], expected_conv, atol=0, rtol=0)
        torch.testing.assert_close(
            mem.req_to_ssm_state.buffer[:, 3], ssm[:, 1, rank * 2 : (rank + 1) * 2], atol=0, rtol=0
        )
        torch.testing.assert_close(mem.req_to_lexical_state[1].ngram[3], lexical.ngram[1], atol=0, rtol=0)
        torch.testing.assert_close(mem.req_to_lexical_state[1].convolution[3], lexical.convolution[1], atol=0, rtol=0)


def test_pd_rejects_invalid_task_shape_before_copying(runtime):
    mem = runtime.mem
    with pytest.raises(ValueError, match="positive"):
        mem.alloc_paged_kv_move_buffer(1, 0)
    mem.alloc_paged_kv_move_buffer(2, 4)
    with pytest.raises(ValueError, match="request slot"):
        mem.write_mem_to_page_kv_move_buffer([], 1, 0, [mem], 1, "linear_att_state")
    with pytest.raises(ValueError, match="capacity"):
        mem.write_mem_to_page_kv_move_buffer([0] * 5, 0, 0, [mem], 1)
    with pytest.raises(ValueError, match="page kind"):
        mem.write_mem_to_page_kv_move_buffer([], 0, 0, [mem], 1, "unknown")
    with pytest.raises(ValueError, match="page index"):
        mem.write_mem_to_page_kv_move_buffer([], -1, 0, [mem], 1)


def test_pd_accepted_state_changes_tp_and_speculative_state_span(config):
    torch.manual_seed(3107)
    config = dict(config, linear_num_key_heads=4, linear_num_value_heads=4)
    kv = torch.zeros(1, 16, 4, 8, device="cuda", dtype=torch.bfloat16)
    conv = torch.randn(3, 5, 1536, 5, device="cuda", dtype=torch.bfloat16)
    ssm = torch.randn(3, 15, 4, 128, 128, device="cuda", dtype=torch.float32)
    raw = torch.zeros(1, 16, 16, device="cuda", dtype=torch.bfloat16)
    rotary = torch.zeros(16, 4, device="cuda", dtype=torch.bfloat16)
    lexical = LexicalSnapshot(
        torch.zeros(5, 2, device="cuda", dtype=torch.int64), torch.zeros(5, 9, 64, device="cuda", dtype=torch.bfloat16)
    )
    source = shard_managers(config, 4, kv, conv, ssm, raw, raw, rotary, rotary, lexical)
    destination = shard_managers(config, 2, kv, conv[..., :4] * 0, ssm[:, :10] * 0, raw, raw, rotary, rotary, lexical)
    for mem in source:
        mem.req_to_mtp_state_index = torch.full((5,), 2, dtype=torch.int32, device="cuda")
    for mem in destination:
        mem.req_to_mtp_state_index = torch.ones(5, dtype=torch.int32, device="cuda")
    outgoing = source[0].alloc_paged_kv_move_buffer(1, 4)
    incoming = destination[0].alloc_paged_kv_move_buffer(1, 4)
    source[0].write_mem_to_page_kv_move_buffer([], 0, 0, source, 4, "linear_att_state", 1)
    incoming.copy_(outgoing)
    destination[0].read_page_kv_move_buffer_to_mem([], 0, 0, destination, 2, "linear_att_state", 3)
    for rank, mem in enumerate(destination):
        expected_conv = torch.cat(
            [
                conv[:, 1, rank * 256 : (rank + 1) * 256, 2:5],
                conv[:, 1, 512 + rank * 256 : 512 + (rank + 1) * 256, 2:5],
                conv[:, 1, 1024 + rank * 256 : 1024 + (rank + 1) * 256, 2:5],
            ],
            1,
        )
        torch.testing.assert_close(mem.req_to_conv_state.buffer[:, 3, ..., :3], expected_conv, atol=0, rtol=0)
        torch.testing.assert_close(
            mem.req_to_ssm_state.buffer[:, 6], ssm[:, 5, rank * 2 : (rank + 1) * 2], atol=0, rtol=0
        )
        assert mem.req_to_mtp_state_index[3] == 0
    assert source[0].req_to_mtp_state_index[1] == 2


def test_pd_rejects_lexical_mismatch_before_changing_recurrent_state(runtime):
    mem = runtime.mem
    buffer = mem.alloc_paged_kv_move_buffer(2, 4)
    buffer.fill_(37)
    before = buffer.clone()
    history = mem.req_to_lexical_state[1]
    mem.req_to_lexical_state[1] = LexicalSnapshot(history.ngram, history.convolution[:, :-1])
    with pytest.raises(ValueError, match="lexical state"):
        mem.write_mem_to_page_kv_move_buffer([], 1, 0, [mem], 1, "linear_att_state", 1)
    torch.testing.assert_close(buffer, before, atol=0, rtol=0)
    conv = mem.req_to_conv_state.buffer.clone()
    ssm = mem.req_to_ssm_state.buffer.clone()
    with pytest.raises(ValueError, match="lexical state"):
        mem.read_page_kv_move_buffer_to_mem([], 1, 0, [mem], 1, "linear_att_state", 1)
    torch.testing.assert_close(mem.req_to_conv_state.buffer, conv, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(mem.req_to_ssm_state.buffer, ssm, atol=0, rtol=0, equal_nan=True)
