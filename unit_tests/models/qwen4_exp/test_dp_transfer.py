from types import SimpleNamespace

import pytest
import torch

from lightllm.models.qwen4_exp.dp_transfer import copy_requests
from lightllm.models.qwen4_exp.lexical_cache import LexicalSnapshot
from lightllm.models.qwen4_exp.mem_manager import Qwen4ExpMemManager, Qwen4ExpMemOperator
from lightllm.server.router.model_infer.mode_backend.dp_backend.dp_shared_kv_trans import TransTask


def owner(seed, mtp=2, rotary=4, conv_dtype=torch.bfloat16):
    mem = Qwen4ExpMemManager.__new__(Qwen4ExpMemManager)
    mem.size = 16
    mem.linear_config = SimpleNamespace(conv_kernel_size=3)

    def values(shape, dtype=torch.bfloat16):
        count = torch.Size(shape).numel()
        return (torch.arange(count).reshape(shape) % 31 + seed * 32).to(dtype)

    mem.kv_buffer = values((2, 17, 2, 4))
    mem.qsa_raw_keys = values((2, 17, 3))
    mem.qsa_pooled_keys = values((2, 17, 3)) + 1
    mem.qsa_key_cos = values((17, rotary))
    mem.qsa_key_sin = values((17, rotary)) + 2
    mem.req_to_token_indexs = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]] * 4, dtype=torch.int32)
    mem.req_to_conv_state = SimpleNamespace(buffer=values((2, 4, 3, 2 + mtp), conv_dtype))
    mem.req_to_ssm_state = SimpleNamespace(buffer=values((2, 4 * (mtp + 1), 2, 2, 2), torch.float32))
    mem.req_to_mtp_state_index = torch.tensor([mtp, min(1, mtp), 0, 0], dtype=torch.int32) if mtp else None
    mem.req_to_lexical_state = {1: LexicalSnapshot(values((4, 3), torch.int64), values((4, 2, 8)))}
    return mem


def task(source, target_request=0, source_request=1, destination_slots=(2, 4, 6), dp=1, rank=0):
    return TransTask(
        req=SimpleNamespace(req_idx=target_request, cur_kv_len=1),
        mem_indexes=torch.tensor(destination_slots, dtype=torch.int32),
        max_kv_len_dp_rank=dp,
        max_kv_len_mem_manager_index=dp * 2 + rank,
        max_kv_len_mem_indexes=source.req_to_token_indexs[source_request, 1:4].clone(),
        source_req_idx=source_request,
        source_kv_len=4,
    )


def snapshot(mem):
    result = {name: tensor.clone() for name, (tensor, _) in mem._slot_buffers().items()}
    result.update(conv=mem.req_to_conv_state.buffer.clone(), ssm=mem.req_to_ssm_state.buffer.clone())
    result["mapping"] = mem.req_to_token_indexs.clone()
    if mem.req_to_mtp_state_index is not None:
        result["offsets"] = mem.req_to_mtp_state_index.clone()
    for layer, state in mem.req_to_lexical_state.items():
        result[f"{layer}:ngram"] = state.ngram.clone()
        result[f"{layer}:convolution"] = state.convolution.clone()
    return result


@pytest.mark.parametrize("source_mtp,target_mtp", [(0, 0), (0, 2), (2, 0), (2, 2)])
@pytest.mark.parametrize("rotary", [0, 4])
@pytest.mark.parametrize("conv_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("prefix_start", [0, 1])
def test_complete_copy_preserves_neighbors_and_only_commits_accepted_state(
    source_mtp, target_mtp, rotary, conv_dtype, prefix_start
):
    target = owner(0, target_mtp, rotary, conv_dtype)
    sources = [target, owner(9), owner(2, source_mtp, rotary, conv_dtype), owner(8)]
    source = sources[2]
    if source_mtp:
        source.req_to_mtp_state_index[1] = source_mtp
    before_source, before_target = snapshot(source), snapshot(target)
    move = task(source)
    move.req.cur_kv_len = prefix_start
    move.max_kv_len_mem_indexes = source.req_to_token_indexs[1, prefix_start:4].clone()
    move.mem_indexes = torch.tensor([2, 4, 6, 8][: 4 - prefix_start], dtype=torch.int32)
    source_offset = int(source.req_to_mtp_state_index[1]) if source_mtp else 0
    calls = []
    original = source.get_index_kv_buffer

    def bounded(indices):
        calls.append(indices.numel())
        return original(indices)

    source.get_index_kv_buffer = bounded
    copy_requests(target, [], sources, 2, 0)
    copy_requests(target, [move], sources, 2, 0, token_chunk_size=2)
    assert calls == ([2, 1] if prefix_start else [2, 2])
    for name, (tensor, axis) in target._slot_buffers().items():
        expected = before_target[name].clone()
        expected.index_copy_(
            axis, move.mem_indexes.long(), before_source[name].index_select(axis, move.max_kv_len_mem_indexes.long())
        )
        torch.testing.assert_close(tensor, expected, atol=0, rtol=0)
    expected_conv = before_target["conv"].clone()
    expected_conv[:, 0, ..., :2] = before_source["conv"][:, 1, ..., source_offset : source_offset + 2]
    expected_ssm = before_target["ssm"].clone()
    expected_ssm[:, 0] = before_source["ssm"][:, (source_mtp + 1) + source_offset]
    torch.testing.assert_close(target.req_to_conv_state.buffer, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(target.req_to_ssm_state.buffer, expected_ssm, atol=0, rtol=0)
    for field in ("ngram", "convolution"):
        expected = before_target[f"1:{field}"].clone()
        expected[0] = before_source[f"1:{field}"][1]
        torch.testing.assert_close(getattr(target.req_to_lexical_state[1], field), expected, atol=0, rtol=0)
    if target_mtp:
        expected = before_target["offsets"].clone()
        expected[0] = 0
        torch.testing.assert_close(target.req_to_mtp_state_index, expected, atol=0, rtol=0)
    for name, value in snapshot(source).items():
        torch.testing.assert_close(value, before_source[name], atol=0, rtol=0)
    torch.testing.assert_close(target.req_to_token_indexs, before_target["mapping"], atol=0, rtol=0)
    assert move.req.cur_kv_len == prefix_start


def test_multiple_sources_select_matching_tp_partition():
    managers = [owner(i) for i in range(6)]
    moves = [task(managers[2]), task(managers[4], target_request=1, destination_slots=(8, 10, 12), dp=2)]
    copy_requests(managers[0], moves, managers, 3, 0)
    for move in moves:
        source = managers[move.max_kv_len_mem_manager_index]
        torch.testing.assert_close(
            managers[0].req_to_lexical_state[1].ngram[move.req.req_idx],
            source.req_to_lexical_state[1].ngram[1],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            managers[0].req_to_ssm_state.buffer[:, move.req.req_idx * 3],
            source.req_to_ssm_state.buffer[:, 4],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize(
    "damage",
    [
        "missing_request",
        "wrong_boundary",
        "wrong_binding",
        "wrong_tp",
        "invalid_offset",
        "wrong_qsa",
        "wrong_capacity",
        "wrong_lexical",
        "invalid_slot",
        "duplicate_slot",
        "hold_request",
    ],
)
def test_bad_late_request_fails_before_any_destination_write(damage):
    managers = [owner(i) for i in range(6)]
    first = task(managers[2])
    bad = task(managers[4], target_request=1, destination_slots=(8, 10, 12), dp=2)
    if damage == "missing_request":
        bad.source_req_idx = None
    elif damage == "wrong_boundary":
        bad.source_kv_len = 3
    elif damage == "wrong_binding":
        bad.max_kv_len_mem_indexes[0] = 0
    elif damage == "wrong_tp":
        bad.max_kv_len_mem_manager_index = 5
    elif damage == "invalid_offset":
        managers[4].req_to_mtp_state_index[1] = 3
    elif damage == "wrong_qsa":
        managers[4].qsa_raw_keys = managers[4].qsa_raw_keys.double()
    elif damage == "wrong_capacity":
        managers[4].qsa_key_sin = managers[4].qsa_key_sin[:-1].contiguous()
    elif damage == "wrong_lexical":
        managers[4].req_to_lexical_state.clear()
    elif damage == "invalid_slot":
        bad.mem_indexes[0] = 16
    elif damage == "duplicate_slot":
        bad.mem_indexes[0] = first.mem_indexes[0]
    elif damage == "hold_request":
        bad.source_req_idx = 3
    before = snapshot(managers[0])
    with pytest.raises(ValueError):
        copy_requests(managers[0], [first, bad], managers, 3, 0)
    for name, value in snapshot(managers[0]).items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_legacy_task_defaults_cannot_silently_copy_token_only():
    target, source = owner(0), owner(1)
    move = TransTask(SimpleNamespace(req_idx=0, cur_kv_len=1), torch.tensor([1]), 1, 1, torch.tensor([3]))
    assert move.source_req_idx is None and move.source_kv_len is None
    with pytest.raises(ValueError, match="live source"):
        copy_requests(target, [move], [target, source], 2, 0)
    with pytest.raises(NotImplementedError, match="derived index keys"):
        Qwen4ExpMemOperator(target).copy_kv_from_other_dp_ranks()


def test_empty_receiver_does_not_require_colocated_global_tp_partition():
    target = owner(0)
    before = snapshot(target)
    copy_requests(target, [], [target], dp_size_in_node=1, rank_in_dp=5)
    for name, value in snapshot(target).items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
