from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
from sortedcontainers import SortedDict
import torch

from lightllm.models.qwen4_exp.dp_prefix_transfer import _pool_fields, copy_with_prefix_cache
from lightllm.models.qwen4_exp.lexical_cache import LexicalSnapshot
from lightllm.server.router.dynamic_prompt import linear_att_radix_cache as radix
from lightllm.server.router.model_infer import infer_batch
from lightllm.server.router.model_infer.mode_backend.dp_backend.dp_shared_kv_trans import TransTask
from .test_dp_transfer import owner, snapshot


class Pool:
    def __init__(self, mem, size=8):
        conv, ssm = mem.req_to_conv_state.buffer, mem.req_to_ssm_state.buffer
        self.conv = torch.zeros((size, conv.shape[0], *conv.shape[2:-1], 2), dtype=conv.dtype)
        self.ssm = torch.zeros((size, ssm.shape[0], *ssm.shape[2:]), dtype=ssm.dtype)
        self.extra_state_cache = SimpleNamespace(
            histories={
                layer: LexicalSnapshot(
                    torch.zeros((size, *state.ngram.shape[1:]), dtype=state.ngram.dtype),
                    torch.zeros((size, *state.convolution.shape[1:]), dtype=state.convolution.dtype),
                )
                for layer, state in mem.req_to_lexical_state.items()
            }
        )
        self.free_list = deque(range(size))

    def get_state_cache(self, index):
        return self.conv[index], self.ssm[index]

    def alloc_one_state_cache(self):
        return self.free_list.popleft() if self.free_list else None

    def free_state_cache(self, indices):
        self.free_list.extend(indices)

    def get_free_cache_num(self):
        return len(self.free_list)


def fill(pool, index, marker):
    for value in _pool_fields(pool, index).values():
        value.fill_(marker)


def setup_contexts(monkeypatch, end=10, shared=0, inflight=False, source_mtp=2, target_mtp=2):
    monkeypatch.setattr(radix, "SharedArray", lambda *args, **kwargs: SimpleNamespace(arr=np.zeros(1, dtype=np.int64)))
    memories = [owner(0, target_mtp), owner(1, source_mtp)]
    caches = []
    for rank, mem in enumerate(memories):
        mem.req_to_token_indexs = torch.arange(16, dtype=torch.int32).repeat(4, 1)
        mem.linear_att_big_page_buffers = Pool(mem)
        small = Pool(mem)
        caches.append(radix.LinearAttPagedRadixCache("cpu-dp", 16, rank, 2, 2, mem, small))
    source, destination = memories[1], memories[0]
    source_ids = SortedDict()
    for boundary in range(4, end + 1, 4):
        index = source.linear_att_big_page_buffers.alloc_one_state_cache()
        source_ids[boundary] = index
        fill(source.linear_att_big_page_buffers, index, boundary)
    key, hashes = torch.arange(end), list(range(100, 100 + end // 2))
    source_node = None
    if not inflight:
        small_indices = [None] * len(hashes)
        if end % 4:
            small_id = caches[1].linear_att_small_page_buffers.alloc_one_state_cache()
            fill(caches[1].linear_att_small_page_buffers, small_id, end)
            small_indices[-1] = small_id
        caches[1].insert(key, key, hashes, small_indices, source_ids)
        source_node, _, _ = caches[1].match_prefix(key, hashes, update_refs=True)
        if not source_node.is_big_page_node():
            source_node = caches[1].deref_to_first_big_page_node(source_node)
    source_request = SimpleNamespace(
        req_id=12, req_idx=0, cur_kv_len=end, shared_kv_node=source_node, linear_att_len_to_big_page_id=source_ids
    )
    offset = int(source.req_to_mtp_state_index[0]) if source_mtp else 0
    source.req_to_conv_state.buffer[:, 0, ..., offset : offset + 2].fill_(end)
    source.req_to_ssm_state.buffer[:, offset].fill_(end)
    source.req_to_lexical_state[1].ngram[0].fill_(end)
    source.req_to_lexical_state[1].convolution[0].fill_(end)
    target_slots = torch.arange(15, 15 - end, -1, dtype=torch.int32)
    target_node = None
    if shared:
        index = destination.linear_att_big_page_buffers.alloc_one_state_cache()
        fill(destination.linear_att_big_page_buffers, index, shared)
        caches[0].insert(
            key[:shared],
            target_slots[:shared].long(),
            hashes[: shared // 2],
            [None] * (shared // 2),
            SortedDict({shared: index}),
        )
        target_node, _, _ = caches[0].match_prefix(key[:shared], hashes[: shared // 2], update_refs=True)
    request = SimpleNamespace(
        req_id=12,
        req_idx=1,
        cur_kv_len=shared,
        shared_kv_node=target_node,
        linear_att_len_to_big_page_id=SortedDict(),
        linear_att_cache_len=end,
        tail_linear_att_small_page_buffer_id=None,
    )
    contexts = [
        SimpleNamespace(
            radix_cache=cache,
            args=SimpleNamespace(linear_att_hash_page_size=2, linear_att_page_block_num=2),
            requests_mapping={12: req},
        )
        for cache, req in zip(caches, (request, source_request))
    ]
    move = TransTask(request, target_slots[shared:], 1, 1, source.req_to_token_indexs[0, shared:end], 0, end)
    return memories, contexts, move, key, hashes, target_slots


class Node:
    def __init__(self, size=2):
        self.barrier = Barrier(size, timeout=10)
        self.values = [None] * size
        self.payload = None
        self.payload_bytes = []

    def peer(self, rank):
        parent = self

        class Collective:
            def __init__(self):
                self.rank = rank

            def gather(self, value):
                parent.values[rank] = value
                parent.barrier.wait()
                result = copy.deepcopy(parent.values)
                parent.barrier.wait()
                return result

            def broadcast(self, source, value):
                if rank == source:
                    parent.payload = value
                    if "fields" in value:
                        for tensor in value["fields"].values():
                            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
                        parent.payload_bytes.append(
                            sum(tensor.numel() * tensor.element_size() for tensor in value["fields"].values())
                        )
                parent.barrier.wait()
                result = copy.deepcopy(parent.payload)
                parent.barrier.wait()
                return result

        return Collective()


def run(memories, contexts, move):
    node = Node()

    def execute(rank):
        copy_with_prefix_cache(
            memories[rank],
            [move] if rank == 0 else [],
            memories,
            2,
            0,
            contexts[rank],
            None,
            collective=node.peer(rank),
        )

    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(execute, rank) for rank in (0, 1)]
        errors = []
        for future in futures:
            try:
                future.result(timeout=15)
            except Exception as error:
                errors.append(error)
    return node, errors


@pytest.mark.parametrize("end", [8, 10])
@pytest.mark.parametrize("shared", [0, 4])
@pytest.mark.parametrize("inflight", [False, True])
@pytest.mark.parametrize("source_mtp,target_mtp", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_remote_jump_can_insert_and_restore_full_native_prefix(
    monkeypatch, end, shared, inflight, source_mtp, target_mtp
):
    memories, contexts, move, key, hashes, target_slots = setup_contexts(
        monkeypatch, end, shared, inflight, source_mtp, target_mtp
    )
    node, errors = run(memories, contexts, move)
    assert not errors
    cache, request = contexts[0].radix_cache, move.req
    expected_boundaries = list(range(shared + 4, end + 1, 4))
    assert list(request.linear_att_len_to_big_page_id) == expected_boundaries
    for boundary, index in request.linear_att_len_to_big_page_id.items():
        for tensor in _pool_fields(cache.linear_att_big_page_buffers, index).values():
            assert bool((tensor == boundary).all())
    small_indices = [None] * len(hashes)
    if end % 4:
        index = request.tail_linear_att_small_page_buffer_id
        assert index is not None
        for tensor in _pool_fields(cache.linear_att_small_page_buffers, index).values():
            assert bool((tensor == end).all())
        small_indices[-1] = index

    cache.insert(key, target_slots.long(), hashes, small_indices, request.linear_att_len_to_big_page_id)
    assert not request.linear_att_len_to_big_page_id
    leaf, length, slots = cache.match_prefix(key, hashes, update_refs=True)
    assert leaf is not None and length == end
    torch.testing.assert_close(slots, target_slots.long(), atol=0, rtol=0)
    cache.dec_node_ref_counter(leaf)
    assert len(node.payload_bytes) == sum(boundary < end for boundary in expected_boundaries)
    assert move.req.cur_kv_len == shared


@pytest.mark.parametrize("shared", [0, 4])
@pytest.mark.parametrize("inflight", [False, True])
@pytest.mark.parametrize("source_mtp,target_mtp", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_abort_before_final_prompt_boundary_releases_transferred_snapshots(
    monkeypatch, shared, inflight, source_mtp, target_mtp
):
    memories, contexts, move, _, _, target_slots = setup_contexts(
        monkeypatch, 6, shared, inflight, source_mtp, target_mtp
    )
    context, request = contexts[0], move.req
    request.linear_att_cache_len = 10
    pool = context.radix_cache.linear_att_big_page_buffers
    free_before = pool.get_free_cache_num()
    _, errors = run(memories, contexts, move)
    assert not errors and request.tail_linear_att_small_page_buffer_id is None
    assert list(request.linear_att_len_to_big_page_id) == ([4] if shared == 0 else [])

    context.req_manager = SimpleNamespace(req_to_token_indexs=memories[0].req_to_token_indexs)
    context.is_linear_att_mixed_model = True
    context.req_manager.req_to_token_indexs[request.req_idx, :6] = target_slots
    request.cur_kv_len = 6
    monkeypatch.setattr(infer_batch, "g_infer_context", context)
    monkeypatch.setattr(infer_batch, "get_env_start_args", lambda: context.args)
    freed = []
    infer_batch.InferenceContext._linear_att_free_req(context, freed, request)
    assert pool.get_free_cache_num() == free_before
    assert not request.linear_att_len_to_big_page_id
    assert request.cur_kv_len == shared and request.shared_kv_node is None
    torch.testing.assert_close(torch.cat(freed), target_slots[shared:], atol=0, rtol=0)


@pytest.mark.parametrize("missing_partition", [False, True])
def test_two_dp_replicas_preserve_tp_partition_and_collective_failure(monkeypatch, missing_partition):
    pairs = [setup_contexts(monkeypatch) for _ in range(2)]
    memories = [pair[0][role] for role in (0, 1) for pair in pairs]
    contexts = [pair[1][role] for role in (0, 1) for pair in pairs]
    moves = [pair[2] for pair in pairs]
    for partition, move in enumerate(moves):
        move.max_kv_len_mem_manager_index = 2 + partition
    source = memories[3]
    for tensor, _ in source._slot_buffers().values():
        tensor.add_(32)
    source.req_to_conv_state.buffer[:, 0, ..., 2:4].add_(32)
    source.req_to_ssm_state.buffer[:, 2].add_(32)
    for history in source.req_to_lexical_state.values():
        history.ngram[0].add_(32)
        history.convolution[0].add_(32)

    source_node = contexts[3].requests_mapping[12].shared_kv_node
    while source_node is not None and source_node.big_page_buffer_idx is not None:
        fill(
            source.linear_att_big_page_buffers, source_node.big_page_buffer_idx, source_node.node_prefix_total_len + 32
        )
        source_node = source_node.parent
    if missing_partition:
        contexts[3].requests_mapping[12].shared_kv_node.parent.big_page_buffer_idx = None
    before = [snapshot(memory) for memory in memories]
    free = [context.radix_cache.linear_att_big_page_buffers.get_free_cache_num() for context in contexts]
    node = Node(4)

    def execute(rank):
        copy_with_prefix_cache(
            memories[rank],
            [moves[rank]] if rank < 2 else [],
            memories,
            2,
            rank % 2,
            contexts[rank],
            None,
            collective=node.peer(rank),
        )

    with ThreadPoolExecutor(4) as executor:
        futures = [executor.submit(execute, rank) for rank in range(4)]
        errors = []
        for future in futures:
            try:
                future.result(timeout=15)
            except Exception as error:
                errors.append(error)
    if missing_partition:
        assert len(errors) == 4 and all(isinstance(error, RuntimeError) for error in errors)
        for rank, context in enumerate(contexts):
            assert context.radix_cache.linear_att_big_page_buffers.get_free_cache_num() == free[rank]
            for name, value in snapshot(memories[rank]).items():
                torch.testing.assert_close(value, before[rank][name], atol=0, rtol=0)
        return
    assert not errors
    for rank, move in enumerate(moves):
        cache, request = contexts[rank].radix_cache, move.req
        assert list(request.linear_att_len_to_big_page_id) == [4, 8]
        for boundary, index in request.linear_att_len_to_big_page_id.items():
            assert all(
                bool((value == boundary + rank * 32).all())
                for value in _pool_fields(cache.linear_att_big_page_buffers, index).values()
            )
        assert all(
            bool((value == 10 + rank * 32).all())
            for value in _pool_fields(
                cache.linear_att_small_page_buffers, request.tail_linear_att_small_page_buffer_id
            ).values()
        )
        for name, (tensor, axis) in memories[rank]._slot_buffers().items():
            expected = before[rank][name].clone()
            expected.index_copy_(
                axis,
                move.mem_indexes.long(),
                before[rank + 2][name].index_select(axis, move.max_kv_len_mem_indexes.long()),
            )
            torch.testing.assert_close(tensor, expected, atol=0, rtol=0)


@pytest.mark.parametrize("damage", ["missing_snapshot", "wrong_identity", "exhausted_pool", "wrong_pool_dtype"])
def test_collective_failure_releases_unpublished_slots_and_preserves_live_requests(monkeypatch, damage):
    memories, contexts, move, *_ = setup_contexts(monkeypatch)
    if damage == "missing_snapshot":
        contexts[1].requests_mapping[12].shared_kv_node.parent.big_page_buffer_idx = None
    elif damage == "wrong_identity":
        contexts[1].requests_mapping[12].req_id = 99
    elif damage == "exhausted_pool":
        contexts[0].radix_cache.linear_att_big_page_buffers.free_list.clear()
    elif damage == "wrong_pool_dtype":
        contexts[1].radix_cache.linear_att_big_page_buffers.ssm = contexts[
            1
        ].radix_cache.linear_att_big_page_buffers.ssm.double()
    before = snapshot(memories[0])
    free = contexts[0].radix_cache.linear_att_big_page_buffers.get_free_cache_num()
    _, errors = run(memories, contexts, move)
    assert len(errors) == 2 and all(isinstance(error, RuntimeError) for error in errors)
    assert contexts[0].radix_cache.linear_att_big_page_buffers.get_free_cache_num() == free
    assert not move.req.linear_att_len_to_big_page_id and move.req.tail_linear_att_small_page_buffer_id is None
    for name, value in snapshot(memories[0]).items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
