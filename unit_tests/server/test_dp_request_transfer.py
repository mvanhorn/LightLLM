import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def pipeline(monkeypatch):
    module = importlib.import_module("lightllm.server.router.model_infer.mode_backend.dp_backend.dp_shared_kv_trans")
    batch = importlib.import_module("lightllm.server.router.model_infer.infer_batch")
    context = SimpleNamespace(
        req_manager=SimpleNamespace(req_to_token_indexs=torch.full((4, 8), -1, dtype=torch.int32)),
        get_can_alloc_token_num=lambda: 100,
        radix_cache=SimpleNamespace(free_radix_cache_to_get_enough_token=lambda count: None),
    )
    monkeypatch.setattr(module, "g_infer_context", context)
    monkeypatch.setattr(batch, "g_infer_context", context)
    instance = module.DPKVSharedMoudle.__new__(module.DPKVSharedMoudle)
    instance.dp_rank_in_node = 0
    instance.backend = SimpleNamespace(
        model=SimpleNamespace(mem_manager=SimpleNamespace(operator=SimpleNamespace())),
        mem_managers=[SimpleNamespace(req_to_token_indexs=torch.arange(24).reshape(3, 8)) for _ in range(4)],
        dp_world_size=2,
        dp_size_in_node=2,
        rank_in_dp=0,
        node_nccl_group=object(),
        is_master_in_dp=True,
    )
    return module, instance, context


def test_complete_copy_finishes_before_cursor_publication(pipeline, monkeypatch):
    module, instance, context = pipeline
    events = []
    request = SimpleNamespace(req_idx=0, cur_kv_len=1, shm_req=SimpleNamespace(shm_cur_kv_len=1))
    move = module.TransTask(request, torch.tensor([3, 5, 7]), 1, 2, torch.tensor([2, 4, 6]), 2, 4)

    def copy(tasks, managers, dp_size, rank):
        assert tasks == [move] and managers is instance.backend.mem_managers
        assert (dp_size, rank) == (2, 0)
        events.append("copy")

    def barrier(group):
        assert group is instance.backend.node_nccl_group
        assert request.cur_kv_len == request.shm_req.shm_cur_kv_len == 1
        assert bool((context.req_manager.req_to_token_indexs == -1).all())
        events.append("barrier")

    instance.backend.model.mem_manager.operator.copy_requests_from_other_dp_ranks = copy
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: events.append("sync"))
    )
    monkeypatch.setattr(module.dist, "barrier", barrier)
    instance.kv_trans([move])
    assert events == ["copy", "sync", "barrier"]
    assert request.cur_kv_len == request.shm_req.shm_cur_kv_len == 4
    assert context.req_manager.req_to_token_indexs[0].tolist() == [-1, 3, 5, 7, -1, -1, -1, -1]


@pytest.mark.parametrize("complete", [False, True])
def test_empty_receivers_join_barrier_only_for_complete_request_protocol(pipeline, monkeypatch, complete):
    module, instance, _ = pipeline
    events = []
    if complete:
        instance.backend.model.mem_manager.operator.copy_requests_from_other_dp_ranks = (
            lambda tasks, *args: events.append(("copy", len(tasks)))
        )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: events.append("sync"))
    )
    monkeypatch.setattr(module.dist, "barrier", lambda group: events.append("barrier"))
    instance.kv_trans([])
    assert events == ([("copy", 0), "sync", "barrier"] if complete else [])


def test_failed_complete_copy_does_not_publish_destination(pipeline, monkeypatch):
    module, instance, context = pipeline
    request = SimpleNamespace(req_idx=0, cur_kv_len=1, shm_req=SimpleNamespace(shm_cur_kv_len=1))
    move = module.TransTask(request, torch.tensor([3]), 1, 2, torch.tensor([2]), 2, 2)

    def fail(*args):
        raise ValueError("incompatible state")

    instance.backend.model.mem_manager.operator.copy_requests_from_other_dp_ranks = fail
    with pytest.raises(ValueError, match="incompatible state"):
        instance.kv_trans([move])
    assert request.cur_kv_len == request.shm_req.shm_cur_kv_len == 1
    assert bool((context.req_manager.req_to_token_indexs == -1).all())


@pytest.mark.parametrize("complete,total,expected_count", [(False, 4, 1), (True, 4, 0), (True, 5, 1)])
def test_complete_request_protocol_never_clips_remote_state_boundary(
    pipeline, monkeypatch, complete, total, expected_count
):
    module, instance, context = pipeline
    if complete:
        instance.backend.model.mem_manager.operator.copy_requests_from_other_dp_ranks = lambda *args: None
    allocations = []

    def allocate(count):
        allocations.append(count)
        return torch.arange(count, dtype=torch.int32)

    instance.backend.model.mem_manager.alloc = allocate
    instance.shared_req_infos = SimpleNamespace(arr=np.array([[[1, 0], [4, 2]]], dtype=np.int64))
    monkeypatch.setattr(module.dist, "barrier", lambda group: None)
    request = SimpleNamespace(req_idx=0, cur_kv_len=1, get_cur_total_len=lambda: total)
    result = instance.build_shared_kv_trans_tasks([request], [0])
    assert len(result) == expected_count
    if result:
        assert allocations == [min(4, total - 1) - 1]
        assert result[0].source_req_idx == 2 and result[0].source_kv_len == 4
        assert result[0].max_kv_len_mem_manager_index == 2
    else:
        assert not allocations
