import importlib
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
import torch


class VaryingMetadata:
    def __init__(self, sizes):
        self.sizes = sizes
        self.next = 0

    def __getstate__(self):
        ordinal = self.next
        self.next += 1
        return {"ordinal": ordinal, "payload": bytes([ordinal]) * self.sizes[ordinal]}


@pytest.mark.parametrize(
    "sizes",
    [
        [32] * 8,
        [1, 1024, 3, 4096, 16, 64, 5, 8192],
        [8192, 5, 64, 16, 4096, 3, 1024, 1],
    ],
    ids=["equal", "first-shortest", "first-longest"],
)
def test_independent_pickle_slots_round_trip(monkeypatch, sizes):
    importlib.import_module("lightllm.common.basemodel")
    module = importlib.import_module("lightllm.common.kv_cache_mem_manager.mem_manager")
    name = "test-memory-manager-" + uuid.uuid4().hex
    shm_name = name + "_mem_manager_0"
    monkeypatch.setattr(module, "get_unique_server_name", lambda: name)
    monkeypatch.setattr(module, "get_node_world_size", lambda: 4)
    monkeypatch.setattr(module, "get_current_rank_in_node", lambda: 0)
    monkeypatch.setattr(module, "kv_trans_use_p2p", lambda: False)
    manager = module.MemoryManager.__new__(module.MemoryManager)
    manager.metadata = VaryingMetadata(sizes)
    indices = torch.arange(24, dtype=torch.int32).reshape(2, 12)
    try:
        manager.write_to_shm(SimpleNamespace(req_to_token_indexs=indices))
        for ordinal in reversed(range(8)):
            restored = module.MemoryManager.loads_from_shm(0)
            assert restored.metadata.ordinal == ordinal
            assert restored.metadata.payload == bytes([ordinal]) * sizes[ordinal]
            assert torch.equal(restored.req_to_token_indexs, indices)
        with pytest.raises(AssertionError):
            module.MemoryManager.loads_from_shm(0)
    finally:
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except FileNotFoundError:
            pass
        else:
            shm.close()
            shm.unlink()
        for lock in (name + "_mem_manager_lock", shm_name + ".lock"):
            Path("/tmp", lock).unlink(missing_ok=True)
