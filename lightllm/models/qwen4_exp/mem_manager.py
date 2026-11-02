import torch

from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager
from lightllm.common.kv_cache_mem_manager.operator.base import BaseMemManagerOperator
from lightllm.common.kv_cache_mem_manager.operator.linear_att import LinearAttMemOperator
from .state import QSAExecutionState
from .lexical_cache import LexicalSnapshot


class Qwen4ExpMemOperator(LinearAttMemOperator):
    def __init__(self, mem_manager):
        BaseMemManagerOperator.__init__(self, mem_manager)
        self.linear_config = mem_manager.linear_config

    def copy_mem_to_mem(self, src_mem_index, dst_mem_index):
        self.mem_manager.load_index_kv_buffer(dst_mem_index, self.mem_manager.get_index_kv_buffer(src_mem_index))

    def load_extra_cpu_cache(self, mem_indexes, state_indexes, page_indexes, cpu_cache_tensor):
        from .cpu_transfer import transfer_extra_pages

        transfer_extra_pages(self.mem_manager, mem_indexes, state_indexes, page_indexes, cpu_cache_tensor)

    def offload_extra_cpu_cache(self, mem_indexes, state_indexes, page_indexes, page_readies, cpu_cache_tensor):
        from .cpu_transfer import transfer_extra_pages

        transfer_extra_pages(self.mem_manager, mem_indexes, state_indexes, page_indexes, cpu_cache_tensor, page_readies)

    def copy_extra_prefix_state(self, source_pool, source_index, destination_pool, destination_index):
        from .cpu_transfer import copy_prefix_extension

        copy_prefix_extension(
            source_pool, source_index, destination_pool, destination_index, self.mem_manager.kv_buffer.device
        )

    def copy_kv_from_other_dp_ranks(self, *args, **kwargs):
        raise NotImplementedError("Qwen4Exp cross-DP cache movement must include derived index keys")


class Qwen4ExpMemManager(Qwen3NextMemManager):
    operator_class = Qwen4ExpMemOperator

    def __init__(self, *args, indexer_head_dim: int, rotary_dim: int, **kwargs):
        if indexer_head_dim < 1 or rotary_dim < 0 or rotary_dim % 2 or rotary_dim > indexer_head_dim:
            raise ValueError("Invalid Qwen4Exp index key or rotary dimensions")
        self.indexer_head_dim = indexer_head_dim
        self.rotary_dim = rotary_dim
        super().__init__(*args, **kwargs)

    def get_cell_size(self):
        layers = self.linear_config.get_full_att_kv_layer_num_with_draft_model()
        extra_elements = 2 * layers * self.indexer_head_dim + 2 * self.rotary_dim
        return super().get_cell_size() + extra_elements * self.dtype.itemsize

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        layers = self.linear_config.get_full_att_kv_layer_num_with_draft_model()
        self.qsa_raw_keys = torch.empty((layers, size + 1, self.indexer_head_dim), dtype=dtype, device="cuda")
        self.qsa_pooled_keys = torch.empty_like(self.qsa_raw_keys)
        self.qsa_key_cos = torch.empty((size + 1, self.rotary_dim), dtype=dtype, device="cuda")
        self.qsa_key_sin = torch.empty_like(self.qsa_key_cos)
        interval = self.linear_config.full_attention_interval
        main_layers = self.linear_config.get_main_model_full_att_layer_num()
        layer_indices = [(index + 1) * interval - 1 for index in range(main_layers)]
        layer_indices += [self.linear_config.all_layer_num + index for index in range(layers - main_layers)]
        self.qsa_states = {
            layer: QSAExecutionState(
                self.qsa_raw_keys[index], self.qsa_pooled_keys[index], self.qsa_key_cos, self.qsa_key_sin
            )
            for index, layer in enumerate(layer_indices)
        }

    def _free_buffers(self):
        self.qsa_states = {}
        self.qsa_raw_keys = self.qsa_pooled_keys = self.qsa_key_cos = self.qsa_key_sin = None
        super()._free_buffers()

    def store_positions(self, slots, cos, sin):
        self.qsa_key_cos.index_copy_(0, slots, cos)
        self.qsa_key_sin.index_copy_(0, slots, sin)

    def _slot_buffers(self):
        return {
            "kv_buffer": (self.kv_buffer, 1),
            "qsa_raw_keys": (self.qsa_raw_keys, 1),
            "qsa_pooled_keys": (self.qsa_pooled_keys, 1),
            "qsa_key_cos": (self.qsa_key_cos, 0),
            "qsa_key_sin": (self.qsa_key_sin, 0),
        }

    def _indices(self, index):
        return torch.as_tensor(index, device=self.kv_buffer.device, dtype=torch.int64).reshape(-1)

    def get_index_kv_buffer(self, index):
        indices = self._indices(index)
        return {name: tensor.index_select(axis, indices) for name, (tensor, axis) in self._slot_buffers().items()}

    def load_index_kv_buffer(self, index, load_tensor_dict):
        indices = self._indices(index)
        buffers = self._slot_buffers()

        for name, (tensor, axis) in buffers.items():
            source = load_tensor_dict[name]
            expected = list(tensor.shape)
            expected[axis] = indices.numel()
            if source.shape != tuple(expected) or source.dtype != tensor.dtype:
                raise ValueError(f"Incompatible Qwen4Exp cache snapshot: {name}")
        for name, (tensor, axis) in buffers.items():
            tensor.index_copy_(axis, indices, load_tensor_dict[name].to(tensor.device, non_blocking=True))

    def attach_request_state(self, req_manager):
        self.req_to_conv_state = req_manager.req_to_conv_state
        self.req_to_ssm_state = req_manager.req_to_ssm_state
        self.req_to_mtp_state_index = req_manager.req_to_mtp_state_index
        self.req_to_lexical_state = {
            layer: LexicalSnapshot(cache.ngram, cache.convolution)
            for layer, cache in req_manager.lexical_caches.items()
        }

    def write_to_shm(self, req_manager):
        self.attach_request_state(req_manager)
        return super().write_to_shm(req_manager)

    def alloc_paged_kv_move_buffer(self, page_num, page_size):
        from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextLinearAttPageHelper
        from .pd_transfer import Qwen4ExpPDLayout

        if page_num < 1 or page_size < 1:
            raise ValueError("Qwen PD page count and capacity must be positive")
        helper = Qwen3NextLinearAttPageHelper(self)
        self.pd_layout = Qwen4ExpPDLayout(self.linear_config, page_size, helper.state_nbytes)
        self.kv_move_buffer = torch.empty(
            (page_num, page_size, 1, 1, self.pd_layout.row_bytes), dtype=torch.uint8, device="cuda"
        )
        self._buffer_mem_indexes_tensors = [
            torch.empty(page_size, dtype=torch.int64, device="cpu", pin_memory=True) for _ in range(page_num)
        ]
        helper.assert_page_size()
        return self.kv_move_buffer

    def write_mem_to_page_kv_move_buffer(
        self, mem_indexes, page_index, dp_index, mem_managers, dp_world_size, page_kind="kv", req_idx=None
    ):
        from .pd_transfer import Qwen4ExpPDPageHelper

        Qwen4ExpPDPageHelper(self).transfer(
            mem_indexes, page_index, dp_index, mem_managers, dp_world_size, page_kind, req_idx, write=True
        )

    def read_page_kv_move_buffer_to_mem(
        self, mem_indexes, page_index, dp_index, mem_managers, dp_world_size, page_kind="kv", req_idx=None
    ):
        from .pd_transfer import Qwen4ExpPDPageHelper

        Qwen4ExpPDPageHelper(self).transfer(
            mem_indexes, page_index, dp_index, mem_managers, dp_world_size, page_kind, req_idx, write=False
        )
