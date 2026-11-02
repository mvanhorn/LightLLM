import torch

from lightllm.common.req_manager import ReqManagerForMamba
from .lexical_cache import LexicalRequestCache
from .speculative_cache import LexicalSpeculativeCache


class Qwen4ExpReqManager(ReqManagerForMamba):
    def __init__(self, max_request_num, max_sequence_length, mem_manager, linear_config, layer_weights):
        super().__init__(max_request_num, max_sequence_length, mem_manager, linear_config)
        self.lexical_caches = {
            weight.layer_num_: LexicalRequestCache(weight, max_request_num + 1)
            for weight in layer_weights
            if hasattr(weight, "ple")
        }
        self.speculative_lexical_caches = (
            {layer: LexicalSpeculativeCache(cache, self.mtp_step + 1) for layer, cache in self.lexical_caches.items()}
            if self.mtp_step > 0
            else {}
        )

    def stage_speculative_lexical(self, layer, metadata, batch):
        self.speculative_lexical_caches[layer].stage(metadata, batch)

    def commit_speculative_state(self, requests, starts, accepted_lengths):
        for cache in self.speculative_lexical_caches.values():
            cache.commit(requests, starts, accepted_lengths, self.HOLD_REQUEST_ID)

    def init_linear_att_state(self, req):
        super().init_linear_att_state(req)
        for cache in self.lexical_caches.values():
            cache.reset(req.req_idx)

    def prepare_lexical(self, metadata, reset_empty_prefix=False):
        return {
            layer: cache.prepare(metadata, reset_empty_prefix=reset_empty_prefix)
            for layer, cache in self.lexical_caches.items()
        }

    def commit_lexical(self, metadata, batches, accepted_lengths=None):
        for layer, cache in self.lexical_caches.items():
            cache.commit(metadata, batches[layer], accepted_lengths)

    def snapshot_lexical(self, request):
        return {layer: cache.snapshot(request) for layer, cache in self.lexical_caches.items()}

    def restore_lexical(self, request, snapshots):
        for layer, cache in self.lexical_caches.items():
            cache.restore(request, snapshots[layer])

    def _prefix_extension(self, prefix_cache):
        extension = prefix_cache.extra_state_cache
        if extension is None or set(extension.histories) != set(self.lexical_caches):
            raise ValueError("Qwen4Exp prefix snapshot is missing its lexical state")
        extension.validate_caches(self.lexical_caches)
        return extension

    def copy_extra_state_to_prefix_cache(self, requests, destinations, prefix_cache):
        extension = self._prefix_extension(prefix_cache)
        device = self.req_to_conv_state.buffer.device
        requests = torch.as_tensor(requests, dtype=torch.int32, device=device).contiguous()
        destinations = torch.as_tensor(destinations, dtype=torch.int32, device=device).contiguous()
        if self.mtp_step > 0:
            from .recurrent_state import copy_accepted_to_prefix

            copy_accepted_to_prefix(self, requests, destinations, prefix_cache)
        extension.copy_from_requests(self.lexical_caches, requests, destinations)

    def copy_big_page_buffer_to_linear_att_state(self, big_page_buffer_idx, req):
        cache = self.mem_manager.linear_att_big_page_buffers
        extension = self._prefix_extension(cache)
        super().copy_big_page_buffer_to_linear_att_state(big_page_buffer_idx, req)
        extension.restore_request(self.lexical_caches, req.req_idx, big_page_buffer_idx)

    def copy_small_page_buffer_to_linear_att_state(self, req, linear_att_small_page_buffers):
        extension = self._prefix_extension(linear_att_small_page_buffers)
        super().copy_small_page_buffer_to_linear_att_state(req, linear_att_small_page_buffers)
        extension.restore_request(self.lexical_caches, req.req_idx, req.shared_kv_node.small_page_buffer_idx)
