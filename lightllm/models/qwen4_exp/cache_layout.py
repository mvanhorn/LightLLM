from dataclasses import dataclass
from math import prod

import torch

from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.utils.envs_utils import get_env_start_args


def align(value, boundary):
    return (value + boundary - 1) // boundary * boundary


@dataclass(frozen=True)
class PageField:
    offset: int
    shape: tuple
    dtype: torch.dtype

    @property
    def bytes(self):
        return prod(self.shape) * self.dtype.itemsize

    def view(self, pages):
        return pages[:, self.offset : self.offset + self.bytes].view(self.dtype).view(pages.shape[0], *self.shape)


class Qwen4ExpPageLayout:
    def __init__(self, config):
        self.base_bytes = LinearAttCacheConfig.get_cpu_cache_big_page_bytes(config)
        self.fields = {}
        offset = self.base_bytes

        def add(name, shape, dtype):
            nonlocal offset
            offset = align(offset, dtype.itemsize)
            field = PageField(offset, shape, dtype)
            self.fields[name] = field
            offset += field.bytes

        tokens = get_env_start_args().cpu_cache_token_page_size
        layers = config.get_full_att_kv_layer_num_with_draft_model()
        for name in ("qsa_raw_keys", "qsa_pooled_keys"):
            add(name, (tokens, layers, config.indexer_head_dim), config.full_att_dtype)
        for name in ("qsa_key_cos", "qsa_key_sin"):
            add(name, (tokens, 1, config.rotary_dim), config.full_att_dtype)
        for layer in config.lexical_layers:
            add(f"lexical.{layer}.ngram", (config.ngram_context,), torch.int64)
            add(
                f"lexical.{layer}.convolution", (config.lexical_context, config.residual_width), config.conv_state_dtype
            )
        self.page_bytes = align(offset, 16)

    def views(self, tensor):
        if tensor.dtype != torch.uint8 or not tensor.is_pinned():
            raise ValueError("Qwen CPU pages require pinned byte storage")
        pages = tensor.view(tensor.shape[0], -1)
        if pages.shape[1] != self.page_bytes:
            raise ValueError("Qwen CPU page size does not match its state layout")
        return {name: field.view(pages) for name, field in self.fields.items()}
