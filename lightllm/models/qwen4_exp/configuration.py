from dataclasses import dataclass

import torch

from .checkpoint import ngram_eos_token_id
from lightllm.common.build_utils import repair_config

from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig


def normalize_model_config(checkpoint, finetune_config=None):
    config = dict(checkpoint.get("text_config", checkpoint))
    for aliases in (
        ["num_attention_heads", "n_head"],
        ["hidden_size", "n_embd", "n_embed"],
        ["num_hidden_layers", "n_layer"],
    ):
        repair_config(config, same_names=aliases)
    rope = config.get("rope_parameters")
    if isinstance(rope, dict):
        for name in ("rope_theta", "partial_rotary_factor"):
            if name in rope:
                config.setdefault(name, rope[name])
        config.setdefault("rope_scaling", dict(rope))
    for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        if name in checkpoint:
            config[name] = checkpoint[name]
    config.setdefault("norm_topk_prob", True)
    if finetune_config is not None:
        config["vocab_size"] = finetune_config.vocab_size
    return config, checkpoint.get("vision_config")


@dataclass(kw_only=True)
class Qwen4ExpCacheConfig(LinearAttCacheConfig):
    lexical_layers: tuple
    ngram_context: int
    lexical_context: int
    residual_width: int
    eos_token_id: int
    indexer_head_dim: int
    rotary_dim: int

    def get_cpu_state_allocator(self):
        from .pinned_memory import pinned_zeros

        return pinned_zeros

    def create_extra_state_cache(self, size):
        from .prefix_cache import LexicalPrefixCache

        return LexicalPrefixCache(size, self)

    def get_cpu_cache_extra_bytes(self):
        from .cache_layout import Qwen4ExpPageLayout

        layout = Qwen4ExpPageLayout(self)
        return layout.page_bytes - layout.base_bytes

    def get_cpu_cache_big_page_bytes(self):
        from .cache_layout import Qwen4ExpPageLayout

        return Qwen4ExpPageLayout(self).page_bytes


def linear_cache_config(config: dict, tp_world_size: int, dtype: torch.dtype, ssm_dtype: torch.dtype, draft_layers=0):
    interval = config["full_attention_interval"]
    layers = config.get("n_layer") or config["num_hidden_layers"]
    expected = ["full_attention" if (layer + 1) % interval == 0 else "linear_attention" for layer in range(layers)]
    actual = ["full_attention" if kind == "indexed_attention" else kind for kind in config.get("layer_types", expected)]
    if actual != expected:
        raise ValueError("Qwen4Exp native GDN cache requires the configured periodic layer layout")
    return Qwen4ExpCacheConfig(
        lexical_layers=tuple(sorted({layer - 1 for layer in config.get("ple_layer_ids", [])})),
        ngram_context=config["ngram_size"] - 1,
        lexical_context=(config["ple_conv_kernel_size"] - 1) * config["ngram_size"],
        residual_width=config["hc_count"] * config["hidden_size"],
        eos_token_id=ngram_eos_token_id(config),
        indexer_head_dim=config["indexer_head_dim"],
        rotary_dim=int(config["head_dim"] * config["partial_rotary_factor"]),
        tp_world_size=tp_world_size,
        full_att_all_num_kv_heads=config["num_key_value_heads"],
        full_att_dtype=dtype,
        full_att_num_kv_heads=max(1, config["num_key_value_heads"] // tp_world_size),
        full_att_head_dim=config["head_dim"],
        global_linear_k_heads=config["linear_num_key_heads"],
        global_linear_v_heads=config["linear_num_value_heads"],
        num_linear_k_heads=max(1, config["linear_num_key_heads"] // tp_world_size),
        num_linear_v_heads=max(1, config["linear_num_value_heads"] // tp_world_size),
        head_linear_k_dim=config["linear_key_head_dim"],
        head_linear_v_dim=config["linear_value_head_dim"],
        conv_kernel_size=config["linear_conv_kernel_dim"],
        linear_layer_num=layers - layers // interval,
        conv_state_dtype=dtype,
        ssm_state_dtype=ssm_dtype,
        full_attention_interval=interval,
        all_layer_num=layers,
        draft_full_att_kv_layer_num=draft_layers,
    )
