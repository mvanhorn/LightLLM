import json
from pathlib import Path

from lightllm.models.qwen3_5_moe.model import Qwen3_5MOETpPartModel
from lightllm.models.registry import ModelRegistry
from lightllm.common.basemodel.attention.triton.fp import TritonAttBackend
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num
from lightllm.utils.torch_dtype_utils import get_torch_dtype
from .configuration import linear_cache_config, normalize_model_config
from .infer_struct import Qwen4ExpInferStateInfo
from .mem_manager import Qwen4ExpMemManager
from .req_manager import Qwen4ExpReqManager
from .layer_weights.pre_and_post_layer_weight import Qwen4ExpPreAndPostLayerWeight
from .layer_weights.transformer_layer_weight import Qwen4ExpTransformerLayerWeight
from .layer_infer.pre_layer_infer import Qwen4ExpPreLayerInfer
from .layer_infer.post_layer_infer import Qwen4ExpPostLayerInfer
from .layer_infer.transformer_layer_infer import Qwen4ExpTransformerLayerInfer


@ModelRegistry("qwen4_exp", is_multimodal=True)
@ModelRegistry("qwen4_exp_text")
class Qwen4ExpTpPartModel(Qwen3_5MOETpPartModel):
    linear_att_mtp_match_decode = True
    pre_and_post_weight_class = Qwen4ExpPreAndPostLayerWeight
    transformer_weight_class = Qwen4ExpTransformerLayerWeight
    pre_layer_infer_class = Qwen4ExpPreLayerInfer
    post_layer_infer_class = Qwen4ExpPostLayerInfer
    transformer_layer_infer_class = Qwen4ExpTransformerLayerInfer
    infer_state_class = Qwen4ExpInferStateInfo

    def _init_config(self):
        with (Path(self.weight_dir_) / "config.json").open() as stream:
            checkpoint = json.load(stream)
        self.config, self.vision_config = normalize_model_config(checkpoint, self.finetune_config)
        self.num_kv_heads = max(self.config["num_key_value_heads"] // self.tp_world_size_, 1)

    def _init_req_manager(self):
        self.linear_config = linear_cache_config(
            self.config,
            self.tp_world_size_,
            self.data_type,
            get_torch_dtype(self.args.linear_att_ssm_data_type),
            draft_layers=get_added_mtp_kv_layer_num(),
        )
        capacity = max(self.batch_max_tokens or 0, self.max_seq_length or 0)
        self.req_manager = Qwen4ExpReqManager(
            self.max_req_num, capacity, None, self.linear_config, self.trans_layers_weight
        )

    def _init_mem_manager(self):
        self.mem_manager = Qwen4ExpMemManager(
            size=self.max_total_token_num,
            dtype=self.data_type,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.config["head_dim"],
            full_att_layer_num=self.linear_config.get_full_att_kv_layer_num_with_draft_model(),
            linear_config=self.linear_config,
            indexer_head_dim=self.config["indexer_head_dim"],
            rotary_dim=int(self.config["head_dim"] * self.config["partial_rotary_factor"]),
            mem_fraction=self.mem_fraction,
        )

        self.mem_manager.attach_request_state(self.req_manager)

    def _init_att_backend(self):
        self.prefill_att_backend = self.decode_att_backend = TritonAttBackend(model=self)
