import torch
import triton

from lightllm.common.kv_trans_kernel.nixl_kv_trans import page_io
from .cache_layout import PageField, align
from .triton_kernel.cache_pages import copy_slot_pages
from .recurrent_state import Qwen4ExpRecurrentPageHelper


class Qwen4ExpPDLayout:
    def __init__(self, config, page_size, gdn_bytes):
        self.tokens = page_size
        self.kv_fields, self.state_fields = {}, {}
        offset = 0

        def add(fields, name, shape, dtype):
            nonlocal offset
            offset = align(offset, dtype.itemsize)
            field = PageField(offset, shape, dtype)
            fields[name] = field
            offset += field.bytes

        add(
            self.kv_fields,
            "kv_buffer",
            (
                page_size,
                config.get_full_att_kv_layer_num_with_draft_model(),
                2 * config.full_att_all_num_kv_heads,
                config.full_att_head_dim,
            ),
            config.full_att_dtype,
        )
        for name in ("qsa_raw_keys", "qsa_pooled_keys"):
            add(
                self.kv_fields,
                name,
                (page_size, config.get_full_att_kv_layer_num_with_draft_model(), config.indexer_head_dim),
                config.full_att_dtype,
            )
        for name in ("qsa_key_cos", "qsa_key_sin"):
            add(self.kv_fields, name, (page_size, 1, config.rotary_dim), config.full_att_dtype)
        self.kv_bytes = offset

        offset = gdn_bytes
        for layer in config.lexical_layers:
            add(self.state_fields, f"lexical.{layer}.ngram", (config.ngram_context,), torch.int64)
            add(
                self.state_fields,
                f"lexical.{layer}.convolution",
                (config.lexical_context, config.residual_width),
                config.conv_state_dtype,
            )
        self.state_bytes = offset
        self.row_bytes = align(max(self.kv_bytes, self.state_bytes), page_size * 16) // page_size
        self.page_bytes = self.row_bytes * page_size

    def views(self, buffer, state=False):
        pages = buffer.view(buffer.shape[0], -1)
        if pages.shape[1] != self.page_bytes or buffer.dtype != torch.uint8:
            raise ValueError("Qwen PD buffer does not match the complete cache layout")
        fields = self.state_fields if state else self.kv_fields
        return {name: field.view(pages) for name, field in fields.items()}


class Qwen4ExpPDPageHelper:
    def __init__(self, mem_manager):
        self.mem = mem_manager
        self.gdn = Qwen4ExpRecurrentPageHelper(mem_manager)
        self.layout = mem_manager.pd_layout

    def _dp_mems(self, mem_managers, dp_index, dp_world_size):
        managers = self.gdn.get_dp_mems(mem_managers, dp_index, dp_world_size)
        for manager in managers:
            config = manager.linear_config
            if (
                config.full_att_all_num_kv_heads != self.mem.linear_config.full_att_all_num_kv_heads
                or config.full_att_head_dim != self.mem.linear_config.full_att_head_dim
                or config.full_att_dtype != self.mem.linear_config.full_att_dtype
                or config.indexer_head_dim != self.mem.linear_config.indexer_head_dim
                or config.rotary_dim != self.mem.linear_config.rotary_dim
                or set(manager.req_to_lexical_state) != set(self.mem.req_to_lexical_state)
            ):
                raise ValueError("Qwen PD ranks must share global KV, index and lexical geometry")
        return managers

    def transfer(self, mem_indexes, page_index, dp_index, mem_managers, dp_world_size, page_kind, req_idx, write):
        if not 0 <= page_index < self.mem.kv_move_buffer.shape[0]:
            raise ValueError("Qwen PD page index is out of range")
        managers = self._dp_mems(mem_managers, dp_index, dp_world_size)
        if page_kind == "linear_att_state":
            if req_idx is None or len(mem_indexes):
                raise ValueError("Qwen recurrent-state pages require a request slot and no token slots")
            lexical = self._lexical_pairs(page_index, req_idx, managers, write)
            if write:
                self.gdn.write_req_to_page(page_index, req_idx, managers)
            else:
                self.gdn.read_page_to_req(page_index, req_idx, managers)
            for live, cached in lexical:
                destination, source = (cached, live) if write else (live, cached)
                destination.copy_(source, non_blocking=True)
        elif page_kind == "kv":
            self._tokens(mem_indexes, page_index, managers, dp_world_size, write)
        else:
            raise ValueError(f"Unknown Qwen PD page kind: {page_kind}")

    def _tokens(self, mem_indexes, page_index, managers, dp_world_size, write):
        if len(mem_indexes) > self.layout.tokens:
            raise ValueError("Qwen PD token list exceeds page capacity")
        if not len(mem_indexes):
            return
        indices = self.mem._buffer_mem_indexes_tensors[page_index]
        indices.fill_(-1)
        indices.numpy()[: len(mem_indexes)] = mem_indexes
        indices = indices.cuda(non_blocking=True)
        views = self.layout.views(self.mem.kv_move_buffer)
        kv_page = views.pop("kv_buffer")[page_index]
        repeat = dp_world_size * managers[0].head_num // self.mem.linear_config.full_att_all_num_kv_heads
        for rank, manager in enumerate(managers):
            if not write or rank % repeat == 0:
                page_io(
                    indices[: len(mem_indexes)],
                    kv_page,
                    manager.kv_buffer,
                    rank,
                    dp_world_size,
                    mode="write" if write else "read",
                )

        page = torch.tensor([page_index], dtype=torch.int64, device=self.mem.kv_move_buffer.device)
        readies = torch.zeros(1, dtype=torch.bool, device=page.device) if write else None
        for manager in managers[:1] if write else managers:
            for name, (slots, axis) in manager._slot_buffers().items():
                if name == "kv_buffer":
                    continue
                cached = views[name]
                tokens, layers, dim = cached.shape[1:]
                if dim:
                    copy_slot_pages[(1, triton.cdiv(tokens * layers * dim, 256))](
                        slots,
                        cached,
                        indices,
                        page,
                        readies,
                        slots.stride(axis),
                        slots.stride(0) if axis == 1 else 0,
                        cached.stride(0),
                        tokens,
                        layers,
                        dim,
                        write,
                        256,
                    )

    def _lexical_pairs(self, page_index, request, managers, write):
        views = self.layout.views(self.mem.kv_move_buffer, state=True)
        pairs = []
        for manager in managers[:1] if write else managers:
            for layer, history in manager.req_to_lexical_state.items():
                for kind in ("ngram", "convolution"):
                    state = getattr(history, kind)
                    if not 0 <= request < state.shape[0]:
                        raise ValueError("Qwen PD request slot is out of range")
                    live = state[request]
                    cached = views[f"lexical.{layer}.{kind}"][page_index]
                    if live.shape != cached.shape or live.dtype != cached.dtype:
                        raise ValueError("Qwen PD lexical state does not match its page layout")
                    pairs.append((live, cached))
        return pairs
