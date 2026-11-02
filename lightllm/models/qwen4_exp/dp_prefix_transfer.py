from dataclasses import dataclass

import torch
import torch.distributed as dist

from .dp_transfer import _plan, copy_requests
from .recurrent_state import accepted_state


class _NodeCollective:
    def __init__(self, group, device):
        self.group, self.device = group, device
        self.rank = dist.get_rank(group)
        self.members = dist.get_process_group_ranks(group)

    def gather(self, value):
        values = [None] * len(self.members)
        with torch.cuda.device(self.device):
            dist.all_gather_object(values, value, group=self.group)
        return values

    def broadcast(self, source, value):
        values = [value]
        dist.broadcast_object_list(values, src=self.members[source], group=self.group, device=self.device)
        return values[0]


@dataclass
class _Reservation:
    request: object
    boundary: int
    pool: object
    index: int
    endpoint: bool
    small: bool


def _pool_fields(pool, index):
    conv, ssm = pool.get_state_cache(index)
    result = {"conv": conv, "ssm": ssm}
    extension = pool.extra_state_cache
    if extension is not None:
        for layer, history in extension.histories.items():
            result[f"lexical.{layer}.ngram"] = history.ngram[index]
            result[f"lexical.{layer}.convolution"] = history.convolution[index]
    return result


def _validate_pool(owner, pool, index):
    fields = _pool_fields(pool, index)
    conv, ssm = owner.req_to_conv_state.buffer, owner.req_to_ssm_state.buffer
    expected = {
        "conv": (conv.shape[:1] + conv.shape[2:-1] + (owner.linear_config.conv_kernel_size - 1,), conv.dtype),
        "ssm": (ssm.shape[:1] + ssm.shape[2:], ssm.dtype),
    }
    for layer, state in owner.req_to_lexical_state.items():
        for name in ("ngram", "convolution"):
            tensor = getattr(state, name)
            expected[f"lexical.{layer}.{name}"] = (tensor.shape[1:], tensor.dtype)
    if fields.keys() != expected.keys() or any(
        tuple(fields[name].shape) != tuple(shape) or fields[name].dtype != dtype
        for name, (shape, dtype) in expected.items()
    ):
        raise ValueError("Cross-DP prefix pool differs from accepted request geometry")


def _pack(fields):
    return {name: value.detach().cpu().clone(memory_format=torch.contiguous_format) for name, value in fields.items()}


def _live_fields(owner, request):
    conv, ssm = accepted_state(owner, request)
    result = {"conv": conv, "ssm": ssm}
    for layer, state in owner.req_to_lexical_state.items():
        result[f"lexical.{layer}.ngram"] = state.ngram[request]
        result[f"lexical.{layer}.convolution"] = state.convolution[request]
    return result


def _store(pool, index, fields):
    targets = _pool_fields(pool, index)
    if targets.keys() != fields.keys() or any(
        targets[name].shape != value.shape or targets[name].dtype != value.dtype for name, value in fields.items()
    ):
        raise ValueError("Cross-DP prefix snapshot geometry differs")

    for name, value in fields.items():
        targets[name].copy_(value)


def _source_index(request, boundary):
    index = request.linear_att_len_to_big_page_id.get(boundary)
    if index is not None:
        return index
    node = request.shared_kv_node
    while node is not None and node.node_prefix_total_len > boundary:
        node = node.parent
    if node is None or node.node_prefix_total_len != boundary or not node.is_big_page_node():
        raise ValueError("Cross-DP source lacks an intermediate prefix snapshot")
    if node.big_page_buffer_idx is None:
        raise ValueError("Cross-DP source prefix snapshot has no owner")
    return node.big_page_buffer_idx


def _release(reservations):
    for reservation in reservations:
        reservation.pool.free_state_cache([reservation.index])


def _reserve(owner, context, tasks, rank, reservations):
    cache = context.radix_cache
    big_tokens = context.args.linear_att_hash_page_size * context.args.linear_att_page_block_num
    descriptors = []
    for task in tasks:
        request, end = task.req, task.source_kv_len
        shared = 0 if request.shared_kv_node is None else request.shared_kv_node.node_prefix_total_len
        limit = min(end, request.linear_att_cache_len)
        boundaries = list(range((shared // big_tokens + 1) * big_tokens, limit + 1, big_tokens))
        pending = []
        for boundary in boundaries:
            if boundary not in request.linear_att_len_to_big_page_id:
                pending.append((boundary, cache.linear_att_big_page_buffers, False))
        if (
            end == request.linear_att_cache_len
            and end % big_tokens
            and request.tail_linear_att_small_page_buffer_id is None
        ):
            cache.free_one_small_page_linear_att_buffer()
            pending.append((end, cache.linear_att_small_page_buffers, True))
        intermediate = []
        for boundary, pool, small in pending:
            index = pool.alloc_one_state_cache()
            if index is None:
                raise RuntimeError("No native prefix snapshot slot for cross-DP transfer")
            reservation = _Reservation(request, boundary, pool, index, boundary == end, small)
            reservations.append(reservation)
            _validate_pool(owner, pool, index)
            if not reservation.endpoint:
                intermediate.append(boundary)
        descriptors.append(
            dict(
                source=task.max_kv_len_mem_manager_index,
                source_request=task.source_req_idx,
                destination=rank,
                request_id=request.req_id,
                end=end,
                boundaries=intermediate,
            )
        )
    return descriptors


def _check_errors(collective, error, reservations):
    errors = collective.gather(error)
    if any(errors):
        _release(reservations)
        raise RuntimeError("Cross-DP prefix transfer failed: " + "; ".join(value for value in errors if value))


def copy_with_prefix_cache(owner, tasks, mem_managers, dp_size_in_node, rank_in_dp, context, group, *, collective=None):
    collective = collective if collective is not None else _NodeCollective(group, owner.kv_buffer.device)
    reservations, local, error = [], [], None
    try:
        _plan(owner, tasks, mem_managers, dp_size_in_node, rank_in_dp)
        local = _reserve(owner, context, tasks, collective.rank, reservations)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    _check_errors(collective, error, reservations)
    gathered = collective.gather(local)
    descriptors = [descriptor for values in gathered for descriptor in values]
    requests = {request.req_idx: request for request in context.requests_mapping.values()}
    sources = {}
    error = None
    try:
        for descriptor in descriptors:
            if descriptor["source"] != collective.rank:
                continue
            request = requests[descriptor["source_request"]]
            if request.req_id != descriptor["request_id"] or request.cur_kv_len != descriptor["end"]:
                raise ValueError("Cross-DP source request identity or accepted boundary changed")
            for boundary in descriptor["boundaries"]:
                index = _source_index(request, boundary)
                pool = context.radix_cache.linear_att_big_page_buffers
                _validate_pool(owner, pool, index)
                sources[(descriptor["destination"], descriptor["request_id"], boundary)] = index
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    _check_errors(collective, error, reservations)
    incoming = {(item.request.req_id, item.boundary): item for item in reservations if not item.endpoint}
    error = None
    for descriptor in descriptors:
        for boundary in descriptor["boundaries"]:
            payload = None
            if collective.rank == descriptor["source"]:
                try:
                    key = (descriptor["destination"], descriptor["request_id"], boundary)
                    payload = dict(
                        fields=_pack(_pool_fields(context.radix_cache.linear_att_big_page_buffers, sources[key]))
                    )
                except Exception as failure:
                    payload = dict(error=f"{type(failure).__name__}: {failure}")
            payload = collective.broadcast(descriptor["source"], payload)
            if payload.get("error"):
                error = payload["error"]
            elif collective.rank == descriptor["destination"] and error is None:
                try:
                    reservation = incoming[(descriptor["request_id"], boundary)]
                    _store(reservation.pool, reservation.index, payload["fields"])
                except Exception as failure:
                    error = f"{type(failure).__name__}: {failure}"
    _check_errors(collective, error, reservations)
    error = None
    try:
        copy_requests(owner, tasks, mem_managers, dp_size_in_node, rank_in_dp)
        for reservation in reservations:
            if reservation.endpoint:
                fields = _pack(_live_fields(owner, reservation.request.req_idx))
                _store(reservation.pool, reservation.index, fields)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    _check_errors(collective, error, reservations)
    for reservation in reservations:
        if reservation.small:
            reservation.request.tail_linear_att_small_page_buffer_id = reservation.index
        else:
            reservation.request.linear_att_len_to_big_page_id[reservation.boundary] = reservation.index
