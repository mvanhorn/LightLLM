from dataclasses import dataclass

import torch

from .recurrent_state import accepted_state, state_span


@dataclass(frozen=True)
class _RequestTransfer:
    source: object
    source_request: int
    destination_request: int
    source_slots: torch.Tensor
    destination_slots: torch.Tensor


def _indices(value, capacity):
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
        raise ValueError("Cross-DP slots must be integer vectors")
    result = value.detach().to(device="cpu", dtype=torch.int64, copy=True)
    if not result.numel() or bool(((result < 0) | (result >= capacity)).any()):
        raise ValueError("Cross-DP slots must be nonempty and within their owner")
    return result


def _same_rows(source, destination, axis):
    if source.ndim != destination.ndim or source.dtype != destination.dtype:
        return False
    return all(left == right for dim, (left, right) in enumerate(zip(source.shape, destination.shape)) if dim != axis)


def _validate_owner(owner):
    conv, ssm = owner.req_to_conv_state.buffer, owner.req_to_ssm_state.buffer
    if (
        conv.ndim < 3
        or ssm.ndim < 3
        or conv.shape[1] < 2
        or ssm.shape[1] % conv.shape[1]
        or conv.shape[0] != ssm.shape[0]
        or owner.req_to_token_indexs.ndim != 2
        or owner.req_to_token_indexs.shape[0] != conv.shape[1]
        or owner.req_to_token_indexs.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Invalid cross-DP recurrent request layout")
    context = owner.linear_config.conv_kernel_size - 1
    if context < 0 or context > conv.shape[-1] or state_span(owner) < 1:
        raise ValueError("Invalid cross-DP accepted recurrent window")
    slot_buffers = owner._slot_buffers()
    if any(tensor.shape[axis] != owner.kv_buffer.shape[1] for tensor, axis in slot_buffers.values()):
        raise ValueError("Cross-DP token cache fields have different physical slot capacities")
    buffers = [tensor for tensor, _ in slot_buffers.values()]
    buffers += [conv, ssm, owner.req_to_token_indexs]
    for snapshot in owner.req_to_lexical_state.values():
        buffers.extend((snapshot.ngram, snapshot.convolution))
        if snapshot.ngram.shape[0] != conv.shape[1] or snapshot.convolution.shape[0] != conv.shape[1]:
            raise ValueError("Lexical and recurrent request counts differ")
    if any(tensor.device != owner.kv_buffer.device or not tensor.is_contiguous() for tensor in buffers):
        raise ValueError("Cross-DP owner buffers must be contiguous on the owner's device")
    indices = getattr(owner, "req_to_mtp_state_index", None)
    if indices is not None and (
        indices.shape != (conv.shape[1],)
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.device != conv.device
    ):
        raise ValueError("Invalid cross-DP accepted-state indices")


def _validate_pair(source, destination):
    _validate_owner(source)
    source_buffers, target_buffers = source._slot_buffers(), destination._slot_buffers()
    if source_buffers.keys() != target_buffers.keys():
        raise ValueError("Cross-DP token cache fields differ")
    for name, (target, axis) in target_buffers.items():
        tensor, source_axis = source_buffers[name]
        if axis != source_axis or not _same_rows(tensor, target, axis):
            raise ValueError(f"Cross-DP token cache geometry differs: {name}")
    source_conv, target_conv = source.req_to_conv_state.buffer, destination.req_to_conv_state.buffer
    source_ssm, target_ssm = source.req_to_ssm_state.buffer, destination.req_to_ssm_state.buffer
    if (
        source.linear_config.conv_kernel_size != destination.linear_config.conv_kernel_size
        or source_conv.dtype != target_conv.dtype
        or source_ssm.dtype != target_ssm.dtype
        or source_conv.shape[:1] + source_conv.shape[2:-1] != target_conv.shape[:1] + target_conv.shape[2:-1]
        or source_ssm.shape[:1] + source_ssm.shape[2:] != target_ssm.shape[:1] + target_ssm.shape[2:]
    ):
        raise ValueError("Cross-DP recurrent shard geometry differs")
    if source.req_to_lexical_state.keys() != destination.req_to_lexical_state.keys():
        raise ValueError("Cross-DP lexical layers differ")
    for layer, target in destination.req_to_lexical_state.items():
        snapshot = source.req_to_lexical_state[layer]
        if not _same_rows(snapshot.ngram, target.ngram, 0) or not _same_rows(
            snapshot.convolution, target.convolution, 0
        ):
            raise ValueError("Cross-DP lexical geometry differs")


def _plan(destination, tasks, mem_managers, dp_size_in_node, rank_in_dp):
    if not tasks:
        return []
    if type(dp_size_in_node) is not int or dp_size_in_node < 1 or len(mem_managers) % dp_size_in_node:
        raise ValueError("Cross-DP managers must contain complete equal-size TP groups")
    tp_size = len(mem_managers) // dp_size_in_node
    if type(rank_in_dp) is not int or not 0 <= rank_in_dp < tp_size:
        raise ValueError("Invalid cross-DP TP rank")
    _validate_owner(destination)
    plans, destination_slots, destination_requests, validated_sources = [], set(), set(), set()
    for task in tasks:
        source_dp = task.max_kv_len_dp_rank
        source_index = task.max_kv_len_mem_manager_index
        if not 0 <= source_dp < dp_size_in_node or source_index != source_dp * tp_size + rank_in_dp:
            raise ValueError("Cross-DP source does not match its TP partition")
        source = mem_managers[source_index]
        if source.kv_buffer.device == destination.kv_buffer.device and (
            source.kv_buffer.data_ptr() == destination.kv_buffer.data_ptr()
        ):
            raise ValueError("Cross-DP source and destination must have independent storage")
        if source_index not in validated_sources:
            _validate_pair(source, destination)
            validated_sources.add(source_index)
        source_request = task.source_req_idx
        target_request = task.req.req_idx
        if (
            type(source_request) is not int
            or not 0 <= source_request < source.req_to_conv_state.buffer.shape[1] - 1
            or type(target_request) is not int
            or not 0 <= target_request < destination.req_to_conv_state.buffer.shape[1] - 1
        ):
            raise ValueError("Cross-DP copies require live source and destination request slots")
        source_slots = _indices(task.max_kv_len_mem_indexes, getattr(source, "size", source.kv_buffer.shape[1]))
        target_slots = _indices(task.mem_indexes, getattr(destination, "size", destination.kv_buffer.shape[1]))
        boundary = task.source_kv_len
        prefix_start = task.req.cur_kv_len
        capacity = min(source.req_to_token_indexs.shape[1], destination.req_to_token_indexs.shape[1])
        if (
            type(boundary) is not int
            or source_slots.numel() != target_slots.numel()
            or boundary != prefix_start + target_slots.numel()
            or prefix_start < 0
            or boundary <= prefix_start
            or boundary > capacity
        ):
            raise ValueError("Cross-DP recurrent state must match the complete transferred prefix boundary")
        bound_slots = source.req_to_token_indexs[source_request, prefix_start:boundary].detach().cpu().long()
        if not torch.equal(bound_slots, source_slots):
            raise ValueError("Cross-DP token slots do not belong to the declared source prefix")
        indices = getattr(source, "req_to_mtp_state_index", None)
        offset = int(indices[source_request]) if indices is not None else 0
        if not 0 <= offset < state_span(source) or offset + source.linear_config.conv_kernel_size - 1 > (
            source.req_to_conv_state.buffer.shape[-1]
        ):
            raise ValueError("Cross-DP source has an invalid accepted-state offset")
        slots = target_slots.tolist()
        if (
            len(set(slots)) != len(slots)
            or destination_slots.intersection(slots)
            or target_request in destination_requests
        ):
            raise ValueError("Cross-DP destination slots and requests must be unique")
        destination_slots.update(slots)
        destination_requests.add(target_request)
        plans.append(_RequestTransfer(source, source_request, target_request, source_slots, target_slots))
    return plans


def copy_requests(destination, tasks, mem_managers, dp_size_in_node, rank_in_dp, *, token_chunk_size=1024):
    if type(token_chunk_size) is not int or token_chunk_size < 1:
        raise ValueError("Cross-DP token chunk size must be positive")
    plans = _plan(destination, tasks, mem_managers, dp_size_in_node, rank_in_dp)
    device = destination.kv_buffer.device
    for plan in plans:
        source = plan.source
        for start in range(0, plan.source_slots.numel(), token_chunk_size):
            end = start + token_chunk_size
            snapshot = source.get_index_kv_buffer(plan.source_slots[start:end])

            snapshot = {name: tensor.to(device, non_blocking=False) for name, tensor in snapshot.items()}
            destination.load_index_kv_buffer(plan.destination_slots[start:end], snapshot)
        conv, ssm = accepted_state(source, plan.source_request)
        context = destination.linear_config.conv_kernel_size - 1
        destination.req_to_conv_state.buffer[:, plan.destination_request, ..., :context].copy_(
            conv.to(device, non_blocking=False)
        )
        destination.req_to_ssm_state.buffer[:, plan.destination_request * state_span(destination)].copy_(
            ssm.to(device, non_blocking=False)
        )
        for layer, target in destination.req_to_lexical_state.items():
            snapshot = source.req_to_lexical_state[layer]
            target.ngram[plan.destination_request].copy_(snapshot.ngram[plan.source_request].to(device))
            target.convolution[plan.destination_request].copy_(snapshot.convolution[plan.source_request].to(device))
        indices = getattr(destination, "req_to_mtp_state_index", None)
        if indices is not None:
            indices[plan.destination_request] = 0
