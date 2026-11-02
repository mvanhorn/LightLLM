from types import SimpleNamespace

import pytest
import torch

from lightllm.models.qwen4_exp.req_manager import Qwen4ExpReqManager
from lightllm.models.qwen4_exp.speculative_cache import LexicalSpeculativeCache
from lightllm.server.router.model_infer.mtp_speculative.utils import verify_mtp_tokens
from .lexical_helpers import make_cache, metadata, reference_segment
from .reference import reference_step
from .runtime import state_for
from .prefix import prefix_context, prefix_request
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.models.qwen4_exp.layer_infer.transformer_layer_infer import Qwen4ExpTransformerLayerInfer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("capture", [False, True])
def test_actual_verifier_commits_only_accepted_lexical_candidates(config, capture):
    torch.manual_seed(1907)
    cache, source = make_cache(config)
    cache.ngram[2].copy_(torch.tensor([7, 11], device="cuda"))
    cache.ngram[5].copy_(torch.tensor([13, 17], device="cuda"))
    cache.convolution.normal_()
    untouched = cache.snapshot(7)
    speculative = LexicalSpeculativeCache(cache, 4)
    req = Qwen4ExpReqManager.__new__(Qwen4ExpReqManager)
    req.speculative_lexical_caches = {1: speculative}
    req.req_sampling_params_manager = SimpleNamespace(
        req_to_next_token_ids=torch.full((8, 4), -1, device="cuda", dtype=torch.int64)
    )
    req.req_to_mtp_state_index = torch.zeros(8, device="cuda", dtype=torch.int32)
    req.HOLD_REQUEST_ID = 7
    backend = SimpleNamespace(model=SimpleNamespace(req_manager=req), is_linear_att_mixed_model=True, max_draft_step=3)
    meta = metadata([5, 5, 5, 2, 2, 7], [11, 12, 13, 11, 12, 1], [1] * 6)
    tokens = torch.empty(6, device="cuda", dtype=torch.int64)
    residual = torch.empty(6, 64, device="cuda", dtype=torch.bfloat16)

    def run():
        batch = cache.prepare(meta)
        result = batch.apply(residual, tokens, cache.layer_weight, 1, None)
        speculative.stage(meta, batch)
        return result

    tokens.fill_(19)
    residual.normal_()
    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run()

    cases = (
        ([5, 5, 5, 2, 2, 7], [2, 1], [19, 63, 23, 29, 31, 37]),
        ([2, 2, 5, 5, 5, 7], [2, 3], [41, 43, 47, 63, 5, 7]),
        ([5, 5, 5, 5, 2, 7], [1, 1], [11, 13, 17, 19, 23, 29]),
    )
    for requests, accepts, ids in cases:
        starts = [0] + [i for i in range(1, len(requests)) if requests[i] != requests[i - 1]]
        counts = [end - start for start, end in zip(starts, starts[1:] + [len(requests)])]
        lengths = []
        for count in counts:
            lengths.extend(range(11, 11 + count))
        meta.copy_from(metadata(requests, lengths, [1] * 6))
        tokens.copy_(torch.tensor(ids, device="cuda"))
        residual.normal_()
        before_ngram, before_conv = cache.ngram.clone(), cache.convolution.clone()
        if capture:
            graph.replay()
            actual = result
        else:
            actual = run()
        torch.testing.assert_close(cache.ngram, before_ngram, atol=0, rtol=0)
        torch.testing.assert_close(cache.convolution, before_conv, atol=0, rtol=0)

        next_ids = torch.arange(101, 107, device="cuda", dtype=torch.int64)
        proposals = req.req_sampling_params_manager.req_to_next_token_ids
        proposals.fill_(-1)
        expected_mask = []
        for start, count, accepted in zip(starts, counts, accepts + [1]):
            request = requests[start]
            proposals[request, 1:accepted] = next_ids[start : start + accepted - 1]
            expected_mask.extend([1] * accepted + [0] * (count - accepted))
        accept_lengths, accepted_index = verify_mtp_tokens(
            backend,
            next_ids,
            torch.tensor(requests, device="cuda", dtype=torch.int32),
            torch.tensor(starts, device="cuda", dtype=torch.int32),
            torch.tensor(sum((list(range(count)) for count in counts), []), device="cuda", dtype=torch.int32),
        )
        assert accept_lengths.tolist() == accepts + [1]
        assert accepted_index.tolist() == expected_mask
        for start, count, accepted in zip(starts[:2], counts[:2], accepts):
            request = requests[start]
            expected, states = reference_segment(
                residual[start : start + count],
                tokens[start : start + count],
                source,
                config,
                before_ngram[request : request + 1],
                before_conv[request : request + 1],
            )
            torch.testing.assert_close(actual[start : start + count], expected, atol=0.016, rtol=0.016)
            ngram, convolution = states[accepted]
            torch.testing.assert_close(cache.ngram[request], ngram[0], atol=0, rtol=0)
            torch.testing.assert_close(cache.convolution[request], convolution[0], atol=0.016, rtol=0.016)
            assert req.req_to_mtp_state_index[request] == accepted - 1
        torch.testing.assert_close(cache.ngram[7], untouched.ngram, atol=0, rtol=0)
        torch.testing.assert_close(cache.convolution[7], untouched.convolution, atol=0, rtol=0)
    if capture:
        graph.reset()


def test_microbatch_candidates_survive_reverse_verification(config):
    cache, source = make_cache(config)
    speculative = LexicalSpeculativeCache(cache, 3)
    batches = []
    for request, ids in ((5, [19, 63, 23]), (2, [29, 31, 37])):
        meta = metadata([request] * 3, [11, 12, 13], [1] * 3)
        tokens = torch.tensor(ids, device="cuda")
        residual = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
        batch = cache.prepare(meta)
        batch.apply(residual, tokens, cache.layer_weight, 1, None)
        speculative.stage(meta, batch)
        batches.append((request, residual, tokens))
    for request, residual, tokens in reversed(batches):
        _, states = reference_segment(
            residual,
            tokens,
            source,
            config,
            cache.ngram[request : request + 1].clone(),
            cache.convolution[request : request + 1].clone(),
        )
        speculative.commit(
            torch.tensor([request] * 3, device="cuda", dtype=torch.int32),
            torch.tensor([0], device="cuda", dtype=torch.int32),
            torch.tensor([2], device="cuda", dtype=torch.int32),
            7,
        )
        torch.testing.assert_close(cache.ngram[request], states[2][0][0], atol=0, rtol=0)
        torch.testing.assert_close(cache.convolution[request], states[2][1][0], atol=0.016, rtol=0.016)


@pytest.mark.parametrize(
    "runtime", [{"mtp_step": 2, "mtp_mode": "eagle_with_att", "mtp_dynamic_verify": True}], indirect=True
)
@pytest.mark.parametrize("transfer", [None, "prefix", "pd"])
@pytest.mark.parametrize("accepted_offset", [1, 2])
def test_native_target_decoder_resumes_after_speculative_rejection(runtime, config, transfer, accepted_offset):
    Qwen3NextTpPartModel._init_triton(None)
    runtime.model.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda is_draft: 2)
    req, weight = runtime.req, runtime.weight
    infer = Qwen4ExpTransformerLayerInfer(1, config)
    backend = SimpleNamespace(model=runtime.model, is_linear_att_mixed_model=True, max_draft_step=2)
    references = {}
    for request in (1, 3):
        req.init_linear_att_state(SimpleNamespace(req_idx=request))
        references[request] = (
            weight.ngram_hasher.component.empty_history(1),
            weight.ple.component.empty_history(1),
            torch.zeros(1, 512, 3, device="cuda", dtype=torch.bfloat16),
            torch.zeros(1, 2, 128, 128, device="cuda", dtype=torch.float32),
        )
    cases = (
        ([1, 3], [2, 2], [7, 11, 13, 17], [0, 2], [2, 2], [0, 0]),
        ([1, 1, 1, 3, 3, 3], [3, 4, 5, 3, 4, 5], [19, 63, 23, 29, 31, 37], [0, 3], [accepted_offset + 1, 1], None),
        ([3, 3, 1], [4, 5, 4 + accepted_offset], [41, 43, 47], [0, 2], [2, 1], None),
    )
    for requests, lengths, ids, starts, accepted, ready in cases:
        counts = [end - start for start, end in zip(starts, starts[1:] + [len(ids)])]
        mtp_indices = sum((list(range(count)) for count in counts), []) if ready is None else None
        state = state_for(runtime, requests, lengths, ids, ready=ready, mtp_indices=mtp_indices)
        residual = torch.randn(len(ids), 64, device="cuda", dtype=torch.bfloat16) * 0.05
        before_ngram = req.lexical_caches[1].ngram.clone()
        before_conv = req.lexical_caches[1].convolution.clone()
        actual = (infer.context_forward if ready is not None else infer.token_forward)(residual, state, weight)
        if ready is None:
            torch.testing.assert_close(req.lexical_caches[1].ngram, before_ngram, atol=0, rtol=0)
            torch.testing.assert_close(req.lexical_caches[1].convolution, before_conv, atol=0, rtol=0)
        for start, count, accept in zip(starts, counts, accepted):
            request = requests[start] if ready is None else requests[starts.index(start)]
            saved = [references[request]]
            expected = []
            for row in range(start, start + count):
                result = reference_step(
                    residual[row : row + 1], state.input_ids[row : row + 1], runtime.source, config, *saved[-1]
                )
                expected.append(result[0])
                saved.append(result[1:])
            torch.testing.assert_close(actual[start : start + count], torch.cat(expected), atol=0.016, rtol=0.016)
            references[request] = saved[accept]
        if ready is None:
            proposals = req.req_sampling_params_manager.req_to_next_token_ids
            proposals.fill_(-1)
            next_ids = torch.arange(101, 101 + len(ids), device="cuda", dtype=torch.int64)
            for start, accept in zip(starts, accepted):
                proposals[requests[start], 1:accept] = next_ids[start : start + accept - 1]
            actual_accepted, _ = verify_mtp_tokens(
                backend,
                next_ids,
                state.b_req_idx,
                torch.tensor(starts, device="cuda", dtype=torch.int32),
                state.b_mtp_index,
            )
            assert actual_accepted.tolist() == accepted
            if transfer is not None and len(ids) == 6:
                assert req.req_to_mtp_state_index[1] == accepted_offset
                expected_conv = req.req_to_conv_state.buffer[:, 1, ..., accepted_offset : accepted_offset + 3].clone()
                expected_ssm = req.req_to_ssm_state.buffer[:, 3 + accepted_offset].clone()
                if transfer == "prefix":
                    context = prefix_context(runtime)
                    saved = prefix_request(1, 3 + accepted_offset)
                    context.copy_linear_att_state_to_cache_buffer(
                        torch.tensor([1], device="cuda", dtype=torch.int32), [saved]
                    )
                else:
                    runtime.mem.alloc_paged_kv_move_buffer(1, 4)
                    runtime.mem.write_mem_to_page_kv_move_buffer([], 0, 0, [runtime.mem], 1, "linear_att_state", 1)
                req.init_linear_att_state(SimpleNamespace(req_idx=1))
                if transfer == "prefix":
                    if saved.linear_att_len_to_big_page_id:
                        page = saved.linear_att_len_to_big_page_id[3 + accepted_offset]
                        req.copy_big_page_buffer_to_linear_att_state(page, SimpleNamespace(req_idx=1))
                    else:
                        req.copy_small_page_buffer_to_linear_att_state(
                            SimpleNamespace(
                                req_idx=1,
                                shared_kv_node=SimpleNamespace(
                                    small_page_buffer_idx=saved.tail_linear_att_small_page_buffer_id
                                ),
                            ),
                            context.radix_cache.linear_att_small_page_buffers,
                        )
                else:
                    runtime.mem.read_page_kv_move_buffer_to_mem([], 0, 0, [runtime.mem], 1, "linear_att_state", 1)
                torch.testing.assert_close(req.req_to_conv_state.buffer[:, 1, ..., :3], expected_conv, atol=0, rtol=0)
                torch.testing.assert_close(req.req_to_ssm_state.buffer[:, 3], expected_ssm, atol=0, rtol=0)
                assert req.req_to_mtp_state_index[1] == 0
        for request, (ngram, conv, _, _) in references.items():
            torch.testing.assert_close(req.lexical_caches[1].ngram[request], ngram[0], atol=0, rtol=0)
            torch.testing.assert_close(req.lexical_caches[1].convolution[request], conv[0], atol=0.016, rtol=0.016)
