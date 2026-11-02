import torch
import triton
import triton.language as tl

from .lexical_cache import LexicalSnapshot


@triton.jit
def _copy_rows(Source, Target, SourceRows, TargetRows, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    target = tl.load(TargetRows + row).to(tl.int64)
    if target >= 0:
        source = tl.load(SourceRows + row).to(tl.int64)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(Source + source * WIDTH + col, col < WIDTH, other=0)
        tl.store(Target + target * WIDTH + col, value, col < WIDTH)


class LexicalPrefixCache:
    def __init__(self, size, config):
        self.config = config
        self.histories = {
            layer: LexicalSnapshot(
                torch.empty((size, config.ngram_context), dtype=torch.int64, device="cpu", pin_memory=True),
                torch.empty(
                    (size, config.lexical_context, config.residual_width),
                    dtype=config.conv_state_dtype,
                    device="cpu",
                    pin_memory=True,
                ),
            )
            for layer in config.lexical_layers
        }
        self.clear_to_init_state()

    def clear_to_init_state(self):
        for history in self.histories.values():
            history.ngram.fill_(self.config.eos_token_id)
            history.convolution.zero_()

    def validate_caches(self, caches):
        if set(caches) != set(self.histories):
            raise ValueError("Lexical prefix cache layers do not match the request cache")
        for layer, history in self.histories.items():
            cache = caches[layer]
            for source, target in ((cache.ngram, history.ngram), (cache.convolution, history.convolution)):
                if source.shape[1:] != target.shape[1:] or source.dtype != target.dtype:
                    raise ValueError("Lexical prefix cache geometry does not match request state")

    def copy_from_requests(self, caches, requests, destinations):
        self.validate_caches(caches)
        if requests.shape != destinations.shape or requests.ndim != 1:
            raise ValueError("Prefix snapshot requests and destinations must be matching vectors")
        for layer, history in self.histories.items():
            cache = caches[layer]
            for source, target in ((cache.ngram, history.ngram), (cache.convolution, history.convolution)):
                width = source[0].numel()
                if requests.numel() and width:
                    _copy_rows[(requests.numel(), triton.cdiv(width, 256))](
                        source, target, requests, destinations, width, 256
                    )

    def restore_request(self, caches, request, source):
        self.validate_caches(caches)
        for layer, cache in caches.items():
            history = self.histories[layer]
            cache.restore(request, LexicalSnapshot(history.ngram[source], history.convolution[source]))
