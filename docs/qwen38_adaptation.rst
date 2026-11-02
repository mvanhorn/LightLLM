Qwen3.8-Flash-Next
==================

Qwen3.8-Flash-Next checkpoints use the ``qwen4_exp`` model type. The model
implementation combines gated residuals, n-gram embeddings, QSA and GDN
attention, with dedicated multimodal, MTP and cache-state integration.
Reusable operations live in ``lightllm/common/gated_residual``,
``lightllm/common/ngram_embedding`` and ``lightllm/common/qsa``.

CPU embedding offload is disabled by default. The optional QSA TopK selector
is enabled by the model only for its qualified H100/Torch/Triton geometry;
other configurations use Torch. Shared-prefix admission is limited to the
model's single-node, DP1 normal serving mode with one-page state blocks.

Validation
----------

Historical H100 measurements and their configuration are recorded in
``qwen38_h100_performance.rst``. They cover TP8 BF16, FP32 SSM states,
fixed MTP2 and one-page linear-attention state blocks. These results do not
qualify PD/cross-DP serving or other cache block configurations.

The four-page-block pressure-recovery configuration has unresolved text
differences. Low-load latency still trails SGLang in some metrics. Archived
HC and sampling experiments are not part of the runtime implementation.
