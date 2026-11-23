# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SpaceByte model flavors.

Every block is a ``Qwen3TransformerBlock`` with Qwen3's init, norms and RoPE,
so a SpaceByte flavor differs from its Qwen3 counterpart only in the byte-level
hierarchy. See ``README.md`` for how the "0.6B" flavor is matched to Qwen3-0.6B.
"""

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.tokenizer import ByteTokenizer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import (
    Embedding,
    Linear,
    StaticShapeCosSinRoPE,
    TransformerBlock,
)
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    InnerAttention,
    SlidingWindowFlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.qwen3.model import Qwen3TransformerBlock

from .model import HierarchyLevel, SpaceByteModel, SpacelikeBoundary

__all__ = ["MODEL_FLAVORS", "SpaceByteModel", "build_model_config"]


# Same initialization, norm and RoPE as qwen3/flavors.py.
_LINEAR_INIT: dict[str, Callable] = {
    "weight": partial(nn.init.normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}
_EPS = 1e-6
_ROPE_THETA = 1000000.0

# On 65536-byte microbatches (scripts/spacebyte_data_stats.py):
#   C4 (en):     5.78 bytes per patch; 5.0 leaves 0.03% of bytes without a
#                global slot and 14% of slots as padding.
#   FineWeb-Edu: 6.07 bytes per patch; 0.003% of bytes, 18% padding.
_C4_CAPACITY_BYTES_PER_PATCH = 5.0
# nizarun English train, 38,464-byte microbatches (scripts/spacebyte_capacity.py,
# 374K microbatches): 6.067 bytes per patch; 5.5 leaves 0.004% of bytes without
# a global slot (0.16% of microbatches overflow) and 10% of slots as padding,
# vs 0.00015% and 19% at 5.0.
_NIZARUN_CAPACITY_BYTES_PER_PATCH = 5.5


def _norm(dim: int, *, sharding_config=None) -> RMSNorm.Config:
    return RMSNorm.Config(
        normalized_shape=dim,
        eps=_EPS,
        param_init=_NORM_INIT,
        sharding_config=sharding_config,
    )


def _sliding_window_attention(
    attn_backend: str, sliding_window_size: int
) -> InnerAttention.Config:
    inner_attention = get_attention_config(attn_backend)
    if isinstance(inner_attention, VarlenInnerAttention.Config):
        return SlidingWindowVarlenInnerAttention.Config(
            sdpa_backends=inner_attention.sdpa_backends,
            sliding_window_size=sliding_window_size,
        )
    assert isinstance(inner_attention, FlexInnerAttention.Config)
    return SlidingWindowFlexInnerAttention.Config(
        block_size=inner_attention.block_size,
        kernel_options=inner_attention.kernel_options,
        sliding_window_size=sliding_window_size,
    )


def _build_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    inner_attention: InnerAttention.Config,
    max_context_length: int,
    enable_sp: bool,
) -> list[TransformerBlock.Config]:
    """Dense Qwen3 blocks, as in ``qwen3/flavors.py``."""
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    # Static-shape compile: RoPE runs at two lengths (bytes, patches); see
    # StaticShapeCosSinRoPE.
    rope = StaticShapeCosSinRoPE.Config(
        dim=head_dim, max_context_length=max_context_length, theta=_ROPE_THETA
    )
    return [
        Qwen3TransformerBlock.Config(
            attention_norm=_norm(dim, sharding_config=norm_sharding),
            ffn_norm=_norm(dim, sharding_config=norm_sharding),
            attention=make_gqa_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_LINEAR_INIT,
                inner_attention=inner_attention,
                rope=rope,
                qk_norm=_norm(head_dim),
                enable_sp=enable_sp,
            ),
            feed_forward=make_ffn_config(
                dim=dim,
                hidden_dim=hidden_dim,
                w13_param_init=_LINEAR_INIT,
                w2_param_init=_LINEAR_INIT,
                enable_sp=enable_sp,
            ),
        )
        for _ in range(n_layers)
    ]


def _spacebyte(
    *,
    local_dim: int,
    local_n_heads: int,
    local_n_kv_heads: int,
    local_head_dim: int,
    local_hidden_dim: int,
    local_window: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    global_dim: int,
    global_n_heads: int,
    global_n_kv_heads: int,
    global_head_dim: int,
    global_hidden_dim: int,
    num_global_layers: int,
    attn_backend: str,
    seq_len: int,
    parallelism: ParallelismConfig,
    capacity_tokens_per_patch: float = _C4_CAPACITY_BYTES_PER_PATCH,
) -> SpaceByteModel.Config:
    """One-level SpaceByte: local byte layers around global patch layers."""
    enable_sp = parallelism.enable_sequence_parallel
    vocab_size = ByteTokenizer.vocab_size

    def local_layers(n_layers: int) -> list[TransformerBlock.Config]:
        return _build_layers(
            n_layers=n_layers,
            dim=local_dim,
            n_heads=local_n_heads,
            n_kv_heads=local_n_kv_heads,
            head_dim=local_head_dim,
            hidden_dim=local_hidden_dim,
            inner_attention=_sliding_window_attention(attn_backend, local_window),
            max_context_length=seq_len,
            enable_sp=enable_sp,
        )

    global_layers = _build_layers(
        n_layers=num_global_layers,
        dim=global_dim,
        n_heads=global_n_heads,
        n_kv_heads=global_n_kv_heads,
        head_dim=global_head_dim,
        hidden_dim=global_hidden_dim,
        inner_attention=get_attention_config(attn_backend),
        # A document has at most as many patches as bytes.
        max_context_length=seq_len,
        enable_sp=enable_sp,
    )
    return SpaceByteModel.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=local_dim,
        norm=_norm(
            local_dim,
            sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp),
        ),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=local_dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=local_dim,
            out_features=vocab_size,
            param_init=_LINEAR_INIT,
            sharding_config=lm_head_sharding_config(),
        ),
        layers=local_layers(num_encoder_layers)
        + global_layers
        + local_layers(num_decoder_layers),
        levels=[
            HierarchyLevel(
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                inner_dim=global_dim,
                boundary=SpacelikeBoundary.Config(),
                capacity_tokens_per_patch=capacity_tokens_per_patch,
            )
        ],
    )


def _debugmodel(
    attn_backend: str, *, seq_len: int, parallelism: ParallelismConfig
) -> SpaceByteModel.Config:
    return _spacebyte(
        local_dim=128,
        local_n_heads=2,
        local_n_kv_heads=1,
        local_head_dim=64,
        local_hidden_dim=384,
        local_window=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        global_dim=256,
        global_n_heads=4,
        global_n_kv_heads=2,
        global_head_dim=64,
        global_hidden_dim=768,
        num_global_layers=2,
        attn_backend=attn_backend,
        seq_len=seq_len,
        parallelism=parallelism,
    )


def _0_6b(
    attn_backend: str, *, seq_len: int, parallelism: ParallelismConfig
) -> SpaceByteModel.Config:
    """Global stack = Qwen3-0.6B's 28 blocks; local stacks at half width."""
    return _spacebyte(
        local_dim=512,
        local_n_heads=4,
        local_n_kv_heads=2,
        local_head_dim=128,
        local_hidden_dim=1536,
        local_window=512,
        num_encoder_layers=14,
        num_decoder_layers=14,
        global_dim=1024,
        global_n_heads=16,
        global_n_kv_heads=8,
        global_head_dim=128,
        global_hidden_dim=3072,
        num_global_layers=28,
        attn_backend=attn_backend,
        seq_len=seq_len,
        parallelism=parallelism,
    )


def _0_6b_flops(
    attn_backend: str, *, seq_len: int, parallelism: ParallelismConfig
) -> SpaceByteModel.Config:
    """"0.6B" with the same training FLOPs per text byte as Qwen3-0.6B.

    Counted algorithmically (scripts/flops_per_byte.py): 3 x forward, forward =
    2 x matmul multiply-accumulates plus QK^T and AV over the pairs the masks
    allow; no norms, elementwise ops, padding slots or kernel tile waste. On the
    nizarun English train split (6.064 bytes per patch, 0.21309 Qwen3 tokens per
    byte) Qwen3-0.6B costs 893.5 MFLOP per text byte and "0.6B" 1.2533x that.
    The global blocks stay Qwen3-0.6B's; the local stacks shrink from 14+14 to
    9+9 layers, and the window from 512 to 432 bytes absorbs the remainder:
    1.00008x Qwen3. Re-measure when the data changes.
    """
    return _spacebyte(
        local_dim=512,
        local_n_heads=4,
        local_n_kv_heads=2,
        local_head_dim=128,
        local_hidden_dim=1536,
        local_window=432,
        num_encoder_layers=9,
        num_decoder_layers=9,
        global_dim=1024,
        global_n_heads=16,
        global_n_kv_heads=8,
        global_head_dim=128,
        global_hidden_dim=3072,
        num_global_layers=28,
        attn_backend=attn_backend,
        seq_len=seq_len,
        parallelism=parallelism,
        capacity_tokens_per_patch=_NIZARUN_CAPACITY_BYTES_PER_PATCH,
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 2048),
    "0.6B": (_0_6b, 16384),
    "0.6B-flops": (_0_6b_flops, 16384),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "varlen",
) -> SpaceByteModel.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    return get_config(
        attn_backend=attn_backend, seq_len=context_len, parallelism=parallelism
    )
