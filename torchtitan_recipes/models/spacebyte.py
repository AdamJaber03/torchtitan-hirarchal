# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SpaceByte model recipes (see torchtitan/models/spacebyte/README.md)."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import ByteTokenizer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.spacebyte import build_model_config, SpaceByteModel
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity


def _global_layers_pattern(model_config: SpaceByteModel.Config) -> str:
    """Regex for the parameters of the blocks above the byte level."""
    indices = [i for i, d in enumerate(model_config.layer_depths()) if d > 0]
    return rf"(^|\.)layers\.({'|'.join(map(str, indices))})\."


def spacebyte_0_6b(
    seq_len: int | None = None,
    attn_backend: str = "varlen",
    *,
    flavor: str = "0.6B",
    lr: float = 3e-4,
    local_lr_multiplier: float = 1.0,
) -> Trainer.Config:
    """SpaceByte matched to Qwen3-0.6B. ``seq_len`` and "tokens" are bytes.

    The global blocks train at ``lr``, as Qwen3-0.6B's blocks would. Everything
    at byte level (local blocks, embedding, final norm, lm_head) trains at
    ``lr * local_lr_multiplier``: these are half as wide, and their best learning
    rate need not match the global one.

    ``flavor="debugmodel"`` builds the same recipe around the debug model.
    """
    parallelism = ParallelismConfig()
    model_config = build_model_config(
        flavor, seq_len=seq_len, parallelism=parallelism, attn_backend=attn_backend
    )
    num_tokens_per_microbatch_per_dp_rank = 4 * model_config.max_context_length
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        tokenizer=ByteTokenizer.Config(),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(
                num_tokens_per_microbatch_per_dp_rank
            ),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(pattern=_global_layers_pattern(model_config), lr=lr),
                    AdamW.Config(pattern=r".*", lr=lr * local_lr_multiplier),
                ]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            max_context_length=model_config.max_context_length,
            steps=1000,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=FullActivationRematConfig(),
    )
