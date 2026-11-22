"""Local recipes for the mcluster11 RTX PRO 6000 (SM120) nodes.

The default ``varlen`` attention backend dispatches to FA4, which PyTorch only
allows on SM 9.x/10.x, so these recipes select the ``flex`` backend instead.

``spacebyte_0_6b_flex`` and ``qwen3_0_6b_flex_baseline`` share optimizer,
schedule and validation settings, so their runs differ only in the model (and
its tokenizer). Compare validation bits per byte:
  SpaceByte: val_loss / ln(2) * BYTE_LABELS_PER_TEXT_BYTE
  Qwen3:     val_loss / ln(2) * QWEN3_LABELS_PER_TEXT_BYTE
"""

import dataclasses

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    TextAugmentation,
    TokenAugmentation,
)
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.components.validator import Validator
from torchtitan.config.function import CallableFunction
from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.qwen3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import (
    c4_document_capacity,
    local_fineweb_edu,
    with_augmentations,
)
from torchtitan_recipes.models.spacebyte import spacebyte_0_6b
from torchtitan_recipes.tests.models.qwen3 import qwen3_0_6b

# English FineWeb-Edu, 1200 x 25k documents (see torchtitan_recipes/data.py
# for the content-hash train/validation split).
_FINEWEB_EDU_DIR = "/home/adamga/leshemg/adamga/data/fineweb_translated/en-original"

# Training-data augmentations, applied in order by TextProcessor: text ones to
# the raw text, token ones to the tokenized document. Validation stays clean.
TEXT_AUGMENTATIONS: tuple[TextAugmentation.Config, ...] = ()
TOKEN_AUGMENTATIONS: tuple[TokenAugmentation.Config, ...] = ()

# Measured on 20k FineWeb-Edu validation docs by scripts/spacebyte_data_stats.py.
QWEN3_LABELS_PER_TEXT_BYTE = 0.21295  # 4.696 text bytes per Qwen3 token
BYTE_LABELS_PER_TEXT_BYTE = 1.00020  # one EOS per document

# The whole validation split: 998,183 documents (counted once over all files).
_FULL_VALIDATION_TEXT_BYTES = 4_891_202_282

# Validation scores ~1.5M text bytes per DP rank for both models: the same
# validation documents, in the same order.
_VALIDATION_BYTES_PER_RANK = 24 * 65536

# Mixes documents across about one source file (~70 MB of text per rank).
_SHUFFLE_BUFFER_DOCS = 25_000


def _use_fineweb_edu(config: Trainer.Config) -> Trainer.Config:
    """Train on the FineWeb-Edu train split, with the augmentations above."""
    config.dataloader.dataset = ConcatThenSplitPackingConfig(
        dataset=with_augmentations(
            local_fineweb_edu(_FINEWEB_EDU_DIR, split="train"),
            text=TEXT_AUGMENTATIONS,
            token=TOKEN_AUGMENTATIONS,
        )
    )
    config.dataloader.streaming_shuffle_buffer_size = _SHUFFLE_BUFFER_DOCS
    return config


def _validator(num_tokens_per_microbatch: int, steps: int) -> Validator.Config:
    return Validator.Config(
        freq=500,
        steps=steps,
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(num_tokens_per_microbatch),
            dataset=ConcatThenSplitPackingConfig(
                dataset=local_fineweb_edu(_FINEWEB_EDU_DIR, split="validation")
            ),
            repeat=True,
        ),
    )


def qwen3_0_6b_flex(seq_len: int = 4096) -> Trainer.Config:
    config = qwen3_0_6b(seq_len=seq_len)
    config.model = build_model_config(
        "0.6B", seq_len=seq_len, parallelism=config.parallelism, attn_backend="flex"
    )
    return config


# Peak LR and global batch for 10B Qwen3 tokens, near the Step Law
# (arXiv:2503.04715) and DeepSeek LLM (arXiv:2401.02954) optima for this size:
# 0.3-1M tokens per step, peak LR 1e-3 to 1.5e-3.
_LR = 1e-3
_WARMUP_STEPS = 1000
_QWEN3_TOKENS_PER_STEP = 32 * 16384  # 524,288: 16 microbatches per rank on 2 GPUs
_STEPS = 19_073  # 10.0B Qwen3 tokens

# SpaceByte trains on the same text bytes as Qwen3, step for step. On the train
# files these runs read, a Qwen3 token is 4.67586 +- 0.00096 byte-stream tokens
# (BOS/EOS included; scripts/qwen3_bytes_per_token.py), so one Qwen3 step is
# 2,451,496 bytes. 19 microbatches of 64512 bytes per rank on 2 GPUs give
# 2,451,456 per step, 0.002% fewer. Both read the same documents in the same
# order, so they see the same text, up to the last few documents.
_SPACEBYTE_BYTES_PER_MICROBATCH = 64512
_SPACEBYTE_BYTES_PER_STEP = 38 * _SPACEBYTE_BYTES_PER_MICROBATCH


def spacebyte_0_6b_flex(
    seq_len: int = 16384, local_lr_multiplier: float = 1.0
) -> Trainer.Config:
    config = spacebyte_0_6b(
        seq_len=seq_len,
        attn_backend="flex",
        lr=_LR,
        local_lr_multiplier=local_lr_multiplier,
    )
    # WSD: hold the peak LR after warmup, then decay to 0 over the last 20% of
    # steps (SmolLM2's small models, arXiv:2502.02737) with the 1-sqrt shape
    # (arXiv:2405.18392). Before the decay, a run can be extended without
    # restarting the schedule.
    config.optim.lr_scheduler = dataclasses.replace(
        config.optim.lr_scheduler,
        warmup_steps=_WARMUP_STEPS,
        decay_ratio=0.2,
        decay_type="sqrt",
    )
    config.training = dataclasses.replace(
        config.training,
        num_tokens_per_microbatch_per_dp_rank=_SPACEBYTE_BYTES_PER_MICROBATCH,
        num_tokens_per_train_step=_SPACEBYTE_BYTES_PER_STEP,
        steps=_STEPS,
    )
    config.dataloader.max_num_documents = c4_document_capacity(
        _SPACEBYTE_BYTES_PER_MICROBATCH
    )
    _use_fineweb_edu(config)
    num_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    config.validator = _validator(
        num_tokens, steps=_VALIDATION_BYTES_PER_RANK // num_tokens
    )
    return config


# Each named run writes logs, traces and W&B files to _OUTPUT_ROOT/<id>_<name>
# and checkpoints to _OUTPUT_ROOT/checkpoints/<id>_<name>. Relaunching a run
# resumes from its latest checkpoint and appends to the same W&B run, so give
# a fresh run a new id rather than reusing one.
_OUTPUT_ROOT = "/home/adamga/leshemg/adamga/torchtitan_hirarchal_outputs"
_WANDB_ENTITY = "adam-gaber-weizmann-institute-of-science"
_WANDB_PROJECT = "torchtitan-hirarchal"


# The last checkpoint before the WSD decay (it starts at 0.8 * _STEPS = 15258).
# Resuming from it with more training.steps extends a run without a second
# decay in the middle.
_PERSISTENT_CHECKPOINT_STEPS = (15_000,)


def _named_run(
    config: Trainer.Config,
    *,
    run_id: int,
    name: str,
    activation_remat: ActivationRematConfig | None,
    persistent_checkpoint_steps: tuple[int, ...] = (),
) -> Trainer.Config:
    """``activation_remat`` replaces the base recipe's policy; it changes memory
    and speed, not numerics. ``persistent_checkpoint_steps`` are kept on top of
    the latest 5; each must be a multiple of the checkpoint interval, or it is
    never saved."""
    run = f"{run_id}_{name}"
    config.activation_remat = activation_remat
    # About 55 min of Qwen3 training on 2 L40S, so a job killed at the 12 h
    # SLURM limit loses less than an hour.
    interval = 250
    if any(step % interval for step in persistent_checkpoint_steps):
        raise ValueError(
            f"persistent_checkpoint_steps {persistent_checkpoint_steps} must be "
            f"multiples of the checkpoint interval {interval}."
        )
    config.dump_folder = f"{_OUTPUT_ROOT}/{run}"
    config.checkpointer = CheckpointManager.Config(
        folder=f"{_OUTPUT_ROOT}/checkpoints/{run}",
        interval=interval,
        keep_latest_k=5,
        purge_exempt=CallableFunction.Config(
            fn=frozenset(persistent_checkpoint_steps).__contains__
        ),
        # Keep the optimizer in the final checkpoint too, so a finished run
        # can be extended.
        last_save_model_only=False,
    )
    config.metrics = dataclasses.replace(
        config.metrics,
        log_freq=10,
        enable_wandb=True,
        wandb_entity=_WANDB_ENTITY,
        wandb_project=_WANDB_PROJECT,
        wandb_run_name=run,
        wandb_run_id=run,
    )
    return config


def spacebyte_0_6b_flex_local_lr_x2() -> Trainer.Config:
    """Local LR scaled like muP's 1/width for Adam: global dim / local dim = 2."""
    return spacebyte_0_6b_flex(local_lr_multiplier=2.0)


def qwen3_0_6b_flex_baseline(seq_len: int = 4096) -> Trainer.Config:
    """Qwen3-0.6B with the same schedule and validation as SpaceByte, and its
    single learning rate equal to SpaceByte's global-layer learning rate."""
    spacebyte = spacebyte_0_6b_flex()
    config = qwen3_0_6b_flex(seq_len=seq_len)
    config.optim = dataclasses.replace(
        spacebyte.optim,
        optimizer=OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=_LR)]
        ),
    )
    config.training = dataclasses.replace(
        spacebyte.training,
        num_tokens_per_microbatch_per_dp_rank=config.training.num_tokens_per_microbatch_per_dp_rank,
        num_tokens_per_train_step=_QWEN3_TOKENS_PER_STEP,
        max_context_length=config.training.max_context_length,
    )
    config.activation_remat = FullActivationRematConfig()
    _use_fineweb_edu(config)
    num_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    validation_tokens = _VALIDATION_BYTES_PER_RANK * QWEN3_LABELS_PER_TEXT_BYTE
    config.validator = _validator(
        num_tokens, steps=max(1, round(validation_tokens / num_tokens))
    )
    return config


def qwen3_0_6b_fineweb_edu_run() -> Trainer.Config:
    return _named_run(
        qwen3_0_6b_flex_baseline(),
        run_id=1,
        name="qwen3-0.6b_fineweb-edu",
        # No remat: 20.6K tokens/s per L40S at 27 GiB (full: 17.6K, 9 GiB).
        activation_remat=None,
        persistent_checkpoint_steps=_PERSISTENT_CHECKPOINT_STEPS,
    )


def spacebyte_0_6b_fineweb_edu_run() -> Trainer.Config:
    return _named_run(
        spacebyte_0_6b_flex(),
        run_id=2,
        name="spacebyte-0.6b_fineweb-edu",
        # Saves matmul outputs, recomputes the rest: 64.2K bytes/s per L40S at
        # 34 GiB (full: 56.1K, 8 GiB). Without remat it runs out of memory.
        activation_remat=DefaultActivationRematConfig(),
        persistent_checkpoint_steps=_PERSISTENT_CHECKPOINT_STEPS,
    )


def spacebyte_debug_flex(local_lr_multiplier: float = 1.0) -> Trainer.Config:
    config = spacebyte_0_6b(
        attn_backend="flex",
        flavor="debugmodel",
        lr=_LR,
        local_lr_multiplier=local_lr_multiplier,
    )
    _use_fineweb_edu(config)
    config.training.steps = 10
    config.optim.lr_scheduler.warmup_steps = 2
    return config


def _smoke(config: Trainer.Config, *, steps: int = 20) -> Trainer.Config:
    """Short run for checking throughput, memory and validation wiring."""
    config.training.steps = steps
    config.training.num_tokens_per_train_step = -1  # one microbatch per step
    config.optim.lr_scheduler.warmup_steps = 2
    config.metrics = MetricsProcessor.Config(log_freq=1)
    if config.validator is not None:
        config.validator.freq = steps
        config.validator.steps = 2
    return config


def _full_validation(
    config: Trainer.Config, *, labels_per_text_byte: float, num_dp_ranks: int
) -> Trainer.Config:
    """Validate once, after the last step, on the whole validation split.

    Each rank streams its own files, so steps are sized from the split's mean
    bytes per rank; ranks cover their files to within one microbatch or so.
    """
    num_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    labels_per_rank = (
        _FULL_VALIDATION_TEXT_BYTES * labels_per_text_byte / num_dp_ranks
    )
    config.validator.freq = config.training.steps
    config.validator.steps = round(labels_per_rank / num_tokens)
    return config


def spacebyte_0_6b_flex_full_validation() -> Trainer.Config:
    """For 2 GPUs: the whole 1M-document split takes about 3 hours to score."""
    return _full_validation(
        spacebyte_0_6b_flex(),
        labels_per_text_byte=BYTE_LABELS_PER_TEXT_BYTE,
        num_dp_ranks=2,
    )


def qwen3_0_6b_flex_baseline_full_validation() -> Trainer.Config:
    """For 2 GPUs: the whole 1M-document split."""
    return _full_validation(
        qwen3_0_6b_flex_baseline(),
        labels_per_text_byte=QWEN3_LABELS_PER_TEXT_BYTE,
        num_dp_ranks=2,
    )


def spacebyte_0_6b_flex_smoke() -> Trainer.Config:
    return _smoke(spacebyte_0_6b_flex())


def qwen3_0_6b_flex_baseline_smoke() -> Trainer.Config:
    return _smoke(qwen3_0_6b_flex_baseline())


def spacebyte_debug_flex_deterministic() -> Trainer.Config:
    config = _smoke(spacebyte_debug_flex(), steps=10)
    config.debug.seed = 42
    config.debug.deterministic = True
    return config


def spacebyte_debug_flex_local_lr_x2_smoke() -> Trainer.Config:
    return _smoke(spacebyte_debug_flex(local_lr_multiplier=2.0), steps=10)


# Profiling: one traced step (step 10) on one GPU, no validation.
def _profile(config: Trainer.Config, name: str) -> Trainer.Config:
    config.training.steps = 12
    config.training.num_tokens_per_train_step = -1  # one microbatch per step
    config.optim.lr_scheduler.warmup_steps = 2
    config.metrics = MetricsProcessor.Config(log_freq=1)
    config.validator = None
    config.profiler = dataclasses.replace(
        config.profiler, enable_profiling=True, profile_freq=10
    )
    config.dump_folder = f"./outputs/profile_{name}"
    return config


def spacebyte_0_6b_flex_profile() -> Trainer.Config:
    return _profile(spacebyte_0_6b_flex(), "spacebyte")


def qwen3_0_6b_flex_baseline_profile() -> Trainer.Config:
    return _profile(qwen3_0_6b_flex_baseline(), "qwen3")
