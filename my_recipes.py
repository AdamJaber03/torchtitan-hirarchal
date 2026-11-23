"""Recipes for the Qwen3-0.6B vs SpaceByte-0.6B runs (GOLEM L40S nodes; first
written for the mcluster11 RTX PRO 6000 nodes). Launch with slurm/train.sh.

The default ``varlen`` attention backend dispatches to FA4, which PyTorch only
allows on SM 9.x/10.x (L40S is SM 8.9), so these recipes select ``flex``.

``spacebyte_0_6b_flex`` and ``qwen3_0_6b_flex_baseline`` share optimizer,
schedule and validation settings, so their runs differ only in the model (and
its tokenizer). Compare validation bits per byte:
  SpaceByte: val_loss / ln(2) * BYTE_LABELS_PER_TEXT_BYTE
  Qwen3:     val_loss / ln(2) * QWEN3_LABELS_PER_TEXT_BYTE
"""

import dataclasses
import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    TextAugmentation,
    TokenAugmentation,
)
from torchtitan.components.loss import ChunkedLossWrapper, loss_config_from_spec
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

# English FineWeb-Edu side of the paired en/ar set: train/ (960 files, 23,969,882
# docs) and validation/ (192 files, 824,543 docs), pre-split and pre-shuffled.
# The file counts divide evenly by DP degrees 8, 16, 24, 32 and 64.
_FINEWEB_EDU_DIR = os.path.expanduser("~/data/fineweb-edu-ar-nizarun/en")

# Training-data augmentations, applied in order by TextProcessor: text ones to
# the raw text, token ones to the tokenized document. Validation stays clean.
TEXT_AUGMENTATIONS: tuple[TextAugmentation.Config, ...] = ()
TOKEN_AUGMENTATIONS: tuple[TokenAugmentation.Config, ...] = ()

# Measured on 20k FineWeb-Edu validation docs by scripts/spacebyte_data_stats.py.
QWEN3_LABELS_PER_TEXT_BYTE = 0.21295  # 4.696 text bytes per Qwen3 token
BYTE_LABELS_PER_TEXT_BYTE = 1.00020  # one EOS per document

# The whole validation split: 824,543 documents (counted once over all files).
_FULL_VALIDATION_TEXT_BYTES = 3_967_216_214

# Validation scores ~1.5M text bytes per DP rank for both models: the same
# validation documents, in the same order.
_VALIDATION_BYTES_PER_RANK = 24 * 65536

# The files are already shuffled; this window only adds extra local mixing.
_SHUFFLE_BUFFER_DOCS = 25_000


def set_loss(config: Trainer.Config, spec: dict) -> Trainer.Config:
    """Set the training loss from a dict spec (see loss_config_from_spec in
    torchtitan/components/loss.py), e.g.::

        set_loss(config, {"loss_type": "mixed", "losses": [
            {"weight": 1.0, "loss": {"loss_type": "cross_entropy"}},
            {"weight": 0.1, "loss": {"loss_type": "mse", "name": "aux"}},
        ]})

    The spec replaces the loss inside ChunkedLossWrapper, which keeps its
    lm_head chunking. Each MixedLoss term is logged as loss_metrics/<name>.
    For separate validation losses, set config.validator.losses to a list of
    specs, each reported as validation_metrics/<name>.
    """
    loss = loss_config_from_spec(spec)
    if isinstance(config.loss, ChunkedLossWrapper.Config):
        config.loss = dataclasses.replace(config.loss, loss_fn=loss)
    else:
        config.loss = loss
    return config


def use_eos_as_bos(config: Trainer.Config) -> Trainer.Config:
    """Start every training and validation document with the EOS token as a
    BOS (TextProcessor.Config.eos_as_bos). Qwen3 and ByteTokenizer have no BOS,
    so otherwise a document's first token is never predicted: about 3.7 bytes
    per document for Qwen3 but 1 for SpaceByte, which slightly favors Qwen3 in
    a BPB comparison. Changes the data format, so only for new runs, and use it
    for both models of a comparison."""
    loaders = [config.dataloader]
    if config.validator is not None:
        loaders.append(config.validator.dataloader)
    for loader in loaders:
        packing = loader.dataset
        single = packing.dataset
        loader.dataset = dataclasses.replace(
            packing,
            dataset=dataclasses.replace(
                single,
                processor=dataclasses.replace(single.processor, eos_as_bos=True),
            ),
        )
    return config


def _hsdp_for_launch(config: Trainer.Config) -> Trainer.Config:
    """Multi-node launches use HSDP: FSDP within a node, replicas across nodes.

    Weights are all-gathered over a node's PCIe GPUs only, and gradients cross
    InfiniBand once per step. Reads torchrun's WORLD_SIZE / LOCAL_WORLD_SIZE,
    so a single-node launch (or no launcher, e.g. --print-config) stays plain
    FSDP. Mutates config.parallelism in place: the model config shares it.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
    if world > local:
        if world % local:
            raise ValueError(f"WORLD_SIZE {world} is not a multiple of {local}.")
        config.parallelism.data_parallel_replicate_degree = world // local
        config.parallelism.data_parallel_shard_degree = local
    return config


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


# Validation scores cross-entropy (kept under the name "loss", as before) and
# exact bits per byte: summed bits over the UTF-8 bytes the labels cover, read
# off the labels with the tokenizer's byte-length table. BPB is comparable
# between Qwen3 and SpaceByte as is; with use_eos_as_bos both also predict
# exactly the same bytes (every byte of every document).
_VALIDATION_LOSSES = (
    {"loss_type": "cross_entropy", "name": "loss"},
    {"loss_type": "bits_per_byte", "name": "bpb"},
)


def _validator(num_tokens_per_microbatch: int, steps: int) -> Validator.Config:
    return Validator.Config(
        freq=500,
        steps=steps,
        losses=[dict(spec) for spec in _VALIDATION_LOSSES],
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

# SpaceByte and Qwen3 are compute-matched and data-matched: the "0.6B-flops"
# flavor costs the same training FLOPs per text byte as Qwen3-0.6B (see its
# docstring; scripts/flops_per_byte.py), and each step reads the same text.
# On the nizarun English train split a Qwen3 token is 4.695697 +- 0.000618
# byte-stream tokens (EOS included; scripts/qwen3_bytes_per_token.py, all 960
# files, 2026-11-23), so one Qwen3 step is 2,461,927 bytes. 64 microbatches of
# 38,464 bytes give 2,461,696, 0.009% fewer, on up to 64 GPUs. With
# _reduce_gradient_traffic and patch capacity 5.5, one L40S node: 88.8K bytes/s
# per GPU at 16.6 GiB (capacity 5.0: 84.5K at 17 GiB;
# 76,928-byte microbatches: 80K at 31 GiB; neither setting: 75K).
_SPACEBYTE_FLAVOR = "0.6B-flops"
_SPACEBYTE_BYTES_PER_MICROBATCH = 38_464
_SPACEBYTE_BYTES_PER_STEP = 64 * _SPACEBYTE_BYTES_PER_MICROBATCH


def spacebyte_0_6b_flex(
    seq_len: int = 16384,
    local_lr_multiplier: float = 1.0,
    bytes_per_microbatch: int = _SPACEBYTE_BYTES_PER_MICROBATCH,
) -> Trainer.Config:
    """``bytes_per_microbatch`` must divide the step over the DP ranks; the
    default allows up to 64 GPUs."""
    config = spacebyte_0_6b(
        seq_len=seq_len,
        flavor=_SPACEBYTE_FLAVOR,
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
        num_tokens_per_microbatch_per_dp_rank=bytes_per_microbatch,
        num_tokens_per_train_step=_SPACEBYTE_BYTES_PER_STEP,
        steps=_STEPS,
    )
    config.dataloader.max_num_documents = c4_document_capacity(bytes_per_microbatch)
    _reduce_gradient_traffic(config)
    _use_fineweb_edu(config)
    num_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    config.validator = _validator(
        num_tokens, steps=_VALIDATION_BYTES_PER_RANK // num_tokens
    )
    _hsdp_for_launch(config)
    return config


# Each named run writes logs, traces and W&B files to _OUTPUT_ROOT/<id>_<name>
# and checkpoints to _OUTPUT_ROOT/checkpoints/<id>_<name>. Relaunching a run
# resumes from its latest checkpoint and appends to the same W&B run, so give
# a fresh run a new id rather than reusing one.
_OUTPUT_ROOT = os.environ.get(
    "TORCHTITAN_OUTPUT_ROOT",
    os.path.expanduser("~/torchtitan_hirarchal_outputs"),
)
_SEED = 42
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
    the latest 3; each must be a multiple of the checkpoint interval, or it is
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
    # A fixed seed makes a launch repeatable (initial weights; GPU kernels are
    # still not bit-deterministic). Resuming loads weights, so it has no effect
    # on a run already started (run 3 began with a random seed).
    config.debug = dataclasses.replace(config.debug, seed=_SEED)
    config.checkpointer = CheckpointManager.Config(
        folder=f"{_OUTPUT_ROOT}/checkpoints/{run}",
        interval=interval,
        keep_latest_k=3,
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


def _reduce_gradient_traffic(config: Trainer.Config) -> Trainer.Config:
    """Less FSDP traffic over the L40S nodes' PCIe: weights stay gathered from
    forward to backward, gradients are reduce-scattered once per step instead
    of once per microbatch, and in bf16. Qwen3-0.6B, 16384-token microbatches,
    1 node: 19.5K -> 22.2K-22.6K tokens/s per GPU for +2 GiB (benchmarked
    2026-11-23). bf16 reduction alone is ~+6% but sums gradients across GPUs
    in bf16 instead of fp32."""
    config.parallelism.fsdp_reshard_after_forward = "never"
    config.parallelism.fsdp_defer_gradient_reduction = True
    config.training.mixed_precision_reduce = "bfloat16"
    return config


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
    _reduce_gradient_traffic(config)
    _use_fineweb_edu(config)
    num_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    validation_tokens = _VALIDATION_BYTES_PER_RANK * QWEN3_LABELS_PER_TEXT_BYTE
    config.validator = _validator(
        num_tokens, steps=max(1, round(validation_tokens / num_tokens))
    )
    _hsdp_for_launch(config)
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


def qwen3_0_6b_fineweb_edu_nizarun_run() -> Trainer.Config:
    """qwen3_0_6b_fineweb_edu_run on GOLEM with the paired-corpus English
    side (~/data/fineweb-edu-ar-nizarun/en). A new id, since the data changed
    and run 1's W&B run and checkpoints belong to the old cluster."""
    config = qwen3_0_6b_flex_baseline()
    # 8192-token microbatches (2 contexts) instead of 16384, so the same
    # 524,288-token step splits over up to 64 GPUs (8 nodes): one microbatch
    # per rank on 64 GPUs, two on 32. Global batch, LR and schedule unchanged.
    num_tokens = 2 * config.training.max_context_length
    config.training.num_tokens_per_microbatch_per_dp_rank = num_tokens
    config.dataloader.max_num_documents = c4_document_capacity(num_tokens)
    validation_tokens = _VALIDATION_BYTES_PER_RANK * QWEN3_LABELS_PER_TEXT_BYTE
    config.validator = _validator(
        num_tokens, steps=max(1, round(validation_tokens / num_tokens))
    )
    return _named_run(
        config,
        run_id=3,
        name="qwen3-0.6b_fineweb-edu-nizarun",
        activation_remat=None,
        persistent_checkpoint_steps=_PERSISTENT_CHECKPOINT_STEPS,
    )



def qwen3_0_6b_fineweb_edu_nizarun_seed42_run() -> Trainer.Config:
    """The Qwen3 side of the compute-matched pair with run 4 (SpaceByte):
    fixed seed, _reduce_gradient_traffic, and 16384-token microbatches (one per
    GPU on 4 nodes; up to 32 GPUs). Run 3 started before these and can only
    resume on 8 nodes."""
    config = qwen3_0_6b_flex_baseline()
    return _named_run(
        config,
        run_id=5,
        name="qwen3-0.6b_fineweb-edu-nizarun-seed42",
        activation_remat=None,
        persistent_checkpoint_steps=_PERSISTENT_CHECKPOINT_STEPS,
    )

def spacebyte_0_6b_fineweb_edu_run() -> Trainer.Config:
    return _named_run(
        spacebyte_0_6b_flex(),
        # Run 2 was the old 14+14-layer flavor on the old data and cluster.
        run_id=4,
        name="spacebyte-0.6b-flops_fineweb-edu-nizarun",
        # Old 14+14-layer flavor, 64512-byte microbatches: 64.2K bytes/s per
        # L40S at 34 GiB (full remat: 56.1K, 8 GiB); no remat ran out of memory.
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
    _hsdp_for_launch(config)
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
    """For 2 GPUs: the whole 825k-document split takes about 2.5 hours to score."""
    return _full_validation(
        spacebyte_0_6b_flex(),
        labels_per_text_byte=BYTE_LABELS_PER_TEXT_BYTE,
        num_dp_ranks=2,
    )


def qwen3_0_6b_flex_baseline_full_validation() -> Trainer.Config:
    """For 2 GPUs: the whole 825k-document split."""
    return _full_validation(
        qwen3_0_6b_flex_baseline(),
        labels_per_text_byte=QWEN3_LABELS_PER_TEXT_BYTE,
        num_dp_ranks=2,
    )


def spacebyte_0_6b_flex_smoke() -> Trainer.Config:
    return _smoke(spacebyte_0_6b_flex())


def qwen3_0_6b_flex_baseline_smoke() -> Trainer.Config:
    return _smoke(qwen3_0_6b_flex_baseline())


def qwen3_bpb_smoke() -> Trainer.Config:
    """Validation with exact BPB, documents starting with EOS-as-BOS."""
    return use_eos_as_bos(_smoke(qwen3_0_6b_flex_baseline()))


def spacebyte_bpb_smoke() -> Trainer.Config:
    """Validation with exact BPB, documents starting with EOS-as-BOS."""
    return use_eos_as_bos(_smoke(spacebyte_0_6b_flex()))


def qwen3_mixed_loss_smoke() -> Trainer.Config:
    """Checks MixedLoss wiring: CE + 0.5 x CE (named "aux"), so the logged
    loss is 1.5 x loss_metrics/cross_entropy; validation scores named losses."""
    config = _smoke(qwen3_0_6b_flex_baseline())
    config.metrics = MetricsProcessor.Config(log_freq=1, enable_tensorboard=True)
    config.dump_folder = "./outputs/mixed_loss_smoke"
    set_loss(
        config,
        {
            "loss_type": "mixed",
            "losses": [
                {"weight": 1.0, "loss": {"loss_type": "cross_entropy"}},
                {"weight": 0.5, "loss": {"loss_type": "cross_entropy", "name": "aux"}},
            ],
        },
    )
    config.validator.losses = [
        {"loss_type": "cross_entropy", "name": "ce"},
        {"loss_type": "cross_entropy", "name": "ce_again"},
    ]
    return config


def spacebyte_debug_flex_deterministic() -> Trainer.Config:
    config = _smoke(spacebyte_debug_flex(), steps=10)
    config.debug.seed = 42
    config.debug.deterministic = True
    return config


def spacebyte_debug_flex_local_lr_x2_smoke() -> Trainer.Config:
    return _smoke(spacebyte_debug_flex(local_lr_multiplier=2.0), steps=10)


def qwen3_0_6b_fineweb_edu_bench() -> Trainer.Config:
    """Throughput of qwen3_0_6b_fineweb_edu_run (same batch, remat and data)
    over 60 steps, without validation, W&B or periodic checkpoints."""
    config = qwen3_0_6b_fineweb_edu_run()
    config.training.steps = 60
    config.validator = None
    config.metrics = MetricsProcessor.Config(log_freq=5)
    config.dump_folder = "./outputs/bench_qwen3"
    # No checkpoints: a final save would make the next bench resume from it.
    config.checkpointer = None
    return config


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


def _spacebyte_bench(bytes_per_microbatch: int, name: str) -> Trainer.Config:
    """Throughput of spacebyte_0_6b_fineweb_edu_run over 30 steps, without
    validation, W&B or checkpoints."""
    config = spacebyte_0_6b_fineweb_edu_run()
    if bytes_per_microbatch != config.training.num_tokens_per_microbatch_per_dp_rank:
        config.training.num_tokens_per_microbatch_per_dp_rank = bytes_per_microbatch
        config.dataloader.max_num_documents = c4_document_capacity(bytes_per_microbatch)
    config.training.steps = 30
    config.validator = None
    config.metrics = MetricsProcessor.Config(log_freq=5)
    config.dump_folder = f"./outputs/{name}"
    config.checkpointer = None
    return config


def spacebyte_0_6b_fineweb_edu_bench() -> Trainer.Config:
    return _spacebyte_bench(_SPACEBYTE_BYTES_PER_MICROBATCH, "bench_spacebyte")


