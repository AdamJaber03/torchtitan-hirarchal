# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Literal, TypeAlias, TypeVar

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.config import Function
from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.distributed.local_compile import local_compile
from torchtitan.distributed.spmd_types import current_spmd_mesh, spmd_mesh_size

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]
LossResult: TypeAlias = tuple[torch.Tensor, dict[str, torch.Tensor]]
LossInput: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]
LossInputT = TypeVar("LossInputT", contravariant=True)


class Loss(Function[LossResult], Generic[LossInputT], ABC):
    """Callable loss contract used by trainers and loss wrappers."""

    normalizer: ClassVar[Literal["tokens", "bytes"]] = "tokens"
    """What a summed loss is divided by: valid label tokens (the trainer passes
    their global count as ``global_loss_token_counts``), or the UTF-8 bytes the
    labels cover (counted by the validator; see ``BitsPerByteLoss``)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Function.Config):
        """Base configuration type for loss components."""

        name: str | None = None
        """Label this loss reports under (inside a ``MixedLoss`` or as a named
        validation loss). Defaults to its ``loss_type`` in ``LOSS_TYPES``."""

    @abstractmethod
    def __call__(
        self,
        pred: LossInputT,
        labels: LossInputT,
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> LossResult:
        """Return the scaled loss and any metrics computed by the loss."""
        ...


@local_compile("loss", batch_invariant=False)
def cross_entropy_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    *,
    global_vocab_size: int | None = None,
    reduction: Literal["sum", "none"] = "sum",
) -> torch.Tensor:
    """Cross-entropy over ``pred[T, V]`` and ``labels[T]``.

    ``pred`` is upcast to fp32 inside the loss rather than in the model forward,
    so that compiling the loss region fuses the forward and backward casts into
    the loss kernels, which saves both time and memory. The vocab-parallel path
    takes ``pred`` as is and upcasts internally, so the compiled region can save
    the low-precision input for backward instead of an fp32 copy.
    """
    if reduction not in ("sum", "none"):
        raise ValueError(f"Unsupported cross-entropy reduction: {reduction}")
    if spmd_mesh_size("tp") > 1:
        if global_vocab_size is None:
            raise ValueError(
                "global_vocab_size is required for vocab-parallel cross-entropy"
            )
        return _LossParallelCrossEntropy.apply(
            pred,
            labels,
            current_spmd_mesh().get_group("tp"),  # pyrefly: ignore[missing-attribute]
            global_vocab_size,
            reduction,
        )

    return torch.nn.functional.cross_entropy(
        pred.float(),
        labels,
        reduction=reduction,
        ignore_index=IGNORE_INDEX,
    )


# Target probability at and above which the target gradient is -p_non_target.
# Below it p_target - 1 does not cancel and rounds once, so it is more accurate
# than -p_non_target, which rounds the sum and the divide.
_HIGH_PROB_THRESHOLD = 1 / 4


def _target_mask(target_lane: torch.Tensor, local_vocab_size: int) -> torch.Tensor:
    """``[T, V_local]`` mask of each row's local target lane (none where -1)."""
    lanes = torch.arange(local_vocab_size, device=target_lane.device)
    return lanes == target_lane.unsqueeze(-1)


class _LossParallelCrossEntropy(torch.autograd.Function):
    """
    Vocab-parallel cross-entropy on local ``[T, V_local]`` logits.

    Replaces ``torch.distributed.tensor.parallel.loss_parallel()`` with an
    explicit autograd Function so that SPMD code can operate on local tensors
    and process groups directly, without the DTensor-based context manager.

    Supports uneven vocab sharding (last TP rank may hold fewer classes) and
    ``IGNORE_INDEX`` labels. Forward uses three TP all-reduces (max,
    non-target sumexp, target logit) to aggregate intermediate results in
    distributed softmax; backward needs zero collectives. Forward saves the
    input and per-row statistics, and backward recomputes the gradient.

    The NLL is ``softplus(logsumexp_{j != t}(z_j) - z_t)`` rather than
    ``logsumexp(z) - z_t``, so confident targets keep a nonzero loss and target
    gradient instead of both rounding to exactly 0 (fp32 rounds once the target
    margin exceeds ~16.6) while non-target gradients survive.

    All inputs and outputs are plain ``torch.Tensor`` (not DTensor).
    """

    @staticmethod
    def spmd_typecheck(
        result: torch.Tensor,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> None:
        """
        SPMD type: logits S(-1)@TP, labels I@TP -> loss I@TP.
        Non-TP axes are passed through from logits to the output.
        """
        spmd.assert_type(logits, {tp_group: spmd.S(logits.dim() - 1)})
        spmd.assert_type(labels, {tp_group: spmd.I})
        spmd.assert_local_type_like(
            result,
            logits,
            {tp_group: spmd.I},  # pyrefly: ignore [bad-argument-type]
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
        global_vocab_size: int,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute exact CE from local vocab shards via TP all-reduces.

        ``reduction="sum"`` returns the scalar summed loss (SFT/CE).
        ``reduction="none"`` returns the per-token NLL ``[T]``, which GRPO
        negates to get per-token logprobs without all-gathering the vocab.
        """
        logits_dtype = logits.dtype

        # Compute this rank's vocab shard bounds for the local logits.
        tp_world_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        chunk_size = (global_vocab_size + tp_world_size - 1) // tp_world_size
        vocab_start = min(global_vocab_size, chunk_size * tp_rank)
        vocab_end = min(global_vocab_size, vocab_start + chunk_size)
        local_vocab_size = max(0, vocab_end - vocab_start)
        if logits.shape[-1] != local_vocab_size:
            raise ValueError(
                "_LossParallelCrossEntropy expected local vocab size "
                f"{local_vocab_size} for global vocab size {global_vocab_size}, "
                f"got {logits.shape[-1]}."
            )
        if local_vocab_size == 0:
            raise ValueError(
                "_LossParallelCrossEntropy does not support empty vocab shards."
            )

        torch._assert_async(
            torch.all(
                (labels == IGNORE_INDEX)
                | ((labels >= 0) & (labels < global_vocab_size))
            ),
            f"labels must be {IGNORE_INDEX} or in [0, {global_vocab_size})",
        )

        # All-reduce max for numerically stable distributed log-softmax.
        local_max = torch.amax(logits, dim=-1, keepdim=True).float()
        local_max = funcol.all_reduce(
            local_max, reduceOp=dist.ReduceOp.MAX.name, group=tp_group
        )

        # Ignored labels are treated as targets that live on no shard: they get
        # no target term, so their loss and gradient stay finite and are zeroed
        # by the label mask. ``target_lane`` is -1 on rows whose target is not
        # on this rank.
        ignored = labels == IGNORE_INDEX
        safe_labels = torch.where(ignored, 0, labels)
        out_of_range = (
            ignored
            | (safe_labels < vocab_start)
            | (safe_labels >= vocab_start + local_vocab_size)
        )
        local_labels = torch.where(out_of_range, 0, safe_labels - vocab_start)
        target_lane = torch.where(out_of_range, -1, local_labels)
        # The TP all-reduce selects the owner rank's shifted target logit.
        local_target_shifted = (
            logits.gather(-1, local_labels.unsqueeze(-1)).squeeze(-1).float()
            - local_max[:, 0]
        )
        target_shifted = torch.where(out_of_range, 0, local_target_shifted)
        target_shifted = funcol.all_reduce(
            target_shifted.unsqueeze(-1),
            reduceOp=dist.ReduceOp.SUM.name,
            group=tp_group,
        )

        # Sum of exp over the non-target classes only. The owner rank drops the
        # target term before summing: once the target dominates, summing it in
        # and subtracting it afterwards loses the small terms to rounding
        # against its exp(0) = 1.
        is_target = _target_mask(target_lane, local_vocab_size)
        exp_shifted = torch.exp(logits.float() - local_max)
        non_target_sumexp = torch.where(is_target, 0, exp_shifted).sum(
            dim=-1, keepdim=True
        )
        non_target_sumexp = funcol.all_reduce(
            non_target_sumexp, reduceOp=dist.ReduceOp.SUM.name, group=tp_group
        )

        # NLL = softplus(logsumexp_{j != t}(z_j) - z_t), which stays accurate
        # when log_softmax at the target rounds to 0. With no non-target mass,
        # log(0) = -inf and the NLL is softplus(-inf) = 0.
        nll = F.softplus(torch.log(non_target_sumexp) - target_shifted)
        result = torch.where(ignored, 0, nll.squeeze(-1))

        # The CE gradient is softmax - onehot. The owner rank's target slot gets
        # p_t - 1, formed by the formula that rounds once: p_t - 1 cancels once
        # the target dominates, so use -p_non_target from _HIGH_PROB_THRESHOLD
        # on.
        target_exp = torch.exp(target_shifted)
        sum_exp = non_target_sumexp + torch.where(ignored.unsqueeze(-1), 0, target_exp)
        target_prob = target_exp / sum_exp
        target_grad = torch.where(
            target_prob < _HIGH_PROB_THRESHOLD,
            target_prob - 1,
            -non_target_sumexp / sum_exp,
        )

        ctx.save_for_backward(
            labels, logits, target_lane, local_max, sum_exp, target_grad
        )
        ctx.logits_dtype = logits_dtype
        ctx.reduction = reduction
        if reduction == "none":
            return result
        return result.sum()

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        (
            labels,
            logits,
            target_lane,
            local_max,
            sum_exp,
            target_grad,
        ) = ctx.saved_tensors
        probs = torch.exp(logits.float() - local_max) / sum_exp
        is_target = _target_mask(target_lane, logits.shape[-1])
        grad_logits = torch.where(is_target, target_grad, probs)

        # reduction="none" gives a per-token ``[T]`` upstream grad; unsqueeze to
        # ``[T, 1]`` to broadcast over the local vocab. "sum" gives the scalar
        # loss grad, which broadcasts as-is.
        if ctx.reduction == "none":
            grad_output = grad_output.unsqueeze(-1)
        grad_output = torch.where(
            (labels != IGNORE_INDEX).unsqueeze(-1), grad_output, 0
        )
        # Scale the gradient in place; it has no other consumer.
        grad_logits.mul_(grad_output)
        return grad_logits.to(ctx.logits_dtype), None, None, None, None


class CrossEntropyLoss(Loss[torch.Tensor]):
    """Cross-entropy loss with sum reduction for token-based normalization."""

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        global_vocab_size: int | None = None
        """Full vocabulary size, needed for spmd_types loss-parallel CE."""

    def __init__(self, config: Config):
        self._loss_fn: LossFunction = cross_entropy_loss
        self.global_vocab_size = config.global_vocab_size

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LossResult:
        del kwargs
        loss = self._loss_fn(pred, labels, global_vocab_size=self.global_vocab_size)
        # loss: V->P, annotate global_loss_token_counts
        if current_spmd_mesh() is not None:
            spmd.assert_type(loss, {"dp": spmd.P, "cp": spmd.P})
            if global_loss_token_counts is not None:
                spmd.assert_type(
                    global_loss_token_counts,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
                )
        if global_loss_token_counts is not None:
            loss = loss / global_loss_token_counts
        return loss, {}


@local_compile("loss", batch_invariant=False)
def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MSE loss with sum reduction for Transformer models training."""
    return torch.nn.functional.mse_loss(
        pred.float(), labels.float().detach(), reduction="sum"
    )


class MSELoss(Loss[torch.Tensor]):
    """MSE loss with sum reduction for Transformer models training (e.g. Flux)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        pass

    def __init__(self, config: Config):
        del config
        self._loss_fn: LossFunction = mse_loss

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LossResult:
        del kwargs
        loss = self._loss_fn(pred, labels)
        # loss: V->P, annotate global_loss_token_counts
        if current_spmd_mesh() is not None:
            spmd.assert_type(loss, {"dp": spmd.P, "cp": spmd.P})
            if global_loss_token_counts is not None:
                spmd.assert_type(
                    global_loss_token_counts,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.I},
                )
        if global_loss_token_counts is not None:
            loss = loss / global_loss_token_counts
        return loss, {}


def combine_loss_metrics(
    current: dict[str, torch.Tensor],
    values: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Merge one loss call's metrics into ``current`` (in place) and return it.

    Mean/fraction metrics must already be normalized by the global valid-token
    count, so summing the contributions of sequence chunks, microbatches and
    ranks gives the global value. Keys ending in ``/max`` or ``/min`` take the
    max or min. Any other suffix is rejected because its reduction is unknown.
    """
    for key, value in values.items():
        previous = current.get(key)
        if previous is None:
            current[key] = value
        elif key.endswith(("/mean", "/frac", "_mean", "_frac", "/per_byte")):
            current[key] = previous + value
        elif key.endswith("/max"):
            current[key] = torch.maximum(previous, value)
        elif key.endswith("/min"):
            current[key] = torch.minimum(previous, value)
        else:
            raise ValueError(
                f"Do not know how to reduce metric '{key}'. "
                "Use a /mean, /frac, _mean, _frac, /per_byte, /max, or /min suffix."
            )
    return current


@dataclass(kw_only=True, slots=True)
class WeightedLoss:
    """One term of a ``MixedLoss``: ``weight * loss``."""

    weight: float
    loss: Loss.Config


def loss_name(config: Loss.Config) -> str:
    """The label a loss reports under: its ``name``, else its ``loss_type``."""
    if config.name is not None:
        return config.name
    for loss_type, loss_cls in LOSS_TYPES.items():
        if type(config) is loss_cls.Config:
            return loss_type
    owner = type(config)._owner
    return owner.__name__ if owner is not None else type(config).__qualname__


class MixedLoss(Loss[LossInput]):
    """Weighted sum of losses: ``sum_i weight_i * loss_i``. Itself a ``Loss``,
    so mixes nest and can be used wherever a loss is expected.

    Every term gets the same ``pred``, ``labels``, ``global_loss_token_counts``
    and extra ``loss_inputs``. Metrics report each term's unweighted value as
    ``<name>/mean`` (``<name>/per_byte`` for byte-normalized terms) and re-key the term's own metrics under ``<name>/``, so a
    nested mix reports ``<outer>/<inner>/mean``.

    Use it inside ``ChunkedLossWrapper`` (as its ``loss_fn``), never around it:
    the wrapper runs backward per chunk, so weights applied outside it would
    not reach the gradients.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        losses: list[WeightedLoss]
        """Weighted terms. Sibling names must be unique."""

        def __post_init__(self):
            if not self.losses:
                raise ValueError("MixedLoss needs at least one loss term.")
            names = []
            for term in self.losses:
                if not isinstance(term, WeightedLoss):
                    raise TypeError(
                        f"MixedLoss terms must be WeightedLoss, got {type(term).__name__}."
                    )
                if not math.isfinite(term.weight):
                    raise ValueError(f"MixedLoss weight must be finite, got {term.weight}.")
                if isinstance(term.loss, ChunkedLossWrapper.Config):
                    raise ValueError(
                        "ChunkedLossWrapper cannot be a MixedLoss term: it runs "
                        "backward internally, so the weight would not apply. Put "
                        "the MixedLoss inside ChunkedLossWrapper instead."
                    )
                names.append(loss_name(term.loss))
            duplicates = sorted({n for n in names if names.count(n) > 1})
            if duplicates:
                raise ValueError(
                    f"MixedLoss term names must be unique; repeated: {duplicates}. "
                    "Set a distinct 'name' on each."
                )

    def __init__(self, config: Config):
        self.terms: list[tuple[str, float, Callable[..., LossResult]]] = [
            (loss_name(term.loss), term.weight, term.loss.build())
            for term in config.losses
        ]

    def __call__(
        self,
        pred: LossInput,
        labels: LossInput,
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> LossResult:
        total: torch.Tensor | None = None
        metrics: dict[str, torch.Tensor] = {}
        for name, weight, loss_fn in self.terms:
            loss, term_metrics = loss_fn(
                pred, labels, global_loss_token_counts, **loss_inputs
            )
            weighted = loss * weight
            total = weighted if total is None else total + weighted
            suffix = "per_byte" if loss_fn.normalizer == "bytes" else "mean"
            metrics[f"{name}/{suffix}"] = loss.detach()
            for key, value in term_metrics.items():
                metrics[f"{name}/{key}"] = value
        assert total is not None
        return total, metrics


class BitsPerByteLoss(Loss[torch.Tensor]):
    """Cross-entropy in bits, to be divided by the UTF-8 bytes of the text the
    labels cover: exact bits per byte, comparable across tokenizers.

    The loss returns the summed bits; the validator divides by the byte count
    it reads off the labels with ``BaseTokenizer.token_byte_lengths``. Only
    for validation (``validator.losses``): training normalizes by tokens.
    """

    normalizer = "bytes"

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        global_vocab_size: int | None = None
        """Full vocabulary size, needed for spmd_types loss-parallel CE."""

    def __init__(self, config: Config):
        self.global_vocab_size = config.global_vocab_size

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_loss_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LossResult:
        del kwargs
        if global_loss_token_counts is not None:
            raise ValueError(
                "bits_per_byte is normalized by text bytes, which only the "
                "validator counts; use it in validator.losses, not for training."
            )
        nats = cross_entropy_loss(
            pred, labels, global_vocab_size=self.global_vocab_size
        )
        return nats / math.log(2), {}


def uses_byte_normalizer(config: Loss.Config) -> bool:
    """Whether a loss config, or any term of a mixed one, is byte-normalized."""
    if isinstance(config, MixedLoss.Config):
        return any(uses_byte_normalizer(term.loss) for term in config.losses)
    owner = type(config)._owner
    return owner is not None and getattr(owner, "normalizer", "tokens") == "bytes"


class _VocabParallelEntropy(torch.autograd.Function):
    """Exact per-token entropy from vocab-sharded logits without a gather."""

    @staticmethod
    def spmd_typecheck(
        result: torch.Tensor,
        *,
        logits: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> None:
        """SPMD type: logits S(-1)@TP -> entropy I@TP."""
        spmd.assert_type(logits, {tp_group: spmd.S(logits.dim() - 1)})
        spmd.assert_local_type_like(
            result,
            logits,
            {tp_group: spmd.I},  # pyrefly: ignore [bad-argument-type]
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        logits: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        del ctx
        logits = logits.float()

        local_max = torch.amax(logits, dim=-1, keepdim=True)
        global_max = funcol.all_reduce(
            local_max, reduceOp=dist.ReduceOp.MAX.name, group=tp_group
        )

        shifted = logits - global_max
        shifted_exp = torch.exp(shifted)
        local_sumexp = shifted_exp.sum(dim=-1)
        # Avoid 0 * -inf for finite distributions with masked logits while
        # preserving NaNs for invalid distributions such as all -inf logits.
        shifted_weighted = torch.where(
            torch.isneginf(shifted),
            torch.zeros_like(shifted),
            shifted_exp * shifted,
        )
        local_weighted_sum = shifted_weighted.sum(dim=-1)
        global_stats = funcol.all_reduce(
            torch.stack((local_sumexp, local_weighted_sum)),
            reduceOp=dist.ReduceOp.SUM.name,
            group=tp_group,
        )
        sumexp, weighted_sum = global_stats.unbind()
        return torch.log(sumexp) - weighted_sum / sumexp


@local_compile("loss", batch_invariant=False)
def compute_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_parallel_group: dist.ProcessGroup | None,
    return_entropy: bool = False,
    global_vocab_size: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-token logprobs from ``logits[T, V]`` and ``labels[T]``.

    When ``return_entropy`` is set, also returns per-token Shannon entropy
    ``H(p) = logsumexp(logits) - sum(softmax(logits) * logits)``, with shape
    ``[T]``. ``vocab_parallel_group`` explicitly describes the logits layout:
    a process group means that logits contain a local vocabulary shard, while
    ``None`` means they contain the full vocabulary. Batch-invariant mode
    gathers shards so trainer and vLLM generator perform the same operation
    sequence. Otherwise, statistics are computed directly from the shards.
    Entropy is a metric only, so it is computed under ``no_grad``: it never
    contributes gradient and must not build an autograd graph over the logits
    softmax.

    Returns ``logprobs`` when ``return_entropy`` is False, else
    ``(logprobs, entropy)``.
    """
    if vocab_parallel_group is not None:
        if global_vocab_size is None:
            raise ValueError(
                "global_vocab_size is required for vocab-parallel policy statistics"
            )
        if not is_in_batch_invariant_mode():
            logprobs = -_LossParallelCrossEntropy.apply(
                logits,
                labels,
                vocab_parallel_group,
                global_vocab_size,
                "none",
            )
            if not return_entropy:
                return logprobs
            with torch.no_grad():
                entropy = _VocabParallelEntropy.apply(
                    logits,
                    vocab_parallel_group,
                )
            return logprobs, entropy

        # The model returns a plain local vocab shard. Labels are global token
        # ids, so batch-invariant cross_entropy needs full-vocab logits.
        # dst=I, not R: the vocab all-gather's grad is the replicated upstream
        # grad sliced back to this rank's vocab shard (I's backward), not an
        # all-reduce (R's backward), which would over-count by the TP degree.
        logits = spmd.redistribute(
            logits,
            vocab_parallel_group,
            src=spmd.S(-1),
            dst=spmd.I,
        )

    # Single bf16->fp32 upcast, reused by both logprobs and (optionally) entropy.
    logits = logits.float()
    logprobs = -F.cross_entropy(
        logits,
        labels,
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    if not return_entropy:
        return logprobs
    with torch.no_grad():
        log_normalizer = torch.logsumexp(logits, dim=-1)
        # Match _VocabParallelEntropy's zero contribution for masked logits.
        weighted_logits = (torch.softmax(logits, dim=-1) * logits).nan_to_num_(
            nan=0.0, posinf=torch.inf, neginf=-torch.inf
        )
        entropy = log_normalizer - weighted_logits.sum(dim=-1)
        entropy.masked_fill_(~torch.isfinite(log_normalizer), torch.nan)
    return logprobs, entropy


class _DecoderOutputGradientBackProp(torch.autograd.Function):
    """Bridges chunked lm_head backward with decoder backward via autograd.

    Forward takes hidden states (connected to the decoder graph), their
    accumulated gradients from chunked lm_head backward, and the loss value.
    Returns a detached loss with this Function as its grad_fn.

    Backward returns each accumulated gradient for its corresponding hidden
    state. Autograd then propagates them through the decoder layers
    automatically -- no explicit hidden_states.backward() needed.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, num_predictions: int, *args: torch.Tensor) -> torch.Tensor:
        # args = (*model_outputs, *accumulated_grads, total_loss), with N
        # model outputs followed by N corresponding accumulated gradients.
        if len(args) != 2 * num_predictions + 1:
            raise ValueError(
                "Chunked-loss autograd bridge expected "
                f"{2 * num_predictions + 1} tensor arguments for "
                f"{num_predictions} predictions, got {len(args)}."
            )
        ctx.num_predictions = num_predictions
        ctx.save_for_backward(*args[num_predictions : 2 * num_predictions])
        return args[-1].detach()

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        # Return each accumulated gradient for its corresponding hidden state.
        # Autograd then propagates them through the hidden states' existing
        # decoder graph -- equivalent to
        # torch.autograd.backward(hidden_states, accumulated_grads), but expressed
        # as return values so autograd handles the traversal in a single pass
        # (no "backward through graph twice" error).
        # Note: this is not safe if downstream accidentally runs tensor ops after
        # the loss returns, which would produce a non-trivial grad_output that we
        # need to properly handle. The complicated part is that grad_output might
        # not be on the same device mesh as the accumulated gradients.
        del grad_output
        accumulated_grads = ctx.saved_tensors
        return (
            None,
            *accumulated_grads,
            *(None for _ in range(ctx.num_predictions)),
            None,
        )


class GradAccumulator:
    """Accumulates chunk gradients into a pre-allocated buffer.

    Instead of collecting chunk gradients in a list and concatenating at the end,
    this uses a pre-allocated buffer with in-place copies for better memory efficiency.

    Args:
        reference: Reference tensor for shape and device.
        num_chunks: Number of chunks that will be added.
        seq_dim: The sequence dimension along which chunks are accumulated.
        dtype: Dtype for the buffer.

    Usage:
        accumulator = GradAccumulator(hidden_states, num_chunks=4, dtype=torch.float32)
        for chunk_grad in chunk_grads:
            accumulator.add(chunk_grad)
        full_grad = accumulator.buffer
    """

    def __init__(
        self,
        reference: torch.Tensor,
        *,
        num_chunks: int,
        seq_dim: int = 0,
        dtype: torch.dtype,
    ):
        self.num_chunks = num_chunks
        self.seq_dim = seq_dim
        self._next_idx = 0
        self.buffer = torch.zeros_like(reference, dtype=dtype)

    def add(self, chunk_grad: torch.Tensor) -> None:
        """Add the next chunk gradient sequentially.

        Chunks must be added in order (0, 1, 2, ..., num_chunks - 1).
        """
        if self._next_idx >= self.num_chunks:
            raise ValueError(f"Already added {self.num_chunks} chunks, cannot add more")

        if chunk_grad.dtype != self.buffer.dtype:
            chunk_grad = chunk_grad.to(self.buffer.dtype)

        chunk_seq_len = chunk_grad.shape[self.seq_dim]
        start = self._next_idx * chunk_seq_len
        end = start + chunk_seq_len

        slices = [slice(None)] * self.buffer.ndim
        slices[self.seq_dim] = slice(start, end)
        self.buffer[tuple(slices)] = chunk_grad

        self._next_idx += 1


class ChunkedLossWrapper(Loss[LossInput]):
    """Chunked loss wrapper that splits the sequence dimension to reduce peak memory.

    Instead of materializing the full [T, V] logits tensor at once, this splits
    the hidden states into N chunks along the token dimension and computes
    lm_head + loss on each chunk sequentially. This reduces peak memory
    from O(T*V) to O(T/N*V).

    The inner ``loss_fn`` defaults to ``CrossEntropyLoss`` and is called once per
    chunk on logits from that chunk. ``pred`` and ``labels`` may be aligned
    tuples; their tensor or tuple structure is preserved when calling the inner
    loss. Additional per-token ``loss_inputs`` are chunked along the same
    sequence dimension and forwarded to the inner loss.

    The flow:
    1. Model forward with _skip_lm_head=True to get one or more hidden states [T, D]
    2. Split each hidden state and its labels into N chunks along seq dim
    3. Detach each hidden-state chunk at the lm_head boundary
    4. Disable FSDP reshard on lm_head across all outputs and chunks
    5. For each chunk: lm_head on each output -> loss_fn(logits, labels, gvt) -> backward()
    6. Assemble one full gradient [T, D] per output via GradAccumulator
    7. Backward through the decoder once with all accumulated gradients

    FSDP2 composability:
        The lm_head's FSDP reshard-after-forward and reshard-after-backward are
        temporarily disabled during the chunked loop so that the weight stays
        unsharded across all outputs and chunks (avoiding repeated all-gathers).
        Gradient synchronization remains disabled until the final chunk, so one
        reduce-scatter processes the accumulated lm_head parameter gradients.

    TP / SP composability:
        The root decoder norm emits hidden states that are replicated on the
        TP axis before chunking, so each chunk enters the lm_head as
        ``Replicate()`` input regardless of whether SP is enabled.

        When loss parallel is applied, each TP rank
        computes partial CE on its ``V/tp`` slice, with an internal
        all-reduce for the correct log-sum-exp.

    CP: Further chunks the local sequence dimension. Works out of the box.

    Compile: the inner ``loss_fn`` can be compiled independently; lm_head is not compiled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        num_chunks: int = 8
        """Number of chunks to split the sequence into."""

        loss_fn: Loss.Config = field(default_factory=CrossEntropyLoss.Config)
        """Loss applied to each chunk's logits."""

    def __init__(self, config: Config):
        self.num_chunks = config.num_chunks
        self.loss_fn: Callable[..., LossResult] = config.loss_fn.build()
        self.lm_head: nn.Module | None = None

    def set_lm_head(self, lm_head: nn.Module) -> None:
        """Set the lm_head module. Must be called before the first __call__."""
        self.lm_head = lm_head

    def __call__(
        self,
        pred: LossInput,
        labels: LossInput,
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute chunked loss.

        Every prediction represented by ``pred`` must come from model forward
        with ``_skip_lm_head=True``. Tensor inputs must be paired with tensor
        labels; tuple inputs must contain one labels tensor per prediction.

        Per-chunk backward runs when the decoder outputs or head require
        gradients, including training the lm_head on frozen decoder outputs.
        With frozen decoder outputs, ``no_grad`` runs chunked forward only.

        Returns a differentiable loss and metrics. When ``.backward()`` is called
        on the loss, it triggers backward through the decoder via a custom
        autograd Function. With frozen decoder outputs and a trainable head,
        head gradients are already accumulated. The returned loss is a leaf
        whose backward accumulates only its own gradient.
        """
        from torch.distributed._composable.fsdp import FSDPModule

        num_chunks = self.num_chunks
        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling ChunkedLossWrapper"
        if isinstance(pred, torch.Tensor) and isinstance(labels, torch.Tensor):
            is_multi_output = False
            pred = (pred,)
            labels = (labels,)
        elif (
            isinstance(pred, tuple)
            and isinstance(labels, tuple)
            and len(pred) > 0
            and len(pred) == len(labels)
        ):
            is_multi_output = True
        else:
            raise ValueError(
                "ChunkedLossWrapper requires either one prediction/labels "
                "tensor pair or non-empty aligned tuples."
            )
        decoder_requires_grad = pred[0].requires_grad
        if any(
            prediction.requires_grad != decoder_requires_grad for prediction in pred[1:]
        ):
            raise ValueError(
                "All chunked-loss predictions must agree on whether gradients "
                "are required."
            )
        run_backward = decoder_requires_grad or (
            torch.is_grad_enabled()
            and any(p.requires_grad for p in lm_head.parameters())
        )

        # Chunking operates on the local tensor. Equal chunk sizes match
        # GradAccumulator's sequential slice
        # writes, which use one chunk length for each write offset.
        def _chunk_local(t):
            seq_len = t.shape[0]
            torch._check(
                seq_len % num_chunks == 0,
                lambda: "ChunkedLossWrapper sequence length must be divisible by num_chunks",
            )
            chunk_len = seq_len // num_chunks
            return tuple(
                c.contiguous() for c in torch.split(t, [chunk_len] * num_chunks, dim=0)
            )

        with spmd.local():
            # ``detach`` + ``requires_grad_`` makes each chunk a leaf so it
            # accumulates ``.grad`` for ``GradAccumulator``.
            hidden_state_chunks_per_output = tuple(
                tuple(
                    chunk.detach().requires_grad_(decoder_requires_grad)
                    for chunk in _chunk_local(hidden_state)
                )
                for hidden_state in pred
            )
            label_chunks_per_output = tuple(_chunk_local(label) for label in labels)
            input_chunks = {
                key: _chunk_local(value) if isinstance(value, torch.Tensor) else value
                for key, value in loss_inputs.items()
            }
            grad_accumulators = (
                tuple(
                    GradAccumulator(
                        hidden_state,
                        num_chunks=num_chunks,
                        dtype=hidden_state.dtype,
                    )
                    for hidden_state in pred
                )
                if decoder_requires_grad
                else ()
            )

            total_loss = pred[0].new_zeros((), dtype=torch.float32)
            if spmd.is_type_checking():
                total_loss = spmd.mutate_type(
                    total_loss,
                    src=spmd.R,
                    dst={"dp": spmd.P, "cp": spmd.P, "tp": spmd.I},
                )
            metrics: dict[str, torch.Tensor] = {}

            fsdp_enabled = isinstance(lm_head, FSDPModule)
            # Disable FSDP reshard on lm_head to keep its weight unsharded across
            # all outputs and chunks, avoiding repeated all-gathers. Coalesce
            # gradient synchronization into one reduce-scatter at the final chunk
            # by disabling it for chunks 0..N-2.
            if fsdp_enabled:
                lm_head.set_reshard_after_forward(False)
                lm_head.set_reshard_after_backward(False)
                lm_head.set_requires_gradient_sync(False, recurse=False)
                # An implicit unshard stores an all-gather event in FSDP's shared
                # all_gather_state for the next FSDP module to consume. Since
                # lm_head is the final FSDP forward in this loop, eager warmup
                # leaves that state uncleared, and CUDA graph capture cannot wait
                # on its eager event. Explicitly unshard while FSDP is idle to
                # avoid populating the shared state.
                with spmd.no_typecheck():
                    lm_head.unshard()

            for chunk_index in range(num_chunks):
                if fsdp_enabled and chunk_index == num_chunks - 1:
                    lm_head.set_requires_gradient_sync(  # pyrefly: ignore[not-callable]
                        True, recurse=False
                    )

                h_chunks = tuple(
                    chunks[chunk_index] for chunks in hidden_state_chunks_per_output
                )
                label_chunks: LossInput = tuple(
                    chunks[chunk_index] for chunks in label_chunks_per_output
                )
                loss_inputs = {
                    key: chunks[chunk_index] if isinstance(chunks, tuple) else chunks
                    for key, chunks in input_chunks.items()
                }
                # TODO: compile lm_head together with loss_fn (only loss_fn is
                # compiled today): frees the fp32 dlogits right after the split, 1.2 GiB per
                # Qwen3-8B chunk. Blocked: compiling HiMidLoLinear rounds grad_weight to bf16
                # (https://github.com/pytorch/pytorch/pull/197381). With FSDP2, fullgraph also
                # fails at lm_head's hooks, which can't be traced.
                logits: LossInput = tuple(lm_head(h_chunk) for h_chunk in h_chunks)
                if not is_multi_output:
                    logits = logits[0]
                    label_chunks = label_chunks[0]
                chunk_loss, chunk_metrics = self.loss_fn(
                    logits,
                    label_chunks,
                    global_loss_token_counts,
                    **loss_inputs,
                )
                # Free logits before backward.
                del logits
                metrics = self._combine_chunk_metrics(metrics, chunk_metrics)
                total_loss = total_loss + chunk_loss.detach()

                if run_backward:
                    with spmd.no_typecheck():
                        chunk_loss.backward()
                        if decoder_requires_grad:
                            for h_chunk, grad_accumulator in zip(
                                h_chunks, grad_accumulators, strict=True
                            ):
                                assert h_chunk.grad is not None
                                grad_accumulator.add(h_chunk.grad)
                                h_chunk.grad = None

            if fsdp_enabled:
                lm_head.set_reshard_after_forward(True)
                lm_head.set_reshard_after_backward(True)
                lm_head.reshard()
            if not decoder_requires_grad:
                # Scaling this leaf cannot rescale already accumulated head gradients.
                return total_loss.requires_grad_(run_backward), metrics

            accumulated_grads = tuple(
                grad_accumulator.buffer for grad_accumulator in grad_accumulators
            )

        with spmd.no_typecheck():
            loss = self._gradient_backprop(
                pred,
                accumulated_grads,
                total_loss,
            )
        return loss, metrics

    @staticmethod
    def _combine_chunk_metrics(
        current: dict[str, torch.Tensor],
        values: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Combine metrics from one sequence chunk into the local accumulator.

        See ``combine_loss_metrics``. The trainer still performs the cross-rank
        loss-mesh reduction on the returned metrics.
        """
        return combine_loss_metrics(current, values)

    def _gradient_backprop(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        accumulated_grads: tuple[torch.Tensor, ...],
        total_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Return a differentiable loss via _DecoderOutputGradientBackProp.
        When ``.backward()`` is called (by the trainer or PP schedule),
        autograd calls ``_DecoderOutputGradientBackProp.backward`` which
        returns each accumulated gradient for its corresponding hidden state,
        propagating through the decoder. Subclasses override to swap in a
        different autograd Function.
        """
        return _DecoderOutputGradientBackProp.apply(
            len(hidden_states),
            *hidden_states,
            *accumulated_grads,
            total_loss,
        )


# Loss types that recipes can name in a loss spec. ChunkedLossWrapper is not
# here: it is the outer wrapper that owns the lm_head, not a composable term.
LOSS_TYPES: dict[str, type[Loss]] = {
    "cross_entropy": CrossEntropyLoss,
    "mse": MSELoss,
    "mixed": MixedLoss,
    "bits_per_byte": BitsPerByteLoss,
}


def loss_config_from_spec(spec: Mapping[str, Any], *, path: str = "loss") -> Loss.Config:
    """Build a loss config from a dict spec: ``{"loss_type": <LOSS_TYPES key>,
    <config fields>...}``. Unknown or missing fields raise.

    A ``"mixed"`` spec lists its terms as ``{"weight": float, "loss": <spec>}``,
    recursively::

        {"loss_type": "mixed", "losses": [
            {"weight": 1.0, "loss": {"loss_type": "cross_entropy"}},
            {"weight": 0.1, "loss": {"loss_type": "mse", "name": "aux"}},
        ]}
    """
    if not isinstance(spec, Mapping):
        raise TypeError(f"{path}: a loss spec must be a dict, got {type(spec).__name__}.")
    fields_spec = dict(spec)
    loss_type = fields_spec.pop("loss_type", None)
    if loss_type not in LOSS_TYPES:
        problem = "is missing 'loss_type'" if loss_type is None else f"has unknown loss_type {loss_type!r}"
        raise ValueError(f"{path} {problem}; known types: {sorted(LOSS_TYPES)}.")
    config_cls = LOSS_TYPES[loss_type].Config
    allowed = {
        f.name: f
        for f in dataclasses.fields(config_cls)
        if f.init and not f.name.startswith("_")
    }
    unknown = sorted(set(fields_spec) - set(allowed))
    if unknown:
        raise ValueError(
            f"{path} ({loss_type!r}) has unknown fields {unknown}; "
            f"allowed: {sorted(allowed)}."
        )
    missing = sorted(
        name
        for name, f in allowed.items()
        if name not in fields_spec
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    )
    if missing:
        raise ValueError(f"{path} ({loss_type!r}) is missing required fields {missing}.")
    if config_cls is MixedLoss.Config:
        terms = fields_spec["losses"]
        if not isinstance(terms, (list, tuple)):
            raise TypeError(f"{path}.losses must be a list of {{'weight', 'loss'}} dicts.")
        fields_spec["losses"] = [
            _weighted_loss_from_spec(term, path=f"{path}.losses[{i}]")
            for i, term in enumerate(terms)
        ]
    return config_cls(**fields_spec)


def _weighted_loss_from_spec(spec: Any, *, path: str) -> WeightedLoss:
    if not isinstance(spec, Mapping):
        raise TypeError(f"{path}: a mixed-loss term must be a dict, got {type(spec).__name__}.")
    keys = set(spec)
    if keys != {"weight", "loss"}:
        raise ValueError(
            f"{path} must have exactly the keys ['loss', 'weight']; "
            f"missing {sorted({'weight', 'loss'} - keys)}, unknown {sorted(keys - {'weight', 'loss'})}."
        )
    weight = spec["weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise TypeError(f"{path}.weight must be a number, got {weight!r}.")
    return WeightedLoss(
        weight=float(weight),
        loss=loss_config_from_spec(spec["loss"], path=f"{path}.loss"),
    )
