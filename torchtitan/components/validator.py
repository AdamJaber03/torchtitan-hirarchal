# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import torch
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.components.loss import (
    ChunkedLossWrapper,
    combine_loss_metrics,
    IGNORE_INDEX,
    loss_config_from_spec,
    LossFunction,
    uses_byte_normalizer,
)
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Configurable
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext, utils as dist_utils
from torchtitan.distributed.accelerator_graph import (
    accelerator_graph_manager,
    accelerator_graphs_supported,
    wrap_with_accelerator_graph,
)
from torchtitan.distributed.fsdp import post_validate_fsdp_for_accelerator_graph
from torchtitan.distributed.pipeline_parallel import (
    prepare_pipeline_schedule_for_accelerator_graph,
)
from torchtitan.models.base.model import BaseModel
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.tools import utils


logger = logging.getLogger(__name__)


class BaseValidator(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        freq: int = 10
        """Frequency of validation"""

        def __post_init__(self) -> None:
            if self.freq <= 0:
                raise ValueError(
                    f"validation frequency must be positive, got {self.freq}"
                )

    def __init__(
        self,
        config: Config,
        **kwargs,
    ):
        self.config = config

    def validate(self, model_parts: list[BaseModel], step: int) -> None:
        raise NotImplementedError("validate method not implemented")

    def should_validate(self, step: int) -> bool:
        return step == 1 or step % self.config.freq == 0


class Validator(BaseValidator):
    # TODO: Simplify validation by limiting its options and matching the training
    # forward workload. See Tianyu's review:
    # https://github.com/pytorch/torchtitan/pull/5241#discussion_r4301298181
    """
    Simple validator focused on correctness and integration.

    Args:
        config: Validator.Config configuration
        parallelism: ParallelismConfig configuration
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Tokenizer
        parallelism_context: Parallel dimensions
        loss_fn: Loss function to use for validation
        metrics_processor: Metrics processor
        pp_schedule: Pipeline schedule (optional)
        pp_has_first_stage: Whether this rank has the first PP stage (optional)
        pp_has_last_stage: Whether this rank has the last PP stage (optional)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseValidator.Config):
        steps: int = -1
        """
        Number of validation steps. -1 consumes the finite dataset once
        (dataloader repeat=False). Ranks then stop independently, so this
        requires data-parallel degree 1; otherwise validation collectives hang.
        Use a positive count when DP > 1 so every rank runs the same number of
        steps with repeat=True.
        """

        dataloader: BaseDataLoader.Config = field(
            default_factory=lambda: GrainDataLoader.Config(
                dataset=ConcatThenSplitPackingConfig(
                    dataset=DATASETS["c4_validation"],
                ),
                repeat=False,
            )
        )
        """DataLoader configuration for validation"""

        losses: list[dict[str, Any]] | None = None
        """Named losses to score instead of the training loss, as loss specs
        (see ``loss_config_from_spec``), e.g.
        ``[{"loss_type": "cross_entropy", "name": "ce"}]``. Each is reported as
        ``validation_metrics/<name>`` from one forward pass. ``None`` scores the
        training loss (and, for a ``MixedLoss``, also each of its terms)."""

        def __post_init__(self):
            BaseValidator.Config.__post_init__(self)
            if self.losses is not None:
                # Fail at config time on bad specs or repeated names.
                self._named_losses_config()
            if not (self.steps > 0 or self.steps == -1):
                raise ValueError(
                    f"validation steps must be positive or -1, got {self.steps}"
                )

        def _named_losses_config(self):
            """The named losses as one weight-1 ``MixedLoss`` config."""
            if not self.losses:
                raise ValueError("validator.losses must be None or a non-empty list.")
            return loss_config_from_spec(
                {
                    "loss_type": "mixed",
                    "losses": [{"weight": 1.0, "loss": spec} for spec in self.losses],
                },
                path="validator.losses",
            )

    # TODO: improve the constructor signature
    def __init__(
        self,
        config: Config,
        *,
        parallelism: ParallelismConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallelism_context: ParallelismContext,
        loss_fn: LossFunction,
        metrics_processor: MetricsProcessor,
        seq_len: int,
        num_tokens_per_microbatch: int,
        accelerator_graph_enabled: bool = False,
        cuda_graph_annotation_config: dict[str, Any] | None = None,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
        **kwargs,
    ):
        super().__init__(config=config)
        self.parallelism = parallelism
        self.tokenizer = tokenizer
        self.parallelism_context = parallelism_context
        self.loss_fn = loss_fn
        # With named losses, score them (and only them) with their own wrapper
        # that shares the training wrapper's lm_head and chunking.
        self.named_losses = config.losses is not None
        # Byte length of each token id, when a named loss is normalized by the
        # UTF-8 bytes its labels cover (bits_per_byte).
        self._token_byte_lengths: torch.Tensor | None = None
        if self.named_losses:
            if parallelism_context.pp_enabled:
                raise ValueError(
                    "validator.losses is not supported with pipeline parallelism."
                )
            named = config._named_losses_config()
            if uses_byte_normalizer(named):
                self._token_byte_lengths = torch.tensor(
                    tokenizer.token_byte_lengths(), dtype=torch.int64
                )
            if isinstance(loss_fn, ChunkedLossWrapper):
                assert loss_fn.lm_head is not None, "lm_head must be set first"
                wrapper = ChunkedLossWrapper.Config(
                    num_chunks=loss_fn.num_chunks, loss_fn=named
                ).build()
                wrapper.set_lm_head(loss_fn.lm_head)
                self.loss_fn = wrapper
            else:
                self.loss_fn = named.build()
        # A bounded validation run repeats data; steps=-1 consumes one finite pass.
        self.dl_config = replace(config.dataloader, repeat=config.steps != -1)
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        if config.steps == -1 and self.dp_world_size > 1:
            raise ValueError(
                "validation.steps=-1 runs one finite pass (dataloader "
                "repeat=False). With data-parallel degree > 1, ranks can exhaust "
                "at different steps and hang on validation collectives. Got "
                f"dp_world_size={self.dp_world_size}. Set validation.steps to a "
                "positive count so every rank runs the same number of steps, "
                "or run with data-parallel degree 1."
            )
        self.seq_len = seq_len
        self.num_tokens_per_microbatch = num_tokens_per_microbatch
        self.metrics_processor = metrics_processor
        self.pp_schedule = pp_schedule
        self.pp_has_first_stage = pp_has_first_stage
        self.pp_has_last_stage = pp_has_last_stage
        self._run_eval = None
        self._cuda_graph_annotation_config = cuda_graph_annotation_config
        self._accelerator_graph_enabled = (
            accelerator_graph_enabled and accelerator_graphs_supported()
        )
        if self._accelerator_graph_enabled:
            # Keep training and optimizer eager until eval captures.
            accelerator_graph_manager.capture_allowed = False
        elif accelerator_graph_enabled:
            logger.warning(
                "Graph capture is unavailable on this accelerator; "
                "using eager validation."
            )

    def should_validate(self, step: int) -> bool:
        """Return whether validation is scheduled or eval capture is pending.

        With eval accelerator graphs enabled, validate after each training update until
        eval captures. These passes warm up eval and capture it before training capture.
        """
        return super().should_validate(step) or (
            self._accelerator_graph_enabled
            and not accelerator_graph_manager.capture_allowed
        )

    def _init_validate_fn(self, model_parts):
        evaluate = partial(self._validate_body, model_parts)
        if not self._accelerator_graph_enabled:
            return evaluate

        if self._run_eval is None:

            def prepare_for_capture(prepared) -> None:
                """Prepare eval PP metadata outside capture.

                Training changes the schedule mode after eval warmup. Rebuilding
                metadata can synchronize with the CPU, which capture does not allow.
                """
                assert self.pp_schedule is not None
                inputs, labels, kwargs = prepared[0]
                with self.parallelism_context.activate_spmd():
                    prepare_pipeline_schedule_for_accelerator_graph(
                        self.pp_schedule,
                        args=(inputs,) if self.pp_has_first_stage else (),
                        kwargs=kwargs,
                        target=labels if self.pp_has_last_stage else None,
                        loss_kwargs={},
                        for_eval=True,
                    )

            # Keep the first validation pass eager, including all its groups.
            self._run_eval = wrap_with_accelerator_graph(
                evaluate,
                num_warmup_iterations=1,
                prepare_for_capture=prepare_for_capture
                if self.parallelism_context.pp_enabled
                else None,
                name="validation",
                annotation_config=self._cuda_graph_annotation_config,
            )
        elif (
            not accelerator_graph_manager.capture_allowed
            and accelerator_graph_manager.all_wrappers_warmed_up
        ):
            accelerator_graph_manager.capture_allowed = True
        return self._run_eval

    def _validate_body(self, model_parts, prepared):
        with self.parallelism_context.activate_spmd():
            if self.parallelism_context.pp_enabled:
                assert self.pp_schedule is not None
                losses = [] if self.pp_has_last_stage else None
                self.pp_schedule.eval(
                    arg_mbs=[(inputs,) for inputs, _, _ in prepared]
                    if self.pp_has_first_stage
                    else None,
                    kwarg_mbs=[kwargs for _, _, kwargs in prepared],
                    target_mbs=[labels for _, labels, _ in prepared]
                    if self.pp_has_last_stage
                    else None,
                    losses=losses,
                    return_outputs=False,
                )
                metrics: dict[str, torch.Tensor] = {}
                if self.pp_has_last_stage:
                    assert losses is not None
                    loss_sum = torch.stack(losses).sum().to(utils.device_type)
                else:
                    loss_sum = torch.full((1,), -1.0, device=utils.device_type)
                # Eval has no backward pass to release stage outputs.
                if isinstance(self.pp_schedule, PipelineScheduleSingle):
                    self.pp_schedule._stage.clear_runtime_states()
                elif isinstance(self.pp_schedule, PipelineScheduleMulti):
                    for stage in self.pp_schedule._stages:
                        stage.clear_runtime_states()
            else:
                assert len(prepared) == len(model_parts) == 1
                inputs, labels, kwargs = prepared[0]
                predictions = model_parts[0](inputs, **kwargs)
                loss_sum, metrics = self.loss_fn(predictions, labels)
                del predictions
            post_validate_fsdp_for_accelerator_graph(model_parts)
        return loss_sum.detach(), {k: v.detach() for k, v in metrics.items()}

    def _count_label_bytes(self, labels: torch.Tensor) -> torch.Tensor:
        """UTF-8 bytes of the text covered by the valid labels."""
        assert self._token_byte_lengths is not None
        if not isinstance(labels, torch.Tensor):
            raise ValueError("bits_per_byte needs a single labels tensor.")
        table = self._token_byte_lengths
        if table.device != labels.device:
            table = self._token_byte_lengths = table.to(labels.device)
        valid = labels != IGNORE_INDEX
        return torch.where(valid, table[labels.clamp(min=0)], 0).sum()

    def _reduce_metrics(
        self,
        metrics: dict[str, torch.Tensor],
        num_global_valid_tokens: int,
        num_global_label_bytes: int | None = None,
    ) -> dict[str, float]:
        """Reduce summed per-rank loss metrics to global values.

        Validation passes no token counts to the loss, so ``/mean`` and
        ``/frac`` values are sums: they are summed over the loss mesh and
        divided by the global valid tokens (``/per_byte``: by the global label
        bytes). ``<name>/mean`` and ``<name>/per_byte`` are reported as
        ``<name>``. ``/max`` and ``/min`` keys are reduced as such.
        """
        parallelism_context = self.parallelism_context
        loss_mesh = (
            parallelism_context.get_optional_mesh("loss")
            if parallelism_context.dp_cp_enabled
            else None
        )
        reduced: dict[str, float] = {}
        for key in sorted(metrics):  # same order on every rank for the collectives
            value = metrics[key]
            if key.endswith("/max"):
                out = dist_utils.dist_max(value, loss_mesh) if loss_mesh else value.item()
            elif key.endswith("/min"):
                out = -dist_utils.dist_max(-value, loss_mesh) if loss_mesh else value.item()
            else:
                total = dist_utils.dist_sum(value, loss_mesh) if loss_mesh else value.item()
                if key.endswith("/per_byte"):
                    if not num_global_label_bytes:
                        raise ValueError(
                            f"{key} needs the label byte count, which is zero or "
                            "was not counted."
                        )
                    out = total / num_global_label_bytes
                else:
                    out = total / num_global_valid_tokens
            name = key
            for suffix in ("/mean", "/per_byte"):
                if key.endswith(suffix):
                    name = key[: -len(suffix)]
            reduced[name] = float(out)
        return reduced

    @sl.log_trace_span("eval")
    @torch.no_grad()
    def validate(
        self,
        model_parts: list[BaseModel],
        step: int,
    ) -> None:
        modes = [
            (module, module.training) for m in model_parts for module in m.modules()
        ]
        try:
            for model in model_parts:
                model.eval()
            sl.add_step_tag("eval")
            self.metrics_processor.reset()

            parallelism_context = self.parallelism_context

            accumulated_loss: torch.Tensor | None = None
            accumulated_metrics: dict[str, torch.Tensor] = {}
            evaluate = None
            device_type = utils.device_type
            local_label_bytes = torch.zeros((), dtype=torch.int64, device=device_type)
            total_global_valid_tokens = torch.zeros(
                (), dtype=torch.int64, device=device_type
            )
            num_steps = 0
            num_pp_microbatches = (
                self.parallelism.num_pp_microbatches
                if parallelism_context.pp_enabled
                else 1
            )

            validation_dataloader = self.dl_config.build(
                dp_world_size=self.dp_world_size,
                dp_rank=self.dp_rank,
                tokenizer=self.tokenizer,
                max_context_length=self.seq_len,
                num_tokens_per_microbatch=self.num_tokens_per_microbatch,
            )

            validation_iterator = iter(
                iterate_and_close_dataloader(validation_dataloader)
            )
            while True:
                # pyrefly: ignore [missing-attribute, unsupported-operation]
                if self.config.steps != -1 and num_steps >= self.config.steps:
                    break

                try:
                    microbatch_group = []
                    local_loss_token_count = torch.zeros((), dtype=torch.int64)
                    for _ in range(num_pp_microbatches):
                        microbatch = next(validation_iterator)
                        local_loss_token_count.add_(
                            microbatch.loss_token_counts.reshape(-1)[0]
                        )
                        self.metrics_processor.ntokens_since_last_log += (
                            microbatch.labels.numel()
                        )
                        input_dict = microbatch.to_input_dict(device_type)
                        microbatch_group.append(input_dict)
                except StopIteration:
                    break

                # All-reduce token count across DP ranks while keeping it on device.
                local_loss_token_count = local_loss_token_count.to(device_type)
                if parallelism_context.dp_enabled:
                    dp_mesh = parallelism_context.get_mesh("dp")
                    global_valid_tokens = dist_utils.dist_sum_tensor(
                        local_loss_token_count, dp_mesh, None
                    )
                else:
                    global_valid_tokens = local_loss_token_count

                prepared = []
                for input_dict in microbatch_group:
                    with parallelism_context.activate_spmd():
                        inputs, labels, kwargs, _ = model_parts[0].preprocess_inputs(
                            input_dict,
                            parallelism_context=parallelism_context,
                            parallelism=self.parallelism,
                            max_num_documents=self.dl_config.max_num_documents,
                            max_context_length=self.seq_len,
                        )
                    prepared.append((inputs, labels, kwargs))
                    if self._token_byte_lengths is not None:
                        local_label_bytes.add_(self._count_label_bytes(labels))
                if evaluate is None:
                    evaluate = self._init_validate_fn(model_parts)
                loss_sum, step_metrics = evaluate(prepared)

                if accumulated_loss is None:
                    accumulated_loss = loss_sum.clone()
                else:
                    accumulated_loss.add_(loss_sum)
                # Clone: under accelerator graphs the outputs alias graph buffers.
                combine_loss_metrics(
                    accumulated_metrics,
                    {k: v.clone() for k, v in step_metrics.items()},
                )
                total_global_valid_tokens.add_(global_valid_tokens)
                num_steps += 1

            if accumulated_loss is None:
                raise ValueError(
                    "Validation ran zero batches on this rank. This happens when the "
                    "validation dataset supplies fewer than num_tokens_per_microbatch "
                    "tokens on this rank, because concat-then-split packing drops "
                    "partially filled batches. Decrease "
                    "training.num_tokens_per_microbatch_per_dp_rank or use a larger "
                    "validation dataset."
                )
            num_global_valid_tokens = int(total_global_valid_tokens.item())
            if num_global_valid_tokens == 0:
                raise ValueError(
                    "Validation ran on zero valid tokens; cannot compute an average "
                    "validation loss. Ensure the validation batches contain unmasked "
                    "labels."
                )
            if parallelism_context.dp_cp_enabled:
                global_loss_sum = dist_utils.dist_sum(
                    accumulated_loss, parallelism_context.get_optional_mesh("loss")
                )
            else:
                global_loss_sum = float(accumulated_loss.item())
            global_avg_loss = global_loss_sum / num_global_valid_tokens
            num_global_label_bytes = None
            if self._token_byte_lengths is not None:
                num_global_label_bytes = int(
                    dist_utils.dist_sum(
                        local_label_bytes, parallelism_context.get_optional_mesh("loss")
                    )
                    if parallelism_context.dp_cp_enabled
                    else local_label_bytes.item()
                )
            loss_metrics = self._reduce_metrics(
                accumulated_metrics, num_global_valid_tokens, num_global_label_bytes
            )

            self.metrics_processor.log_validation(
                # Named losses replace the training loss, whose weight-1 sum
                # over them means nothing on its own.
                loss=None if self.named_losses else global_avg_loss,
                step=step,
                loss_metrics=loss_metrics,
            )
        finally:
            for module, training in modes:
                module.training = training


def iterate_and_close_dataloader(
    dataloader: BaseDataLoader,
) -> Iterator[TrainingMicrobatch]:
    """Close a temporary dataloader when its consumer stops iterating."""
    try:
        yield from dataloader
    finally:
        dataloader.close()
