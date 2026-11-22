# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SpaceByte: a hierarchical byte-level decoder built from Qwen3 blocks.

See ``README.md`` for the architecture and the comparison with Qwen3.

Shape suffix legend
(https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):
  T = tokens at the current hierarchy level (bytes at level 0),
  C = patch slots of the next level (fixed capacity per microbatch),
  D = model dimension of the current level.
"""

import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.config import Configurable
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    spmd_mesh_size,
)
from torchtitan.models.common.attention import (
    AttentionMetadata,
    AttentionMetadataMap,
    InnerAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.flops import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.tools.utils import round_up

__all__ = [
    "HierarchyLevel",
    "PatchLevelInputs",
    "SpaceByteModel",
    "SpacelikeBoundary",
]


class SpacelikeBoundary(Configurable):
    """SpaceByte's patch boundary rule (Slagle 2024, Sec. 2).

    A token is spacelike if it is not an ASCII letter, an ASCII digit, or a
    UTF-8 continuation byte. A patch ends at every spacelike token that does
    not follow another spacelike token, so a run like ``", "`` ends one patch.

    At a deeper level the rule sees the tokens at the previous level's patch
    ends (``' '``, ``'.'``, ``'\\n'``, ...), so rules for sentence or paragraph
    patches are sibling ``Configurable`` classes with the same call signature.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        del config

    def __call__(self, tokens_T: torch.Tensor) -> torch.Tensor:
        t = tokens_T
        spacelike_T = (
            (t < ord("0"))
            | ((t > ord("9")) & (t < ord("A")))
            | ((t > ord("Z")) & (t < ord("a")))
            | ((t > ord("z")) & (t < 0b1000_0000))
            | (t >= 0b1100_0000)
        )
        previous_T = torch.cat([spacelike_T.new_zeros(1), spacelike_T[:-1]])
        return spacelike_T & ~previous_T


@dataclass(kw_only=True, slots=True)
class HierarchyLevel:
    """One downsampling step: encoder layers, then a coarser level, then decoder layers.

    The layers of a ``SpaceByteModel`` are stored flat in execution order, and
    ``SpaceByteModel.Config.levels`` (outermost first) says how they split::

        enc[0] .. enc[n-1] | inner | dec[n-1] .. dec[0]
    """

    num_encoder_layers: int
    num_decoder_layers: int
    inner_dim: int
    """Model dimension of the next level. Activations are zero-padded in front
    on the way in and the last ``dim`` features are added back on the way out."""
    boundary: SpacelikeBoundary.Config
    capacity_tokens_per_patch: float
    """Sets the fixed number of patch slots per microbatch:
    ``C = round_up(ceil(T / capacity_tokens_per_patch), 128)``. Shapes must be
    static for compile and accelerator graphs, so the next level always runs on
    C slots. Set it a bit below the data's mean tokens per patch: unused slots
    are padding (wasted compute), and patches beyond C get no inner-level
    contribution (their tokens are still predicted by the local layers)."""

    def __post_init__(self) -> None:
        if self.capacity_tokens_per_patch < 1:
            raise ValueError("capacity_tokens_per_patch must be >= 1.")


class PatchLevelInputs(NamedTuple):
    """Index tables from level k (T tokens) to level k + 1 (C patch slots).

    They depend only on the input bytes, so ``preprocess_inputs`` builds them
    once per microbatch. Slot C (one past the end) is a dummy that marks
    "no patch" in ``token_to_patch_T``.
    """

    patch_index_C: torch.Tensor
    """Level-k index of the token that ends each patch (0 for unused slots)."""
    token_to_patch_T: torch.Tensor
    """Slot whose output is added to each level-k token, or C for none."""
    positions_C: torch.Tensor
    """Patch position within its document; unused slots restart at 0."""
    padding_mask_C: torch.Tensor
    """True for unused slots."""
    tokens_C: torch.Tensor
    """Level-k token at each patch end, for the next level's boundary rule."""
    attention_metadata: AttentionMetadataMap
    """Attention metadata of the level-(k + 1) layers."""


def _patch_capacity(num_tokens: int, tokens_per_patch: float) -> int:
    return min(num_tokens, round_up(math.ceil(num_tokens / tokens_per_patch), 128))


def _build_patch_level(
    boundary: SpacelikeBoundary,
    tokens_T: torch.Tensor,
    positions_T: torch.Tensor,
    padding_mask_T: torch.Tensor | None,
    *,
    capacity: int,
    max_context_length: int,
) -> PatchLevelInputs:
    """Select patch-end tokens into C slots without device-to-host syncs."""
    num_tokens = tokens_T.shape[0]
    device = tokens_T.device
    doc_start_T = positions_T == 0
    # A document's first token always ends a patch, so every document has a
    # patch and patch positions reset with it.
    is_patch_end_T = boundary(tokens_T) | doc_start_T
    if padding_mask_T is not None:
        is_patch_end_T &= ~padding_mask_T
    slot_T = torch.cumsum(is_patch_end_T, dim=0) - 1
    kept_T = is_patch_end_T & (slot_T < capacity)
    token_to_patch_T = torch.where(kept_T, slot_T, capacity)

    patch_index_C = torch.zeros(capacity + 1, dtype=torch.long, device=device)
    patch_index_C.scatter_(
        0, token_to_patch_T, torch.arange(num_tokens, device=device)
    )
    patch_index_C = patch_index_C[:capacity]

    num_valid = kept_T.sum()
    slot_C = torch.arange(capacity, device=device)
    valid_C = slot_C < num_valid
    patch_doc_start_C = doc_start_T[patch_index_C] & valid_C
    doc_start_slot_C = torch.cummax(
        torch.where(patch_doc_start_C, slot_C, 0), dim=0
    ).values
    # Unused slots form trailing padding segments (positions restart at 0, as
    # for data-side padding), so causal masking keeps real patches from
    # attending to them.
    positions_C = torch.where(
        valid_C,
        slot_C - doc_start_slot_C,
        (slot_C - num_valid) % max_context_length,
    )
    return PatchLevelInputs(
        patch_index_C=patch_index_C,
        token_to_patch_T=token_to_patch_T,
        positions_C=positions_C,
        padding_mask_C=~valid_C,
        tokens_C=tokens_T[patch_index_C],
        attention_metadata={},
    )


def _build_attention_metadata(
    layers: list[TransformerBlock.Config],
    positions: torch.Tensor,
    **kwargs: Any,
) -> AttentionMetadataMap:
    """Build one metadata entry per inner-attention backend used by ``layers``."""
    attention_metadata: dict[type[InnerAttention], AttentionMetadata] = {}
    for layer in layers:
        config = layer.attention.inner_attention
        backend = config._owner
        assert backend is not None and issubclass(backend, InnerAttention)
        if backend in attention_metadata:
            continue
        metadata = config.build_attention_metadata(positions, **kwargs)
        if metadata is not None:
            attention_metadata[backend] = metadata
    return attention_metadata


class SpaceByteModel(Decoder):
    """Byte-level decoder whose inner levels run only at patch ends.

    With one level this is SpaceByte: local byte layers, global layers on the
    tokens that end each word-like patch, then local byte layers again. The
    global output at a patch end is added back to that byte; the following
    local layers carry it forward to the next patch's bytes.
    """

    supports_pipeline_parallel = False

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        levels: list[HierarchyLevel]
        """Downsampling steps, outermost (bytes) first."""
        # Same regions as Qwen3Model. Without them the byte-level stacks run
        # SwiGLU and RoPE as separate eager kernels, 1.7x slower end to end.
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=[
                    "loss",
                    "fused_binary_activation",
                    "rope",
                    "fp32_to_bf16_split",
                ]
            )
        )

        def __post_init__(self) -> None:
            Decoder.Config.__post_init__(self)
            if not self.levels:
                raise ValueError("SpaceByteModel needs at least one level.")
            if self.enable_weight_tying:
                raise ValueError("SpaceByteModel does not tie embeddings.")
            depths = self.layer_depths()
            if depths.count(len(self.levels)) == 0:
                raise ValueError("The innermost level has no layers.")
            level_dims = [self.dim] + [level.inner_dim for level in self.levels]
            for i, (layer, depth) in enumerate(zip(self.layers, depths)):
                if layer.attention.dim != level_dims[depth]:
                    raise ValueError(
                        f"layers[{i}] has dim {layer.attention.dim}, but it runs "
                        f"at level {depth} with dim {level_dims[depth]}."
                    )
            for outer, inner in zip(level_dims, level_dims[1:]):
                if inner < outer:
                    raise ValueError("inner_dim must not shrink the model dim.")

        def layer_depths(self) -> list[int]:
            """Hierarchy depth of each entry of ``layers`` (0 = bytes)."""
            num_levels = len(self.levels)
            num_outer = sum(
                level.num_encoder_layers + level.num_decoder_layers
                for level in self.levels
            )
            num_inner = len(self.layers) - num_outer
            if num_inner < 0:
                raise ValueError(
                    f"levels use {num_outer} layers but only {len(self.layers)} exist."
                )
            depths = []
            for depth, level in enumerate(self.levels):
                depths += [depth] * level.num_encoder_layers
            depths += [num_levels] * num_inner
            for depth in reversed(range(num_levels)):
                depths += [depth] * self.levels[depth].num_decoder_layers
            return depths

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            """Training FLOPs per byte.

            A depth-d layer runs on ``prod(1 / capacity_tokens_per_patch)``
            slots per byte (counting padding slots, which are computed), with
            attention over a correspondingly shorter context.
            """
            assert isinstance(model, SpaceByteModel)
            nparams, active_nparams = get_nparams_and_active_nparams(
                model, modules_excluded_from_active_params=[model.layers]
            )
            flops = 6 * active_nparams  # norm and lm_head, run on every byte
            slots_per_byte = [1.0]
            for level in self.levels:
                slots_per_byte.append(
                    slots_per_byte[-1] / level.capacity_tokens_per_patch
                )
            for layer, layer_config, depth in zip(
                model.layers.values(), self.layers, self.layer_depths()
            ):
                attention = layer.attention
                layer_flops = 6 * sum(p.numel() for p in layer.parameters())
                layer_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=attention.head_dim,
                    v_head_dim=attention.head_dim,
                    seq_len=math.ceil(seq_len * slots_per_byte[depth]),
                    sliding_window_size=getattr(
                        layer_config.attention.inner_attention,
                        "sliding_window_size",
                        None,
                    ),
                )
                flops += layer_flops * slots_per_byte[depth]
            return nparams, int(flops)

    def __init__(self, config: Config):
        for axis in ("tp", "cp"):
            if spmd_mesh_size(axis) > 1:
                raise ValueError(f"SpaceByteModel does not support {axis.upper()}.")
        super().__init__(config)
        self.boundaries = [level.boundary.build() for level in config.levels]
        keys = list(self.layers.keys())
        depths = config.layer_depths()
        num_levels = len(config.levels)
        num_encoder = [level.num_encoder_layers for level in config.levels]
        start = 0
        self._encoder_keys: list[list[str]] = []
        for n in num_encoder:
            self._encoder_keys.append(keys[start : start + n])
            start += n
        num_inner = depths.count(num_levels)
        self._inner_keys = keys[start : start + num_inner]
        start += num_inner
        self._decoder_keys: list[list[str]] = [[] for _ in range(num_levels)]
        for depth in reversed(range(num_levels)):
            n = config.levels[depth].num_decoder_layers
            self._decoder_keys[depth] = keys[start : start + n]
            start += n

    def _run_layers(
        self,
        keys: list[str],
        h: torch.Tensor,
        attention_metadata: AttentionMetadataMap,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        for key in keys:
            layer = self.layers[key]
            h = layer(
                h,
                attention_metadata.get(type(layer.attention.inner_attention)),
                positions,
            )
        return h

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: AttentionMetadataMap,
        *,
        level_inputs: list[PatchLevelInputs],
        padding_mask: torch.Tensor | None = None,
    ):
        del padding_mask  # dense blocks ignore it; patch slots carry their own
        h_TD = self.tok_embeddings(tokens)

        # Going in: run each level's encoder, then gather its patch-end tokens
        # into the next level's slots, zero-padded in front to the next dim.
        residuals: list[torch.Tensor] = []
        level_positions, level_metadata = positions, attention_metadata
        for depth, level in enumerate(self.config.levels):
            h_TD = self._run_layers(
                self._encoder_keys[depth], h_TD, level_metadata, level_positions
            )
            residuals.append(h_TD)
            patches = level_inputs[depth]
            h_CD = h_TD[patches.patch_index_C]
            h_TD = F.pad(h_CD, (level.inner_dim - h_CD.shape[-1], 0))
            level_positions = patches.positions_C
            level_metadata = patches.attention_metadata

        h_TD = self._run_layers(self._inner_keys, h_TD, level_metadata, level_positions)

        # Going out: add each patch's last ``dim`` features to the token that
        # ended it (slot C is zeros, for tokens that end no patch), then run
        # the level's decoder.
        for depth in reversed(range(len(self.config.levels))):
            patches = level_inputs[depth]
            residual_TD = residuals.pop()
            dim = residual_TD.shape[-1]
            patch_out_CD = torch.cat([h_TD[:, -dim:], h_TD.new_zeros(1, dim)])
            h_TD = residual_TD + patch_out_CD[patches.token_to_patch_T]
            if depth == 0:
                level_positions, level_metadata = positions, attention_metadata
            else:
                level_positions = level_inputs[depth - 1].positions_C
                level_metadata = level_inputs[depth - 1].attention_metadata
            h_TD = self._run_layers(
                self._decoder_keys[depth], h_TD, level_metadata, level_positions
            )

        h_TD = self.norm(h_TD)
        if self._skip_lm_head:
            return h_TD
        return self.lm_head(h_TD)

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context: ParallelismContext,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        global_routing_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
        """Build per-level patch tables and attention metadata, then SPMD-wrap."""
        del kwargs, global_routing_token_counts
        loss_kwargs = input_dict.pop("loss_kwargs", {})
        tokens_T = input_dict["input"]
        positions_T = input_dict.get("positions")
        if positions_T is None:
            raise ValueError("SpaceByteModel requires per-document positions.")
        padding_mask_T = input_dict.get("padding_mask")
        if max_context_length is None:
            max_context_length = self.config.max_context_length

        depths = self.config.layer_depths()
        layers_at_depth = [
            [layer for layer, d in zip(self.config.layers, depths) if d == depth]
            for depth in range(len(self.config.levels) + 1)
        ]
        metadata_kwargs = {
            "max_num_documents": max_num_documents,
            "max_context_length": max_context_length,
        }
        with torch.no_grad():
            input_dict["attention_metadata"] = _build_attention_metadata(
                layers_at_depth[0],
                positions_T,
                padding_mask=padding_mask_T,
                **metadata_kwargs,
            )
            level_inputs = []
            for depth, (level, boundary) in enumerate(
                zip(self.config.levels, self.boundaries)
            ):
                patches = _build_patch_level(
                    boundary,
                    tokens_T,
                    positions_T,
                    padding_mask_T,
                    capacity=_patch_capacity(
                        tokens_T.shape[0], level.capacity_tokens_per_patch
                    ),
                    max_context_length=max_context_length,
                )
                patches = patches._replace(
                    attention_metadata=_build_attention_metadata(
                        layers_at_depth[depth + 1],
                        patches.positions_C,
                        padding_mask=patches.padding_mask_C,
                        **metadata_kwargs,
                    )
                )
                level_inputs.append(patches)
                tokens_T = patches.tokens_C
                positions_T = patches.positions_C
                padding_mask_T = patches.padding_mask_C
        input_dict["level_inputs"] = level_inputs

        input_shardings = decoder_input_sharding()
        input_shardings.update(dict.fromkeys(loss_kwargs, input_shardings["labels"]))
        input_dict.update(loss_kwargs)
        input_dict = annotate_input_spmd_types(
            parallelism_context, input_dict, input_shardings
        )
        loss_kwargs = {name: input_dict.pop(name) for name in loss_kwargs}
        inputs = input_dict.pop("input")
        labels = input_dict.pop("labels")
        return inputs, labels, input_dict, loss_kwargs
