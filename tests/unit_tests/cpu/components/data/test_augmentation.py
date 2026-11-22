# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TextProcessor augmentation hooks.

Augmentation bugs are silent: identical random choices on every rank, a resume
that replays different noise, or a moved EOS that merges packed documents all
still train. These tests pin the contract instead.
"""

from dataclasses import dataclass

import grain.python as grain
import numpy as np
import pytest
import torch

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    DatasetBuildContext,
    DatasetIterationPolicy,
    GrainDataLoader,
    SingleDatasetConfig,
    TextAugmentation,
    TextCollator,
    TokenAugmentation,
)
from torchtitan.components.data.text_datasets import TextProcessor


class _Tokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, add_bos=False, add_eos=False):
        tokens = [ord(char) % 250 + 10 for char in text]
        return [self.bos_id] * add_bos + tokens + [self.eos_id] * add_eos


_CONTEXT = DatasetBuildContext(
    tokenizer=_Tokenizer(),
    max_context_length=16,
    num_tokens_per_microbatch=32,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)


class RandomUppercase(TextAugmentation):
    @dataclass(kw_only=True, slots=True)
    class Config(TextAugmentation.Config):
        prob: float

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del context
        self.prob = config.prob

    def __call__(self, text, rng):
        flips = rng.random(len(text)) < self.prob
        return "".join(c.upper() if f else c for c, f in zip(text, flips))


class ShiftInteriorTokens(TokenAugmentation):
    @dataclass(kw_only=True, slots=True)
    class Config(TokenAugmentation.Config):
        shift: int

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del context
        self.shift = config.shift

    def __call__(self, token_ids, rng):
        del rng
        out = token_ids.copy()
        out[1:-1] += self.shift
        return out


class DropLastToken(TokenAugmentation):
    @dataclass(kw_only=True, slots=True)
    class Config(TokenAugmentation.Config):
        pass

    def __call__(self, token_ids, rng):
        del rng
        return token_ids[:-1]


@dataclass(frozen=True)
class _StreamingRows:
    rows: tuple[dict, ...]

    def build(self):
        return self

    def build_iter_dataset(self, *, dataset_iteration_policy: DatasetIterationPolicy):
        policy = dataset_iteration_policy
        dataset = grain.MapDataset.source(self.rows)[
            policy.dp_rank :: policy.dp_world_size
        ]
        if policy.repeat:
            dataset = dataset.repeat()
        return dataset.to_iter_dataset()


def _policy(**overrides) -> DatasetIterationPolicy:
    values = {
        "seed": 42,
        "shuffle": False,
        "repeat": False,
        "dp_rank": 0,
        "dp_world_size": 1,
        "streaming_shuffle_buffer_size": 4,
    }
    return DatasetIterationPolicy(**(values | overrides))


def _texts(dataset_config, **policy_overrides) -> list[list[int]]:
    dataset = dataset_config.build(
        context=_CONTEXT, dataset_iteration_policy=_policy(**policy_overrides)
    )
    return [sequence.input_ids.tolist() for sequence in dataset]


def _augmented_dataset(rows, **processor_fields) -> SingleDatasetConfig:
    return SingleDatasetConfig(
        source=_StreamingRows(rows=rows),
        processor=TextProcessor.Config(**processor_fields),
    )


def test_text_then_token_augmentations_apply_in_order():
    processor = TextProcessor.Config(
        text_augmentations=(RandomUppercase.Config(prob=1.0),),
        token_augmentations=(ShiftInteriorTokens.Config(shift=1000),),
    ).build(context=_CONTEXT)
    sequence = processor({"text": "abc"}, np.random.default_rng(0))

    expected = _Tokenizer().encode("ABC", add_bos=True, add_eos=True)
    expected[1:-1] = [t + 1000 for t in expected[1:-1]]
    assert sequence.input_ids.tolist() == expected[:-1]
    assert sequence.labels.tolist() == expected[1:]


def test_no_augmentations_matches_plain_tokenization():
    processor = TextProcessor.Config().build(context=_CONTEXT)
    sequence = processor({"text": "hello"}, np.random.default_rng(0))
    expected = _Tokenizer().encode("hello", add_bos=True, add_eos=True)
    assert sequence.input_ids.tolist() == expected[:-1]
    assert sequence.labels.tolist() == expected[1:]


def test_token_augmentation_must_keep_eos_last():
    processor = TextProcessor.Config(
        token_augmentations=(DropLastToken.Config(),)
    ).build(context=_CONTEXT)
    with pytest.raises(ValueError, match="BOS first and EOS last"):
        processor({"text": "hello"}, np.random.default_rng(0))


def test_augmentation_draws_are_reproducible_and_independent_across_streams():
    # Every rank (and mixture child) sees the same texts here, so any shared
    # random stream would show up as identical augmentations.
    rows = tuple({"text": "abcdefghijklmnopqrstuvwxyz" * 2} for _ in range(8))
    config = _augmented_dataset(
        rows, text_augmentations=(RandomUppercase.Config(prob=0.5),)
    )

    assert _texts(config, seed=7) == _texts(config, seed=7)
    rank0 = _texts(config, seed=7, dp_rank=0, dp_world_size=2)
    rank1 = _texts(config, seed=7, dp_rank=1, dp_world_size=2)
    assert rank0 != rank1
    # Mixture child 1 on rank 0 is seeded seed + 1; child 0 on rank 1 has
    # rank offset 1. They must not share a stream.
    child1_rank0 = _texts(config, seed=8, dp_rank=0, dp_world_size=2)
    assert child1_rank0 != rank1
    # Within one stream, each document gets its own draws.
    assert len(set(map(tuple, rank0))) == len(rank0)


def test_loader_resume_replays_identical_augmentations():
    rows = tuple({"text": f"document number {index} " * 2} for index in range(40))
    config = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=_augmented_dataset(
                rows,
                text_augmentations=(RandomUppercase.Config(prob=0.5),),
                token_augmentations=(ShiftInteriorTokens.Config(shift=300),),
            )
        ),
        collator=TextCollator.Config(),
        shuffle=False,
        repeat=True,
        num_prefetch_microbatches=1,
    )

    def build():
        return config.build(
            dp_world_size=1,
            dp_rank=0,
            tokenizer=_Tokenizer(),
            max_context_length=16,
            num_tokens_per_microbatch=32,
        )

    loader = build()
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    state = loader.state_dict()
    expected = [next(iterator) for _ in range(3)]

    restored = build()
    restored.load_state_dict(state)
    restored_iterator = iter(restored)
    for want in expected:
        got = next(restored_iterator)
        assert torch.equal(want.input, got.input)
        assert torch.equal(want.labels, got.labels)
