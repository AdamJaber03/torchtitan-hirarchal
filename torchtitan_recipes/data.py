# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset-specific defaults shared by training recipes."""

import dataclasses
import glob
import logging
import os
from dataclasses import dataclass
from typing import Literal

import grain.python as grain

from torchtitan.components.data import (
    DatasetIterationPolicy,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
    TextAugmentation,
    TokenAugmentation,
)
from torchtitan.components.data.text_datasets import TextProcessor
from torchtitan.observability.logging import warn_once


logger = logging.getLogger(__name__)


def c4_document_capacity(num_tokens_per_microbatch: int) -> int:
    """Estimate C4 document capacity from the per-rank token budget."""
    warn_once(
        logger,
        "C4 document capacity is an estimate. Tune dataloader.max_num_documents "
        "for your workload to balance attention overhead and packing padding.",
    )
    # Estimate 16 document slots per 8K tokens, with at least 16 for short batches.
    return max(16, (num_tokens_per_microbatch + 511) // 512)


def with_augmentations(
    dataset: SingleDatasetConfig,
    *,
    text: tuple[TextAugmentation.Config, ...] = (),
    token: tuple[TokenAugmentation.Config, ...] = (),
) -> SingleDatasetConfig:
    """Return ``dataset`` with these augmentations on its ``TextProcessor``."""
    if not isinstance(dataset.processor, TextProcessor.Config):
        raise ValueError(
            "augmentations are applied by TextProcessor, but the dataset's "
            f"processor is {type(dataset.processor).__name__}."
        )
    processor = dataclasses.replace(
        dataset.processor, text_augmentations=text, token_augmentations=token
    )
    return dataclasses.replace(dataset, processor=processor)


@dataclass(frozen=True)
class _FileShardedStreamingSource:
    """Streaming source that requires whole files per DP rank.

    Hugging Face streaming gives each rank a disjoint subset of files only when
    the file count is divisible by the DP degree; otherwise every rank parses
    every file and keeps one row in N. Raise instead of running that slowly.
    """

    source: HuggingFaceStreamingSource.Config
    num_files: int

    def build(self):
        return _FileShardedStream(self.source.build(), self.num_files)


class _FileShardedStream:
    def __init__(self, stream, num_files: int) -> None:
        self._stream = stream
        self._num_files = num_files

    def build_iter_dataset(
        self, *, dataset_iteration_policy: DatasetIterationPolicy
    ) -> grain.IterDataset:
        dp_world_size = dataset_iteration_policy.dp_world_size
        if self._num_files % dp_world_size != 0:
            raise ValueError(
                f"{self._num_files} files cannot be split evenly over "
                f"{dp_world_size} data-parallel ranks."
            )
        return self._stream.build_iter_dataset(
            dataset_iteration_policy=dataset_iteration_policy
        )


def local_fineweb_edu(
    data_dir: str, *, split: Literal["train", "validation"]
) -> SingleDatasetConfig:
    """FineWeb-Edu from ``<data_dir>/<split>/part-*.parquet`` (a "text" column).

    The files come pre-split and pre-shuffled (see BUILD.md next to the data):
    validation is 1 in 30 documents by a keyed content hash of the text, and
    documents are spread over the files in a fixed random order, so every file
    follows the same distribution. Only the "text" column is read.
    """
    split_dir = os.path.join(data_dir, split)
    paths = sorted(glob.glob(os.path.join(split_dir, "part-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no part-*.parquet files in {split_dir!r}")
    return SingleDatasetConfig(
        source=_FileShardedStreamingSource(
            source=HuggingFaceStreamingSource.Config(
                path="parquet",
                split="train",
                load_dataset_kwargs={
                    "data_files": {"train": paths},
                    "columns": ["text"],
                },
            ),
            num_files=len(paths),
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    )
