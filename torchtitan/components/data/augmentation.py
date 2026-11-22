# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-document data augmentations applied by ``TextProcessor``.

``TextProcessor`` runs its text augmentations on the raw text, tokenizes it with
BOS/EOS, then runs its token augmentations on the whole document before the
input/label shift. See ``README.md`` ("Augmentations") for the randomness and
checkpoint contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.config import Configurable

__all__ = ["TextAugmentation", "TokenAugmentation"]


class TextAugmentation(Configurable, ABC):
    """Transforms a document's text before tokenization."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del config, context

    @abstractmethod
    def __call__(self, text: str, rng: np.random.Generator) -> str:
        """Return the augmented text. Draw all randomness from ``rng``."""
        ...


class TokenAugmentation(Configurable, ABC):
    """Transforms a tokenized document, BOS and EOS included, before the shift.

    Inputs and labels both come from the returned ids, so a transform changes
    them consistently. The length may change. The BOS and EOS that
    ``TextProcessor`` added must stay first and last.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del config, context

    @abstractmethod
    def __call__(self, token_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return the augmented ids. Draw all randomness from ``rng``."""
        ...
