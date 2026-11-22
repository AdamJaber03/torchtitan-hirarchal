# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SpaceByte patch selection: the boundary rule and the index tables built by
``preprocess_inputs``. A wrong table silently mixes bytes across patches or
documents, so these check the tables against a direct Python reference."""

import dataclasses
import unittest
from unittest.mock import patch

import torch

from torchtitan.components.tokenizer import ByteTokenizer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.spacebyte import build_model_config
from torchtitan.models.spacebyte.model import SpacelikeBoundary


def _reference_is_spacelike(byte: int) -> bool:
    """Paper Sec. 2: not a letter, number, or UTF-8 continuation byte."""
    char = chr(byte)
    is_alnum = byte < 0x80 and (char.isascii() and char.isalnum())
    is_continuation = 0x80 <= byte < 0xC0
    return not (is_alnum or is_continuation)


def _reference_patch_ends(tokens: list[int], positions: list[int]) -> list[int]:
    ends = []
    for i, token in enumerate(tokens):
        spacelike = _reference_is_spacelike(token)
        previous = i > 0 and _reference_is_spacelike(tokens[i - 1])
        if positions[i] == 0 or (spacelike and not previous):
            ends.append(i)
    return ends


_TEXTS = [
    "the enemy!” he exclaimed. “Their capture must be prevented.",
    "where $q_1=q_2=\\dots=q_\\kappa$ and $V_1=V_2$. In this way,",
    "    exp += 2;\n\n    mbf[3] = exp;\n",
    "héllo wörld 中文字 ok",
]


class TestSpacelikeBoundary(unittest.TestCase):
    def test_matches_reference_rule(self):
        rule = SpacelikeBoundary.Config().build()
        for text in _TEXTS:
            tokens = ByteTokenizer().encode(text)
            got = rule(torch.tensor(tokens)).nonzero().flatten().tolist()
            # Without doc starts, position 0 is a boundary only if spacelike.
            expected = _reference_patch_ends(tokens, [1] * len(tokens))
            self.assertEqual(got, expected, text)


class TestPatchTables(unittest.TestCase):
    def setUp(self):
        config = build_model_config(
            "debugmodel", parallelism=ParallelismConfig(), attn_backend="varlen"
        )
        self.max_context_length = config.max_context_length
        with torch.device("meta"):
            self.model = config.build()

    def _microbatch(self, texts: list[str], num_padding: int):
        tokenizer = ByteTokenizer()
        tokens, positions = [], []
        for text in texts:
            doc = tokenizer.encode(text, add_eos=True)
            tokens += doc
            positions += list(range(len(doc)))
        padding = [False] * len(tokens) + [True] * num_padding
        tokens += [0] * num_padding
        positions += list(range(num_padding))
        return {
            "input": torch.tensor(tokens),
            "labels": torch.tensor(tokens),
            "positions": torch.tensor(positions),
            "padding_mask": torch.tensor(padding),
        }

    def _preprocess(self, input_dict):
        with patch(
            "torchtitan.models.spacebyte.model.annotate_input_spmd_types",
            lambda _ctx, d, _s: d,
        ):
            _, _, model_kwargs, _ = self.model.preprocess_inputs(
                input_dict,
                parallelism_context=None,
                parallelism=ParallelismConfig(),
                max_context_length=self.max_context_length,
            )
        (patches,) = model_kwargs["level_inputs"]
        return patches

    def _check(self, input_dict, patches, capacity_limit=None):
        tokens = input_dict["input"].tolist()
        positions = input_dict["positions"].tolist()
        padding = input_dict["padding_mask"].tolist()
        ends = [
            i
            for i in _reference_patch_ends(tokens, positions)
            if not padding[i]
        ]
        capacity = patches.patch_index_C.shape[0]
        kept = ends[: capacity_limit or capacity]
        num_kept = len(kept)

        self.assertEqual(patches.patch_index_C[:num_kept].tolist(), kept)
        self.assertTrue(patches.padding_mask_C[num_kept:].all())
        self.assertFalse(patches.padding_mask_C[:num_kept].any())
        # Every kept end maps to its slot; every other byte maps to the dummy.
        expected_slot = [capacity] * len(tokens)
        for slot, i in enumerate(kept):
            expected_slot[i] = slot
        self.assertEqual(patches.token_to_patch_T.tolist(), expected_slot)
        # Patch positions count patches within each document.
        doc_patch = []
        for i in kept:
            doc_patch.append(0 if positions[i] == 0 else doc_patch[-1] + 1)
        self.assertEqual(patches.positions_C[:num_kept].tolist(), doc_patch)
        # Unused slots start their own segment so real patches never see them.
        if num_kept < capacity:
            self.assertEqual(patches.positions_C[num_kept].item(), 0)
        self.assertEqual(
            patches.tokens_C[:num_kept].tolist(), [tokens[i] for i in kept]
        )

    def test_packed_documents_with_padding(self):
        input_dict = self._microbatch(_TEXTS, num_padding=37)
        patches = self._preprocess(dict(input_dict))
        self._check(input_dict, patches)
        # The global layers get metadata built from patch positions.
        self.assertEqual(len(patches.attention_metadata), 1)

    def test_overflow_keeps_first_patches(self):
        # 1 byte per patch: far more patch ends than the 128-slot capacity.
        input_dict = self._microbatch([" a" * 300], num_padding=0)
        level = self.model.config.levels[0]
        self.model.config.levels[0] = dataclasses.replace(
            level, capacity_tokens_per_patch=4.0
        )
        try:
            patches = self._preprocess(dict(input_dict))
        finally:
            self.model.config.levels[0] = level
        self.assertEqual(patches.patch_index_C.shape[0], 256)
        self._check(input_dict, patches, capacity_limit=256)
        self.assertFalse(patches.padding_mask_C.any())


if __name__ == "__main__":
    unittest.main()
