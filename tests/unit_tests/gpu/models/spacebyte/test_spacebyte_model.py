# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end SpaceByte checks with real attention kernels.

The gather into patch slots and the add-back are easy to get subtly acausal
(a byte seeing a later patch) or to leak across packed documents; the loss
would still go down. These tests catch both, and that every block trains.
"""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from torchtitan.components.tokenizer import ByteTokenizer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.spacebyte import build_model_config

_DOCS = [
    "The first document talks about cats, and nothing else at all. " * 3,
    "A second one: x = f(y) + 2; return x * 3;\n" * 4,
    "Finally, the third document is here to check isolation. " * 3,
]


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSpaceByteModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        config = build_model_config(
            "debugmodel", parallelism=ParallelismConfig(), attn_backend="flex"
        )
        self.max_context_length = config.max_context_length
        with torch.device("cuda"):
            self.model = config.build()
        self.model.init_states(buffer_device=torch.device("cuda"))

        tokenizer = ByteTokenizer()
        tokens, positions, self.doc_spans = [], [], []
        for text in _DOCS:
            doc = tokenizer.encode(text, add_eos=True)
            self.doc_spans.append((len(tokens), len(tokens) + len(doc)))
            tokens += doc
            positions += list(range(len(doc)))
        num_padding = 1024 - len(tokens)
        assert num_padding > 0
        self.tokens = torch.tensor(tokens + [0] * num_padding, device="cuda")
        self.positions = torch.tensor(
            positions + list(range(num_padding)), device="cuda"
        )
        self.padding_mask = torch.arange(1024, device="cuda") >= len(tokens)

    def _logits(self, tokens: torch.Tensor) -> torch.Tensor:
        input_dict = {
            "input": tokens,
            "labels": tokens,
            "positions": self.positions,
            "padding_mask": self.padding_mask,
        }
        with patch(
            "torchtitan.models.spacebyte.model.annotate_input_spmd_types",
            lambda _ctx, d, _s: d,
        ):
            inputs, _, model_kwargs, _ = self.model.preprocess_inputs(
                input_dict,
                parallelism_context=None,
                parallelism=ParallelismConfig(),
                max_context_length=self.max_context_length,
            )
        return self.model(inputs, **model_kwargs)

    def test_causal_and_document_isolated(self):
        start, end = self.doc_spans[1]
        # Turn a letter in the middle of a word into a space: this moves patch
        # boundaries, so it exercises the patch tables, not just local layers.
        t = next(
            i
            for i in range(start + 40, end)
            if all(chr(self.tokens[j].item()).isalpha() for j in (i - 1, i, i + 1))
        )
        perturbed = self.tokens.clone()
        perturbed[t] = ord(" ")
        with torch.no_grad():
            logits = self._logits(self.tokens)
            logits_perturbed = self._logits(perturbed)

        tolerance = {"atol": 1e-5, "rtol": 1e-5}
        torch.testing.assert_close(logits[:t], logits_perturbed[:t], **tolerance)
        later_start, later_end = self.doc_spans[2]
        torch.testing.assert_close(
            logits[later_start:later_end],
            logits_perturbed[later_start:later_end],
            **tolerance,
        )
        # The perturbation does reach the rest of its own document.
        self.assertGreater((logits[t:end] - logits_perturbed[t:end]).abs().max(), 1e-3)

    def test_every_parameter_gets_gradient(self):
        logits = self._logits(self.tokens)
        labels = torch.roll(self.tokens, -1).masked_fill(self.padding_mask, -100)
        F.cross_entropy(logits, labels, ignore_index=-100).backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(param.grad.abs().max().item(), 0, name)


if __name__ == "__main__":
    unittest.main()
