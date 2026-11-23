# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure the data statistics that SpaceByte's config and BPB comparison use.

1. Bytes per patch on packed C4 microbatches, to set
   ``HierarchyLevel.capacity_tokens_per_patch``: report the overflow (bytes
   whose patch gets no global slot) and the padding waste for candidate values.
2. Label tokens per UTF-8 text byte for the Qwen3 tokenizer and for bytes, to
   convert each model's per-label loss to bits per byte.

Usage:
    python scripts/spacebyte_data_stats.py --num_docs 20000
"""

import argparse
import math

import numpy as np
import torch
import pyarrow.parquet as pq
from datasets import load_dataset

from torchtitan.components.tokenizer import ByteTokenizer, HuggingFaceTokenizer
from torchtitan.models.spacebyte.model import SpacelikeBoundary
from torchtitan_recipes.data import local_fineweb_edu


def _fineweb_edu_validation(data_dir: str):
    """Validation documents in the order the training recipes read files."""
    dataset = local_fineweb_edu(data_dir, split="validation")
    for path in dataset.source.source.load_dataset_kwargs["data_files"]["train"]:
        yield from pq.read_table(path, columns=["text"]).to_pylist()


def _packed_rows(docs, *, row_len: int, max_context_length: int):
    """Mimic ConcatThenSplit packing: concat docs, split into full rows,
    restart positions at each document and every ``max_context_length``."""
    tokens, positions = [], []
    for doc in docs:
        tokens += doc
        positions += [i % max_context_length for i in range(len(doc))]
        while len(tokens) >= row_len:
            yield torch.tensor(tokens[:row_len]), torch.tensor(positions[:row_len])
            tokens, positions = tokens[row_len:], positions[row_len:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=20000)
    parser.add_argument("--row_len", type=int, default=65536)
    parser.add_argument("--max_context_length", type=int, default=16384)
    parser.add_argument("--hf_assets_path", default="./assets/hf/Qwen3-0.6B")
    parser.add_argument(
        "--fineweb_edu_dir",
        default=None,
        help="Measure the FineWeb-Edu validation split under this directory "
        "files instead of C4 validation.",
    )
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=[4.0, 4.25, 4.5, 4.75, 5.0]
    )
    args = parser.parse_args()

    byte_tokenizer = ByteTokenizer()
    qwen_tokenizer = HuggingFaceTokenizer(tokenizer_path=args.hf_assets_path)
    if args.fineweb_edu_dir is None:
        samples = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    else:
        samples = _fineweb_edu_validation(args.fineweb_edu_dir)
    texts = [sample["text"] for _, sample in zip(range(args.num_docs), samples)]

    text_bytes = sum(len(t.encode("utf-8")) for t in texts)
    qwen_labels = sum(len(qwen_tokenizer.encode(t, add_eos=True)) for t in texts)
    byte_docs = [byte_tokenizer.encode(t, add_eos=True) for t in texts]
    byte_labels = sum(len(d) for d in byte_docs)
    print(f"docs={len(texts)}  text bytes={text_bytes}")
    print(f"qwen3 labels per text byte  = {qwen_labels / text_bytes:.5f}")
    print(f"qwen3 text bytes per token  = {text_bytes / qwen_labels:.4f}")
    print(f"byte  labels per text byte  = {byte_labels / text_bytes:.5f}")

    rule = SpacelikeBoundary.Config().build()
    patches_per_row = []
    for tokens, positions in _packed_rows(
        byte_docs, row_len=args.row_len, max_context_length=args.max_context_length
    ):
        patches_per_row.append(int((rule(tokens) | (positions == 0)).sum()))
    patches = np.array(patches_per_row)
    print(
        f"rows={len(patches)}  mean bytes/patch={args.row_len / patches.mean():.3f}  "
        f"patches/row: mean={patches.mean():.0f} p99={np.percentile(patches, 99):.0f} "
        f"max={patches.max()}"
    )
    for ratio in args.ratios:
        capacity = min(
            args.row_len, 128 * math.ceil(math.ceil(args.row_len / ratio) / 128)
        )
        overflow = np.clip(patches - capacity, 0, None)
        # Bytes in overflowed patches, approximating each patch by the mean size.
        overflow_bytes = overflow.sum() * args.row_len / patches.mean()
        print(
            f"ratio={ratio:<5} capacity={capacity:<6} "
            f"rows overflowing={np.mean(overflow > 0):.4f}  "
            f"bytes w/o global={overflow_bytes / (len(patches) * args.row_len):.5f}  "
            f"padding slots={1 - np.minimum(patches, capacity).mean() / capacity:.3f}"
        )


if __name__ == "__main__":
    main()
