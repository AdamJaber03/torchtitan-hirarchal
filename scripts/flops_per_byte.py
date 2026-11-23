"""Data statistics for theoretical training FLOPs per text byte, Qwen3 vs SpaceByte.

FLOPs here are algorithmic, independent of kernels: training = 3 x forward,
forward = 2 x the multiply-accumulates of every matmul (linear layers, lm_head)
plus attention's QK^T and AV over the (query, key) pairs the masks allow
(causal, same document segment, sliding window for SpaceByte's local layers).
Norms, RoPE, activations, softmax and embedding lookups are ignored, and so are
SpaceByte's padding slots and kernel tile waste: they are library artifacts.

This script measures the data-dependent counts per text byte, packed as the
training recipes pack them (ConcatThenSplit; segments restart at each document
and every max_context_length; a row cut splits a segment):
  - Qwen3: tokens and causal attention pairs.
  - SpaceByte: bytes, patches, global attention pairs, and local attention
    pairs for several sliding windows.

Usage:
    python scripts/flops_per_byte.py --num_files 64 --docs_per_file 4000
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan_recipes.data import local_fineweb_edu

_FINEWEB_EDU_DIR = os.path.expanduser("~/data/fineweb-edu-ar-nizarun/en")
_HF_ASSETS = "./assets/hf/Qwen3-0.6B"
_WINDOWS = (64, 128, 256, 384, 400, 416, 432, 448, 464, 480, 496, 512, 768, 1024)

# Rows as the recipes build them.
_QWEN_ROW, _QWEN_CONTEXT = 16384, 4096
_BYTE_ROW, _BYTE_CONTEXT = 64512, 16384

def _segments(doc_lengths, *, row_len: int, max_context_length: int):
    """Yield (row_offset, length, starts_document) for each attention segment."""
    filled = 0
    for length in doc_lengths:
        pos = 0
        while pos < length:
            seg = min(length - pos, max_context_length - pos % max_context_length)
            seg = min(seg, row_len - filled)
            yield pos, seg
            pos += seg
            filled = (filled + seg) % row_len

def _window_pairs(n: int, w: int) -> int:
    """Pairs in a causal segment of n tokens with a window of w (self included)."""
    if n <= w:
        return n * (n + 1) // 2
    return w * (w + 1) // 2 + (n - w) * w

def _spacelike(b: np.ndarray) -> np.ndarray:
    return (
        (b < ord("0"))
        | ((b > ord("9")) & (b < ord("A")))
        | ((b > ord("Z")) & (b < ord("a")))
        | ((b > ord("z")) & (b < 0x80))
        | (b >= 0xC0)
    )

def _count(args) -> dict:
    path, docs_per_file = args
    qwen = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS)
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    texts = texts[:docs_per_file]
    out = {"text_bytes": 0, "qwen_tokens": 0, "qwen_pairs": 0, "bytes": 0,
           "patches": 0, "global_pairs": 0}
    out.update({f"local_pairs_{w}": 0 for w in _WINDOWS})

    qwen_lengths, byte_docs = [], []
    for text in texts:
        raw = text.encode("utf-8")
        out["text_bytes"] += len(raw)
        qwen_lengths.append(len(qwen.encode(text, add_eos=True)))
        # Byte stream: text bytes + EOS (id outside 0..255, spacelike).
        byte_docs.append(np.frombuffer(raw, dtype=np.uint8).astype(np.int32))

    out["qwen_tokens"] = sum(qwen_lengths)
    for _, seg in _segments(qwen_lengths, row_len=_QWEN_ROW,
                            max_context_length=_QWEN_CONTEXT):
        out["qwen_pairs"] += seg * (seg + 1) // 2

    byte_lengths = [len(d) + 1 for d in byte_docs]
    out["bytes"] = sum(byte_lengths)
    segs = _segments(byte_lengths, row_len=_BYTE_ROW,
                     max_context_length=_BYTE_CONTEXT)
    # Patch ends per document (EOS counts as spacelike), then per segment.
    for doc, length in zip(byte_docs, byte_lengths):
        spacelike = np.append(_spacelike(doc), True)
        prev = np.concatenate([[False], spacelike[:-1]])
        ends = spacelike & ~prev
        pos = 0
        while pos < length:
            _, seg = next(segs)
            # A patch also ends at each segment's first byte (the model's
            # rule at position 0), so every segment has a global slot there.
            patches = int(ends[pos + 1:pos + seg].sum()) + 1
            out["patches"] += patches
            out["global_pairs"] += patches * (patches + 1) // 2
            for w in _WINDOWS:
                out[f"local_pairs_{w}"] += _window_pairs(seg, w)
            pos += seg
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_files", type=int, default=64)
    parser.add_argument("--docs_per_file", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    dataset = local_fineweb_edu(_FINEWEB_EDU_DIR, split="train")
    paths = dataset.source.source.load_dataset_kwargs["data_files"]["train"]
    step = max(1, len(paths) // args.num_files)
    chosen = paths[::step][: args.num_files]
    with Pool(args.workers) as pool:
        parts = pool.map(_count, [(p, args.docs_per_file) for p in chosen])
    total = {k: sum(p[k] for p in parts) for k in parts[0]}
    per_byte = {k: v / total["text_bytes"] for k, v in total.items()}
    print(json.dumps({"total": total, "per_text_byte": per_byte}, indent=1))

if __name__ == "__main__":
    main()
