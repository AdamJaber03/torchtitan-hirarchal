"""Measure Qwen3 and byte stream tokens per UTF-8 text byte on FineWeb-Edu train.

Samples every ``--stride``-th train document of the first ``--num_files`` files
in the order the training recipes read them, and tokenizes them as
``TextProcessor`` does (BOS and EOS added). The ratio of the two stream lengths
sets how many bytes SpaceByte must train on to see the same text as a Qwen3 run
of a given token count.

Usage:
    python scripts/qwen3_bytes_per_token.py --num_files 720 --stride 10
"""

import argparse
import math
import os
from multiprocessing import Pool

import pyarrow.parquet as pq

from torchtitan.components.tokenizer import ByteTokenizer, HuggingFaceTokenizer
from torchtitan_recipes.data import local_fineweb_edu

_FINEWEB_EDU_DIR = os.path.expanduser("~/data/fineweb-edu-ar-nizarun/en")
_HF_ASSETS = "./assets/hf/Qwen3-0.6B"


def _count(args) -> list[tuple[int, int, int]]:
    path, stride = args
    qwen = HuggingFaceTokenizer(tokenizer_path=_HF_ASSETS)
    byte = ByteTokenizer()
    out = []
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    for text in texts[::stride]:
        out.append(
            (
                len(text.encode("utf-8")),
                len(qwen.encode(text, add_bos=True, add_eos=True)),
                len(byte.encode(text, add_bos=True, add_eos=True)),
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_files", type=int, default=720)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    dataset = local_fineweb_edu(_FINEWEB_EDU_DIR, split="train")
    paths = dataset.source.source.load_dataset_kwargs["data_files"]["train"]
    with Pool(args.workers) as pool:
        per_file = pool.map(
            _count, [(p, args.stride) for p in paths[: args.num_files]]
        )

    docs = [d for f in per_file for d in f]
    text = sum(d[0] for d in docs)
    qwen = sum(d[1] for d in docs)
    byte = sum(d[2] for d in docs)
    ratio = byte / qwen
    # Standard error of the ratio of sums, by the delta method over documents.
    mean_q, mean_b = qwen / len(docs), byte / len(docs)
    var = sum((b - ratio * q) ** 2 for _, q, b in docs) / (len(docs) - 1)
    stderr = math.sqrt(var / len(docs)) / mean_q
    # The same ratio per file, to see the spread across crawl dumps.
    file_ratios = sorted(
        sum(d[2] for d in f) / sum(d[1] for d in f) for f in per_file if f
    )
    print(f"docs={len(docs)}  text bytes={text}  mean byte stream/doc={mean_b:.0f}")
    print(f"qwen3 stream tokens per text byte = {qwen / text:.6f}")
    print(f"byte  stream tokens per text byte = {byte / text:.6f}")
    print(f"byte stream tokens per qwen3 token = {ratio:.6f} +- {stderr:.6f}")
    print(
        f"per-file ratio: min={file_ratios[0]:.4f} "
        f"median={file_ratios[len(file_ratios) // 2]:.4f} max={file_ratios[-1]:.4f}"
    )


if __name__ == "__main__":
    main()
