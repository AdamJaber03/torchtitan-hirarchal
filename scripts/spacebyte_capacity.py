"""Overflow vs padding for SpaceByte's patch capacity at a given microbatch size.

The global blocks run on C = round_up(ceil(T / capacity_tokens_per_patch), 128)
slots per T-byte microbatch. Patches past slot C get no global step: the bytes
from the (C+1)-th patch end to the microbatch's end are predicted from stale
global output. Slots past the microbatch's patch count are padding.

Packs train documents per file as ConcatThenSplit does (byte stream with EOS;
a patch also ends where positions restart: each document start and every
max_context_length bytes) and reports, per capacity:
  - microbatches that overflow,
  - bytes without a global slot (exact, as a fraction of all bytes),
  - padding slots (fraction of computed slots),
  - global-slot compute relative to the real patches (1.0 = no waste).

Usage:
    python scripts/spacebyte_capacity.py --num_files 120 --row_len 38464
"""

import argparse
import json
import math
import os
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

from torchtitan_recipes.data import local_fineweb_edu

_FINEWEB_EDU_DIR = os.path.expanduser("~/data/fineweb-edu-ar-nizarun/en")
_RATIOS = (4.5, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5)


def _spacelike(b: np.ndarray) -> np.ndarray:
    return (
        (b < ord("0"))
        | ((b > ord("9")) & (b < ord("A")))
        | ((b > ord("Z")) & (b < ord("a")))
        | ((b > ord("z")) & (b < 0x80))
        | (b >= 0xC0)
    )


def _rows(args) -> list[np.ndarray]:
    """Patch-end byte offsets of each full row packed from one file."""
    path, row_len, max_context_length = args
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    ends_stream, filled, rows = [], 0, []
    for text in texts:
        raw = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
        # EOS: outside 0..255, spacelike.
        spacelike = np.append(_spacelike(raw), True)
        ends = spacelike & ~np.concatenate([[False], spacelike[:-1]])
        ends[::max_context_length] = True  # positions restart
        ends_stream.append(ends)
        filled += len(ends)
        if filled >= row_len:
            stream = np.concatenate(ends_stream)
            while len(stream) >= row_len:
                rows.append(np.flatnonzero(stream[:row_len]))
                stream = stream[row_len:]
            ends_stream, filled = [stream], len(stream)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_files", type=int, default=120)
    parser.add_argument("--row_len", type=int, default=38464)
    parser.add_argument("--max_context_length", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=120)
    args = parser.parse_args()

    dataset = local_fineweb_edu(_FINEWEB_EDU_DIR, split="train")
    paths = dataset.source.source.load_dataset_kwargs["data_files"]["train"]
    chosen = paths[:: max(1, len(paths) // args.num_files)][: args.num_files]
    with Pool(args.workers) as pool:
        parts = pool.map(
            _rows, [(p, args.row_len, args.max_context_length) for p in chosen]
        )
    rows = [r for part in parts for r in part]
    counts = np.array([len(r) for r in rows])
    T = args.row_len
    out = {
        "rows": len(rows),
        "row_len": T,
        "bytes_per_patch": T / counts.mean(),
        "patches_per_row": {
            "mean": counts.mean(),
            "p99": float(np.percentile(counts, 99)),
            "p99.9": float(np.percentile(counts, 99.9)),
            "max": int(counts.max()),
        },
        "by_ratio": [],
    }
    for ratio in _RATIOS:
        C = min(T, 128 * math.ceil(math.ceil(T / ratio) / 128))
        lost = sum(T - r[C] for r in rows if len(r) > C)
        kept = np.minimum(counts, C)
        out["by_ratio"].append(
            {
                "ratio": ratio,
                "capacity": C,
                "rows_overflowing": float(np.mean(counts > C)),
                "bytes_without_global": lost / (len(rows) * T),
                "padding_slots": 1 - kept.mean() / C,
                "slot_compute_vs_real_patches": C / counts.mean(),
            }
        )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
