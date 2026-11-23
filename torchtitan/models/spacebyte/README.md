# SpaceByte

A byte-level decoder (Slagle 2024, [arXiv:2404.14408](https://arxiv.org/abs/2404.14408),
reference code [kjslag/spacebyte](https://github.com/kjslag/spacebyte)) built from
Qwen3 blocks. The `0.6B` flavor is meant to be compared directly with Qwen3-0.6B.

## Architecture

```
bytes  _ s p a c e _ i s _ a l l _
         |         |     |       |
         v         v     v       v
       embedding (256 -> 512)
       14 local Qwen3 blocks  (dim 512, sliding window 512 bytes)
         |         |     |       |      gather patch-end bytes into C slots,
         |         `-----+-------'      zero-pad in front 512 -> 1024
         |               v
         |     28 global Qwen3 blocks  (= Qwen3-0.6B: dim 1024, full causal
         |               |               attention over patches)
         |               v              last 512 dims added back at the
         +<--------------'              patch-end byte
       14 local Qwen3 blocks
       norm, lm_head (512 -> 256)
```

A **patch** ends at every *spacelike* byte (not an ASCII letter, digit or UTF-8
continuation byte) that does not follow another spacelike byte, and at each
document start. On English text a patch is roughly a word plus its trailing
space. The global blocks run once per patch, at its last byte, where the next
word's first byte is predicted.

### Static shapes: patch capacity

Compile and accelerator graphs need fixed shapes, so the global blocks always
run on `C = round_up(ceil(T / capacity_tokens_per_patch), 128)` slots for a
microbatch of `T` bytes. `preprocess_inputs` builds the gather and scatter
tables once per microbatch, sync-free:
- Unused slots are trailing padding segments that real patches never attend to.
- Patches beyond `C` contribute nothing globally. Their bytes are still
  predicted by the local blocks, and labels are untouched.

With `capacity_tokens_per_patch=5.0` and 65536-byte microbatches:

| Dataset | Bytes per patch | Bytes without a global slot | Padding slots |
|---|---|---|---|
| C4 (en) | 5.78 | 0.03% | 14% |
| FineWeb-Edu | 6.07 | 0.003% | 18% |

Measure with `scripts/spacebyte_data_stats.py` when you change the dataset.

### Code layout

All blocks are stored flat in `layers`, in execution order
(`enc | inner | dec`), so FSDP units, activation remat, init, and RoPE cache
sharing work exactly as for any `Decoder`. `SpaceByteModel.Config.levels` says
how `layers` splits.

A second level (for example sentence patches over word patches) means:
- another `HierarchyLevel` in `levels`, and its layers in `layers`;
- a boundary rule class next to `SpacelikeBoundary`. Its input at level k+1 is
  the bytes that ended level-k patches.

`forward` and `preprocess_inputs` already loop over levels.

## Comparison with Qwen3-0.6B

The paper compares models at equal training FLOPs and similar inference
FLOPs-per-byte, not at equal parameter count. Its 1B comparison gives the global
stack the subword model's width, uses `D_local = D/2`, `L_local = L_global`
split evenly, and a local window of `D_local`. Here:

| | Qwen3-0.6B | SpaceByte `0.6B` |
|---|---|---|
| Blocks | 28 x (1024, 16/8 heads, FFN 3072) | same 28 global blocks + 28 local (512, 4/2 heads, FFN 1536) |
| Vocabulary | 151936 (tied) | 256 (untied) |
| Parameters (total / non-embedding) | 596M / 440M | ~528M / ~528M |
| Context | 4096 tokens (~18.8K bytes) | 16384 bytes |
| Training FLOPs per byte (MFU convention, C4) | 6.4G / 4.586 = 1.39G | 1.60G (global blocks at 1/5 of bytes, padding slots included) |
| Measured on 1x L40S, flex, FineWeb-Edu | 96K bytes/s, 36% MFU | 58K bytes/s, 25% MFU |

Where a SpaceByte step goes (one L40S, 65536 bytes):
- Global blocks: 41% of GPU time. They run as efficiently as Qwen3's blocks.
- Local blocks: 58%. They do about as many matmul FLOPs as the global blocks
  but carry 2.5x the activations (28 x 65536 x 512 vs 28 x 13184 x 1024), so
  their norms, residual adds, RoPE and SwiGLU are memory-bound and make up 64%
  of their time.

Per byte, SpaceByte costs about 15% more FLOPs than Qwen3-0.6B. For a
compute-matched comparison (the paper's setup), give both runs the same
training FLOPs, not the same bytes.

### `0.6B-flops`: compute-matched flavor

`my_recipes.py` trains `0.6B-flops`, which costs the same training FLOPs per
text byte as Qwen3-0.6B, so equal text also means equal total compute. FLOPs
are counted algorithmically by `scripts/flops_per_byte.py`:
- training = 3 x forward;
- forward = 2 x the multiply-accumulates of every matmul, plus QK^T and AV
  over the (query, key) pairs the masks allow;
- norms, elementwise ops, padding slots and kernel tile waste are ignored.

On the nizarun English train split (0.21309 Qwen3 tokens and 0.16490 patches
per text byte):

| | Qwen3-0.6B | SpaceByte `0.6B` | SpaceByte `0.6B-flops` |
|---|---|---|---|
| Local layers | - | 14 + 14, window 512 | 9 + 9, window 432 |
| MFLOP per text byte | 893.5 | 1119.9 (1.2533x) | 893.6 (1.00008x) |

The global blocks are unchanged, still Qwen3-0.6B's. Re-measure when the data
changes.

`0.6B-flops` also uses patch capacity 5.5 instead of 5.0. On 38,464-byte
microbatches of this data, 0.004% of bytes lose their global slot and 10% of
slots are padding, against 19% at 5.0: about 5% faster
(`scripts/spacebyte_capacity.py`). Its RoPE uses `StaticShapeCosSinRoPE`; with
the dynamic-shape compile, activation recompute crashed at startup on some
launches.

Everything else is Qwen3: RMSNorm with eps 1e-6, QK-norm, CosSinRoPE with theta
1e6, SwiGLU, and std-0.02 init. Deliberate differences from the reference
implementation, made to keep Qwen3 as the only other variable:
- No learned absolute position embeddings; RoPE only.
- No `1/sqrt(d_in)` init with per-parameter learning-rate multipliers.
- SwiGLU instead of ReLU.
- Packed documents with document masking instead of BOS-aligned crops.

### Bits per byte

Both models log loss per predicted label. Convert with each tokenizer's labels
per text byte on the validation data (`bpb = loss / ln 2 * ratio`):

| Validation data | SpaceByte ratio | Qwen3 ratio |
|---|---|---|
| C4 (en) | 1.00046 | 0.21806 (4.586 bytes/token) |
| FineWeb-Edu | 1.00020 | 0.21295 (4.696 bytes/token) |

`my_recipes.py` trains both models on FineWeb-Edu and gives them validators
that score the same validation bytes.

## Status

- Parallelism: FSDP/HSDP only. TP, CP and PP raise.
- Attention: `flex` (any GPU) and `varlen` (FA4; SM 9.x/10.x).
- No HF checkpoint, so there is no state-dict adapter.
- Tests:
  - `tests/unit_tests/cpu/models/spacebyte`: boundary rule and patch tables
    against a Python reference.
  - `tests/unit_tests/gpu/models/spacebyte`: causality, document isolation,
    and gradients in every block.
