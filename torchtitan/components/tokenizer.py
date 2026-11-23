# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from tokenizers import AddedToken, Tokenizer
from torchtitan.config import Configurable


logger = logging.getLogger(__name__)


class BaseTokenizer(ABC, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    # base tokenizer interface, for typing purpose mainly
    eos_id: int | None

    def __init__(self):
        self.eos_id = None
        self._chat_template = None

    @abstractmethod
    def encode(self, *args, **kwargs) -> list[int]:
        ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str:
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        ...

    def token_byte_lengths(self) -> list[int]:
        """UTF-8 byte length of the text each token id stands for, indexed by
        id (0 for special tokens such as EOS). Summed over a document's labels,
        it gives the exact number of text bytes the model predicts, which is
        the denominator of bits per byte."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide token byte lengths."
        )

    def set_chat_template(self, template: str) -> None:
        """Compile and store a Jinja chat template."""
        import json

        import jinja2
        import jinja2.ext
        import jinja2.sandbox

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        def tojson(
            x, ensure_ascii=False, indent=None, separators=None, sort_keys=False
        ):
            return json.dumps(
                x,
                ensure_ascii=ensure_ascii,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
            )

        def strftime_now(fmt):
            from datetime import datetime

            return datetime.now().strftime(fmt)

        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols],
        )
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = strftime_now
        env.filters["tojson"] = tojson
        self._chat_template = env.from_string(template)

    def apply_chat_template(
        self, messages: Sequence[Mapping[str, Any]], **kwargs
    ) -> str:
        """Render messages through the Jinja chat template. Returns formatted text.

        Messages should be a list of dicts with "role" and "content" keys, e.g.
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}].

        Additional template variables (e.g. add_generation_prompt, tools) can be
        passed as kwargs.
        """
        if self._chat_template is None:
            raise ValueError("No chat template set. Call set_chat_template() first.")
        return self._chat_template.render(messages=messages, **kwargs)


class HuggingFaceTokenizer(BaseTokenizer):
    """
    A tokenizer wrapper that handles BOS/EOS token inference and encoding.

    This class loads tokenizer files and automatically infers BOS/EOS tokens from
    a configuration file (tokenizer_config.json) as well as specific formatting related to
    chat templates. Its encode method suppresses the tokenizer's own special tokens
    and adds BOS/EOS itself, controlled by the add_bos/add_eos arguments.

    Args:
        config (Config): Configurable config (currently empty).
        tokenizer_path (str): Path to directory containing tokenizer files
    """

    # Following HF transformers convention (CHAT_TEMPLATE_FILE in transformers/utils/hub.py),
    # a standalone Jinja file at the model root takes priority over an inline template in
    # tokenizer_config.json. Models like GPT-OSS use this pattern.
    CHAT_TEMPLATE_FILE = "chat_template.jinja"

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        pass  # no fields — tokenizer_path passed at build time

    def __init__(
        self,
        config: Config | None = None,
        *,
        tokenizer_path: str,
    ):
        super().__init__()
        self.tokenizer_path = tokenizer_path

        # Initialize BOS/EOS token attributes (frequently used)
        self.bos_id = None
        self.eos_id = None
        self.bos_token = None
        self.eos_token = None

        # Load the underlying tokenizer
        self.tokenizer = self._load_tokenizer_from_path(tokenizer_path)

        # Load configuration files
        self._hf_config = self._load_config(
            os.path.join(tokenizer_path, "tokenizer_config.json")
        )
        if self._hf_config is None:
            logger.warning(
                "No tokenizer_config.json found at %s. "
                "Special token inference and chat template auto-loading disabled.",
                tokenizer_path,
            )

        # Infer special tokens from config (if available) and BOS/EOS behavior
        if self._hf_config is not None:
            self._infer_special_tokens()
        self._infer_should_add_bos_eos()

        # Auto-load chat template: standalone .jinja file takes priority
        # (e.g. GPT-OSS), then fall back to inline in tokenizer_config.json
        # (e.g. Llama3, Qwen3, DeepSeek V3).
        if self._hf_config is not None:
            jinja_path = os.path.join(tokenizer_path, self.CHAT_TEMPLATE_FILE)
            if os.path.exists(jinja_path):
                with open(jinja_path) as f:
                    self.set_chat_template(f.read())
            elif "chat_template" in self._hf_config:
                self.set_chat_template(self._hf_config["chat_template"])

    def _load_config(self, config_path: str) -> dict | None:
        """Load configuration from JSON file if it exists."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        return None

    def _load_tokenizer_from_path(self, tokenizer_path: str) -> Tokenizer:
        """Load tokenizer from various file formats."""
        if not os.path.exists(tokenizer_path):
            if "assets/tokenizer" in tokenizer_path:
                raise FileNotFoundError(
                    "Detected ./assets/tokenizer path which was deprecated in https://github.com/pytorch/torchtitan/pull/1540.\n"
                    "Remove model.tokenizer_path and set hf_assets_path to assets "
                    "downloaded with ./scripts/download_hf_assets.py\n"
                    "See example: https://github.com/pytorch/torchtitan/tree/main/torchtitan/models/deepseek_v3#download-tokenizer"
                )
            else:
                raise FileNotFoundError(
                    f"Tokenizer path '{tokenizer_path}' does not exist"
                )

        # Define paths for different tokenizer file types
        tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
        vocab_txt_path = os.path.join(tokenizer_path, "vocab.txt")
        vocab_json_path = os.path.join(tokenizer_path, "vocab.json")
        merges_txt_path = os.path.join(tokenizer_path, "merges.txt")

        # Strategy 1: Load from tokenizer.json (preferred for modern tokenizers)
        if os.path.exists(tokenizer_json_path):
            logger.info("Loading tokenizer from tokenizer.json")
            return Tokenizer.from_file(tokenizer_json_path)
        # Strategy 2: Load from vocab files (with or without merges.txt)
        elif os.path.exists(vocab_json_path) or os.path.exists(vocab_txt_path):
            # Load vocabulary
            if os.path.exists(vocab_json_path):
                logger.info("Loading vocabulary from vocab.json")
                with open(vocab_json_path, "r") as f:
                    vocab = json.load(f)
                vocab_source = "vocab.json"
            else:
                logger.info("Loading vocabulary from vocab.txt")
                vocab = {}
                with open(vocab_txt_path, "r") as f:
                    for i, line in enumerate(f):
                        token = line.strip()
                        if token:
                            vocab[token] = i
                vocab_source = "vocab.txt"

            # Strategy 2a: Use BPE if merges.txt exists
            if os.path.exists(merges_txt_path):
                logger.info(f"Loading BPE tokenizer from {vocab_source} + merges.txt")
                from tokenizers import decoders, pre_tokenizers, processors
                from tokenizers.models import BPE

                # Load merges from file and convert to tuples
                merges = []
                with open(merges_txt_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith(
                            "#"
                        ):  # Skip comments and empty lines
                            parts = line.split()
                            if len(parts) >= 2:
                                merges.append((parts[0], parts[1]))

                # Create BPE model
                bpe_model = BPE(vocab=vocab, merges=merges)
                tokenizer = Tokenizer(bpe_model)

                # Configure GPT-2 style components for proper space handling
                tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                    add_prefix_space=False
                )
                tokenizer.decoder = decoders.ByteLevel()
                tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

                return tokenizer

            # Strategy 2b: Use WordLevel if no merges.txt
            else:
                logger.info(f"Loading WordLevel tokenizer from {vocab_source}")
                from tokenizers.models import WordLevel

                word_level_model = WordLevel(vocab=vocab, unk_token="[UNK]")
                return Tokenizer(word_level_model)

        else:
            # List available files for debugging
            available_files = [
                f
                for f in os.listdir(tokenizer_path)
                if os.path.isfile(os.path.join(tokenizer_path, f))
            ]
            raise FileNotFoundError(
                f"No supported tokenizer files found in '{tokenizer_path}'. "
                f"Available files: {available_files}. "
                "Looking for: tokenizer.json, vocab.txt+merges.txt, or vocab.json+merges.txt"
            )

    def _get_token_from_config(self, config: dict[str, Any], key: str) -> str | None:
        """
        Parse special tokens from config that can be either strings or dicts.
        HF tokens are stored as either {'bos_token': '<bos>'} or {'bos_token': {'content': '<bos>', ...}}.
        """
        token = config.get(key)
        if isinstance(token, dict):
            if "content" not in token:
                raise ValueError(f"Could not parse {key} from config")
            token = token["content"]
        elif token is not None and not isinstance(token, str):
            raise ValueError(
                f"Could not parse {key} from config - expected string or dict"
            )
        return token

    def _process_special_token(
        self, token_str: str, token_config: dict, token_id: int | None = None
    ) -> AddedToken:
        """
        Process a configured added token and update BOS/EOS attributes if applicable.

        Args:
            token_str: The token string content
            token_config: Token configuration dictionary
            token_id: Optional explicit token ID (for added_tokens_decoder)

        Returns:
            AddedToken object to be added to the tokenizer
        """
        # Get reference BOS/EOS tokens from config for comparison
        config_bos_token = (
            self._get_token_from_config(self._hf_config, "bos_token")
            if self._hf_config
            else None
        )
        config_eos_token = (
            self._get_token_from_config(self._hf_config, "eos_token")
            if self._hf_config
            else None
        )

        # Store BOS/EOS tokens as class attributes if they match
        if token_str == config_bos_token:
            self.bos_token = token_str
            self.bos_id = (
                token_id
                if token_id is not None
                else self.tokenizer.token_to_id(token_str)
            )
        if token_str == config_eos_token:
            self.eos_token = token_str
            self.eos_id = (
                token_id
                if token_id is not None
                else self.tokenizer.token_to_id(token_str)
            )

        # Create AddedToken object based on config format
        if isinstance(token_config, dict):
            if token_config.get("__type") == "AddedToken" or "content" in token_config:
                # Handle both AddedToken format and added_tokens_decoder format
                return AddedToken(
                    content=token_str,
                    single_word=token_config.get("single_word", False),
                    lstrip=token_config.get("lstrip", False),
                    rstrip=token_config.get("rstrip", False),
                    normalized=token_config.get("normalized", True),
                    special=token_config.get("special", True),
                )

        # Fallback to simple special token
        return AddedToken(content=token_str, special=True)

    def _infer_special_tokens(self):
        """
        Read configured tokens and preserve their special-token flags.
        Store BOS/EOS tokens as class attributes since they are frequently used.

        This method handles multiple token configuration formats:
        1. Standard top-level keys (bos_token, eos_token, etc.)
        2. added_tokens_decoder dictionary (including non-special added tokens)
        """
        standard_keys = [
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "sep_token",
            "cls_token",
            "mask_token",
        ]

        # List to collect AddedToken objects for updating the underlying tokenizer
        added_tokens_to_add = []

        if not self._hf_config:
            return

        # Process standard top-level token keys
        for key in standard_keys:
            token_config = self._hf_config.get(key)
            if token_config is not None:
                token_str = self._get_token_from_config(self._hf_config, key)
                if token_str is not None:
                    added_token = self._process_special_token(token_str, token_config)
                    added_tokens_to_add.append(added_token)

        # Process added_tokens_decoder (comprehensive special token definitions)
        added_tokens_decoder = self._hf_config.get("added_tokens_decoder", {})
        for token_id_str, token_config in added_tokens_decoder.items():
            if isinstance(token_config, dict) and "content" in token_config:
                token_str = token_config["content"]
                token_id = int(token_id_str)
                added_token = self._process_special_token(
                    token_str, token_config, token_id
                )
                added_tokens_to_add.append(added_token)

        # add_tokens keeps each AddedToken.special flag; add_special_tokens would
        # set special=True on all of them.
        if added_tokens_to_add:
            self.tokenizer.add_tokens(added_tokens_to_add)

            # Update BOS/EOS token IDs after adding to tokenizer (in case they changed)
            if self.bos_token:
                self.bos_id = self.tokenizer.token_to_id(self.bos_token)
            if self.eos_token:
                self.eos_id = self.tokenizer.token_to_id(self.eos_token)

    def _infer_should_add_bos_eos(self):
        """
        Determine the default BOS/EOS behavior for ``encode``.

        ``encode`` suppresses the tokenizer's own special tokens, so these
        defaults are the only ones applied when the caller does not pass
        add_bos/add_eos explicitly.
        """
        self.default_add_bos = False
        self.default_add_eos = False
        self.hf_adds_bos = False
        self.hf_adds_eos = False

        # First, determine if underlying tokenizer auto-adds BOS/EOS tokens empirically
        encoded_empty_str = self.tokenizer.encode("").ids
        if self.bos_id is not None and self.bos_id in encoded_empty_str:
            self.hf_adds_bos = True
        if self.eos_id is not None and self.eos_id in encoded_empty_str:
            self.hf_adds_eos = True

        # Check tokenizer_config.json for explicit settings
        if self._hf_config:
            config_add_bos = self._hf_config.get("add_bos_token")
            config_add_eos = self._hf_config.get("add_eos_token")
            if config_add_bos is not None:
                self.default_add_bos = bool(config_add_bos)
            if config_add_eos is not None:
                self.default_add_eos = bool(config_add_eos)

    def encode(self, *args, **kwargs) -> list[int]:
        """
        Encode text into token IDs with BOS/EOS handling.

        Args:
            text (str): The text to encode
            add_bos (bool): Whether to add BOS token
            add_eos (bool): Whether to add EOS token

        Returns:
            list[int]: List of token IDs
        """
        # Extract arguments
        if len(args) >= 1:
            text = args[0]
        else:
            text = kwargs.get("text", "")

        add_bos = kwargs.get("add_bos", self.default_add_bos or self.hf_adds_bos)
        add_eos = kwargs.get("add_eos", self.default_add_eos or self.hf_adds_eos)

        # Get base token IDs from the underlying tokenizer
        token_ids = self.tokenizer.encode(text, add_special_tokens=False).ids

        # Add BOS token if requested
        if add_bos and self.bos_id is not None:
            token_ids.insert(0, self.bos_id)

        # Add EOS token if requested
        if add_eos and self.eos_id is not None:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(self, *args, **kwargs) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids (list[int]): List of token IDs to decode
            **kwargs: Additional arguments passed to the underlying tokenizer's decode method
                     (e.g., skip_special_tokens)

        Returns:
            str: Decoded text
        """
        # Extract token_ids from arguments
        if len(args) >= 1:
            token_ids = args[0]
            # Pass through remaining kwargs
            return self.tokenizer.decode(token_ids, **kwargs)
        else:
            token_ids = kwargs.pop("token_ids", [])
            # Pass through remaining kwargs after removing token_ids
            return self.tokenizer.decode(token_ids, **kwargs)

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> dict[str, int]:
        """Get the vocabulary as a dictionary."""
        return self.tokenizer.get_vocab()

    def token_to_id(self, token: str) -> int | None:
        """Convert token to ID."""
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        """Convert ID to token."""
        return self.tokenizer.id_to_token(token_id)

    def token_byte_lengths(self) -> list[int]:
        """Exact UTF-8 byte length of each token id, for byte-level BPE.

        Read from the vocabulary, whose byte-level token strings map each
        character to one raw byte (GPT-2's bytes_to_unicode). Decoding a token
        on its own would not work: a token holding part of a multi-byte
        character decodes to U+FFFD (3 bytes). Special added tokens count 0
        bytes; other added tokens count the bytes of their text.
        """
        decoder = type(self.tokenizer.decoder).__name__
        if decoder != "ByteLevel":
            raise NotImplementedError(
                "token_byte_lengths needs a byte-level BPE tokenizer; this "
                f"tokenizer's decoder is {decoder}."
            )
        byte_of_char = _byte_level_char_to_byte()
        added = self.tokenizer.get_added_tokens_decoder()
        vocab = self.tokenizer.get_vocab(with_added_tokens=True)
        lengths = [0] * (max(vocab.values()) + 1)
        for token, token_id in vocab.items():
            if token_id in added:
                added_token = added[token_id]
                lengths[token_id] = (
                    0 if added_token.special else len(added_token.content.encode())
                )
                continue
            try:
                lengths[token_id] = len(bytes(byte_of_char[char] for char in token))
            except KeyError as error:
                raise ValueError(
                    f"Token {token!r} (id {token_id}) is not a byte-level string."
                ) from error
        return lengths


class ByteTokenizer(BaseTokenizer):
    """UTF-8 byte tokenizer: one token per byte, vocabulary of 256.

    EOS is byte 0xFF, which never occurs in valid UTF-8, so it cannot collide
    with text. There is no BOS: packed documents are delimited by ``positions``
    resetting to 0, which byte-level models use directly.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        pass

    vocab_size: ClassVar[int] = 256

    def __init__(self, config: Config | None = None, *, tokenizer_path: str = ""):
        # The trainer passes ``tokenizer_path`` to every tokenizer; bytes need
        # no assets.
        del config, tokenizer_path
        super().__init__()
        self.eos_id = 0xFF
        self.bos_id = None

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        del add_bos  # no BOS token
        token_ids = list(text.encode("utf-8"))
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(t for t in token_ids if t != self.eos_id).decode(
            "utf-8", errors="replace"
        )

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def token_byte_lengths(self) -> list[int]:
        """Every token is one byte, except EOS (0xFF), which is no text."""
        return [0 if token_id == self.eos_id else 1 for token_id in range(256)]


def _byte_level_char_to_byte() -> dict[str, int]:
    """Inverse of GPT-2's ``bytes_to_unicode``: the printable character that
    byte-level BPE uses for each of the 256 byte values."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values, chars = printable[:], printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            byte_values.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {chr(char): byte for byte, char in zip(byte_values, chars)}


class MultiModalTokenizer(HuggingFaceTokenizer):
    """Single source of truth for multimodal special tokens.

    ``TOKEN_FIELDS`` lists tokens the tokenizer validates and exposes. It includes
    ``pad`` because batching needs ``pad_id``.

    ``LOSS_MASK_TOKEN_FIELDS`` lists tokens masked with ``IGNORE_INDEX`` while an
    unpadded sample is processed. Padding is added later: the packer or collator
    inserts ``pad_id``, sets its labels to ``IGNORE_INDEX``, and sets its
    ``padding_mask`` entries to true. Therefore ``pad`` is not a loss-mask field.

    Models with additional modality tokens should subclass both this class and
    ``Config``. Extend ``TOKEN_FIELDS`` so initialization validates and exposes
    the new tokens. Also extend ``LOSS_MASK_TOKEN_FIELDS`` for placeholder or
    boundary tokens that preprocessing inserts but the language model should not
    learn to predict. Every loss-mask field must also be a token field.

    # TODO: All 5 fields are currently required. If a future VLM doesn't need
    # some (e.g. no video, no vision_start/end markers), consider making fields
    # optional.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTokenizer.Config):
        image_token: str
        """Token string for image placeholders, e.g. ``"<|image_pad|>"``."""

        video_token: str
        """Token string for video placeholders, e.g. ``"<|video_pad|>"``."""

        vision_start_token: str
        """Token string marking the start of a vision sequence."""

        vision_end_token: str
        """Token string marking the end of a vision sequence."""

        pad_token: str
        """Token string for padding."""

    # Config field prefixes that are validated and exposed as token and ID attributes.
    TOKEN_FIELDS: ClassVar[tuple[str, ...]] = (
        "image",
        "video",
        "vision_start",
        "vision_end",
        "pad",
    )
    # TOKEN_FIELDS subset whose occurrences as label targets do not contribute loss.
    LOSS_MASK_TOKEN_FIELDS: ClassVar[tuple[str, ...]] = (
        "image",
        "video",
        "vision_start",
        "vision_end",
    )

    def __init__(self, config: Config, *, tokenizer_path: str):
        super().__init__(config, tokenizer_path=tokenizer_path)

        added_tokens = self.tokenizer.get_added_tokens_decoder()
        token_to_id = {tok.content: tok_id for tok_id, tok in added_tokens.items()}

        for name in self.TOKEN_FIELDS:
            token_str: str = getattr(config, f"{name}_token")
            if token_str not in token_to_id:
                raise ValueError(
                    f"Special token '{token_str}' (config field '{name}_token') "
                    f"not found in tokenizer at '{tokenizer_path}'. "
                    f"Available added tokens: {list(token_to_id.keys())}"
                )
            setattr(self, f"{name}_token", token_str)
            setattr(self, f"{name}_id", token_to_id[token_str])
