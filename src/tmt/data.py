"""Raw byte file selection and document loading. Please keep any data related work or helpers in this module!!!"""

from __future__ import annotations

import base64
import glob
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import validation

if TYPE_CHECKING:
    from .model import Model


@dataclass(frozen=True)
class DataOptions:
    """Training and optional evaluation file selections for one run."""

    pattern: str | PathLike[str]
    eval_data: str | PathLike[str] | None = None
    eval_max_bytes: int = 8192


class ByteFiles:
    """Resolve sorted raw-byte files and verify that each selection is readable."""

    def __init__(self, pattern: str | PathLike[str]) -> None:
        self.paths = sorted(glob.glob(pattern, recursive=True))
        if not self.paths:
            raise FileNotFoundError(f'no files matched {pattern!r}')
        for filename in self.paths:
            with open(filename, 'rb') as file:
                file.read(1)

    def documents(self, max_bytes: int | None = None) -> Iterator[bytes]:
        """Yield separate raw files with an optional global prefix budget."""
        remaining = max_bytes
        for path in self.paths:
            if remaining == 0:
                break
            data = Path(path).read_bytes()
            if remaining is not None:
                data = data[:remaining]
                remaining -= len(data)
            yield data


def documents(
    pattern: str | PathLike[str], max_bytes: int | None = None
) -> Iterator[bytes]:
    """Yield sorted raw files with an optional prefix across all file bytes."""
    yield from ByteFiles(pattern).documents(max_bytes)


def _merge(data: bytes, ranks: dict[tuple[int, int], int]) -> list[int]:
    """Merge the lowest-ranked adjacent pair, left to right, until none remain."""
    tokens = list(data)
    while len(tokens) > 1:
        pair = min(zip(tokens, tokens[1:]), key=lambda item: ranks.get(item, math.inf))
        if pair not in ranks:
            break
        merged, index = [], 0
        while index < len(tokens):
            if tuple(tokens[index:index + 2]) == pair:
                merged.append(ranks[pair])
                index += 2
            else:
                merged.append(tokens[index])
                index += 1
        tokens = merged
    return tokens


class BPETokenizer:
    """Byte IDs 0 to 255, stop ID 256, and ranked merge IDs from 257."""

    def __init__(self, merges: Iterable[Sequence[int]]) -> None:
        self.vocabulary = [bytes([byte]) for byte in range(256)] + [b'']
        self._ranks: dict[tuple[int, int], int] = {}
        for pair in merges:
            validation.require(
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and all(
                    type(token) is int
                    and token != 256
                    and 0 <= token < len(self.vocabulary)
                    for token in pair
                )
                and tuple(pair) not in self._ranks,
                'BPE merges must be distinct pairs of earlier non-stop token IDs',
            )
            self._ranks[tuple(pair)] = len(self.vocabulary)
            self.vocabulary.append(self.vocabulary[pair[0]] + self.vocabulary[pair[1]])
        self.vocab_size = len(self.vocabulary)

    @classmethod
    def from_dict(cls, value: Any) -> BPETokenizer:
        validation.require(
            isinstance(value, dict)
            and set(value) == {'merges'}
            and isinstance(value['merges'], list),
            "BPE tokenizer must be an object with only a 'merges' array",
        )
        return cls(value['merges'])

    def encode(self, data: bytes) -> list[int]:
        return _merge(data, self._ranks)


def huggingface_merges(value: Any) -> list[tuple[int, int]]:
    """Map ByteLevel BPE merges from Hugging Face JSON without preprocessing."""
    validation.require(
        isinstance(value, dict)
        and value.get('normalizer') is None
        and value.get('pre_tokenizer') is None
        and not value.get('added_tokens')
        and isinstance(value.get('model'), dict)
        and value['model'].get('type') == 'BPE'
        and isinstance(value['model'].get('merges'), list),
        'Hugging Face file must be BPE without normalizer, pre-tokenizer, '
        'or added tokens',
    )
    # GPT-2 byte-to-Unicode alphabet used by Hugging Face ByteLevel models.
    visible = [*range(33, 127), *range(161, 173), *range(174, 256)]
    hidden = [byte for byte in range(256) if byte not in visible]
    ids = {chr(byte): byte for byte in visible}
    ids.update({chr(256 + index): byte for index, byte in enumerate(hidden)})
    merges = []
    for item in value['model']['merges']:
        pair = item.split(' ') if isinstance(item, str) else item
        validation.require(
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(symbol, str) and symbol in ids for symbol in pair)
            and ''.join(pair) not in ids,
            f'invalid Hugging Face BPE merge: {item!r}',
        )
        ids[''.join(pair)] = 257 + len(merges)
        merges.append((ids[pair[0]], ids[pair[1]]))
    return merges


def tiktoken_merges(text: str) -> list[tuple[int, int]]:
    """Derive ranked merges from a tiktoken rank file, without its regex split."""
    try:
        rows = [line.split() for line in text.splitlines() if line]
        ranked = sorted(
            (int(rank), base64.b64decode(token, validate=True))
            for token, rank in rows
        )
    except ValueError as error:
        raise ValueError('invalid tiktoken rank file') from error
    validation.require(
        len({token for _, token in ranked if len(token) == 1}) == 256,
        'tiktoken ranks must contain every byte',
    )
    ranks: dict[tuple[int, int], int] = {}
    for _, token in ranked:
        if len(token) > 1:
            pair = tuple(_merge(token, ranks))
            validation.require(
                len(pair) == 2, f'tiktoken token {token!r} is not one BPE merge'
            )
            ranks[pair] = 257 + len(ranks)
    return list(ranks)


def encode_document(model: Model, data: bytes) -> bytes | list[int]:
    """Return byte IDs unchanged, or BPE IDs for a model with a tokenizer."""
    return data if model.tokenizer is None else model.tokenizer.encode(data)


def decode_token(model: Model, token: int) -> bytes:
    tokenizer = model.tokenizer
    return bytes([token]) if tokenizer is None else tokenizer.vocabulary[token]
