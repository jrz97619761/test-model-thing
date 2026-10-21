"""Raw byte file selection and document loading. Please keep any data related work or helpers in this module!!!"""

from __future__ import annotations

import glob
from collections.abc import Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path


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
