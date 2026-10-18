"""Raw byte file selection and document loading."""

from __future__ import annotations

import glob
from collections.abc import Iterator
from os import PathLike
from pathlib import Path


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
