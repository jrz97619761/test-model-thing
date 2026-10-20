"""Model setting defaults and partial JSON configuration files."""

from __future__ import annotations

from os import PathLike
from typing import Any

from . import validation
from .records import read_json

MODEL_DEFAULTS: dict[str, Any] = {
    'dim': 512,
    'layers': 16,
    'spread': 32,
    'temp': 0.75,
    'rate': 0.0005,
    'bound': (40000, 120000),
    'block': 'rtu',
}


def load_config(path: str | PathLike[str] | None = None) -> dict[str, Any]:
    overrides: Any = {} if path is None else read_json(path)
    validation.require(
        isinstance(overrides, dict), 'model config must be a JSON object'
    )
    resolved = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in {**MODEL_DEFAULTS, **overrides}.items()
    }
    return validation.model_settings(resolved, 'model config')
