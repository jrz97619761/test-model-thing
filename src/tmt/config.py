from __future__ import annotations

import json
from os import PathLike
from typing import Any

from . import validation

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
    overrides: Any = {}
    if path is not None:
        with open(path) as file:
            overrides = json.load(file)

    validation.require(
        isinstance(overrides, dict), 'model config must be a JSON object'
    )
    for name in overrides:
        if name not in MODEL_DEFAULTS:
            raise ValueError(
                f'invalid model config: got an unexpected keyword argument {name!r}'
            )
    resolved = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in {**MODEL_DEFAULTS, **overrides}.items()
    }
    return validation.model_settings(resolved, 'model config')
