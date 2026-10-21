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
    'block': 'tmt-v0',
}

_MODEL_SETTING_FIELDS = {
    'dim',
    'layers',
    'spread',
    'temp',
    'rate',
    'bound',
    'block',
}


def group_model_config(settings: dict[str, Any]) -> dict[str, Any]:
    """Group flat model settings for JSON examples and manifests."""
    model: dict[str, Any] = {
        'block': settings['block'],
        'dimensions': {
            name: settings[name] for name in ('dim', 'layers', 'spread')
        },
    }
    extensions = {
        name: value
        for name, value in settings.items()
        if name not in _MODEL_SETTING_FIELDS
    }
    if extensions:
        model['extensions'] = extensions
    return {
        'model': model,
        'hparams': {
            'optimizer': {
                'rate': settings['rate'],
                'bound': settings['bound'],
            },
            'sampling': {'temp': settings['temp']},
        },
    }


def _flatten_grouped_config(overrides: dict[str, Any]) -> dict[str, Any]:
    """Validate grouped fields and convert them to flat model settings."""
    group_names = {'model', 'hparams'}
    unknown_groups = set(overrides) - group_names
    if unknown_groups:
        name = sorted(unknown_groups)[0]
        validation.require(False, f"model config has unknown field {name!r}")

    def group(value: Any, path: str, allowed: set[str]) -> dict[str, Any]:
        validation.require(isinstance(value, dict), f'{path} must be an object')
        unknown = set(value) - allowed
        if unknown:
            name = sorted(unknown)[0]
            validation.require(False, f"{path} has unknown field {name!r}")
        return value

    model = group(overrides.get('model', {}), 'model config.model', {
        'block', 'dimensions', 'extensions'
    })
    hparams = group(overrides.get('hparams', {}), 'model config.hparams', {
        'optimizer', 'sampling'
    })
    dimensions = group(
        model.get('dimensions', {}),
        'model config.model.dimensions',
        {'dim', 'layers', 'spread'},
    )
    extensions = group(
        model.get('extensions', {}),
        'model config.model.extensions',
        set(MODEL_DEFAULTS) - _MODEL_SETTING_FIELDS,
    )
    optimizer = group(
        hparams.get('optimizer', {}),
        'model config.hparams.optimizer',
        {'rate', 'bound'},
    )
    sampling = group(
        hparams.get('sampling', {}),
        'model config.hparams.sampling',
        {'temp'},
    )
    flat: dict[str, Any] = {}
    if 'block' in model:
        flat['block'] = model['block']
    flat.update(dimensions)
    flat.update(extensions)
    flat.update(sampling)
    flat.update(optimizer)
    return flat


def load_config(path: str | PathLike[str] | None = None) -> dict[str, Any]:
    overrides: Any = {} if path is None else read_json(path)
    validation.require(
        isinstance(overrides, dict), 'model config must be a JSON object'
    )
    overrides = _flatten_grouped_config(overrides)
    resolved = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in {**MODEL_DEFAULTS, **overrides}.items()
    }
    return validation.model_settings(resolved, 'model config')
