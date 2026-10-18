"""Model construction and strict Safetensors checkpoint I/O."""

from __future__ import annotations

import json
import random
from os import PathLike
from typing import TYPE_CHECKING, Any, Mapping

from . import validation

if TYPE_CHECKING:
    from .main import Model


def new_model(settings: Mapping[str, Any], seed: int) -> Model:
    validation.seed(seed)
    validation.model_settings(settings, 'model config')
    import mlx.core as mx
    from tmt.main import Model

    random.seed(seed)
    mx.random.seed(seed)
    model = Model(**settings)
    mx.eval(model.parameters())
    return model


def _flatten(tree: Any, prefix: str) -> dict[str, Any]:
    import mlx.utils as util

    return {f'{prefix}.{name}': value for name, value in util.tree_flatten(tree)}


def _unflatten(tensors: Mapping[str, Any], prefix: str) -> Any:
    import mlx.utils as util

    return util.tree_unflatten(
        [
            (name[len(prefix) + 1 :], value)
            for name, value in tensors.items()
            if name.startswith(f'{prefix}.')
        ]
    )


def save_checkpoint(
    path: str | PathLike[str], model: Model, settings: Mapping[str, Any]
) -> None:
    import mlx.core as mx

    raw = json.dumps({'version': 1, 'model': settings})
    validation.metadata(raw)
    model.optimizer.init(model.trainable_parameters())
    tensors = _flatten(model.parameters(), 'm')
    tensors.update(_flatten(model.optimizer.state, 'o'))
    mx.eval(model.parameters(), model.optimizer.state)
    mx.save_safetensors(path, tensors, metadata={'tmt': raw})


def load_checkpoint(
    path: str | PathLike[str], seed: int
) -> tuple[Model, dict[str, Any]]:
    validation.seed(seed)
    import mlx.core as mx

    envelope = validation.checkpoint_header(path)
    settings = validation.metadata(envelope['__metadata__']['tmt'])
    model = new_model(settings, seed)
    model.optimizer.init(model.trainable_parameters())
    expected = _flatten(model.parameters(), 'm')
    expected.update(_flatten(model.optimizer.state, 'o'))
    try:
        tensors = mx.load(path)
    except Exception as error:
        raise ValueError(f'invalid checkpoint: {error}') from error
    validation.checkpoint_arrays(tensors, expected)

    model.update(_unflatten(tensors, 'm'))
    model.optimizer.state = _unflatten(tensors, 'o')
    model.compiled = None
    mx.eval(model.parameters(), model.optimizer.state)
    return model, settings


def load_model(
    path: str | PathLike[str], seed: int
) -> tuple[Model, dict[str, Any]]:
    """Load a checkpoint through the compatibility name."""
    return load_checkpoint(path, seed)
