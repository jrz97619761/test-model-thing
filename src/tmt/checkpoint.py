"""Model construction and strict Safetensors checkpoint I/O."""

from __future__ import annotations

import json
import os
import random
import stat
import tempfile
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from . import validation

if TYPE_CHECKING:
    from .model import Model


def new_model(settings: Mapping[str, Any], seed: int) -> Model:
    validation.seed(seed)
    validation.model_settings(settings, 'model config')
    import mlx.core as mx
    from tmt.model import Model

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
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.safetensors', delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        mx.save_safetensors(temporary, tensors, metadata={'tmt': raw})
        with temporary.open('rb') as file:
            try:
                mode = stat.S_IMODE(path.stat().st_mode)
            except FileNotFoundError:
                pass
            else:
                temporary.chmod(mode)
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | PathLike[str], seed: int
) -> tuple[Model, dict[str, Any]]:
    validation.seed(seed)
    import mlx.core as mx

    envelope = validation.checkpoint_header(path)
    settings = validation.metadata(envelope['__metadata__']['tmt'])
    python_state = random.getstate()
    mlx_state = mx.array(mx.random.state[0])
    try:
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
    finally:
        random.setstate(python_state)
        # Preserve the RNG state object captured by compiled MLX functions.
        mx.random.state[0][...] = mlx_state


def load_model(
    path: str | PathLike[str], seed: int
) -> tuple[Model, dict[str, Any]]:
    """Load a checkpoint through the compatibility name."""
    return load_checkpoint(path, seed)
