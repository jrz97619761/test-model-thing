"""Model construction and strict Safetensors checkpoint I/O."""

from __future__ import annotations

import json
import os
import random
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from . import validation

if TYPE_CHECKING:
    from .model import Model


@contextmanager
def preserved_rng() -> Iterator[None]:
    """Restore the caller's Python and MLX random state after any exit."""
    import mlx.core as mx

    python_state = random.getstate()
    mlx_state = mx.array(mx.random.state[0])
    try:
        yield
    finally:
        random.setstate(python_state)
        # Preserve the RNG state object captured by compiled MLX functions.
        mx.random.state[0][...] = mlx_state


def new_model(settings: Mapping[str, Any], seed: int) -> Model:
    validation.seed(seed)
    validation.model_settings(settings, 'model config')
    import mlx.core as mx

    from .model import Model

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
        # Open before the mode copy, which can remove read permission.
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


def load_checkpoint(path: str | PathLike[str]) -> tuple[Model, dict[str, Any]]:
    """Load strict model and optimizer tensors without a change to the RNG state."""
    import mlx.core as mx

    envelope = validation.checkpoint_header(path)
    settings = validation.metadata(envelope['__metadata__']['tmt'])
    with preserved_rng():
        # The loaded tensors replace every template value, so the seed is arbitrary.
        model = new_model(settings, 0)
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
