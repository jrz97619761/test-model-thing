from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True)
class SavedCheckpoint:
    path: Path
    mx: Any
    settings: dict[str, Any]
    model: Any
    arrays: dict[str, Any]
    metadata: dict[str, str]


@pytest.fixture
def assert_almost_equal():
    """Keep the original decimal rounding and absolute delta checks."""

    def compare(actual, expected, places=7, delta=None):
        if actual == expected:
            return
        difference = abs(actual - expected)
        if delta is not None:
            assert difference <= delta
        else:
            assert round(difference, places) == 0

    return compare


@pytest.fixture
def tiny_settings():
    from tmt.cli import load_config

    settings = load_config()
    settings.update(dim=4, layers=1, spread=4)
    return settings


@pytest.fixture
def model_factory(tiny_settings):
    from tmt.cli import new_model

    def create(seed=11, **overrides):
        return new_model({**tiny_settings, **overrides}, seed)

    return create


@pytest.fixture
def register_test_blocks(monkeypatch):
    from block_support import MatrixCache, NoWeights, Stateless
    from tmt.blocks import BLOCKS

    for name, block in (
        ('stateless-test', Stateless),
        ('matrix-test', MatrixCache),
        ('empty-test', NoWeights),
    ):
        monkeypatch.setitem(BLOCKS, name, block)


@pytest.fixture
def loss_case(model_factory):
    import math

    import mlx.core as mx

    model = model_factory()
    x = [0.0, 0.5, 1.0, 1.5]
    output = mx.arange(256) * 0.001
    stop = mx.array([0.2])
    target = model.encoder(mx.array(67))
    mx.eval(output, stop, target)

    logits = [float(value) for value in output.tolist()]
    x_mean = sum(x) / len(x)
    variance = sum((value - x_mean) ** 2 for value in x) / len(x)
    variance_term = max(0.0, 1.0 - math.sqrt(variance + 1e-4))
    latent = [float(value) for value in target.tolist()]
    latent_term = sum((left - right) ** 2 for left, right in zip(x, latent)) / len(x)
    maximum = max(logits)
    logsumexp = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
    cross_entropy = -logits[67] + logsumexp
    stop_value = float(stop.item())
    terms = {
        'variance': variance_term,
        'latent_prediction': latent_term,
        'cross_entropy': cross_entropy,
        'stop': (stop_value - 1.0) ** 2,
    }
    return model, mx.array(x), output, stop, terms


@pytest.fixture
def evaluation_model():
    import mlx.core as mx

    class FakeBlock:
        @property
        def state_fields(self):
            return ('cache',)

        @property
        def trace_fields(self):
            return ('history',)

        def __init__(self):
            self.cache = mx.array([91.0])
            self.history = mx.array([[93.0]])

        def reset(self):
            self.cache = mx.zeros_like(self.cache)
            self.history = mx.zeros_like(self.history)

        def snapshot_state(self):
            saved = {
                name: mx.array(getattr(self, name))
                for name in self.state_fields + self.trace_fields
            }
            mx.eval(*saved.values())
            return saved

        def restore_state(self, saved):
            for name, value in saved.items():
                setattr(self, name, value)

        def eval_state(self):
            mx.eval(
                *(getattr(self, name) for name in self.state_fields + self.trace_fields)
            )

    class FakeModel:
        def __init__(self):
            self.blocks = [FakeBlock()]
            self.uniform = False

        def reset(self):
            for block in self.blocks:
                block.reset()

        def snapshot_state(self):
            return [block.snapshot_state() for block in self.blocks]

        def restore_state(self, saved):
            for block, block_saved in zip(self.blocks, saved):
                block.restore_state(block_saved)

        def eval_state(self):
            for block in self.blocks:
                block.eval_state()

        def step(self, current, dummies=None, frozen=False):
            assert frozen
            block = self.blocks[0]
            byte = int(current.item())
            state = (int(block.cache.item()) * 3 + byte) % 17
            block.cache = mx.array([float(state)])
            block.history = block.history + mx.array([[float(byte)]])
            logits = (
                mx.zeros((256,))
                if self.uniform
                else mx.array(
                    [state / 5 if index == byte else 0 for index in range(256)]
                )
            )
            return ((None, [], []), (logits, mx.zeros((1,))))

    model = FakeModel()
    saved = model.snapshot_state()

    def unchanged():
        current = model.snapshot_state()
        assert len(current) == len(saved), 'block count changed'
        for block_index, (actual, expected) in enumerate(zip(current, saved)):
            assert actual.keys() == expected.keys(), (
                f'block {block_index} fields changed'
            )
            for name, expected_value in expected.items():
                assert mx.array_equal(actual[name], expected_value).item(), (
                    f'block {block_index} field {name!r} changed'
                )

    yield model
    unchanged()


@pytest.fixture
def forbid_call():
    def fail(*args, **kwargs):
        pytest.fail('unexpected call before input validation')

    return fail


@pytest.fixture
def checkpoint_factory(tiny_settings):
    import mlx.core as mx
    from tmt.cli import new_model, save_checkpoint

    def save(path):
        model = new_model(tiny_settings, 11)
        save_checkpoint(path, model, tiny_settings)
        arrays, metadata = mx.load(path, return_metadata=True)
        mx.eval(arrays)
        return mx, dict(tiny_settings), model, arrays, metadata

    return save


@pytest.fixture
def saved_checkpoint(tmp_path, checkpoint_factory):
    path = tmp_path / 'model.safetensors'
    mx, settings, model, arrays, metadata = checkpoint_factory(path)
    return SavedCheckpoint(
        path=path,
        mx=mx,
        settings=settings,
        model=model,
        arrays=arrays,
        metadata=metadata,
    )
