from __future__ import annotations
import pytest

# general

@pytest.fixture
def forbid_call():
    def fail(*args, **kwargs): pytest.fail('unexpected call before input validation')
    return fail

# model

@pytest.fixture
def model_settings():
    from tmt.runtime.checkpoint import DEFAULT_CONFIG
    return {**DEFAULT_CONFIG, **{'dim': 4, 'layers': 1, 'spread': 4}}

@pytest.fixture
def model_init(model_settings):
    from tmt.runtime.checkpoint import preserve_seed, seed
    from tmt.model.model import Model
    from tmt.model.blocks import Config

    def create(overrides: Config = {}, model_seed = 11):
        with preserve_seed():
            seed(model_seed)
            return Model({**model_settings, **overrides})

    return create

@pytest.fixture
def model_snapshot():
    from mlx.utils import tree_flatten
    from tmt.model.model import Model

    def take(model: Model):
        tree = {
            'parameters': model.trainable_parameters(),
            'optimizer': model.optimizer.opt.state,
            'runtime': model.snapshot(),
        }
        return {name: value.tolist() for name, value in tree_flatten(tree)}

    return take