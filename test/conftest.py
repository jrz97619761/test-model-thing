import pytest


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
    return path, *checkpoint_factory(path)
