import subprocess
import sys

import pytest

from tmt import validation


def test_validation_import_without_model_runtime(tmp_path):
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name in ('tmt.main', 'mlx') or name.startswith('mlx.'):
        raise ImportError('model import forbidden')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tmt import validation
validation.seed(2**64 - 1)
validation.run_name('example')
validation.monitoring(None, 'X', 1, None, None, 1)
assert validation.is_finite_number(1.5)
assert not validation.is_finite_number(10**400)
"""
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_missing_fields_precede_unknown_fields(tiny_settings):
    settings = {**tiny_settings, 'unknown': 1}
    del settings['spread'], settings['dim']
    with pytest.raises(ValueError, match="model config is missing field 'dim'"):
        validation.model_settings(settings, 'model config')


@pytest.mark.parametrize(
    ('changes', 'message'),
    [
        ({'block': 'unknown', 'dim': 0}, r'model config\.block must be one of'),
        (
            {'dim': 0, 'layers': 0, 'spread': 0},
            r'model config\.dim must be a positive integer',
        ),
        (
            {'dim': 4, 'layers': 0, 'spread': 0},
            r'model config\.layers must be a positive integer',
        ),
        (
            {'temp': -1, 'rate': 0},
            r'model config\.temp must be a finite nonnegative number',
        ),
        (
            {'temp': 0.75, 'rate': 0, 'bound': [120000, 40000]},
            r'model config\.rate must be a finite positive number',
        ),
    ],
    ids=[
        'block-before-dim',
        'dim-before-layers-and-spread',
        'layers-before-spread',
        'integers-before-temp-and-rate',
        'rate-before-bound',
    ],
)
def test_model_settings_validation_order(tiny_settings, changes, message):
    settings = {**tiny_settings, **changes}
    with pytest.raises(ValueError, match=message):
        validation.model_settings(settings, 'model config')


def test_core_settings_precede_finite_json_check(tiny_settings, monkeypatch):
    from tmt import config

    monkeypatch.setattr(
        config, 'MODEL_DEFAULTS', {**config.MODEL_DEFAULTS, 'extension': 1}
    )
    settings = {**tiny_settings, 'dim': 0, 'extension': float('nan')}
    with pytest.raises(
        ValueError, match=r'model config\.dim must be a positive integer'
    ):
        validation.model_settings(settings, 'model config')


def test_finite_json_check_precedes_extension_type_check(tiny_settings, monkeypatch):
    from tmt import config

    monkeypatch.setattr(
        config, 'MODEL_DEFAULTS', {**config.MODEL_DEFAULTS, 'extension': 1}
    )
    settings = {**tiny_settings, 'extension': float('nan')}
    with pytest.raises(
        ValueError, match=r'model config\.extension must contain finite JSON values'
    ):
        validation.model_settings(settings, 'model config')
