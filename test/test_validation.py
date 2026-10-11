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
validation.validate_seed(2**64 - 1)
validation.validate_run_name('example')
validation.validate_monitoring(None, 'X', 1, None, None, 1)
assert validation.is_finite_number(1.5)
assert not validation.is_finite_number(10**400)
"""
    result = subprocess.run(
        [sys.executable, '-c', script], cwd=tmp_path,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_missing_fields_precede_unknown_fields(tiny_settings):
    settings = {**tiny_settings, 'unknown': 1}
    del settings['spread'], settings['dim']
    with pytest.raises(ValueError, match="model config is missing field 'dim'"):
        validation.validate_model_settings(settings, 'model config')
