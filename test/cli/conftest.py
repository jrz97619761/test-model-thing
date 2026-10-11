import pytest


@pytest.fixture
def cli_runner(tmp_path):
    import os
    import subprocess
    import sysconfig
    from pathlib import Path

    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    command = str(Path(sysconfig.get_path('scripts')) / 'tmt')

    def run(*args, success=False):
        result = subprocess.run(
            [command, *args], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == (0 if success else 1), result.stderr
        assert 'Traceback' not in result.stderr
        return result

    return run


@pytest.fixture
def cli_workspace(tmp_path, tiny_settings, cli_runner):
    import json

    (tmp_path / 'model.json').write_text(json.dumps(tiny_settings))
    (tmp_path / 'grid.json').write_text('{"seed": [11, 22]}')
    (tmp_path / 'train.bin').write_bytes(b'abc')
    (tmp_path / 'dev.bin').write_bytes(b'xyz')
    return tmp_path, cli_runner


@pytest.fixture
def model_free_cli(tmp_path):
    import subprocess
    import sys

    script = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name in ('tmt.main', 'tmt.benchmark', 'mlx') or name.startswith('mlx.'):
        raise ImportError('model import forbidden')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
runpy.run_module(sys.argv.pop(1), run_name='__main__', alter_sys=True)
"""

    def run(*args, module='tmt'):
        return subprocess.run(
            [sys.executable, '-c', script, module, *args], cwd=tmp_path,
            capture_output=True, text=True, timeout=10,
        )

    return run


@pytest.fixture
def manifest_fields():
    return {
        'config', 'seed', 'data_glob', 'data_files', 'updates', 'log_every', 'objective',
        'completed_updates', 'resume', 'initial_optimizer_step', 'started_at',
        'status', 'last_checkpoint', 'sample', 'evaluation', 'trainable_parameters',
        'temporal_gradient',
    }
