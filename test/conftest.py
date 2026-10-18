import json
from pathlib import Path
from types import SimpleNamespace

import pytest


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
def checkpoint(tmp_path, tiny_settings):
    """Save an untrained tiny checkpoint and return its parts."""
    import mlx.core as mx
    from tmt.cli import new_model, save_checkpoint

    path = tmp_path / 'model.safetensors'
    model = new_model(tiny_settings, 11)
    save_checkpoint(path, model, tiny_settings)
    arrays, metadata = mx.load(str(path), return_metadata=True)
    return SimpleNamespace(
        path=path,
        settings=dict(tiny_settings),
        model=model,
        arrays=arrays,
        metadata=metadata,
    )


@pytest.fixture
def snapshot():
    """Return weights, optimizer state, and block runtime arrays as Python lists."""
    from mlx.utils import tree_flatten

    def take(model):
        tree = {
            'parameters': model.trainable_parameters(),
            'optimizer': model.optimizer.state,
            'runtime': model.snapshot_state(),
        }
        return {name: value.tolist() for name, value in tree_flatten(tree)}

    return take


@pytest.fixture
def jsonl():
    return lambda path: [
        json.loads(line) for line in Path(path).read_text().splitlines()
    ]


@pytest.fixture
def manifest_fields():
    return {
        'config',
        'seed',
        'data_glob',
        'data_files',
        'updates',
        'log_every',
        'objective',
        'completed_updates',
        'resume',
        'initial_optimizer_step',
        'started_at',
        'status',
        'last_checkpoint',
        'sample',
        'evaluation',
        'trainable_parameters',
        'temporal_gradient',
    }


@pytest.fixture
def run_cli(tmp_path, monkeypatch, capsys):
    """Run tmt.cli.main in tmp_path; return (exit code, stdout, stderr)."""
    from tmt.cli import main

    monkeypatch.chdir(tmp_path)

    def run(*args):
        try:
            main(list(args))
            code = 0
        except SystemExit as error:
            code = error.code
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return run


@pytest.fixture
def workspace(tmp_path, monkeypatch, tiny_settings):
    """Run in tmp_path with model.json, grid.json, train.bin, and dev.bin."""
    (tmp_path / 'model.json').write_text(json.dumps(tiny_settings))
    (tmp_path / 'grid.json').write_text('{"seed": [11, 22]}')
    (tmp_path / 'train.bin').write_bytes(b'abc')
    (tmp_path / 'dev.bin').write_bytes(b'xyz')
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def cli(tmp_path):
    import os
    import subprocess
    import sysconfig

    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    command = str(Path(sysconfig.get_path('scripts')) / 'tmt')

    def run(*args, code=0):
        result = subprocess.run(
            [command, *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == code, result.stderr
        assert 'Traceback' not in result.stderr
        return result

    return run


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
            [sys.executable, '-c', script, module, *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )

    return run
