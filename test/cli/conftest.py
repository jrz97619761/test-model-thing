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
            [command, *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
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
            [sys.executable, '-c', script, module, *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )

    return run


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
def manual():
    from importlib.resources import files

    return files('tmt').joinpath('commands.md').read_text(encoding='utf-8')


@pytest.fixture
def trained_model(model_factory):
    model = model_factory()
    model.train_step(65, 66, False, True)
    model.train_step(66, 67, True, True)
    assert model.compiled is not None
    return model


@pytest.fixture
def sampling_runs(tmp_path, tiny_settings):
    from cli_support import model_snapshot, read_jsonl
    from tmt.cli import train

    data = tmp_path / 'train.bin'
    data.write_bytes(b'abcdefghij')
    sampled_checkpoint = tmp_path / 'sampled' / 'model.safetensors'
    plain_checkpoint = tmp_path / 'plain' / 'model.safetensors'

    sampled = train(
        tiny_settings,
        sampled_checkpoint,
        str(data),
        4,
        seed=29,
        ce_only=True,
        log_every=4,
        sample_every=2,
        sample_prompt='The ',
        sample_seed=13,
        sample_bytes=3,
    )
    plain = train(
        tiny_settings,
        plain_checkpoint,
        str(data),
        4,
        seed=29,
        ce_only=True,
        log_every=4,
    )
    sampled_snapshot = model_snapshot(sampled)
    plain_snapshot = model_snapshot(plain)
    sample_rows = read_jsonl(sampled_checkpoint.parent / 'manifest' / 'samples.jsonl')

    return {
        'sample_rows': sample_rows,
        'root': tmp_path,
        'data': data,
        'sampled': sampled,
        'sampled_snapshot': sampled_snapshot,
        'plain': plain,
        'plain_snapshot': plain_snapshot,
        'sampled_checkpoint': sampled_checkpoint,
    }


@pytest.fixture
def sampling_resume(sampling_runs):
    from tmt.cli import train

    return train(
        None,
        sampling_runs['sampled_checkpoint'],
        str(sampling_runs['data']),
        2,
        seed=29,
        ce_only=True,
        resume=True,
        log_every=2,
        sample_every=1,
        sample_prompt='The ',
        sample_seed=13,
        sample_bytes=2,
    )


@pytest.fixture
def evaluation_runs(tmp_path, tiny_settings):
    from cli_support import model_snapshot
    from tmt.cli import train

    training_data = tmp_path / 'training.bin'
    training_data.write_bytes(b'abcdefghij')
    development = tmp_path / 'development'
    development.mkdir()
    first = development / 'a.bin'
    second = development / 'b.bin'
    first.write_bytes(b'abcd')
    second.write_bytes(b'uvwxyz')
    evaluation_pattern = str(development / '*.bin')
    monitored_checkpoint = tmp_path / 'monitored' / 'model.safetensors'
    plain_checkpoint = tmp_path / 'plain' / 'model.safetensors'

    monitored = train(
        tiny_settings,
        monitored_checkpoint,
        str(training_data),
        5,
        seed=43,
        ce_only=True,
        log_every=5,
        eval_data=evaluation_pattern,
        eval_every=2,
        eval_max_bytes=7,
    )
    plain = train(
        tiny_settings,
        plain_checkpoint,
        str(training_data),
        5,
        seed=43,
        ce_only=True,
        log_every=5,
    )

    exact_checkpoint = tmp_path / 'exact' / 'model.safetensors'
    train(
        tiny_settings,
        exact_checkpoint,
        str(training_data),
        4,
        seed=43,
        ce_only=True,
        log_every=4,
        eval_data=evaluation_pattern,
        eval_every=2,
        eval_max_bytes=7,
    )

    return {
        'tmp_path': tmp_path,
        'monitored': monitored,
        'monitored_snapshot': model_snapshot(monitored),
        'plain': plain,
        'plain_snapshot': model_snapshot(plain),
        'monitored_checkpoint': monitored_checkpoint,
        'exact_checkpoint': exact_checkpoint,
        'evaluation_pattern': evaluation_pattern,
        'first': first,
        'second': second,
    }


@pytest.fixture
def loss_run(tmp_path, tiny_settings, monkeypatch):
    from cli_support import read_jsonl
    from tmt.cli import train
    from tmt.main import Model

    data = tmp_path / 'train.bin'
    data.write_bytes(b'abcdefg')
    checkpoint = tmp_path / 'run' / 'model.safetensors'
    per_update = []
    original_train_step = Model.train_step

    def capture(model, *args, metrics=None, **kwargs):
        result = original_train_step(model, *args, metrics=metrics, **kwargs)
        per_update.append(
            {
                'loss': float(result[2].item()),
                'components': dict(metrics),
            }
        )
        return result

    monkeypatch.setattr(Model, 'train_step', capture)
    train(tiny_settings, checkpoint, str(data), 5, seed=37, log_every=2)

    return {
        'per_update': per_update,
        'rows': read_jsonl(checkpoint.parent / 'manifest' / 'loss.jsonl'),
    }
