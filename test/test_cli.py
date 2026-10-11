import contextlib
import inspect
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


MANIFEST_FIELDS = {'config', 'seed', 'data_glob', 'data_files', 'updates', 'log_every', 'objective',
                   'completed_updates', 'resume', 'initial_optimizer_step',
                   'started_at', 'status', 'last_checkpoint', 'sample', 'evaluation',
                   'trainable_parameters', 'temporal_gradient'}


def test_version():
    from io import StringIO
    from tmt.cli import main

    installed = Mock(version='0.1.0')
    for metadata, commit in ((None, 'unknown'), ('{"vcs_info":{"commit_id":"abc123"}}', 'abc123')):
        installed.read_text.return_value = metadata
        stdout = StringIO()
        with patch('importlib.metadata.distribution', return_value=installed), contextlib.redirect_stdout(stdout):
            with pytest.raises(SystemExit) as exit:
                main(['--version'])
        assert exit.value.code == 0
        assert stdout.getvalue() == f'tmt 0.1.0, commit {commit}\n'
    code = """
import builtins, runpy
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'tmt.main' or name == 'mlx' or name.startswith('mlx.'):
        raise ImportError('model import forbidden')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
runpy.run_module('tmt', run_name='__main__', alter_sys=True)
"""
    for args in (['--version'], ['--help']):
        result = subprocess.run([sys.executable, '-c', code, *args], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_command_manual(subtests):
    from importlib.resources import files
    from tmt.cli import command_manual

    manual = files('tmt').joinpath('commands.md').read_text(encoding='utf-8')
    assert command_manual() == manual
    script = """
import builtins, runpy
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'tmt.main' or name == 'mlx' or name.startswith('mlx.'):
        raise ImportError('model import forbidden in help')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
runpy.run_module('tmt', run_name='__main__', alter_sys=True)
"""
    with tempfile.TemporaryDirectory() as directory:
        def run(args):
            return subprocess.run([sys.executable, '-c', script, *args], cwd=directory,
                                  capture_output=True, text=True, timeout=10)
        for topic in ('init', 'train', 'sweep', 'evaluate', 'benchmark', 'generate', 'help'):
            expected = '## ' + topic + '\n' + manual.split('## ' + topic + '\n', 1)[1].split('\n## ', 1)[0].rstrip() + '\n'
            assert command_manual(topic) == expected
            for args in (['help', topic], ['--help', topic], ['-h', topic], [topic, '--help'],
                         [topic, '-h'], [topic, '--data', 'missing/*', '--help']):
                with subtests.test(args=args):
                    result = run(args)
                    assert (result.returncode, result.stdout) == (0, expected), result.stderr
        assert run(['help']).stdout == manual
        for args in (['--help'], ['-h']):
            result = run(args)
            assert result.returncode == 0, result.stderr
            assert 'tmt help' in result.stdout
            assert '## init' not in result.stdout
        for args in (['help', 'nonexistent'], ['--help', 'nonexistent'], ['-h', 'nonexistent'], ['nonexistent', '--help']):
            result = run(args)
            assert result.returncode == 2
            assert 'unknown help topic' in result.stderr
        assert list(Path(directory).iterdir()) == []


def test_installed_workflow():
    import os
    import sysconfig
    from tmt.cli import load_model

    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PATH'] = sysconfig.get_path('scripts') + os.pathsep + env.get('PATH', '')
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        def run(*args):
            result = subprocess.run(['tmt', *args], cwd=root, env=env, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stderr
            return result.stdout
        assert '## init' in run('help', 'init')
        run('init')
        (root / 'model.json').write_text('{"dim":4,"layers":1,"spread":4,"block":"rtu"}')
        (root / 'data/train/tiny.bin').write_bytes(b'abcd')
        (root / 'data/dev/tiny.bin').write_bytes(b'wxyz')
        progress = run('train', '--data', 'data/train/*', '--run', 'first', '--updates', '3', '--log-every', '2',
                       '--sample-every', '2', '--sample-bytes', '4', '--sample-prompt', 'A', '--sample-seed', '7',
                       '--eval-data', 'data/dev/*', '--eval-every', '2', '--eval-max-bytes', '3')
        assert 'updates: 2, objective: tmt, loss:' in progress
        assert 'updates: 3, objective: tmt, loss:' in progress
        checkpoint = root / 'runs/first/model.safetensors'
        assert 'sample at update 2, step 2:' in progress
        assert 'evaluation at update 3: BPB' in progress
        samples = [json.loads(line) for line in (checkpoint.parent / 'manifest/samples.jsonl').read_text().splitlines()]
        assert [(row['completed_updates'], row['seed'], row['bytes'], row['prompt']) for row in samples] == [(2, 7, 4, 'A')]
        assert len(Path(samples[0]['path']).read_bytes()) == 4
        evaluation = [json.loads(line) for line in (checkpoint.parent / 'manifest/evaluation.jsonl').read_text().splitlines()]
        assert [(row['completed_updates'], row['input_bytes'], row['targets']) for row in evaluation] == [(2, 3, 2), (3, 3, 2)]
        model, settings = load_model(checkpoint)
        assert (settings['dim'], settings['layers'], settings['spread'], model.optimizer.state['step'].item()) == (4, 1, 4, 3)
        manifest = checkpoint.parent / 'manifest/run.json'
        fresh = json.loads(manifest.read_text())
        assert (fresh['data_glob'], fresh['data_files'], fresh['updates'], fresh['completed_updates']) == ('data/train/*', [str((root / 'data/train/tiny.bin').resolve())], 3, 3)
        assert (fresh['objective'], fresh['resume'], fresh['initial_optimizer_step']) == ('tmt', False, 0)
        assert (fresh['trainable_parameters'], fresh['temporal_gradient']) == (model.count(), model.temporal_gradient)
        (root / 'data/resume').mkdir()
        (root / 'data/resume/next.bin').write_bytes(b'wxyz')
        run('train', str(checkpoint), '--data', 'data/resume/*', '--resume', '--updates', '2', '--seed', '99', '--ce-only')
        assert load_model(checkpoint)[0].optimizer.state['step'].item() == 5
        record = json.loads(manifest.read_text())
        assert set(record) == MANIFEST_FIELDS
        assert (record['status'], record['config'], record['last_checkpoint']) == ('complete', settings, str(checkpoint.resolve()))
        assert (record['trainable_parameters'], record['temporal_gradient']) == (model.count(), model.temporal_gradient)
        assert (record['data_glob'], record['data_files'], record['updates'], record['completed_updates']) == ('data/resume/*', [str((root / 'data/resume/next.bin').resolve())], 2, 2)
        assert (record['seed'], record['objective'], record['resume'], record['initial_optimizer_step']) == (99, 'ce_only', True, 3)
        run('evaluate', str(checkpoint), '--data', 'data/train/*', '--windows', '1', '2', '--output', 'score.json')
        assert json.loads((root / 'score.json').read_text())['targets'] == 3
        run('generate', str(checkpoint), '--prompt', 'The ', '--output', 'sample.bin', '--bytes', '4')
        assert len((root / 'sample.bin').read_bytes()) == 4
        assert '390 targets' in run('benchmark', str(checkpoint))


def test_installed_failure_paths(subtests):
    import os
    import sysconfig

    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    command = str(Path(sysconfig.get_path('scripts')) / 'tmt')
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        def run(*args, success=False):
            result = subprocess.run([command, *args], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode == (0 if success else 1), result.stderr
            assert 'Traceback' not in result.stderr
            return result

        # Missing data must fail before config loading or run-folder creation.
        for extra in (['--run', 'absent'], []):
            result = run('train', '--data', 'missing/*', *extra)
            assert 'no files matched' in result.stderr
            assert list(root.iterdir()) == []

        (root / 'model.json').write_text(json.dumps(_tiny_settings()))
        (root / 'grid.json').write_text('{"seed": [11, 22]}')
        (root / 'train.bin').write_bytes(b'abc')
        (root / 'dev.bin').write_bytes(b'xyz')
        for extra, message in (
            (['--sample-prompt', ''], 'sample-prompt must not be empty'),
            (['--sample-bytes', '0'], 'sample-bytes must be positive'),
            (['--sample-bytes', '-1'], 'sample-bytes must be positive'),
            (['--eval-max-bytes', '0'], 'eval-max-bytes must be positive'),
            (['--eval-max-bytes', '-1'], 'eval-max-bytes must be positive'),
            (['--sample-every', '0'], 'sample-every must be positive'),
            (['--sample-every', '1', '--sample-prompt', ''], 'sample-prompt must not be empty'),
            (['--sample-every', '1', '--sample-bytes', '-1'], 'sample-bytes must be positive'),
            (['--eval-data', 'dev.bin'], 'eval-data and eval-every must be used together'),
            (['--eval-every', '1'], 'eval-data and eval-every must be used together'),
            (['--eval-data', 'dev.bin', '--eval-every', '0'], 'eval-every must be positive'),
            (['--eval-data', 'dev.bin', '--eval-every', '1', '--eval-max-bytes', '0'], 'eval-max-bytes must be positive'),
        ):
            result = run('train', '--data', 'train.bin', '--run', 'invalid', *extra)
            assert message in result.stderr
            assert not (root / 'runs').exists()
            result = run('sweep', '--grid', 'grid.json', '--data', 'train.bin',
                         '--development', 'dev.bin', '--output', 'invalid', *extra)
            assert message in result.stderr
            assert not (root / 'invalid').exists()
        for name in ('', '.', '..', '../escaped-name', str(root / 'absolute-name'),
                     'nested/name', 'trailing/', 'nested\\name'):
            with subtests.test(name=name):
                result = run('train', '--data', 'train.bin', '--run', name)
                assert 'run must be one relative folder name' in result.stderr
                assert not (root / 'runs').exists()
                assert not (root / 'absolute-name').exists()
        for count in ('0', '-1'):
            result = run('train', '--data', 'train.bin', '--run', 'invalid', '--updates', count)
            assert 'updates must be positive' in result.stderr
            assert not (root / 'runs').exists()
            result = run('sweep', '--grid', 'grid.json', '--data', 'train.bin',
                         '--development', 'dev.bin', '--output', 'invalid', '--updates', count)
            assert 'updates must be positive' in result.stderr
            assert not (root / 'invalid').exists()
            result = run('evaluate', 'missing.safetensors', '--data', 'dev.bin', '--windows', count)
            assert 'evaluation windows must be positive' in result.stderr
            result = run('evaluate', 'missing.safetensors', '--data', 'dev.bin', '--max-bytes', count,
                         '--output', 'score.json')
            assert 'max-bytes must be positive' in result.stderr
            assert not (root / 'score.json').exists()
            result = run('sweep', '--grid', 'grid.json', '--data', 'train.bin',
                         '--development', 'dev.bin', '--output', 'invalid', '--max-bytes', count)
            assert 'max-bytes must be positive' in result.stderr
            assert not (root / 'invalid').exists()
            result = run('generate', 'missing.safetensors', '--prompt', 'x',
                         '--output', 'sample.bin', '--log-every', count)
            assert 'log-every must be positive' in result.stderr
            assert not (root / 'sample.bin').exists()
        result = run('generate', 'missing.safetensors', '--prompt', 'x',
                     '--output', 'sample.bin', '--bytes', '-1')
        assert 'bytes must be nonnegative' in result.stderr
        for temperature in ('nan', 'inf', '-inf', '-1'):
            result = run('generate', 'missing.safetensors', '--prompt', 'x',
                         '--output', 'sample.bin', f'--temperature={temperature}')
            assert 'temperature must be a finite nonnegative number' in result.stderr
            assert not (root / 'sample.bin').exists()
        for epochs in ('0', '-1'):
            result = run('benchmark', 'missing.safetensors', '--cola-data', 'missing.tsv',
                         '--epochs', epochs)
            assert 'epochs must be positive with cola-data' in result.stderr
            result = subprocess.run([sys.executable, '-m', 'tmt.benchmark', 'missing.safetensors',
                                     '--cola-data', 'missing.tsv', '--epochs', epochs], cwd=root,
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode == 1
            assert 'epochs must be positive with cola-data' in result.stderr
            assert 'Traceback' not in result.stderr
        for grid in ([4], None, {'dim': 4}, {'dim': []}, {'dim': [4], 'seed': []},
                     {'dim': [4, 0]}, {'unknown': [1]}, {'seed': [11, '22']},
                     {'seed': [11, -1]}, {'seed': [11, 2**64]}):
            with subtests.test(grid=grid):
                (root / 'bad-grid.json').write_text(json.dumps(grid))
                result = run('sweep', '--grid', 'bad-grid.json', '--data', 'train.bin',
                             '--development', 'dev.bin', '--output', 'bad-sweep', '--updates', '1')
                assert not (root / 'bad-sweep').exists()
        args = ['sweep', '--grid', 'grid.json', '--data', 'train.bin',
                '--development', 'dev.bin', '--output', 'sweep', '--updates', '2']
        (root / 'sweep').mkdir()  # An existing empty folder is valid.
        run(*args, success=True)
        before = {p: p.read_bytes() for p in (root / 'sweep').rglob('*') if p.is_file()}
        result = run(*args[:-1], '1')
        assert 'not empty' in result.stderr
        assert before == {p: p.read_bytes() for p in (root / 'sweep').rglob('*') if p.is_file()}
        checkpoint = root / 'sweep/run-0001/model.safetensors'
        for contents in ('{', '[]', '{"unknown": 1}', '{"bound": 30}', '{"dim": true}', '{"rate": 0}',
                         '{"block": "unknown-block"}'):
            (root / 'bad.json').write_text(contents)
            result = run('train', '--data', 'train.bin', '--config', 'bad.json', '--run', 'bad')
            if 'unknown-block' in contents:
                assert 'model config.block must be one of' in result.stderr
            assert not (root / 'runs/bad').exists()
        for grid, message in (({'bound': [30]}, 'model config.bound must be two integers'),
                              ({'block': ['unknown-block']}, 'model config.block must be one of')):
            (root / 'bad-grid.json').write_text(json.dumps(grid))
            result = run('sweep', '--grid', 'bad-grid.json', '--data', 'train.bin',
                         '--development', 'dev.bin', '--output', 'bad-sweep')
            assert message in result.stderr
            assert not (root / 'bad-sweep').exists()
        result = run('evaluate', str(checkpoint), '--data', 'dev.bin', '--max-bytes', '2',
                     '--output', 'score.json', success=True)
        score = json.loads((root / 'score.json').read_text())
        assert (score['input_bytes'], score['targets']) == (2, 1)
        run('generate', 'missing.safetensors', '--prompt', 'x', '--output', 'sample.bin')
        (root / 'invalid.safetensors').write_bytes(b'invalid')
        run('evaluate', 'invalid.safetensors', '--data', 'dev.bin')
        (root / 'short.bin').write_bytes(b'x')
        run('train', '--data', 'short.bin', '--run', 'no-targets', '--updates', '1')
        record = json.loads((root / 'runs/no-targets/manifest/run.json').read_text())
        assert (record['status'], record['last_checkpoint']) == ('failed', None)
        assert (record['updates'], record['completed_updates'], record['initial_optimizer_step']) == (1, 0, 0)
        result = run('evaluate', str(checkpoint), '--data', 'short.bin')
        assert 'at least one target' in result.stderr


def test_cli_rejects_seeds_before_work(subtests):
    script = """
import builtins, runpy
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'tmt.main' or name == 'tmt.benchmark' or name == 'mlx' or name.startswith('mlx.'):
        raise ImportError('model import forbidden before seed validation')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
runpy.run_module('tmt', run_name='__main__', alter_sys=True)
"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / 'protected.bin'
        output.write_bytes(b'keep')
        commands = (
            ['train', '--data', 'missing/*', '--run', 'invalid'],
            ['train', 'missing.safetensors', '--data', 'missing/*', '--resume'],
            ['generate', 'missing.safetensors', '--prompt', 'X', '--output', str(output)],
            ['benchmark', 'missing.safetensors'],
            ['sweep', '--grid', 'missing.json', '--data', 'missing/*',
             '--development', 'missing/*', '--output', 'invalid'],
        )
        for seed in (-1, 2**64):
            for args in commands:
                options = ('--seed', '--sample-seed') if args[0] in ('train', 'sweep') else ('--seed',)
                for option in options:
                    with subtests.test(command=args[0], option=option, seed=seed):
                        result = subprocess.run([sys.executable, '-c', script, *args, option, str(seed)],
                                                cwd=root, capture_output=True, text=True, timeout=10)
                        assert result.returncode == 1, result.stderr
                        assert f'{option[2:]} must be between 0 and {2 ** 64 - 1}' in result.stderr
                        assert 'Traceback' not in result.stderr
            result = subprocess.run([sys.executable, '-m', 'tmt.benchmark', 'missing.safetensors',
                                     '--seed', str(seed)], cwd=root, capture_output=True, text=True, timeout=10)
            assert result.returncode == 1, result.stderr
            assert 'seed must be between' in result.stderr
            assert 'Traceback' not in result.stderr
            assert output.read_bytes() == b'keep'
            assert list(root.iterdir()) == [output]


def test_seed_validation_before_direct_model_work(subtests):
    from tmt.cli import generate, load_model, new_model, sweep, train
    from tmt.benchmark import run

    with patch('tmt.cli.load_config') as config, patch('tmt.cli._validate_model_settings') as settings:
        for seed in (-1, 2**64, True, 1.5):
            calls = (
                lambda: new_model({}, seed),
                lambda: load_model('missing.safetensors', seed),
                lambda: generate('missing.safetensors', 'X', 'unused.bin', seed),
                lambda: train({}, 'unused/model.safetensors', 'missing/*', seed=seed),
                lambda: train({}, 'unused/model.safetensors', 'missing/*', sample_seed=seed),
                lambda: sweep('missing.json', 'missing/*', 'missing/*', 'unused', seed=seed),
                lambda: sweep('missing.json', 'missing/*', 'missing/*', 'unused', sample_seed=seed),
            )
            for call in calls:
                with subtests.test(seed=seed, call=call), pytest.raises(ValueError, match='seed must be'):
                    call()
            model = Mock()
            with pytest.raises(ValueError, match='seed must be'):
                run('missing.safetensors', model=model, seed=seed)
            model.freeze.assert_not_called()
        config.assert_not_called()
        settings.assert_not_called()


def test_init():
    from tmt.cli import load_config, main
    from tmt.main import Model

    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
        main(['init'])
        saved = json.loads((Path(directory) / 'model.json').read_text())
        assert saved == load_config()
        assert set(saved) == set(inspect.signature(Model).parameters)
        templates = {Path(directory) / name for name in ('model.json', 'grid.json', 'run.example.json')}
        assert set(Path(directory).rglob('*.json')) == templates
        assert all((Path(name).is_dir() for name in ('data/train', 'data/dev', 'runs')))
        assert json.loads(Path('grid.json').read_text()) == {'dim': [4, 8], 'layers': [1], 'spread': [4], 'block': ['rtu'], 'seed': [11, 22]}
        template = json.loads(Path('run.example.json').read_text())
        assert template == json.loads((repo / 'examples' / 'run.example.json').read_text())
        assert set(template) == MANIFEST_FIELDS
        assert (template['config'], template['seed'], template['data_glob'], template['data_files']) == (saved, 11, 'data/train/*', [])
        assert (template['updates'], template['objective'], template['completed_updates']) == (1000, 'tmt', 0)
        assert (template['resume'], template['initial_optimizer_step'], template['started_at'], template['last_checkpoint']) == (False, 0, None, None)

        explicit = Path(directory) / 'custom' / 'settings.json'
        main(['init', '--output', str(explicit)])
        assert json.loads(explicit.read_text()) == saved
        assert set(Path(directory).rglob('*.json')) == templates | {explicit}

    assert load_config(repo / 'examples' / 'model.json') == load_config()
    tiny = load_config(repo / 'examples' / 'tiny-model.json')
    assert (tiny['dim'], tiny['layers'], tiny['spread']) == (4, 1, 4)
    assert json.loads((repo / 'examples' / 'grid.json').read_text()) == {
        'block': ['rtu'], 'dim': [4, 8], 'layers': [1], 'spread': [4], 'seed': [11, 22],
    }


def test_init_preserves_files():
    from tmt.cli import main

    with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
        main(['init'])
        Path('model.json').write_text('{"dim": 4}')
        before = {p: p.read_bytes() for p in Path('.').rglob('*') if p.is_file()}
        for args in (['init'], ['init', '--output', 'model.json']):
            with pytest.raises(SystemExit) as exit: main(args)
            assert exit.value.code == 1
        assert before == {p: p.read_bytes() for p in before}


def test_config():
    from tmt.cli import load_config
    from tmt.main import Model

    defaults = load_config()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model.json'
        path.write_text('{}')
        assert load_config(path) == defaults
        path.write_text('{"dim": 4}')
        settings = load_config(path)
        assert (settings['dim'], settings['layers'], settings['bound']) == (4, 16, [40000, 120000])

    parameters = list(inspect.signature(Model).parameters.values())
    parameters.append(inspect.Parameter('extra', inspect.Parameter.KEYWORD_ONLY, default=17))
    extended = Mock()
    extended.__signature__ = inspect.Signature(parameters)
    with patch('tmt.main.Model', extended):
        assert load_config()['extra'] == 17


def test_invalid_config_values(subtests):
    from tmt.cli import _validate_metadata, load_config, new_model
    from tmt.main import Model

    cases = []
    for name in ('dim', 'layers', 'spread'):
        cases.extend((name, value) for value in (True, 0, -1, 4.0, '4', None, [], {}))
    for name in ('temp', 'rate'):
        cases.extend((name, value) for value in (True, -1, '1', None, [], {},
                                               float('nan'), float('inf'), -float('inf'), 10 ** 400))
    cases.append(('rate', 0))
    cases.append(('block', 'unknown-block'))
    cases.extend(('bound', value) for value in (30, None, '0,1', {}, [], [0], [0, 1, 2],
                                               [-1, 1], [1, 1], [2, 1], [False, 1], [0, 1.0]))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model.json'
        for name, value in cases:
            with subtests.test(name=name, value=value):
                path.write_text(json.dumps({name: value}))
                with pytest.raises(ValueError, match=f'model config.{name}'):
                    load_config(path)
                settings = dict(_tiny_settings(), **{name: value})
                with pytest.raises(ValueError, match=f'model config.{name}'):
                    new_model(settings, 11)
                raw = json.dumps({'version': 1, 'model': settings})
                with pytest.raises(ValueError, match=f'checkpoint.model.{name}'):
                    _validate_metadata(raw)
        with pytest.raises(ValueError, match='unknown block'):
            Model(dim=4, layers=1, spread=4, block='unknown-block')
        path.write_text('{"dim": 1, "layers": 1, "spread": 1, "temp": 0, "rate": 1, "bound": [0, 1]}')
        settings = load_config(path)
        assert (settings['temp'], settings['rate'], settings['bound']) == (0, 1, [0, 1])


def test_evaluation_byte_limits(subtests):
    from tmt.cli import build_parser, documents, evaluate_command, sweep

    for limit in (0, -1):
        with subtests.test(limit=limit), patch('tmt.cli.load_model') as load:
            args = build_parser().parse_args(['evaluate', 'missing.safetensors', '--data', 'missing/*',
                                              '--max-bytes', str(limit)])
            with pytest.raises(ValueError, match='max-bytes must be positive'):
                evaluate_command(args)
            load.assert_not_called()
            with pytest.raises(ValueError, match='max-bytes must be positive'):
                sweep('missing.json', 'missing/*', 'missing/*', 'unused', max_bytes=limit)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'a.bin').write_bytes(b'abcd')
        (root / 'b.bin').write_bytes(b'efgh')
        pattern = str(root / '*.bin')
        assert list(documents(pattern, 6)) == [b'abcd', b'ef']
        assert list(documents(pattern, 1)) == [b'a']


def _tiny_settings():
    return {'dim': 4, 'layers': 1, 'spread': 4, 'temp': 0.75,
            'rate': 0.0005, 'bound': [40000, 120000], 'block': 'rtu'}


def test_train():
    import mlx.core as mx
    from tmt.cli import main, train
    from tmt.main import Model

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / 'train.bin'
        data.write_bytes(bytes((0, 255, 65, 10)))
        calls = []
        train_step, step = Model.train_step, Model.step

        def record_train(model, current, target, end, ce_only=False, **kwargs):
            calls.append((current, target, end))
            return train_step(model, current, target, end, ce_only, **kwargs)

        from io import StringIO
        stdout = StringIO()
        with (contextlib.redirect_stdout(stdout),
              patch.object(Model, 'sample', side_effect=AssertionError('sampled')),
              patch.object(Model, 'train_step', record_train),
              patch.object(Model, 'step', autospec=True, side_effect=step) as steps):
            first = train(_tiny_settings(), root / 'one' / 'model.safetensors', str(data), 3, 11)
            second = train(_tiny_settings(), root / 'two' / 'model.safetensors', str(data), 3, 11)
            resumed = train(None, root / 'one' / 'model.safetensors', str(data), 2, 11, resume=True)

        assert stdout.getvalue().count('run folder:') == 3
        assert stdout.getvalue().count('saved checkpoint:') == 3
        assert stdout.getvalue().count('objective:') == 3
        assert 'seed: 11, config:' in stdout.getvalue()
        assert '\r' not in stdout.getvalue()

        with contextlib.chdir(root), patch('tmt.cli.train') as command:
            Path('model.json').write_text(json.dumps(_tiny_settings()))
            main(['train', '--data', str(data), '--run', 'named', '--updates', '3'])
            assert command.call_args.args[:2] == (_tiny_settings(), Path('runs/named/model.safetensors'))
            Path('runs/named').mkdir(parents=True)  # The mock replaces folder creation in train.
            main(['train', '--data', str(data)])
            assert command.call_args.args[1].parent.parent == Path('runs')
            with pytest.raises(SystemExit) as exit: main(['train', '--data', str(data), '--run', 'named'])
            assert exit.value.code == 1
            for args in (['--run', 'bad', 'model.safetensors'], ['--resume'], ['--resume', '--run', 'bad'],
                         ['model.safetensors', '--resume', '--config', 'model.json']):
                with pytest.raises(SystemExit) as exit: main(['train', '--data', str(data), *args])
                assert exit.value.code == 2

        expected = [(0, 255, False), (255, 65, False), (65, 10, True)]
        assert calls == expected + expected + expected[:2]
        assert [call.args[1].item() for call in steps.call_args_list if call.kwargs.get('frozen')] == [10, 10]
        assert [int(model.optimizer.state['step'].item()) for model in (first, second, resumed)] == [3, 3, 5]
        assert mx.array_equal(first.decoder.decode.weight, second.decoder.decode.weight).item()


def test_manifest(subtests):
    from datetime import datetime, timezone
    from tmt.cli import train
    from tmt.main import Model

    original_dump, records = json.dump, []

    def capture(record, *args, **kwargs):
        records.append(dict(record))
        return original_dump(record, *args, **kwargs)

    train_step = Model.train_step
    for ce_only in (False, True):
        with subtests.test(ce_only=ce_only), tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            root = Path(directory).resolve()
            Path('data').mkdir()
            paths, calls = {}, []
            for name, content in (('a.bin', b'ab'), ('b.bin', b'cd'), ('c.bin', b'ef')):
                data = root / 'data' / name
                data.write_bytes(content)
                paths[content[0]] = str(data)

            def record_train(model, current, target, end, objective=False, **kwargs):
                calls.append((paths[current], objective))
                return train_step(model, current, target, end, objective, **kwargs)

            checkpoint = root / 'run' / 'model.safetensors'
            records.clear()
            with (patch('tmt.cli.json.dump', side_effect=capture) as writes,
                  patch.object(Model, 'train_step', record_train)):
                trained = train(_tiny_settings(), checkpoint, 'data/*.bin', 4, 11, ce_only)

            assert writes.call_count == 2
            assert set(records[0]) == MANIFEST_FIELDS
            assert (records[0]['trainable_parameters'], records[0]['temporal_gradient']) == (
                trained.count(), trained.temporal_gradient
            )
            for record in records:
                assert (record['config'], record['seed'], record['data_glob']) == (_tiny_settings(), 11, 'data/*.bin')
                assert set(record['data_files']) == set(paths.values())
                assert record['data_files'] == [path for path, _ in calls[:3]]
                assert (record['updates'], record['objective'], record['resume'], record['initial_optimizer_step']) == (4, 'ce_only' if ce_only else 'tmt', False, 0)
            assert calls == [(path, ce_only) for path in records[0]['data_files'] + records[0]['data_files'][:1]]
            assert [(record['status'], record['completed_updates'], record['last_checkpoint']) for record in records] == [('running', 0, None), ('complete', 4, str(checkpoint.resolve()))]
            assert datetime.fromisoformat(records[0]['started_at']).utcoffset() == timezone.utc.utcoffset(None)
            assert json.loads((checkpoint.parent / 'manifest' / 'run.json').read_text()) == records[1]
            assert checkpoint.is_file()


def test_manifest_partial_progress(subtests):
    from tmt.cli import train
    from tmt.main import Model

    train_step = Model.train_step
    for error_type, status in ((OSError, 'failed'), (KeyboardInterrupt, 'interrupted')):
        for resume in (False, True):
            with subtests.test(status=status, resume=resume), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = root / 'train.bin'
                data.write_bytes(b'abcdef')
                checkpoint = root / 'run' / 'model.safetensors'
                original = None
                if resume:
                    train(_tiny_settings(), checkpoint, str(data), 2, 11)
                    original = checkpoint.read_bytes()
                completed = 0

                def fail(model, current, target, end, ce_only=False, **kwargs):
                    nonlocal completed
                    if completed == 1:
                        raise error_type('injected update failure')
                    result = train_step(model, current, target, end, ce_only, **kwargs)
                    completed += 1
                    return result

                with patch.object(Model, 'train_step', fail), pytest.raises(error_type):
                    train(None if resume else _tiny_settings(), checkpoint, str(data), 3, 17, True, resume)

                record = json.loads((checkpoint.parent / 'manifest/run.json').read_text())
                assert set(record) == MANIFEST_FIELDS
                assert (record['data_glob'], record['data_files'], record['updates'], record['completed_updates']) == (str(data), [str(data.resolve())], 3, 1)
                assert (record['status'], record['seed'], record['objective'], record['resume'], record['initial_optimizer_step']) == (status, 17, 'ce_only', resume, 2 if resume else 0)
                assert record['last_checkpoint'] == (str(checkpoint.resolve()) if resume else None)
                metrics = [json.loads(line) for line in (checkpoint.parent / 'manifest/loss.jsonl').read_text().splitlines()]
                assert [(row['completed_updates'], row['objective']) for row in metrics] == [(1, 'ce_only')]
                assert math.isfinite(metrics[0]['loss'])
                assert metrics[0]['interval_updates'] == 1
                assert metrics[0]['optimizer_step'] == (3 if resume else 1)
                assert metrics[0]['loss'] == metrics[0]['mean_loss']
                if resume:
                    assert checkpoint.read_bytes() == original
                else:
                    assert not checkpoint.exists()


def test_train_loss_intervals(subtests, assert_almost_equal):
    from io import StringIO
    import mlx.core as mx
    from tmt.cli import train
    from tmt.main import Model

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / 'train.bin'
        data.write_bytes(b'abcdef')
        for updates, interval, expected in ((5, 2, [2, 4, 5]), (4, 2, [2, 4]), (3, 100, [3])):
            with subtests.test(updates=updates, interval=interval):
                checkpoint = root / 'model.safetensors'
                losses, stdout = [], StringIO()
                original = Model.train_step

                def capture(model, *args, **kwargs):
                    result = original(model, *args, **kwargs)
                    losses.append(float(result[2].item()))
                    return result

                with (patch.object(Model, 'train_step', capture), contextlib.redirect_stdout(stdout),
                      patch('tmt.cli.time.perf_counter', side_effect=[0, 2, 4, 6])):
                    model = train(_tiny_settings(), checkpoint, str(data), updates,
                                  ce_only=True, log_every=interval)
                metrics = [json.loads(line) for line in (root / 'manifest/loss.jsonl').read_text().splitlines()]
                assert [{name: row[name] for name in ('completed_updates', 'objective', 'loss')} for row in metrics] == [dict(completed_updates=index, objective='ce_only', loss=losses[index - 1]) for index in expected]
                previous = 0
                for number, row in enumerate(metrics, 1):
                    count = row['completed_updates'] - previous
                    mean = math.fsum(losses[previous:row['completed_updates']]) / count
                    assert_almost_equal(row['mean_loss'], mean)
                    assert_almost_equal(row['mean_loss_components']['cross_entropy'], mean)
                    assert {name: value for name, value in row['mean_loss_components'].items() if name != 'cross_entropy'} == dict(variance=0.0, latent_prediction=0.0, stop=0.0)
                    assert (row['interval_updates'], row['elapsed_seconds'], row['updates_per_second'], row['optimizer_step']) == (count, number * 2, count / 2, row['completed_updates'])
                    previous = row['completed_updates']
                assert stdout.getvalue().count('objective:') == len(expected)
                assert f'updates: {updates}, objective: ce_only, loss: {losses[-1]:.6f}' in stdout.getvalue()
                assert json.loads((root / 'manifest/run.json').read_text())['log_every'] == interval
                with contextlib.redirect_stdout(StringIO()):
                    reference = train(_tiny_settings(), root / 'reference/model.safetensors',
                                      str(data), updates, ce_only=True)
                assert mx.array_equal(model.decoder.decode.weight, reference.decoder.decode.weight).item()


def test_evaluate(assert_almost_equal):
    import mlx.core as mx
    from tmt.benchmark import evaluate

    class FakeBlock:
        state_fields = ('cache',)
        trace_fields = ('history',)

        def __init__(self):
            self.cache = mx.array([91.])
            self.history = mx.array([[93.]])

        def reset(self):
            self.cache = mx.zeros_like(self.cache)
            self.history = mx.zeros_like(self.history)

        def snapshot_state(self):
            saved = {name: mx.array(getattr(self, name))
                     for name in self.state_fields + self.trace_fields}
            mx.eval(*saved.values())
            return saved

        def restore_state(self, saved):
            for name, value in saved.items():
                setattr(self, name, value)

        def eval_state(self):
            mx.eval(*(getattr(self, name)
                      for name in self.state_fields + self.trace_fields))

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
            logits = (mx.zeros((256,)) if self.uniform else
                      mx.array([state / 5 if i == byte else 0 for i in range(256)]))
            return ((None, [], []), (logits, mx.zeros((1,))))

    model = FakeModel()
    saved = model.snapshot_state()
    def unchanged():
        current = model.snapshot_state()
        assert len(current) == len(saved)
        for actual, expected in zip(current, saved):
            assert actual.keys() == expected.keys()
            assert all(mx.array_equal(actual[name], expected[name]).item()
                       for name in expected)
    model.uniform = True
    for windows in ((0,), (-1,), (1, 0)):
        with pytest.raises(ValueError, match='windows must be positive'):
            evaluate(model, [b'ab'], windows)
        unchanged()
    for documents in ([], [b''], [b'a', b'b']):
        with pytest.raises(ValueError, match='at least one target'):
            evaluate(model, documents)
        unchanged()
    uniform_score = evaluate(model, [b'aa'])
    assert_almost_equal(uniform_score['bpb'], 8.0, delta=1e-6)
    assert uniform_score['common_bpb'] is None
    model.uniform, documents, windows = False, [b'aa', b'abcde', b'', b'xy', b'pqrs'], (1, 2)

    def manual(prefix, target):
        state = sum(byte * 3 ** power for byte, power in zip(prefix, reversed(range(len(prefix))))) % 17
        return math.log(255 + math.exp(state / 5)) - (state / 5 if prefix[-1] == target else 0)

    def total(window=None, common=False):
        return math.fsum(manual(document[max(0, offset - window):offset] if window else document[:offset], document[offset])
                         for document in documents for offset in range(1, len(document)) if not common or offset >= 2)
    result = evaluate(model, documents, windows); assert (result['input_bytes'], result['targets'], result['common_targets']) == (13, 9, 5)
    assert_almost_equal(result['bpb'], total() / (9 * math.log(2)), delta=1e-5)
    assert_almost_equal(result['common_bpb'], total(common=True) / (5 * math.log(2)), delta=1e-5)
    for score, window in zip(result['windows'], windows):
        common = total(window, True) / (5 * math.log(2))
        assert_almost_equal(score['bpb'], total(window) / (9 * math.log(2)), delta=1e-5)
        assert_almost_equal(score['common_bpb'], common, delta=1e-5)
        assert_almost_equal(score['delta_bpb'], common - result['common_bpb'], delta=1e-5)
    unchanged()
    short = evaluate(model, [b'ab'], (3,))
    assert (short['common_targets'], short['common_bpb'], short['windows'][0]['delta_bpb']) == (0, None, None)
    with patch('tmt.benchmark._nll', side_effect=RuntimeError('injected score failure')), pytest.raises(RuntimeError, match='injected'): evaluate(model, documents, windows)
    unchanged()


def test_benchmark(assert_almost_equal):
    import mlx.core as mx
    import mlx.utils as util
    from tmt.benchmark import BYTE_DOCUMENTS, CONTEXT_WINDOWS, mcc, run
    from tmt.cli import new_model, save_checkpoint

    settings = {'dim': 4, 'layers': 1, 'spread': 4, 'temp': 0.75, 'rate': 0.0005,
                'bound': [40000, 120000], 'block': 'rtu'}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = new_model(settings, 11)
        checkpoint = root / 'model.safetensors'
        save_checkpoint(checkpoint, model, settings)
        fields = ('states', 'decaytrace', 'embedtrace')
        before = {name: mx.array(value) for name, value in util.tree_flatten(model.parameters()) if not any(field in name for field in fields)}
        optimizer_before = dict(util.tree_flatten(model.optimizer.state))
        mx.eval(*before.values(), *optimizer_before.values())
        data = root / 'cola.tsv'
        data.write_text('src\t1\tx\talpha\nsrc\t0\tx\tbeta\nsrc\t0\tx\tgamma\nsrc\t1\tx\tdelta\n')
        from io import StringIO
        stdout = StringIO()
        states = [mx.array(getattr(model.blocks[0], field)) for field in fields]
        with contextlib.redirect_stdout(stdout):
            byte_only = run(str(checkpoint), model=model)
        assert byte_only['epochs'] == []
        assert byte_only['byte_scores']['targets'] == 390
        assert '104 targets' in stdout.getvalue()
        assert all((mx.array_equal(value, getattr(model.blocks[0], field)).item() for value, field in zip(states, fields)))
        with contextlib.redirect_stdout(stdout):
            result = run(str(checkpoint), 1, 0.5, str(data), model=model, seed=11)
        assert byte_only['byte_scores'] == result['byte_scores']
        assert result['feature_convention'] == 'recurrent-state'
        assert 'CoLA feature convention: recurrent-state' in stdout.getvalue()
        for partition in ('train', 'held'):
            score = result['epochs'][0][partition]
            assert sum((score[key] for key in ('tp', 'tn', 'fp', 'fn'))) == 2
        from unittest.mock import patch
        with (patch('tmt.benchmark.evaluate', return_value={
                  'bpb': 8.0, 'targets': 1, 'common_bpb': 8.0, 'common_targets': 1, 'windows': [],
              }), patch.object(model.blocks[-1], 'features', return_value=mx.zeros((model.dim, 1))),
              pytest.raises(ValueError, match='CoLA features must be an MLX array with shape')):
            run(str(checkpoint), 1, 0.5, str(data), model=model, seed=11)
        with pytest.raises(FileNotFoundError): run(str(checkpoint), data=str(root / 'missing.tsv'), model=model)

    scores = result['byte_scores']
    assert (sum(map(len, BYTE_DOCUMENTS)), scores['input_bytes'], scores['targets'], scores['common_targets']) == (393, 393, 390, 104)
    assert tuple((item['window'] for item in scores['windows'])) == CONTEXT_WINDOWS
    held = result['epochs'][0]['held']
    denominator = math.sqrt((held['tp'] + held['fp']) * (held['tp'] + held['fn']) * (held['tn'] + held['fp']) * (held['tn'] + held['fn']))
    assert_almost_equal(held['mcc'], (held['tp'] * held['tn'] - held['fp'] * held['fn']) * 100 / denominator if denominator else 0)
    assert mcc(1, 1, 0, 0) == 100.0
    after = dict(util.tree_flatten(model.parameters()))
    assert all((mx.array_equal(value, after[name]).item() for name, value in before.items()))
    optimizer_after = dict(util.tree_flatten(model.optimizer.state))
    assert all((mx.array_equal(value, optimizer_after[name]).item() for name, value in optimizer_before.items()))


def _checkpoint(path):
    import mlx.core as mx
    from tmt.cli import new_model, save_checkpoint

    settings = {'dim': 4, 'layers': 1, 'spread': 4, 'temp': 0.75, 'rate': 0.0005,
                'bound': [40000, 120000], 'block': 'rtu'}
    model = new_model(settings, 11)
    save_checkpoint(path, model, settings)
    arrays, metadata = mx.load(path, return_metadata=True)
    mx.eval(arrays)
    return mx, settings, model, arrays, metadata


@pytest.fixture
def saved_checkpoint(tmp_path):
    path = tmp_path / 'model.safetensors'
    mx, settings, model, arrays, metadata = _checkpoint(path)
    return path, mx, settings, model, arrays, metadata


def _assert_rejected(path, mx, load_model, arrays, metadata, *parts):
    mx.save_safetensors(path, arrays, metadata=metadata)
    with pytest.raises(ValueError) as error: load_model(path)
    message = str(error.value)
    for part in parts: assert part in message


def test_checkpoint_names(subtests, saved_checkpoint):
    from tmt.cli import load_model

    (path, mx, _, _, arrays, metadata) = saved_checkpoint
    key = next(name for name in arrays if name.startswith('m.'))
    missing = dict(arrays)
    missing.pop(key)
    extra = dict(arrays)
    extra['o.extra'] = next(iter(arrays.values()))
    cases = [('missing model tensor', missing, key), ('extra optimizer tensor', extra, 'o.extra')]
    for label, tensors, name in cases:
        with subtests.test(label=label): _assert_rejected(path, mx, load_model, tensors, metadata, name)


def test_checkpoint_shapes(saved_checkpoint):
    from tmt.cli import load_model

    (path, mx, _, _, arrays, metadata) = saved_checkpoint
    name = 'm.decoder.decode.weight'
    arrays[name] = mx.zeros((1,), dtype=arrays[name].dtype)
    _assert_rejected(path, mx, load_model, arrays, metadata, 'decoder.decode.weight', 'shape')


def test_checkpoint_dtypes(saved_checkpoint):
    from tmt.cli import load_model

    (path, mx, _, _, arrays, metadata) = saved_checkpoint
    name = next(key for key in arrays if key.startswith('o.') and key.endswith('.m'))
    arrays[name] = mx.zeros(arrays[name].shape, dtype=mx.int32)
    _assert_rejected(path, mx, load_model, arrays, metadata, name[2:], 'dtype')


def test_generate(subtests):
    from io import StringIO
    from tmt.cli import generate

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model.safetensors'
        _checkpoint(path)
        original = path.read_bytes()
        first_path, second_path = Path(directory) / 'first.bin', Path(directory) / 'second.bin'
        stdout = StringIO()
        with contextlib.redirect_stdout(stdout):
            first = generate(path, 'The ', first_path, 11, 16, log_every=5)
        assert [line for line in stdout.getvalue().splitlines() if line.startswith('generated bytes:')] == ['generated bytes: 5/16', 'generated bytes: 10/16', 'generated bytes: 15/16']
        second = generate(path, 'The ', second_path, 11, 16)
        assert (len(first), len(second)) == (16, 16)
        assert first == second
        assert first_path.read_bytes() == first
        assert second_path.read_bytes() == second
        assert path.read_bytes() == original
        for seed in (0, 2**64 - 1):
            with subtests.test(seed=seed):
                assert len(generate(path, 'X', first_path, seed, 1)) == 1
        assert generate(path, 'The ', first_path, count=0) == b''
        assert first_path.read_bytes() == b''
        with pytest.raises(ValueError, match='bytes must be nonnegative'):
            generate(path, 'The ', first_path, count=-1)
        assert first_path.read_bytes() == b''
        minimum = generate(path, 'The ', first_path, count=2, temperature=0.1)
        for temperature in (0, 0.05):
            with subtests.test(temperature=temperature):
                assert generate(path, 'The ', first_path, count=2, temperature=temperature) == minimum


def test_generate_rejects_invalid_temperature_before_output(subtests):
    from tmt.cli import generate

    with tempfile.TemporaryDirectory() as directory, patch('tmt.cli.load_model') as load:
        output = Path(directory) / 'sample.bin'
        for existing in (False, True):
            if existing:
                output.write_bytes(b'keep')
            for temperature in (float('nan'), float('inf'), -float('inf'), -0.1):
                with subtests.test(existing=existing, temperature=temperature):
                    with pytest.raises(ValueError, match='temperature must be a finite nonnegative number'):
                        generate('missing.safetensors', 'The ', output, temperature=temperature)
                    if existing:
                        assert output.read_bytes() == b'keep'
                    else:
                        assert not output.exists()
        load.assert_not_called()


def test_benchmark_rejects_nonpositive_cola_epochs_before_work(subtests):
    from tmt.benchmark import run

    with patch('tmt.cli.load_model') as load, patch('tmt.benchmark.evaluate') as evaluate:
        model = Mock()
        for epochs in (0, -1):
            for supplied in (None, model):
                with subtests.test(epochs=epochs, supplied=supplied is not None):
                    with pytest.raises(ValueError, match='epochs must be positive with cola-data'):
                        run('missing.safetensors', epochs, data='missing.tsv', model=supplied)
        load.assert_not_called()
        evaluate.assert_not_called()
        model.freeze.assert_not_called()


def test_sweep_rejects_invalid_grid_before_output(subtests):
    from tmt.cli import sweep

    cases = [(grid, 'sweep grid must be a JSON object') for grid in (None, 4, [], 'dim')]
    cases.extend(({'dim': axis}, "sweep grid field 'dim' must be a nonempty array")
                 for axis in (4, None, '4', {}, [], True))
    cases.extend((({'dim': [4], 'seed': []}, "sweep grid field 'seed' must be a nonempty array"),
                  ({'dim': [4, 0]}, 'model config.dim must be a positive integer'),
                  ({'seed': [11, True]}, 'sweep grid seed must be an integer'),
                  ({'seed': [11, -1]}, 'sweep grid seed must be between'),
                  ({'seed': [11, 2**64]}, 'sweep grid seed must be between')))
    with tempfile.TemporaryDirectory() as directory, patch('tmt.cli.train') as train:
        root = Path(directory)
        config, grid_path, output = root / 'model.json', root / 'grid.json', root / 'sweep'
        config.write_text(json.dumps(_tiny_settings()))
        for existing in (False, True):
            if existing:
                output.mkdir()
            for grid, message in cases:
                with subtests.test(existing=existing, grid=grid):
                    grid_path.write_text(json.dumps(grid))
                    with pytest.raises(ValueError, match=message):
                        sweep(grid_path, 'missing/*', 'missing/*', output, config, updates=1)
                    if existing:
                        assert list(output.iterdir()) == []
                    else:
                        assert not output.exists()
        train.assert_not_called()


def test_input_failures_allow_retry(subtests):
    from io import StringIO
    from tmt.cli import main, train

    with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
        root = Path(directory)
        Path('model.json').write_text(json.dumps(_tiny_settings()))
        Path('grid.json').write_text('{"seed": [11]}')
        Path('train.bin').write_bytes(b'ab')
        Path('dev.bin').write_bytes(b'cd')
        Path('not-a-file').mkdir()
        for selection in ('absent/*', 'not-a-file'):
            for role in ('train', 'evaluation'):
                args = ['train', '--data', selection if role == 'train' else 'train.bin',
                        '--run', 'retry', '--updates', '1']
                if role == 'evaluation':
                    args += ['--eval-data', selection, '--eval-every', '1']
                with subtests.test(command='train', role=role, selection=selection):
                    with contextlib.redirect_stderr(StringIO()), pytest.raises(SystemExit) as exit:
                        main(args)
                    assert exit.value.code == 1
                    assert not Path('runs/retry').exists()
            for role in ('train', 'evaluation', 'development'):
                args = ['sweep', '--grid', 'grid.json',
                        '--data', selection if role == 'train' else 'train.bin',
                        '--development', selection if role == 'development' else 'dev.bin',
                        '--output', 'retry-sweep', '--updates', '1']
                if role == 'evaluation':
                    args += ['--eval-data', selection, '--eval-every', '1']
                with subtests.test(command='sweep', role=role, selection=selection):
                    with contextlib.redirect_stderr(StringIO()), pytest.raises(SystemExit) as exit:
                        main(args)
                    assert exit.value.code == 1
                    assert not Path('retry-sweep').exists()
        with patch('tmt.cli.new_model') as model:
            with pytest.raises(FileNotFoundError):
                train(_tiny_settings(), root / 'direct/model.safetensors',
                      'train.bin', updates=1, eval_data='absent/*', eval_every=1)
            model.assert_not_called()
            assert not (root / 'direct').exists()
        with contextlib.redirect_stdout(StringIO()):
            main(['train', '--data', 'train.bin', '--run', 'retry', '--updates', '1'])
            main(['sweep', '--grid', 'grid.json', '--data', 'train.bin',
                  '--development', 'dev.bin', '--output', 'retry-sweep', '--updates', '1'])
        assert json.loads(Path('runs/retry/manifest/run.json').read_text())['status'] == 'complete'
        record = json.loads(Path('retry-sweep/sweep.json').read_text())
        assert (record['status'], record['completed_runs'], record['error']) == ('complete', 1, None)


def test_sweep_records_execution_failure_and_preserves_results(subtests):
    from io import StringIO
    from tmt.cli import sweep

    for error_type, status in ((OSError, 'failed'), (KeyboardInterrupt, 'interrupted')):
        with subtests.test(status=status), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, grid, data = root / 'model.json', root / 'grid.json', root / 'data.bin'
            config.write_text(json.dumps(_tiny_settings()))
            grid.write_text('{"seed": [11, 22]}')
            data.write_bytes(b'ab')
            output = root / 'sweep'
            completed_model = Mock()
            completed_model.count.return_value = 7
            completed_model.temporal_gradient = ['rtu-v1']
            with (patch('tmt.cli.train', side_effect=[completed_model, error_type('injected failure')]),
                  patch('tmt.benchmark.evaluate', return_value={'targets': 1, 'bpb': 8.0}),
                  pytest.raises(error_type)):
                sweep(grid, str(data), str(data), output, config, updates=1)
            record = json.loads((output / 'sweep.json').read_text())
            assert (record['status'], record['current_run'], record['completed_runs']) == (status, 2, 1)
            assert record['error'] == {'type': error_type.__name__, 'message': 'injected failure'}
            first_row = json.loads((output / 'results.jsonl').read_text().splitlines()[0])
            assert (first_row['trainable_parameters'], first_row['temporal_gradient']) == (7, ['rtu-v1'])
            rows = (output / 'results.jsonl').read_bytes()
            with pytest.raises(FileExistsError, match='choose a new output folder'):
                sweep(grid, str(data), str(data), output, config, updates=1)
            assert (output / 'results.jsonl').read_bytes() == rows
            with contextlib.redirect_stdout(StringIO()):
                sweep(grid, str(data), str(data), root / 'retry', config, updates=1)
            assert json.loads((root / 'retry/sweep.json').read_text())['status'] == 'complete'


def test_sweep(subtests):
    from io import StringIO
    from tmt.cli import load_model, sweep
    from tmt.blocks import BLOCKS

    from tmt.cli import main
    with patch('tmt.cli.sweep') as command:
        args = ['sweep', '--grid', 'grid.json', '--data', 'train/*', '--development', 'dev/*', '--output', 'runs/grid']
        main(args)
        assert command.call_args.args[4] == 'model.json'
        main([*args, '--config', 'custom.json'])
        assert command.call_args.args[4] == 'custom.json'
        main([*args, '--sample-every', '2', '--sample-bytes', '3',
              '--eval-data', 'held-out/*', '--eval-every', '5'])
        assert command.call_args.kwargs['sample_every'] == 2
        assert command.call_args.kwargs['sample_bytes'] == 3
        assert command.call_args.kwargs['eval_data'] == 'held-out/*'
        assert command.call_args.kwargs['eval_every'] == 5

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        grid, config = root / 'grid.json', root / 'model.json'
        grid.write_text(json.dumps({'block': ['rtu', 'rtu-test'], 'seed': [11, 22]}))
        config.write_text(json.dumps(_tiny_settings()))
        train, development = root / 'train.bin', root / 'development.bin'
        train.write_bytes(b'\x00\x01\x02')
        development.write_bytes(b'\xf0\xf1\xf2\xf3')
        for ce_only in (False, True):
            with subtests.test(ce_only=ce_only), patch.dict(BLOCKS, {'rtu-test': BLOCKS['rtu']}):
                output, stdout = root / ('runs-ce' if ce_only else 'runs'), StringIO()
                with contextlib.redirect_stdout(stdout):
                    best = sweep(grid, str(train), str(development), output, config, 2, 11, ce_only,
                                 max_bytes=4, log_every=1, sample_every=2, sample_bytes=3,
                                 eval_data=str(development), eval_every=2, eval_max_bytes=3)
                rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
                assert [(row['settings']['block'], row['seed']) for row in rows] == [
                    ('rtu', 11), ('rtu', 22), ('rtu-test', 11), ('rtu-test', 22),
                ]
                assert best == min(rows, key=lambda row: row['development_bpb'])
                assert json.dumps(best) in stdout.getvalue()
                for index, row in enumerate(rows, 1):
                    checkpoint = Path(row['checkpoint'])
                    assert (row['objective'], row['development_targets']) == ('ce_only' if ce_only else 'tmt', 3)
                    assert row['trainable_parameters'] > 0
                    assert row['temporal_gradient'] == ['rtu-v1']
                    assert json.loads((checkpoint.parent / 'manifest' / 'model.json').read_text()) == row['settings']
                    manifest = json.loads((checkpoint.parent / 'manifest' / 'run.json').read_text())
                    assert (set(manifest), manifest['status'], manifest['last_checkpoint']) == (MANIFEST_FIELDS, 'complete', str(checkpoint.resolve()))
                    assert manifest['config'] == row['settings']
                    assert (manifest['trainable_parameters'], manifest['temporal_gradient']) == (
                        row['trainable_parameters'], row['temporal_gradient']
                    )
                    assert manifest['log_every'] == 1
                    metrics = [json.loads(line) for line in (checkpoint.parent / 'manifest/loss.jsonl').read_text().splitlines()]
                    assert [metric['completed_updates'] for metric in metrics] == [1, 2]
                    evaluation = json.loads((checkpoint.parent / 'manifest/evaluation.jsonl').read_text())
                    assert (evaluation['completed_updates'], evaluation['targets']) == (2, 2)
                    sample = json.loads((checkpoint.parent / 'manifest/samples.jsonl').read_text())
                    assert (sample['completed_updates'], sample['bytes']) == (2, 3)
                    assert len(Path(sample['path']).read_bytes()) == 3
                    for field in ('seed', 'updates', 'objective', 'data_glob'):
                        assert manifest[field] == row[field]
                    assert (manifest['data_files'], manifest['completed_updates'], manifest['resume'], manifest['initial_optimizer_step']) == ([str(train.resolve())], row['updates'], False, 0)
                    model, settings = load_model(checkpoint, row['seed'])
                    assert settings == row['settings']
                    assert row['trainable_parameters'] == model.count()
                    assert row['temporal_gradient'] == model.temporal_gradient
                    assert int(model.optimizer.state['step'].item()) == 2
                    assert checkpoint.parent.name == f'run-{index:04d}'


def test_checkpoint_header(subtests, saved_checkpoint):
    from tmt.cli import load_model

    (path, mx, settings, model, arrays, metadata) = saved_checkpoint
    restored, loaded = load_model(path)
    assert (loaded, restored.dim, int(restored.optimizer.state['step'].item())) == (settings, 4, 0)
    assert mx.array_equal(restored.decoder.decode.weight, model.decoder.decode.weight).item()

    header = json.loads(metadata['tmt'])
    missing = dict(header)
    missing['model'] = dict(header['model'])
    missing['model'].pop('spread')
    missing_block = dict(header)
    missing_block['model'] = dict(header['model'])
    missing_block['model'].pop('block')
    bad_version = dict(header)
    bad_version['version'] = True
    cases = [('missing metadata', None, '__metadata__'),
             ('missing model field', {'tmt': json.dumps(missing)}, 'spread'),
             ('missing block selector', {'tmt': json.dumps(missing_block)}, 'block'),
             ('bad version', {'tmt': json.dumps(bad_version)}, 'version')]
    for label, invalid, field in cases:
        with subtests.test(label=label):
            _assert_rejected(path, mx, load_model, arrays, invalid, field)
    with subtests.test(label='truncated header'):
        path.write_bytes((8).to_bytes(8, 'little') + b'{}')
        with pytest.raises(ValueError, match='header'): load_model(path)
