import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('extra', (['--run', 'absent'], []))
def test_cli_missing_data(cli_runner, tmp_path, extra):
    root, run = tmp_path, cli_runner
    result = run('train', '--data', 'missing/*', *extra)
    assert 'no files matched' in result.stderr
    assert list(root.iterdir()) == []


@pytest.mark.parametrize('extra,message', (
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
    ))
def test_cli_monitoring_options(cli_workspace, extra, message):
    root, run = cli_workspace
    result = run('train', '--data', 'train.bin', '--run', 'invalid', *extra)
    assert message in result.stderr
    assert not (root / 'runs').exists()
    result = run('sweep', '--grid', 'grid.json', '--data', 'train.bin',
                 '--development', 'dev.bin', '--output', 'invalid', *extra)
    assert message in result.stderr
    assert not (root / 'invalid').exists()


@pytest.mark.parametrize('name', ('', '.', '..', '../escaped-name', '/absolute-name',
                 'nested/name', 'trailing/', 'nested\\name'))
def test_cli_run_names(cli_workspace, name):
    root, run = cli_workspace
    result = run('train', '--data', 'train.bin', '--run', name)
    assert 'run must be one relative folder name' in result.stderr
    assert not (root / 'runs').exists()
    assert not (root / 'absolute-name').exists()


@pytest.mark.parametrize('count', ('0', '-1'))
def test_cli_positive_options(cli_workspace, count):
    root, run = cli_workspace
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


@pytest.mark.parametrize('temperature', ('nan', 'inf', '-inf', '-1'))
def test_cli_temperatures(cli_workspace, temperature):
    root, run = cli_workspace
    result = run('generate', 'missing.safetensors', '--prompt', 'x',
                 '--output', 'sample.bin', f'--temperature={temperature}')
    assert 'temperature must be a finite nonnegative number' in result.stderr
    assert not (root / 'sample.bin').exists()


@pytest.mark.parametrize('epochs', ('0', '-1'))
def test_cli_epochs(cli_workspace, epochs):
    root, run = cli_workspace
    result = run('benchmark', 'missing.safetensors', '--cola-data', 'missing.tsv',
                 '--epochs', epochs)
    assert 'epochs must be positive with cola-data' in result.stderr
    result = subprocess.run([sys.executable, '-m', 'tmt.benchmark', 'missing.safetensors',
                             '--cola-data', 'missing.tsv', '--epochs', epochs], cwd=root,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert 'epochs must be positive with cola-data' in result.stderr
    assert 'Traceback' not in result.stderr


@pytest.mark.parametrize('grid', ([4], None, {'dim': 4}, {'dim': []}, {'dim': [4], 'seed': []},
                 {'dim': [4, 0]}, {'unknown': [1]}, {'seed': [11, '22']},
                 {'seed': [11, -1]}, {'seed': [11, 2**64]}))
def test_cli_grids(cli_workspace, grid):
    root, run = cli_workspace
    (root / 'bad-grid.json').write_text(json.dumps(grid))
    result = run('sweep', '--grid', 'bad-grid.json', '--data', 'train.bin',
                 '--development', 'dev.bin', '--output', 'bad-sweep', '--updates', '1')
    assert not (root / 'bad-sweep').exists()


@pytest.mark.parametrize('contents', ('{', '[]', '{"unknown": 1}', '{"bound": 30}', '{"dim": true}', '{"rate": 0}',
                     '{"block": "unknown-block"}'))
def test_cli_config_files(cli_workspace, contents):
    root, run = cli_workspace
    (root / 'bad.json').write_text(contents)
    result = run('train', '--data', 'train.bin', '--config', 'bad.json', '--run', 'bad')
    if 'unknown-block' in contents:
        assert 'model config.block must be one of' in result.stderr
    assert not (root / 'runs/bad').exists()


@pytest.mark.parametrize('grid,message', (({'bound': [30]}, 'model config.bound must be two integers'),
                          ({'block': ['unknown-block']}, 'model config.block must be one of')))
def test_cli_grid_values(cli_workspace, grid, message):
    root, run = cli_workspace
    (root / 'bad-grid.json').write_text(json.dumps(grid))
    result = run('sweep', '--grid', 'bad-grid.json', '--data', 'train.bin',
                 '--development', 'dev.bin', '--output', 'bad-sweep')
    assert message in result.stderr
    assert not (root / 'bad-sweep').exists()


def test_cli_sweep_and_input_errors(cli_workspace):
    root, run = cli_workspace
    args = ['sweep', '--grid', 'grid.json', '--data', 'train.bin',
            '--development', 'dev.bin', '--output', 'sweep', '--updates', '2']
    (root / 'sweep').mkdir()  # An existing empty folder is valid.
    run(*args, success=True)
    before = {p: p.read_bytes() for p in (root / 'sweep').rglob('*') if p.is_file()}
    result = run(*args[:-1], '1')
    assert 'not empty' in result.stderr
    assert before == {p: p.read_bytes() for p in (root / 'sweep').rglob('*') if p.is_file()}
    checkpoint = root / 'sweep/run-0001/model.safetensors'
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


def test_cli_negative_bytes(cli_workspace):
    root, run = cli_workspace
    result = run('generate', 'missing.safetensors', '--prompt', 'x',
                 '--output', 'sample.bin', '--bytes', '-1')
    assert 'bytes must be nonnegative' in result.stderr


@pytest.mark.parametrize('seed', [-1, 2**64])
@pytest.mark.parametrize('args,option', [
    (['train', '--data', 'missing/*', '--run', 'invalid'], '--seed'),
    (['train', '--data', 'missing/*', '--run', 'invalid'], '--sample-seed'),
    (['train', 'missing.safetensors', '--data', 'missing/*', '--resume'], '--seed'),
    (['train', 'missing.safetensors', '--data', 'missing/*', '--resume'], '--sample-seed'),
    (['generate', 'missing.safetensors', '--prompt', 'X', '--output', 'protected.bin'], '--seed'),
    (['benchmark', 'missing.safetensors'], '--seed'),
    (['sweep', '--grid', 'missing.json', '--data', 'missing/*', '--development', 'missing/*', '--output', 'invalid'], '--seed'),
    (['sweep', '--grid', 'missing.json', '--data', 'missing/*', '--development', 'missing/*', '--output', 'invalid'], '--sample-seed'),
])
def test_cli_seed_validation(tmp_path, model_free_cli, seed, args, option):
    output = tmp_path / 'protected.bin'
    output.write_bytes(b'keep')
    result = model_free_cli(*args, option, str(seed))
    assert result.returncode == 1, result.stderr
    assert f'{option[2:]} must be between 0 and {2**64 - 1}' in result.stderr
    assert 'Traceback' not in result.stderr
    assert output.read_bytes() == b'keep'
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize('seed', [-1, 2**64])
def test_benchmark_seed(tmp_path, seed):
    result = subprocess.run(
        [sys.executable, '-m', 'tmt.benchmark', 'missing.safetensors', '--seed', str(seed)],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 1, result.stderr
    assert 'seed must be between' in result.stderr
    assert 'Traceback' not in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('seed', (-1, 2**64, True, 1.5))
@pytest.mark.parametrize('operation', ['new', 'load', 'generate', 'train', 'train-sample', 'sweep', 'sweep-sample', 'benchmark'])
def test_direct_seed_validation(monkeypatch, seed, operation):
    from tmt.cli import generate, load_model, new_model, sweep, train
    from tmt.benchmark import run
    from tmt import cli

    def unexpected(*args, **kwargs):
        pytest.fail('model configuration was checked before seed validation')

    monkeypatch.setattr(cli, 'load_config', unexpected)
    monkeypatch.setattr(cli.validation, 'validate_model_settings', unexpected)
    monkeypatch.setattr(cli, 'load_model', unexpected)
    monkeypatch.setattr(cli, 'new_model', unexpected)
    monkeypatch.setattr(cli, 'train', unexpected)
    monkeypatch.setattr(cli, 'sweep', unexpected)
    monkeypatch.setattr(cli, 'generate', unexpected)
    monkeypatch.setattr('tmt.benchmark.load_model', unexpected, raising=False)
    calls = {
        'new': lambda: new_model({}, seed),
        'load': lambda: load_model('missing.safetensors', seed),
        'generate': lambda: generate('missing.safetensors', 'X', 'unused.bin', seed),
        'train': lambda: train({}, 'unused/model.safetensors', 'missing/*', seed=seed),
        'train-sample': lambda: train({}, 'unused/model.safetensors', 'missing/*', sample_seed=seed),
        'sweep': lambda: sweep('missing.json', 'missing/*', 'missing/*', 'unused', seed=seed),
        'sweep-sample': lambda: sweep('missing.json', 'missing/*', 'missing/*', 'unused', sample_seed=seed),
        'benchmark': lambda: run('missing.safetensors', model=SimpleNamespace(freeze=unexpected), seed=seed),
    }
    with pytest.raises(ValueError, match='seed must be'):
        calls[operation]()


def invalid_settings():
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
    return cases


@pytest.mark.parametrize(
    "name,value", invalid_settings(),
    ids=lambda value: "overflow" if isinstance(value, int) and value > 10**100 else None,
)
def test_config_validation(tmp_path, tiny_settings, name, value):
    from tmt.cli import load_config, new_model
    from tmt.validation import validate_metadata

    path = tmp_path / 'model.json'
    path.write_text(json.dumps({name: value}))
    with pytest.raises(ValueError, match=f'model config.{name}'):
        load_config(path)
    settings = dict(tiny_settings, **{name: value})
    with pytest.raises(ValueError, match=f'model config.{name}'):
        new_model(settings, 11)
    raw = json.dumps({'version': 1, 'model': settings})
    with pytest.raises(ValueError, match=f'checkpoint.model.{name}'):
        validate_metadata(raw)


def test_config_limits(tmp_path):
    from tmt.cli import load_config
    from tmt.main import Model

    path = tmp_path / 'model.json'
    with pytest.raises(ValueError, match='unknown block'):
        Model(dim=4, layers=1, spread=4, block='unknown-block')
    path.write_text('{"dim": 1, "layers": 1, "spread": 1, "temp": 0, "rate": 1, "bound": [0, 1]}')
    settings = load_config(path)
    assert (settings['temp'], settings['rate'], settings['bound']) == (0, 1, [0, 1])


@pytest.mark.parametrize('limit', [0, -1])
def test_byte_limits(monkeypatch, forbid_call, limit):
    from tmt.cli import build_parser, evaluate_command, sweep

    monkeypatch.setattr('tmt.cli.load_model', forbid_call)
    args = build_parser().parse_args(['evaluate', 'missing.safetensors', '--data', 'missing/*',
                                      '--max-bytes', str(limit)])
    with pytest.raises(ValueError, match='max-bytes must be positive'):
        evaluate_command(args)
    with pytest.raises(ValueError, match='max-bytes must be positive'):
        sweep('missing.json', 'missing/*', 'missing/*', 'unused', max_bytes=limit)


@pytest.mark.parametrize('limit,expected', [(6, [b'abcd', b'ef']), (1, [b'a'])])
def test_document_budget(tmp_path, limit, expected):
    from tmt.cli import documents

    root = tmp_path
    (root / 'a.bin').write_bytes(b'abcd')
    (root / 'b.bin').write_bytes(b'efgh')
    pattern = str(root / '*.bin')
    assert list(documents(pattern, limit)) == expected


@pytest.mark.parametrize('selection', ['absent/*', 'not-a-file'])
@pytest.mark.parametrize('command,role', [
    ('train', 'train'), ('train', 'evaluation'),
    ('sweep', 'train'), ('sweep', 'evaluation'), ('sweep', 'development'),
])
def test_input_retry(cli_workspace, monkeypatch, selection, command, role):
    from tmt.cli import main

    root, _ = cli_workspace
    monkeypatch.chdir(root)
    (root / 'not-a-file').mkdir()
    if command == 'train':
        valid = ['train', '--data', 'train.bin', '--run', 'retry', '--updates', '1']
        output = root / 'runs/retry'
    else:
        valid = ['sweep', '--grid', 'grid.json', '--data', 'train.bin',
                 '--development', 'dev.bin', '--output', 'retry-sweep', '--updates', '1']
        output = root / 'retry-sweep'
    invalid = list(valid)
    if role == 'evaluation':
        invalid += ['--eval-data', selection, '--eval-every', '1']
    else:
        option = '--data' if role == 'train' else '--development'
        invalid[invalid.index(option) + 1] = selection
    with pytest.raises(SystemExit) as exit:
        main(invalid)
    assert exit.value.code == 1
    assert not output.exists()
    main(valid)
    manifest = output / ('manifest/run.json' if command == 'train' else 'sweep.json')
    record = json.loads(manifest.read_text())
    assert record['status'] == 'complete'
    if command == 'sweep':
        assert (record['completed_runs'], record['error']) == (2, None)


def test_direct_input_validation(cli_workspace, monkeypatch, forbid_call, tiny_settings):
    from tmt.cli import train

    root, _ = cli_workspace
    monkeypatch.chdir(root)
    monkeypatch.setattr('tmt.cli.new_model', forbid_call)
    with pytest.raises(FileNotFoundError):
        train(tiny_settings, root / 'direct/model.safetensors',
              'train.bin', updates=1, eval_data='absent/*', eval_every=1)
    assert not (root / 'direct').exists()
