import contextlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli_support import assert_fields, assert_ordered_records, read_jsonl


@pytest.mark.parametrize(
    ('metadata', 'commit'),
    [
        (None, 'unknown'),
        ('{"vcs_info":{"commit_id":"abc123"}}', 'abc123'),
    ],
    ids=['missing-commit-metadata', 'commit-present'],
)
def test_version(monkeypatch, metadata, commit):
    from importlib import metadata as package_metadata
    from io import StringIO
    from tmt.cli import main

    installed = SimpleNamespace(version='0.1.0', read_text=lambda name: metadata)
    monkeypatch.setattr(package_metadata, 'distribution', lambda name: installed)
    stdout = StringIO()
    with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as exit:
        main(['--version'])
    assert exit.value.code == 0
    assert stdout.getvalue() == f'tmt 0.1.0, commit {commit}\n'


@pytest.mark.parametrize('args', [['--version'], ['--help']], ids=['version', 'help'])
def test_model_free_options(model_free_cli, tmp_path, args):
    result = model_free_cli(*args)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_full_manual(manual, model_free_cli, tmp_path):
    from tmt.cli import command_manual

    assert command_manual() == manual
    assert model_free_cli('help').stdout == manual
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    'topic', ['init', 'train', 'sweep', 'evaluate', 'benchmark', 'generate', 'help']
)
@pytest.mark.parametrize(
    'form',
    [
        ['help', '{topic}'],
        ['--help', '{topic}'],
        ['-h', '{topic}'],
        ['{topic}', '--help'],
        ['{topic}', '-h'],
        ['{topic}', '--data', 'missing/*', '--help'],
    ],
    ids=[
        'help-command',
        'long-option-before-topic',
        'short-option-before-topic',
        'long-option-after-topic',
        'short-option-after-topic',
        'help-with-command-option',
    ],
)
def test_manual_topic(manual, model_free_cli, tmp_path, topic, form):
    from tmt.cli import command_manual

    expected = (
        '## '
        + topic
        + '\n'
        + manual.split('## ' + topic + '\n', 1)[1].split('\n## ', 1)[0].rstrip()
        + '\n'
    )
    assert command_manual(topic) == expected
    result = model_free_cli(*(arg.format(topic=topic) for arg in form))
    assert (result.returncode, result.stdout) == (0, expected), result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('option', ['--help', '-h'], ids=['long-help', 'short-help'])
def test_help_index(model_free_cli, tmp_path, option):
    result = model_free_cli(option)
    assert result.returncode == 0, result.stderr
    assert 'tmt help' in result.stdout
    assert '## init' not in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ('args', 'code'),
    [
        (['--help'], 0),
        (['--version'], 2),
        ([], 2),
        (['missing.safetensors', '--unknown'], 2),
        (['missing.safetensors', '--seed', '-1'], 1),
    ],
    ids=['help', 'version', 'missing-command', 'unknown-option', 'invalid-seed'],
)
def test_benchmark_module_argument_dispatch(model_free_cli, tmp_path, args, code):
    expected = model_free_cli('benchmark', *args)
    result = model_free_cli(*args, module='tmt.benchmark')
    assert (result.returncode, result.stdout, result.stderr) == (
        expected.returncode,
        expected.stdout,
        expected.stderr,
    )
    assert result.returncode == code
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    'args',
    [
        ['help', 'nonexistent'],
        ['--help', 'nonexistent'],
        ['-h', 'nonexistent'],
        ['nonexistent', '--help'],
    ],
    ids=[
        'unknown-topic-command',
        'unknown-topic-after-long-option',
        'unknown-topic-after-short-option',
        'unknown-topic-before-help-option',
    ],
)
def test_unknown_topic(model_free_cli, tmp_path, args):
    result = model_free_cli(*args)
    assert result.returncode == 2
    assert 'unknown help topic' in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_installed_workflow(tmp_path, manifest_fields, cli_runner):
    from tmt.cli import load_model

    root = tmp_path

    assert '## init' in cli_runner('help', 'init', success=True).stdout
    cli_runner('init', success=True)
    (root / 'model.json').write_text('{"dim":4,"layers":1,"spread":4,"block":"rtu"}')
    (root / 'data/train/tiny.bin').write_bytes(b'abcd')
    (root / 'data/dev/tiny.bin').write_bytes(b'wxyz')
    progress = cli_runner(
        'train',
        '--data',
        'data/train/*',
        '--run',
        'first',
        '--updates',
        '3',
        '--log-every',
        '2',
        '--sample-every',
        '2',
        '--sample-bytes',
        '4',
        '--sample-prompt',
        'A',
        '--sample-seed',
        '7',
        '--eval-data',
        'data/dev/*',
        '--eval-every',
        '2',
        '--eval-max-bytes',
        '3',
        success=True,
    ).stdout
    assert 'updates: 2, objective: tmt, loss:' in progress
    assert 'updates: 3, objective: tmt, loss:' in progress
    checkpoint = root / 'runs/first/model.safetensors'
    assert 'sample at update 2, step 2:' in progress
    assert 'evaluation at update 3: BPB' in progress
    samples = read_jsonl(checkpoint.parent / 'manifest/samples.jsonl')
    assert_ordered_records(
        samples,
        [
            {
                'completed_updates': 2,
                'seed': 7,
                'bytes': 4,
                'prompt': 'A',
            }
        ],
        context='workflow samples',
    )
    assert len(Path(samples[0]['path']).read_bytes()) == 4
    evaluation = read_jsonl(checkpoint.parent / 'manifest/evaluation.jsonl')
    expected_evaluations = [
        {'completed_updates': 2, 'input_bytes': 3, 'targets': 2},
        {'completed_updates': 3, 'input_bytes': 3, 'targets': 2},
    ]
    assert_ordered_records(
        evaluation, expected_evaluations, context='workflow evaluations'
    )
    model, settings = load_model(checkpoint)
    assert_fields(
        settings,
        {'dim': 4, 'layers': 1, 'spread': 4},
        context='workflow checkpoint settings',
    )
    assert model.optimizer.state['step'].item() == 3, (
        'fresh checkpoint optimizer step differs'
    )
    manifest = checkpoint.parent / 'manifest/run.json'
    fresh = json.loads(manifest.read_text())
    assert_fields(
        fresh,
        {
            'data_glob': 'data/train/*',
            'data_files': [str((root / 'data/train/tiny.bin').resolve())],
            'updates': 3,
            'completed_updates': 3,
            'objective': 'tmt',
            'resume': False,
            'initial_optimizer_step': 0,
            'trainable_parameters': model.count(),
            'temporal_gradient': model.temporal_gradient,
        },
        context='fresh run manifest',
    )
    (root / 'data/resume').mkdir()
    (root / 'data/resume/next.bin').write_bytes(b'wxyz')
    cli_runner(
        'train',
        str(checkpoint),
        '--data',
        'data/resume/*',
        '--resume',
        '--updates',
        '2',
        '--seed',
        '99',
        '--ce-only',
        success=True,
    )
    assert load_model(checkpoint)[0].optimizer.state['step'].item() == 5
    record = json.loads(manifest.read_text())
    assert set(record) == manifest_fields
    assert_fields(
        record,
        {
            'status': 'complete',
            'config': settings,
            'last_checkpoint': str(checkpoint.resolve()),
            'trainable_parameters': model.count(),
            'temporal_gradient': model.temporal_gradient,
            'data_glob': 'data/resume/*',
            'data_files': [str((root / 'data/resume/next.bin').resolve())],
            'updates': 2,
            'completed_updates': 2,
            'seed': 99,
            'objective': 'ce_only',
            'resume': True,
            'initial_optimizer_step': 3,
        },
        context='resumed run manifest',
    )
    cli_runner(
        'evaluate',
        str(checkpoint),
        '--data',
        'data/train/*',
        '--windows',
        '1',
        '2',
        '--output',
        'score.json',
        success=True,
    )
    assert json.loads((root / 'score.json').read_text())['targets'] == 3
    cli_runner(
        'generate',
        str(checkpoint),
        '--prompt',
        'The ',
        '--output',
        'sample.bin',
        '--bytes',
        '4',
        success=True,
    )
    assert len((root / 'sample.bin').read_bytes()) == 4
    assert (
        '390 targets' in cli_runner('benchmark', str(checkpoint), success=True).stdout
    )


def test_init(tmp_path, monkeypatch, manifest_fields):
    from tmt.cli import load_config, main
    from tmt.main import Model

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    main(['init'])
    saved = json.loads((tmp_path / 'model.json').read_text())
    assert saved == load_config()
    assert set(saved) == set(inspect.signature(Model).parameters)
    templates = {
        tmp_path / name for name in ('model.json', 'grid.json', 'run.example.json')
    }
    assert set(tmp_path.rglob('*.json')) == templates
    for name in ('data/train', 'data/dev', 'runs'):
        assert Path(name).is_dir(), f'init did not create {name!r}'
    assert json.loads(Path('grid.json').read_text()) == {
        'dim': [4, 8],
        'layers': [1],
        'spread': [4],
        'block': ['rtu'],
        'seed': [11, 22],
    }
    template = json.loads(Path('run.example.json').read_text())
    assert template == json.loads((repo / 'examples' / 'run.example.json').read_text())
    assert set(template) == manifest_fields
    assert_fields(
        template,
        {
            'config': saved,
            'seed': 11,
            'data_glob': 'data/train/*',
            'data_files': [],
            'updates': 1000,
            'objective': 'tmt',
            'completed_updates': 0,
            'resume': False,
            'initial_optimizer_step': 0,
            'started_at': None,
            'last_checkpoint': None,
        },
        context='init template',
    )

    explicit = tmp_path / 'custom' / 'settings.json'
    main(['init', '--output', str(explicit)])
    assert json.loads(explicit.read_text()) == saved
    assert set(tmp_path.rglob('*.json')) == templates | {explicit}

    assert load_config(repo / 'examples' / 'model.json') == load_config()
    tiny = load_config(repo / 'examples' / 'tiny-model.json')
    assert_fields(tiny, {'dim': 4, 'layers': 1, 'spread': 4}, context='init tiny')
    assert json.loads((repo / 'examples' / 'grid.json').read_text()) == {
        'block': ['rtu'],
        'dim': [4, 8],
        'layers': [1],
        'spread': [4],
        'seed': [11, 22],
    }


@pytest.mark.parametrize(
    'args',
    [['init'], ['init', '--output', 'model.json']],
    ids=['default-output', 'explicit-output'],
)
def test_init_preserves_files(tmp_path, monkeypatch, args):
    from tmt.cli import main

    monkeypatch.chdir(tmp_path)
    main(['init'])
    Path('model.json').write_text('{"dim": 4}')
    before = {p: p.read_bytes() for p in Path('.').rglob('*') if p.is_file()}
    with pytest.raises(SystemExit) as exit:
        main(args)
    assert exit.value.code == 1
    after = {p: p.read_bytes() for p in Path('.').rglob('*') if p.is_file()}
    assert after.keys() == before.keys(), 'init failure changed the set of files'
    assert after == before, 'init failure changed existing file contents'


def test_config(monkeypatch, tmp_path):
    import inspect

    from tmt import config
    from tmt.cli import load_config
    from tmt.main import Model

    defaults = load_config()
    parameters = inspect.signature(Model).parameters
    assert set(defaults) == set(parameters), (
        'config fields differ from constructor fields'
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in parameters.values()
    ), 'direct Model constructor fields must remain required'

    repo = Path(__file__).resolve().parents[2]
    assert json.loads((repo / 'examples/model.json').read_text()) == defaults
    tiny = json.loads((repo / 'examples/tiny-model.json').read_text())
    assert tiny == {**defaults, 'dim': 4, 'layers': 1, 'spread': 4}

    path = tmp_path / 'model.json'
    path.write_text('{}')
    assert load_config(path) == defaults
    path.write_text('{"dim": 4}')
    settings = load_config(path)
    assert_fields(
        settings,
        {
            'dim': 4,
            'layers': 16,
            'bound': [40000, 120000],
        },
        context='config settings',
    )

    with pytest.raises(TypeError, match='required positional arguments'):
        Model(dim=4, layers=1, spread=4, block='rtu')

    monkeypatch.setitem(config.MODEL_DEFAULTS, 'extra', 17)
    assert load_config()['extra'] == 17
