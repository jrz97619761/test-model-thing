import contextlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(('metadata', 'commit'), [
    (None, 'unknown'),
    ('{"vcs_info":{"commit_id":"abc123"}}', 'abc123'),
])
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


@pytest.mark.parametrize('args', [['--version'], ['--help']])
def test_model_free_options(model_free_cli, tmp_path, args):
    result = model_free_cli(*args)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def manual():
    from importlib.resources import files

    return files('tmt').joinpath('commands.md').read_text(encoding='utf-8')


def test_full_manual(manual, model_free_cli, tmp_path):
    from tmt.cli import command_manual

    assert command_manual() == manual
    assert model_free_cli('help').stdout == manual
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('topic', ['init', 'train', 'sweep', 'evaluate', 'benchmark', 'generate', 'help'])
@pytest.mark.parametrize('form', [
    ['help', '{topic}'], ['--help', '{topic}'], ['-h', '{topic}'],
    ['{topic}', '--help'], ['{topic}', '-h'], ['{topic}', '--data', 'missing/*', '--help'],
])
def test_manual_topic(manual, model_free_cli, tmp_path, topic, form):
    from tmt.cli import command_manual

    expected = '## ' + topic + '\n' + manual.split('## ' + topic + '\n', 1)[1].split('\n## ', 1)[0].rstrip() + '\n'
    assert command_manual(topic) == expected
    result = model_free_cli(*(arg.format(topic=topic) for arg in form))
    assert (result.returncode, result.stdout) == (0, expected), result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('option', ['--help', '-h'])
def test_help_index(model_free_cli, tmp_path, option):
    result = model_free_cli(option)
    assert result.returncode == 0, result.stderr
    assert 'tmt help' in result.stdout
    assert '## init' not in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('args,code', [
    (['--help'], 0),
    (['--version'], 2),
    ([], 2),
    (['missing.safetensors', '--unknown'], 2),
    (['missing.safetensors', '--seed', '-1'], 1),
])
def test_benchmark_module_uses_main_cli(model_free_cli, tmp_path, args, code):
    expected = model_free_cli('benchmark', *args)
    result = model_free_cli(*args, module='tmt.benchmark')
    assert (result.returncode, result.stdout, result.stderr) == (
        expected.returncode, expected.stdout, expected.stderr,
    )
    assert result.returncode == code
    assert list(tmp_path.iterdir()) == []


def test_benchmark_module_scores(saved_checkpoint, cli_runner, tmp_path):
    import subprocess
    import sys

    checkpoint = saved_checkpoint[0]
    expected = cli_runner('benchmark', str(checkpoint), success=True)
    result = subprocess.run(
        [sys.executable, '-m', 'tmt.benchmark', str(checkpoint)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert (result.returncode, result.stdout, result.stderr) == (0, expected.stdout, expected.stderr)
    assert '390 targets' in result.stdout


@pytest.mark.parametrize('args', [
    ['help', 'nonexistent'], ['--help', 'nonexistent'],
    ['-h', 'nonexistent'], ['nonexistent', '--help'],
])
def test_unknown_topic(model_free_cli, tmp_path, args):
    result = model_free_cli(*args)
    assert result.returncode == 2
    assert 'unknown help topic' in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_installed_workflow(tmp_path, manifest_fields, cli_runner):
    from tmt.cli import load_model

    root = tmp_path

    def run(*args):
        return cli_runner(*args, success=True).stdout

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
    assert set(record) == manifest_fields
    assert (record['status'], record['config'], record['last_checkpoint']) == ('complete', settings, str(checkpoint.resolve()))
    assert (record['trainable_parameters'], record['temporal_gradient']) == (model.count(), model.temporal_gradient)
    assert (record['data_glob'], record['data_files'], record['updates'], record['completed_updates']) == ('data/resume/*', [str((root / 'data/resume/next.bin').resolve())], 2, 2)
    assert (record['seed'], record['objective'], record['resume'], record['initial_optimizer_step']) == (99, 'ce_only', True, 3)
    run('evaluate', str(checkpoint), '--data', 'data/train/*', '--windows', '1', '2', '--output', 'score.json')
    assert json.loads((root / 'score.json').read_text())['targets'] == 3
    run('generate', str(checkpoint), '--prompt', 'The ', '--output', 'sample.bin', '--bytes', '4')
    assert len((root / 'sample.bin').read_bytes()) == 4
    assert '390 targets' in run('benchmark', str(checkpoint))


def test_init(tmp_path, monkeypatch, manifest_fields):
    from tmt.cli import load_config, main
    from tmt.main import Model

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    main(['init'])
    saved = json.loads((tmp_path / 'model.json').read_text())
    assert saved == load_config()
    assert set(saved) == set(inspect.signature(Model).parameters)
    templates = {tmp_path / name for name in ('model.json', 'grid.json', 'run.example.json')}
    assert set(tmp_path.rglob('*.json')) == templates
    assert all((Path(name).is_dir() for name in ('data/train', 'data/dev', 'runs')))
    assert json.loads(Path('grid.json').read_text()) == {'dim': [4, 8], 'layers': [1], 'spread': [4], 'block': ['rtu'], 'seed': [11, 22]}
    template = json.loads(Path('run.example.json').read_text())
    assert template == json.loads((repo / 'examples' / 'run.example.json').read_text())
    assert set(template) == manifest_fields
    assert (template['config'], template['seed'], template['data_glob'], template['data_files']) == (saved, 11, 'data/train/*', [])
    assert (template['updates'], template['objective'], template['completed_updates']) == (1000, 'tmt', 0)
    assert (template['resume'], template['initial_optimizer_step'], template['started_at'], template['last_checkpoint']) == (False, 0, None, None)

    explicit = tmp_path / 'custom' / 'settings.json'
    main(['init', '--output', str(explicit)])
    assert json.loads(explicit.read_text()) == saved
    assert set(tmp_path.rglob('*.json')) == templates | {explicit}

    assert load_config(repo / 'examples' / 'model.json') == load_config()
    tiny = load_config(repo / 'examples' / 'tiny-model.json')
    assert (tiny['dim'], tiny['layers'], tiny['spread']) == (4, 1, 4)
    assert json.loads((repo / 'examples' / 'grid.json').read_text()) == {
        'block': ['rtu'], 'dim': [4, 8], 'layers': [1], 'spread': [4], 'seed': [11, 22],
    }


@pytest.mark.parametrize('args', [['init'], ['init', '--output', 'model.json']])
def test_init_preserves_files(tmp_path, monkeypatch, args):
    from tmt.cli import main

    monkeypatch.chdir(tmp_path)
    main(['init'])
    Path('model.json').write_text('{"dim": 4}')
    before = {p: p.read_bytes() for p in Path('.').rglob('*') if p.is_file()}
    with pytest.raises(SystemExit) as exit:
        main(args)
    assert exit.value.code == 1
    assert before == {p: p.read_bytes() for p in before}


def test_config(monkeypatch, tmp_path):
    from tmt.cli import load_config
    from tmt.main import Model

    defaults = load_config()
    path = tmp_path / 'model.json'
    path.write_text('{}')
    assert load_config(path) == defaults
    path.write_text('{"dim": 4}')
    settings = load_config(path)
    assert (settings['dim'], settings['layers'], settings['bound']) == (4, 16, [40000, 120000])

    parameters = list(inspect.signature(Model).parameters.values())
    parameters.append(inspect.Parameter('extra', inspect.Parameter.KEYWORD_ONLY, default=17))
    def extended():
        pass

    extended.__signature__ = inspect.Signature(parameters)
    monkeypatch.setattr('tmt.main.Model', extended)
    assert load_config()['extra'] == 17
