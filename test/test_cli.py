import json
import shlex
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest

TOPICS = ('init', 'train', 'sweep', 'evaluate', 'benchmark', 'generate', 'help')
BASE_ARGV = {
    'train': 'train --data train.bin --run invalid',
    'sweep': 'sweep --grid grid.json --data train.bin --development dev.bin '
    '--output invalid',
    'evaluate': 'evaluate missing.safetensors --data dev.bin --output score.json',
    'generate': 'generate missing.safetensors --prompt x --output sample.bin',
    'benchmark': 'benchmark missing.safetensors --cola-data missing.tsv',
}
RUN_NAME = 'run must be one relative folder name'
EVAL_PAIR = 'eval-data and eval-every must be used together'
TEMPERATURE = 'temperature must be a finite nonnegative number'
INVALID_OPTION_VALUES = {  # id: (extra options, expected stderr fragment)
    'train-sample-prompt': ("--sample-prompt ''", 'sample-prompt must not be empty'),
    'train-sample-bytes': ('--sample-bytes 0', 'sample-bytes must be positive'),
    'train-sample-every': ('--sample-every 0', 'sample-every must be positive'),
    'train-eval-data': ('--eval-data dev.bin', EVAL_PAIR),
    'train-eval-every': ('--eval-every 1', EVAL_PAIR),
    'train-eval-every-zero': (
        '--eval-every 0 --eval-data dev.bin',
        'eval-every must be positive',
    ),
    'train-eval-max-bytes': ('--eval-max-bytes 0', 'eval-max-bytes must be positive'),
    'train-updates': ('--updates 0', 'updates must be positive'),
    'run-empty': ("--run ''", RUN_NAME),
    'run-parent': ('--run ..', RUN_NAME),
    'run-absolute': ('--run /absolute-name', RUN_NAME),
    'run-nested': ('--run nested/name', RUN_NAME),
    'run-backslash': ("--run 'nested\\name'", RUN_NAME),
    'train-bad-config': ('--config bad.json', 'model config.dim'),
    'sweep-updates': ('--updates -1', 'updates must be positive'),
    'sweep-max-bytes': ('--max-bytes 0', 'max-bytes must be positive'),
    'sweep-sample-bytes': ('--sample-bytes -1', 'sample-bytes must be positive'),
    'evaluate-windows': ('--windows 0', 'evaluation windows must be positive'),
    'evaluate-max-bytes': ('--max-bytes -1', 'max-bytes must be positive'),
    'generate-log-every': ('--log-every 0', 'log-every must be positive'),
    'generate-negative-bytes': ('--bytes -1', 'bytes must be nonnegative'),
    'generate-temperature-nan': ('--temperature=nan', TEMPERATURE),
    'generate-temperature-positive-infinity': ('--temperature=inf', TEMPERATURE),
    'generate-temperature-negative-infinity': ('--temperature=-inf', TEMPERATURE),
    'generate-temperature-negative': ('--temperature=-1', TEMPERATURE),
    'generate-missing-checkpoint': ('', "'missing.safetensors'"),
    'generate-empty-prompt': ("--prompt ''", 'generation needs a nonempty prompt'),
    'benchmark-epochs': ('--epochs 0', 'epochs must be positive with cola-data'),
}
HELP_ERRORS = {  # id: (argv, expected stderr fragment)
    'unknown-topic-command': ('help nonexistent', 'unknown help topic'),
    'unknown-topic-after-long-option': ('--help nonexistent', 'unknown help topic'),
    'unknown-topic-after-short-option': ('-h nonexistent', 'unknown help topic'),
    'unknown-topic-before-help-option': ('nonexistent --help', 'unknown help topic'),
    'two-topics': ('help train sweep', 'help accepts at most one topic'),
}
ARGUMENT_ERRORS = {
    'checkpoint-with-run': '--run bad model.safetensors',
    'resume-without-checkpoint': '--resume',
    'resume-with-run': '--resume --run bad',
    'resume-with-config': 'model.safetensors --resume --config model.json',
}
SWEEP_MISSING = (
    'sweep --grid missing.json --data missing/* --development missing/* '
    '--output invalid'
)
OPERATIONS = 'new load generate train train-sample sweep sweep-sample benchmark'.split()
SEED_ARGV = {  # even cases get seed -1, odd cases get 2**64
    'train-seed': 'train --data missing/* --run invalid --seed',
    'train-sample-seed': 'train --data missing/* --run invalid --sample-seed',
    'resume-seed': 'train missing.safetensors --data missing/* --resume --seed',
    'resume-sample-seed': 'train missing.safetensors --data missing/* --resume '
    '--sample-seed',
    'generate-seed': 'generate missing.safetensors --prompt X --output protected.bin '
    '--seed',
    'benchmark-seed': 'benchmark missing.safetensors --seed',
    'sweep-seed': SWEEP_MISSING + ' --seed',
    'sweep-sample-seed': SWEEP_MISSING + ' --sample-seed',
}


@pytest.fixture
def manual():
    return files('tmt').joinpath('commands.md').read_text(encoding='utf-8')


@pytest.mark.parametrize(
    ('metadata', 'commit'),
    [(None, 'unknown'), ('{"vcs_info":{"commit_id":"abc123"}}', 'abc123')],
    ids=['missing-commit-metadata', 'commit-present'],
)
def test_version(monkeypatch, capsys, metadata, commit):
    from importlib import metadata as package_metadata
    from tmt.cli import main

    installed = SimpleNamespace(version='0.1.0', read_text=lambda name: metadata)
    monkeypatch.setattr(package_metadata, 'distribution', lambda name: installed)
    with pytest.raises(SystemExit) as error:
        main(['--version'])
    assert error.value.code == 0
    assert capsys.readouterr().out == f'tmt 0.1.0, commit {commit}\n'


@pytest.mark.parametrize(
    ('argv', 'code', 'stdout_start', 'stderr_has'),
    [
        ('--help', 0, 'usage: tmt', ''),
        ('--version', 0, 'tmt ', ''),
        ('help', 0, '# TMT command manual', ''),
        ('help train', 0, '## train', ''),
        ('generate --help', 0, '## generate', ''),
        ('train --data missing/* --seed -1', 1, '', 'seed must be between'),
    ],
    ids=['--help', '--version', 'help', 'help-train', 'generate-help', 'seed-error'],
)
def test_model_free_commands(
    model_free_cli, tmp_path, manual, argv, code, stdout_start, stderr_has
):
    result = model_free_cli(*shlex.split(argv))
    assert result.returncode == code, result.stderr
    assert result.stdout.startswith(stdout_start)
    assert stderr_has in result.stderr
    assert 'Traceback' not in result.stderr
    assert list(tmp_path.iterdir()) == []
    if argv == '--help':
        assert 'tmt help' in result.stdout
        assert '## init' not in result.stdout
    if argv == 'help':
        assert result.stdout == manual


def test_manual_topics(run_cli, manual):
    from tmt.cli import command_manual

    assert command_manual() == manual
    forms = (
        'help {}',
        '--help {}',
        '-h {}',
        '{} --help',
        '{} -h',
        '{} --data missing/* --help',
    )
    for topic in TOPICS:
        section = manual.split('## ' + topic + '\n', 1)[1].split('\n## ', 1)[0]
        expected = '## ' + topic + '\n' + section.rstrip() + '\n'
        assert command_manual(topic) == expected
        for form in forms:
            argv = form.format(topic).split()
            assert run_cli(*argv) == (0, expected, ''), argv
    code, stdout, stderr = run_cli('-h')
    assert (code, stderr) == (0, '')
    assert 'tmt help' in stdout
    assert '## init' not in stdout


@pytest.mark.parametrize('case', HELP_ERRORS)
def test_help_errors(run_cli, tmp_path, case):
    argv, message = HELP_ERRORS[case]
    code, stdout, stderr = run_cli(*argv.split())
    assert (code, stdout) == (2, '')
    assert message in stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('case', ARGUMENT_ERRORS)
def test_argument_errors(workspace, run_cli, case):
    argv = ['train', '--data', 'train.bin', *ARGUMENT_ERRORS[case].split()]
    assert run_cli(*argv)[0] == 2


@pytest.mark.parametrize('case', SEED_ARGV)
def test_seed_checked_first(run_cli, tmp_path, case):
    output = tmp_path / 'protected.bin'
    output.write_bytes(b'keep')
    seed = (-1, 2**64)[list(SEED_ARGV).index(case) % 2]
    argv = shlex.split(SEED_ARGV[case])
    code, stdout, stderr = run_cli(*argv, str(seed))
    assert (code, stdout) == (1, '')
    assert f'{argv[-1][2:]} must be between 0 and {2**64 - 1}' in stderr
    assert output.read_bytes() == b'keep'
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize('operation', OPERATIONS)
def test_direct_seed_validation(monkeypatch, forbid_call, operation):
    from tmt import cli
    from tmt.benchmark import run
    from tmt.cli import generate, load_model, new_model, sweep, train

    monkeypatch.setattr(cli.validation, 'model_settings', forbid_call)
    for name in 'load_config load_model new_model train sweep generate'.split():
        monkeypatch.setattr(cli, name, forbid_call)
    seed = (-1, 2**64, True, 1.5)[OPERATIONS.index(operation) % 4]
    monkeypatch.setattr('tmt.benchmark.load_model', forbid_call, raising=False)
    calls = {
        'new': lambda: new_model({}, seed),
        'load': lambda: load_model('missing.safetensors', seed),
        'generate': lambda: generate('missing.safetensors', 'X', 'unused.bin', seed),
        'train': lambda: train({}, 'unused/model.safetensors', 'missing/*', seed=seed),
        'train-sample': lambda: train(
            {}, 'unused/model.safetensors', 'missing/*', sample_seed=seed
        ),
        'sweep': lambda: sweep(
            'missing.json', 'missing/*', 'missing/*', 'unused', seed=seed
        ),
        'sweep-sample': lambda: sweep(
            'missing.json', 'missing/*', 'missing/*', 'unused', sample_seed=seed
        ),
        'benchmark': lambda: run(
            'missing.safetensors', model=SimpleNamespace(freeze=forbid_call), seed=seed
        ),
    }
    with pytest.raises(ValueError, match='seed must be'):
        calls[operation]()


@pytest.mark.parametrize('case', INVALID_OPTION_VALUES)
def test_invalid_option_values(workspace, run_cli, case):
    extra, message = INVALID_OPTION_VALUES[case]
    command = 'train' if case.startswith('run-') else case.split('-')[0]
    (workspace / 'bad.json').write_text('{"dim": true}')
    kept = case in ('generate-negative-bytes', 'generate-temperature-negative')
    if kept:
        (workspace / 'sample.bin').write_bytes(b'keep')
    code, stdout, stderr = run_cli(*shlex.split(BASE_ARGV[command] + ' ' + extra))
    assert (code, stdout) == (1, '')
    assert message in stderr
    if kept:
        assert (workspace / 'sample.bin').read_bytes() == b'keep'
    absent = ['runs', 'invalid', 'score.json', 'absolute-name']
    for name in absent + ([] if kept else ['sample.bin']):
        assert not (workspace / name).exists(), name


@pytest.mark.parametrize(
    'case',
    [
        'train-data',
        'train-evaluation',
        'sweep-data',
        'sweep-evaluation',
        'sweep-development',
    ],
)
def test_input_errors_leave_no_output(workspace, run_cli, case):
    command, role = case.split('-')
    selection = 'not-a-file' if role == 'evaluation' else 'absent/*'
    (workspace / 'not-a-file').mkdir()
    valid = shlex.split(BASE_ARGV[command].replace('invalid', 'retry') + ' --updates 1')
    output = workspace / ('runs/retry' if command == 'train' else 'retry')
    manifest = output / ('manifest/run.json' if command == 'train' else 'sweep.json')
    invalid = list(valid)
    if role == 'evaluation':
        invalid += ['--eval-data', selection, '--eval-every', '1']
    else:
        option = '--development' if role == 'development' else '--data'
        invalid[invalid.index(option) + 1] = selection
    assert run_cli(*invalid)[0] == 1
    assert not output.exists()
    assert run_cli(*valid)[0] == 0
    record = json.loads(manifest.read_text())
    assert record['status'] == 'complete'
    if command == 'sweep':
        assert record['completed_runs'] == 2
        assert record['error'] is None


@pytest.mark.parametrize('limit,expected', [(6, [b'abcd', b'ef']), (1, [b'a'])])
def test_documents_budget(tmp_path, limit, expected):
    from tmt.cli import documents

    (tmp_path / 'a.bin').write_bytes(b'abcd')
    (tmp_path / 'b.bin').write_bytes(b'efgh')
    assert list(documents(str(tmp_path / '*.bin'), limit)) == expected


def test_generate(checkpoint, tmp_path, capsys):
    from tmt.cli import generate

    path, original = checkpoint.path, checkpoint.path.read_bytes()
    first_path, second_path = tmp_path / 'first.bin', tmp_path / 'second.bin'
    first = generate(path, 'The ', first_path, 11, 16, log_every=5)
    lines = capsys.readouterr().out.splitlines()
    assert [line for line in lines if line.startswith('generated bytes:')] == [
        'generated bytes: 5/16',
        'generated bytes: 10/16',
        'generated bytes: 15/16',
    ]
    second = generate(path, 'The ', second_path, 11, 16)
    assert len(first) == len(second) == 16
    assert first == second
    assert first_path.read_bytes() == first
    assert second_path.read_bytes() == second
    assert path.read_bytes() == original
    first_path.write_bytes(b'previous sample')
    assert generate(path, 'The ', first_path, count=0) == b''
    assert first_path.read_bytes() == b''
    for seed in (0, 2**64 - 1):
        assert len(generate(path, 'X', tmp_path / 'sample.bin', seed, 1)) == 1


def test_generate_temperature_floor(checkpoint, tmp_path):
    from tmt.cli import generate

    path, output = checkpoint.path, tmp_path / 'sample.bin'
    minimum = generate(path, 'The ', output, count=2, temperature=0.1)
    for temperature in (0, 0.05):
        assert generate(path, 'The ', output, 11, 2, temperature) == minimum


def test_evaluate_command(workspace, checkpoint, run_cli):
    code, _, stderr = run_cli(
        *('evaluate', str(checkpoint.path), '--data', 'dev.bin', '--max-bytes', '2'),
        *('--output', 'score.json'),
    )
    assert code == 0, stderr
    score = json.loads((workspace / 'score.json').read_text())
    assert (score['input_bytes'], score['targets']) == (2, 1)
    (workspace / 'invalid.safetensors').write_bytes(b'invalid')
    code, _, stderr = run_cli('evaluate', 'invalid.safetensors', '--data', 'dev.bin')
    assert code == 1
    assert 'tmt: error:' in stderr
    (workspace / 'short.bin').write_bytes(b'x')
    code, _, stderr = run_cli('evaluate', str(checkpoint.path), '--data', 'short.bin')
    assert code == 1
    assert 'at least one target' in stderr


def test_installed_workflow(tmp_path, manifest_fields, cli, jsonl):
    from tmt.cli import load_model

    root = tmp_path
    assert '## init' in cli('help', 'init').stdout
    cli('init')
    (root / 'model.json').write_text('{"dim":4,"layers":1,"spread":4,"block":"rtu"}')
    (root / 'data/train/tiny.bin').write_bytes(b'abcd')
    (root / 'data/dev/tiny.bin').write_bytes(b'wxyz')
    progress = cli(
        *('train', '--data', 'data/train/*', '--run', 'first', '--updates', '3'),
        *('--log-every', '2', '--sample-every', '2', '--sample-bytes', '4'),
        *('--sample-prompt', 'A', '--sample-seed', '7', '--eval-data', 'data/dev/*'),
        *('--eval-every', '2', '--eval-max-bytes', '3'),
    ).stdout
    assert 'updates: 2, objective: tmt, loss:' in progress
    assert 'updates: 3, objective: tmt, loss:' in progress
    checkpoint = root / 'runs/first/model.safetensors'
    assert 'sample at update 2, step 2:' in progress
    assert 'evaluation at update 3: BPB' in progress
    samples = jsonl(checkpoint.parent / 'manifest/samples.jsonl')
    keys = ('completed_updates', 'seed', 'bytes', 'prompt')
    assert [{k: row[k] for k in keys} for row in samples] == [
        {'completed_updates': 2, 'seed': 7, 'bytes': 4, 'prompt': 'A'}
    ]
    assert len(Path(samples[0]['path']).read_bytes()) == 4
    evaluation = jsonl(checkpoint.parent / 'manifest/evaluation.jsonl')
    keys = ('completed_updates', 'input_bytes', 'targets')
    assert [{k: row[k] for k in keys} for row in evaluation] == [
        {'completed_updates': 2, 'input_bytes': 3, 'targets': 2},
        {'completed_updates': 3, 'input_bytes': 3, 'targets': 2},
    ]
    model, settings = load_model(checkpoint)
    expected = {'dim': 4, 'layers': 1, 'spread': 4}
    assert {k: settings[k] for k in expected} == expected
    assert model.optimizer.state['step'].item() == 3
    manifest = checkpoint.parent / 'manifest/run.json'
    fresh = json.loads(manifest.read_text())
    expected = {
        'data_glob': 'data/train/*',
        'data_files': [str((root / 'data/train/tiny.bin').resolve())],
        'updates': 3,
        'completed_updates': 3,
        'objective': 'tmt',
        'resume': False,
        'initial_optimizer_step': 0,
        'trainable_parameters': model.count(),
        'temporal_gradient': model.temporal_gradient,
    }
    assert {k: fresh[k] for k in expected} == expected
    (root / 'data/resume').mkdir()
    (root / 'data/resume/next.bin').write_bytes(b'wxyz')
    cli(
        *('train', str(checkpoint), '--data', 'data/resume/*', '--resume'),
        *('--updates', '2', '--seed', '99', '--ce-only'),
    )
    assert load_model(checkpoint)[0].optimizer.state['step'].item() == 5
    record = json.loads(manifest.read_text())
    assert set(record) == manifest_fields
    expected = {
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
    }
    assert {k: record[k] for k in expected} == expected
    cli(
        *('evaluate', str(checkpoint), '--data', 'data/train/*', '--windows', '1', '2'),
        *('--output', 'score.json'),
    )
    assert json.loads((root / 'score.json').read_text())['targets'] == 3
    cli(
        *('generate', str(checkpoint), '--prompt', 'The ', '--output', 'sample.bin'),
        *('--bytes', '4'),
    )
    assert len((root / 'sample.bin').read_bytes()) == 4
    assert '390 targets' in cli('benchmark', str(checkpoint)).stdout
    (root / 'g.json').write_text('{"seed": [11]}')
    sweep = (
        *('sweep', '--grid', 'g.json', '--data', 'data/train/*'),
        *('--development', 'data/dev/*', '--output', 'sw', '--updates', '1'),
    )
    cli(*sweep)
    assert len(jsonl(root / 'sw/results.jsonl')) == 1
    before = {p: p.read_bytes() for p in (root / 'sw').rglob('*') if p.is_file()}
    assert 'not empty' in cli(*sweep, code=1).stderr
    after = {p: p.read_bytes() for p in (root / 'sw').rglob('*') if p.is_file()}
    assert after == before
