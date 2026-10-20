import argparse
import json
import shlex
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest

TOPICS = (
    'init', 'train', 'sweep', 'evaluate', 'benchmark', 'generate', 'inspect', 'help'
)
BASE_ARGV = {
    'train': 'train --data train.bin --run invalid --seed 11',
    'sweep': 'sweep --grid grid.json --data train.bin --development dev.bin '
    '--output invalid --seed 11',
    'evaluate': 'evaluate missing.safetensors --data dev.bin --output score.json',
    'generate': 'generate missing.safetensors --prompt x --output sample.bin --seed 11',
    'benchmark': 'benchmark missing.safetensors --cola-data missing.tsv --seed 11',
}
RUN_NAME = 'run must be one relative folder name'
EVAL_PAIR = 'eval-data and eval-every must be used together'
TEMPERATURE = 'temperature must be a finite nonnegative number'
SYNTAX = 'invalid JSON in syntax.json: Expecting'
NO_TARGETS = 'evaluation requires at least one target byte pair'
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
    'train-save-every': ('--save-every 0', 'save-every must be positive'),
    'sweep-save-every': ('--save-every -1', 'save-every must be positive'),
    'run-empty': ("--run ''", RUN_NAME),
    'run-parent': ('--run ..', RUN_NAME),
    'run-absolute': ('--run /absolute-name', RUN_NAME),
    'run-nested': ('--run nested/name', RUN_NAME),
    'run-backslash': ("--run 'nested\\name'", RUN_NAME),
    'train-bad-config': ('--config bad.json', 'model config.dim'),
    'train-config-syntax': ('--config syntax.json', SYNTAX),
    'train-eval-no-targets': ('--eval-data short.bin --eval-every 1', NO_TARGETS),
    'sweep-config-syntax': ('--config syntax.json', SYNTAX),
    'sweep-grid-syntax': ('--grid syntax.json', SYNTAX),
    'sweep-eval-no-targets': ('--eval-data short.bin --eval-every 1', NO_TARGETS),
    'sweep-development-no-targets': ('--development short.bin', NO_TARGETS),
    'sweep-updates': ('--updates -1', 'updates must be positive'),
    'sweep-max-bytes': ('--max-bytes 0', 'max-bytes must be positive'),
    'sweep-sample-bytes': ('--sample-bytes -1', 'sample-bytes must be positive'),
    'evaluate-windows': ('--windows 0', 'evaluation windows must be positive'),
    'evaluate-max-bytes': ('--max-bytes -1', 'max-bytes must be positive'),
    'evaluate-no-targets': ('--data short.bin', NO_TARGETS),
    'generate-log-every': ('--log-every 0', 'log-every must be positive'),
    'generate-negative-bytes': ('--bytes -1', 'bytes must be nonnegative'),
    'generate-temperature-nan': ('--temperature=nan', TEMPERATURE),
    'generate-temperature-positive-infinity': ('--temperature=inf', TEMPERATURE),
    'generate-temperature-negative-infinity': ('--temperature=-inf', TEMPERATURE),
    'generate-temperature-negative': ('--temperature=-1', TEMPERATURE),
    'generate-missing-checkpoint': ('', "'missing.safetensors'"),
    'generate-empty-prompt': ("--prompt ''", 'generation needs a nonempty prompt'),
    'generate-output-folder': (
        '--output missing/sample.bin',
        'output folder missing does not exist',
    ),
    'benchmark-epochs': ('--epochs 0', 'epochs must be positive with cola-data'),
}
HELP_ERRORS = {  # id: (argv, expected stderr fragment)
    'unknown-command': ('help nonexistent', 'unknown help topic'),
    'unknown-after-long-help': ('--help nonexistent', 'unknown help topic'),
    'unknown-after-short-help': ('-h nonexistent', 'unknown help topic'),
    'unknown-before-help': ('nonexistent --help', 'unknown help topic'),
    'multiple-topics': ('help train sweep', 'help accepts at most one topic'),
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
OPERATIONS = 'new generate train train-sample sweep sweep-sample benchmark'.split()
SEED_ARGV = {  # even cases get seed -1, odd cases get 2**64
    'train-seed': 'train --data missing/* --run valid --seed',
    'train-sample-seed': 'train --data missing/* --run valid --seed 11 --sample-seed',
    'resume-seed': 'train missing.safetensors --data missing/* --resume --seed',
    'resume-sample-seed': 'train missing.safetensors --data missing/* --resume '
    '--seed 11 --sample-seed',
    'generate-seed': 'generate missing.safetensors --prompt X --output protected.bin '
    '--seed',
    'benchmark-seed': 'benchmark missing.safetensors --seed',
    'sweep-seed': SWEEP_MISSING + ' --seed',
    'sweep-sample-seed': SWEEP_MISSING + ' --seed 11 --sample-seed',
}


@pytest.fixture
def manual():
    return files('tmt').joinpath('commands.md').read_text(encoding='utf-8')


@pytest.mark.parametrize(
    ('argv', 'expected'),
    [
        (
            'train --data train.bin --seed 11',
            {
                'command': 'train',
                'config': None,
                'log_every': 100,
                'save_every': 500,
                'checkpoint': None,
                'resume': False,
            },
        ),
        (
            'sweep --data train.bin --seed 11 --grid grid.json '
            '--development dev.bin --output sweep',
            {
                'command': 'sweep',
                'config': 'model.json',
                'log_every': 100,
                'save_every': 500,
                'max_bytes': 8192,
            },
        ),
        (
            'generate model.safetensors --prompt X --output sample.bin --seed 11',
            {
                'command': 'generate',
                'log_every': 1000,
                'bytes': 512,
                'temperature': None,
            },
        ),
        (
            'evaluate model.safetensors --data dev.bin',
            {'command': 'evaluate', 'max_bytes': 8192, 'windows': (), 'output': None},
        ),
        (
            'benchmark model.safetensors --seed 11',
            {
                'command': 'benchmark',
                'epochs': 1,
                'split': 0.5,
                'cola_data': None,
                'output': None,
            },
        ),
        ('init', {'command': 'init', 'output': None}),
    ],
    ids=['train', 'sweep', 'generate', 'evaluate', 'benchmark', 'init'],
)
def test_parser_command_defaults(argv, expected):
    from tmt.cli import build_parser

    args = vars(build_parser().parse_args(shlex.split(argv)))
    assert {name: args[name] for name in expected} == expected


@pytest.mark.parametrize(
    'argv',
    [
        ['train'],
        ['sweep', '--grid', 'grid.json', '--development', 'dev/*', '--output', 'sweep'],
    ],
    ids=['train', 'sweep'],
)
def test_shared_training_arguments(argv):
    from tmt.cli import build_parser

    options = (
        "--data 'train/*' --updates 3 --log-every 2 --save-every 5 --seed 22 --ce-only "
        '--sample-every 1 --sample-prompt X --sample-seed 33 --sample-bytes 4 '
        "--eval-data 'dev/*' --eval-every 2 --eval-max-bytes 8"
    )
    args = vars(build_parser().parse_args([*argv, *shlex.split(options)]))
    expected = {
        'data': 'train/*',
        'updates': 3,
        'log_every': 2,
        'save_every': 5,
        'seed': 22,
        'ce_only': True,
        'sample_every': 1,
        'sample_prompt': 'X',
        'sample_seed': 33,
        'sample_bytes': 4,
        'eval_data': 'dev/*',
        'eval_every': 2,
        'eval_max_bytes': 8,
    }
    assert {name: args[name] for name in expected} == expected


def test_every_command_has_handler_and_manual_section(manual):
    from tmt.cli import COMMANDS, HELP_TOPICS, build_parser

    parser = build_parser()
    subparsers = next(
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert tuple(subparsers.choices) == HELP_TOPICS
    assert set(HELP_TOPICS) == set(TOPICS)
    for command in COMMANDS:
        subparser = subparsers.choices[command.name]
        assert subparser.get_default('handler') is command.handler
        assert f'\n## {command.name}\n' in manual


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
    ('module', 'argv', 'code', 'stdout_start', 'stderr_has'),
    [
        ('tmt', '--version', 0, 'tmt ', ''),
        ('tmt', '--version --help', 0, 'tmt ', ''),
        ('tmt', '--help --version', 0, 'usage: tmt', ''),
        ('tmt', '--seed 1 --help', 2, '', 'tmt: error:'),
        ('tmt', 'train --data missing/* --seed -1', 1, '', 'seed must be between'),
        ('tmt.benchmark', '--help', 0, '## benchmark', ''),
    ],
    ids=[
        'version',
        'version-first',
        'help-first',
        'unknown-option-first',
        'seed-error',
        'benchmark-module-help',
    ],
)
def test_model_free_commands(
    model_free_cli, tmp_path, module, argv, code, stdout_start, stderr_has
):
    result = model_free_cli(*shlex.split(argv), module=module)
    assert result.returncode == code, result.stderr
    assert result.stdout.startswith(stdout_start)
    assert stderr_has in result.stderr
    assert 'unknown help topic' not in result.stderr
    assert 'Traceback' not in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    'argv',
    [
        ['--help'],
        ['-h'],
        ['help'],
        ['help', 'train'],
        ['help', '--help'],
        ['--help', 'train'],
        ['-h', 'train'],
        ['train', '--help'],
        ['train', '-h'],
        ['train', '--data', 'missing/*', '--help'],
    ],
    ids=[
        'index-long', 'index-short', 'manual', 'manual-topic', 'help-topic',
        'long-topic', 'short-topic', 'command-long', 'command-short',
        'command-with-options',
    ],
)
def test_help_forms_need_no_setup(tmp_path, model_free_cli, manual, argv):
    result = model_free_cli(*argv)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ''
    assert 'Traceback' not in result.stderr
    assert list(tmp_path.iterdir()) == []
    if argv in (['--help'], ['-h']):
        assert result.stdout.startswith('usage: tmt')
        assert 'tmt help' in result.stdout
        assert '## init' not in result.stdout
    elif argv == ['help']:
        assert result.stdout == manual
    else:
        topic = 'help' if argv[:2] == ['help', '--help'] else 'train'
        section = manual.split(f'## {topic}\n', 1)[1].split('\n## ', 1)[0]
        assert result.stdout == f'## {topic}\n{section.rstrip()}\n'


@pytest.mark.parametrize(
    ('module', 'argv'),
    [
        ('tmt', ['train', '--data', 'missing/*', '--run', 'run']),
        (
            'tmt',
            [
                'sweep', '--grid', 'grid.json', '--data', 'missing/*',
                '--development', 'missing/*', '--output', 'sweep',
            ],
        ),
        (
            'tmt',
            [
                'generate', 'missing.safetensors', '--prompt', 'x',
                '--output', 'sample.bin',
            ],
        ),
        ('tmt', ['benchmark', 'missing.safetensors']),
        ('tmt.benchmark', ['missing.safetensors']),
    ],
    ids=['train', 'sweep-with-seed-axis', 'generate', 'benchmark', 'benchmark-module'],
)
def test_primary_seed_required_before_import(
    tmp_path, model_free_cli, module, argv
):
    if argv[0] == 'sweep':
        (tmp_path / 'grid.json').write_text('{"seed": [11, 22]}')
    result = model_free_cli(*argv, module=module)
    assert result.returncode == 2
    assert 'the following arguments are required: --seed' in result.stderr
    assert result.stdout == ''
    assert 'Traceback' not in result.stderr
    expected = ['grid.json'] if argv[0] == 'sweep' else []
    assert sorted(path.name for path in tmp_path.iterdir()) == expected


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


@pytest.mark.parametrize(
    ('argv', 'expected'),
    [
        ([], (False, None)),
        (['help'], (True, None)),
        (['help', 'train'], (True, 'train')),
        (['help', '--help'], (True, 'help')),
        (['-h'], (False, None)),
        (['--help', 'train'], (True, 'train')),
        (['--help', '--version'], (False, None)),
        (['train', '--help'], (True, 'train')),
        (['nonexistent', '-h'], (True, 'nonexistent')),
        (['--version', '--help'], (False, None)),
        (['--seed', '1', '--help'], (False, None)),
        (['train', '--data', 'x'], (False, None)),
    ],
)
def test_manual_topic(argv, expected):
    from tmt.cli import _manual_topic

    assert _manual_topic(argv) == expected


@pytest.mark.parametrize(
    ('argv', 'commands'),
    [
        ('--seed 1 --help', 'train, benchmark, generate, sweep'),
        ('--seed 1 train --data train.bin', 'train, benchmark, generate, sweep'),
        ('--data=train.bin train', 'train, evaluate, sweep'),
        ('--resume train model.safetensors', 'train'),
        ('--windows 2 evaluate model.safetensors', 'evaluate'),
    ],
    ids=['before-help', 'before-command', 'equals-form', 'flag', 'evaluate-only'],
)
def test_command_option_before_command(run_cli, tmp_path, argv, commands):
    code, stdout, stderr = run_cli(*shlex.split(argv))
    assert (code, stdout) == (2, '')
    name = shlex.split(argv)[0].split('=')[0]
    assert (
        f'{name} must come after the command name; commands that accept it: {commands}'
        in stderr
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ('argv', 'code'),
    [('--vers', 0), ('--bogus train', 2), ('-- train', 2)],
    ids=['top-level-abbreviation', 'unknown-option', 'separator'],
)
def test_other_leading_options_use_parser(run_cli, argv, code):
    result = run_cli(*argv.split())
    assert result[0] == code
    assert 'must come after the command name' not in result[2]


@pytest.mark.parametrize('case', HELP_ERRORS)
def test_help_errors(run_cli, tmp_path, case):
    argv, message = HELP_ERRORS[case]
    code, stdout, stderr = run_cli(*argv.split())
    assert (code, stdout) == (2, '')
    assert message in stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('case', ARGUMENT_ERRORS)
def test_argument_errors(workspace, run_cli, case):
    argv = [
        'train', '--data', 'train.bin', '--seed', '11',
        *ARGUMENT_ERRORS[case].split(),
    ]
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


def test_training_validation_order(run_cli, tmp_path):
    code, stdout, stderr = run_cli(
        'train', '--data', 'missing/*', '--seed', '11',
        '--log-every', '0', '--sample-bytes', '0',
    )
    assert (code, stdout) == (1, '')
    assert 'log-every must be positive' in stderr
    assert list(tmp_path.iterdir()) == []


SWEEP_PATHS = ('missing.json', 'missing/*', 'missing/*', 'unused', 'model.json')


@pytest.mark.parametrize('operation', OPERATIONS)
def test_direct_seed_validation(monkeypatch, forbid_call, operation):
    from tmt import runtime, validation
    from tmt.benchmark import run
    from tmt.checkpoint import new_model
    from tmt.runtime import generate, sweep, train

    monkeypatch.setattr(validation, 'model_settings', forbid_call)
    monkeypatch.setattr(validation, 'checkpoint_header', forbid_call)
    for name in 'load_config ByteFiles'.split():
        monkeypatch.setattr(runtime, name, forbid_call)
    monkeypatch.setattr(runtime, 'TrainingRun', forbid_call)
    seed = (-1, 2**64, True, 1.5)[OPERATIONS.index(operation) % 4]
    calls = {
        'new': lambda: new_model({}, seed),
        'generate': lambda: generate('missing.safetensors', 'X', 'unused.bin', seed),
        'train': lambda: train({}, 'unused/model.safetensors', 'missing/*', 1, seed),
        'train-sample': lambda: train(
            {}, 'unused/model.safetensors', 'missing/*', 1, 11, sample_seed=seed
        ),
        'sweep': lambda: sweep(*SWEEP_PATHS, 1, seed),
        'sweep-sample': lambda: sweep(*SWEEP_PATHS, 1, 11, sample_seed=seed),
        'benchmark': lambda: run(
            'missing.safetensors', model=SimpleNamespace(freeze=forbid_call), seed=seed
        ),
    }
    message = 'sample-seed must be' if operation.endswith('-sample') else 'seed must be'
    with pytest.raises(ValueError, match=message):
        calls[operation]()


def test_direct_primary_seed_has_no_default():
    import inspect

    from tmt.benchmark import run
    from tmt.runtime import generate, sweep, train

    for function in (train, sweep, generate, run):
        seed = inspect.signature(function).parameters['seed']
        assert seed.default is inspect.Parameter.empty, function.__name__


@pytest.mark.parametrize('case', INVALID_OPTION_VALUES)
def test_invalid_option_values(workspace, run_cli, case):
    """Each input error stops before model work and leaves no output."""
    extra, message = INVALID_OPTION_VALUES[case]
    command = 'train' if case.startswith('run-') else case.split('-')[0]
    (workspace / 'bad.json').write_text('{"dim": true}')
    (workspace / 'syntax.json').write_text('{"dim": 4,')
    (workspace / 'short.bin').write_bytes(b'x')
    kept = case in ('generate-negative-bytes', 'generate-temperature-negative')
    if kept:
        (workspace / 'sample.bin').write_bytes(b'keep')
    code, stdout, stderr = run_cli(*shlex.split(BASE_ARGV[command] + ' ' + extra))
    assert (code, stdout) == (1, '')
    assert message in stderr
    if kept:
        assert (workspace / 'sample.bin').read_bytes() == b'keep'
    absent = ['runs', 'invalid', 'score.json', 'absolute-name', 'missing']
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


def test_fresh_train_keeps_existing_checkpoint(
    workspace, run_cli, monkeypatch, forbid_call
):
    (workspace / 'model.safetensors').write_bytes(b'keep')
    monkeypatch.setattr('tmt.runtime.new_model', forbid_call)
    code, stdout, stderr = run_cli(
        'train', 'model.safetensors', '--data', 'train.bin', '--seed', '11'
    )
    assert (code, stdout) == (1, '')
    assert stderr == (
        'tmt: error: checkpoint model.safetensors exists; '
        'use --resume, or remove it\n'
    )
    assert (workspace / 'model.safetensors').read_bytes() == b'keep'
    assert not (workspace / 'manifest').exists()


@pytest.mark.parametrize(
    ('selection', 'error'),
    [('*.bin', FileNotFoundError), ('selected', IsADirectoryError)],
    ids=['unmatched', 'directory'],
)
def test_byte_files_rejects_bad_selection(tmp_path, selection, error):
    from tmt.data import ByteFiles

    (tmp_path / 'selected').mkdir()
    with pytest.raises(error):
        ByteFiles(str(tmp_path / selection))


def test_byte_files_keep_empty_files_and_share_budget(tmp_path):
    from tmt.data import ByteFiles, documents

    for name, content in (('a', b''), ('b', b'abcd'), ('c', b''), ('d', b'efgh')):
        (tmp_path / f'{name}.bin').write_bytes(content)
    pattern = str(tmp_path / '*.bin')
    files = ByteFiles(pattern)

    assert [Path(path).name for path in files.paths] == [
        'a.bin',
        'b.bin',
        'c.bin',
        'd.bin',
    ]
    assert list(files.documents()) == [b'', b'abcd', b'', b'efgh']
    assert list(documents(pattern, 6)) == [b'', b'abcd', b'', b'ef']
    assert list(documents(pattern, 1)) == [b'', b'a']


def test_generate(checkpoint, tmp_path, capsys):
    import mlx.core as mx

    from tmt.runtime import generate

    path, original = checkpoint.path, checkpoint.path.read_bytes()
    first_path, second_path = tmp_path / 'first.bin', tmp_path / 'second.bin'
    mx.random.seed(5)
    rng_before = mx.array(mx.random.state[0])
    first = generate(path, 'The ', first_path, 11, count=16, log_every=5)
    assert mx.array_equal(mx.random.state[0], rng_before).item()
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        'generated bytes: 5/16',
        'generated bytes: 10/16',
        'generated bytes: 15/16',
        f'wrote output: {first_path}, bytes: 16',
    ]
    second = generate(path, 'The ', second_path, 11, count=16)
    assert len(first) == len(second) == 16
    assert first == second
    assert first_path.read_bytes() == first
    assert second_path.read_bytes() == second
    assert path.read_bytes() == original
    first_path.write_bytes(b'previous sample')
    assert generate(path, 'The ', first_path, seed=11, count=0) == b''
    assert first_path.read_bytes() == b''
    for seed in (0, 2**64 - 1):
        assert len(generate(path, 'X', tmp_path / 'sample.bin', seed, count=1)) == 1


def test_generate_temperature_floor(checkpoint, tmp_path):
    from tmt.runtime import generate

    path, output = checkpoint.path, tmp_path / 'sample.bin'
    minimum = generate(path, 'The ', output, seed=11, count=2, temperature=0.1)
    for temperature in (0, 0.05):
        sample = generate(path, 'The ', output, 11, count=2, temperature=temperature)
        assert sample == minimum


def test_inspect_command(workspace, checkpoint, run_cli):
    code, _, stderr = run_cli(
        'train', '--data', 'train.bin', '--run', 'a', '--updates', '3', '--seed', '11'
    )
    assert code == 0, stderr
    code, stdout, stderr = run_cli('inspect', 'runs/a/model.safetensors')
    assert (code, stderr) == (0, '')
    settings = json.loads((workspace / 'model.json').read_text())
    assert stdout.splitlines() == [
        f'checkpoint: runs/a/model.safetensors, step: 3, '
        f'trainable parameters: {checkpoint.model.count()}',
        f'config: {json.dumps(settings)}',
        'run status: complete, objective: tmt, seed: 11, updates: 3',
    ]
    code, stdout, _ = run_cli('inspect', str(checkpoint.path))
    assert code == 0
    assert stdout.splitlines()[0].startswith(f'checkpoint: {checkpoint.path}, step: 0,')
    assert stdout.splitlines()[-1] == 'run manifest: none'
    (workspace / 'invalid.safetensors').write_bytes(b'invalid')
    code, stdout, stderr = run_cli('inspect', 'invalid.safetensors')
    assert (code, stdout) == (1, '')
    assert stderr.startswith('tmt: error: invalid checkpoint:')


def test_evaluate_command(workspace, checkpoint, run_cli):
    code, _, stderr = run_cli(
        *('evaluate', str(checkpoint.path), '--data', 'dev.bin', '--max-bytes', '2'),
        *('--output', 'score.json'),
    )
    assert code == 0, stderr
    score = json.loads((workspace / 'score.json').read_text())
    assert (score['input_bytes'], score['targets']) == (2, 1)
    assert list(score)[-2:] == ['checkpoint', 'optimizer_step']
    assert (score['checkpoint'], score['optimizer_step']) == (str(checkpoint.path), 0)
    (workspace / 'invalid.safetensors').write_bytes(b'invalid')
    code, _, stderr = run_cli('evaluate', 'invalid.safetensors', '--data', 'dev.bin')
    assert code == 1
    assert 'tmt: error:' in stderr
    (workspace / 'short.bin').write_bytes(b'x')
    code, _, stderr = run_cli('evaluate', str(checkpoint.path), '--data', 'short.bin')
    assert code == 1
    assert 'at least one target' in stderr


@pytest.mark.parametrize(
    'launch',
    [
        'from tmt.cli import console_main\nconsole_main(ARGV)',
        "import runpy, sys\nsys.argv = ['tmt', *ARGV]\n"
        "runpy.run_module('tmt', run_name='__main__', alter_sys=True)",
    ],
    ids=['console-script', 'python-m'],
)
@pytest.mark.parametrize('signal_name', ['SIGINT', 'SIGTERM'])
def test_train_signal_saves_checkpoint(workspace, snapshot, launch, signal_name):
    import signal
    import subprocess
    import sys

    from tmt.checkpoint import load_checkpoint
    from tmt.runtime import train

    script = """
import os
import signal
from tmt.model import Model

update = Model.update_gradients
def interrupt(model, grads):
    os.kill(os.getpid(), signal.SIGNAL_NAME)
    return update(model, grads)
Model.update_gradients = interrupt
ARGV = ['train', '--data', 'train.bin', '--run', 'interrupted',
        '--seed', '11', '--updates', '3']
""".replace('SIGNAL_NAME', signal_name) + launch
    result = subprocess.run(
        [sys.executable, '-c', script], cwd=workspace,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == -signal.SIGINT, result.stderr
    assert 'Traceback' not in result.stderr
    assert result.stderr.splitlines()[-1] == 'tmt: interrupted'
    checkpoint = workspace / 'runs/interrupted/model.safetensors'
    assert 'saved checkpoint:' in result.stdout
    record = json.loads((checkpoint.parent / 'manifest/run.json').read_text())
    assert record['status'] == 'interrupted'
    assert record['completed_updates'] == 1
    loaded, settings = load_checkpoint(checkpoint)
    expected = train(
        settings, workspace / 'reference/model.safetensors', 'train.bin', 1, 11
    )
    assert snapshot(loaded) == snapshot(expected)


def raise_interrupt(*args, **kwargs):
    raise KeyboardInterrupt


def test_console_main_reports_interrupt_once(monkeypatch, capsys):
    import sys

    from tmt import cli

    previous = []
    monkeypatch.setattr(sys, 'excepthook', lambda kind, *rest: previous.append(kind))
    monkeypatch.setattr(cli, 'main', raise_interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli.console_main(['train'])
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    sys.excepthook(ValueError, ValueError('other'), None)
    assert capsys.readouterr().err == 'tmt: interrupted\n'
    assert previous == [ValueError]


def test_main_leaves_interrupt_to_caller(monkeypatch):
    import sys

    from tmt import cli

    hook = sys.excepthook
    monkeypatch.setattr(cli, 'evaluate_checkpoint', raise_interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli.main(['evaluate', 'model.safetensors', '--data', 'dev.bin'])
    assert sys.excepthook is hook


def test_console_script_uses_console_main():
    from importlib.metadata import entry_points

    (script,) = entry_points(group='console_scripts', name='tmt')
    assert script.value == 'tmt.cli:console_main'


def test_installed_workflow(tmp_path, manifest_fields, cli, jsonl):
    from tmt.checkpoint import load_checkpoint

    root = tmp_path
    assert '## init' in cli('help', 'init').stdout
    cli('init')
    (root / 'model.json').write_text('{"dim":4,"layers":1,"spread":4,"block":"rtu"}')
    (root / 'data/train/tiny.bin').write_bytes(b'abcd')
    (root / 'data/dev/tiny.bin').write_bytes(b'wxyz')
    progress = cli(
        *(
            'train', '--data', 'data/train/*', '--run', 'first',
            '--updates', '3', '--seed', '11',
        ),
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
    model, settings = load_checkpoint(checkpoint)
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
    assert load_checkpoint(checkpoint)[0].optimizer.state['step'].item() == 5
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
        *('--bytes', '4', '--seed', '11'),
    )
    assert len((root / 'sample.bin').read_bytes()) == 4
    assert 'targets: 390' in cli('benchmark', str(checkpoint), '--seed', '11').stdout
    (root / 'g.json').write_text('{"seed": [11]}')
    sweep = (
        *('sweep', '--grid', 'g.json', '--data', 'data/train/*'),
        *(
            '--development', 'data/dev/*', '--output', 'sw', '--updates', '1',
            '--seed', '11',
        ),
    )
    (root / 'sw').mkdir()
    cli(*sweep)
    assert len(jsonl(root / 'sw/results.jsonl')) == 1
    before = {p: p.read_bytes() for p in (root / 'sw').rglob('*') if p.is_file()}
    assert 'not empty' in cli(*sweep, code=1).stderr
    after = {p: p.read_bytes() for p in (root / 'sw').rglob('*') if p.is_file()}
    assert after == before
