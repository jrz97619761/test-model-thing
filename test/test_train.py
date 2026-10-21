import json
import math
import signal
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

INVALID_GRIDS = {
    'sweep grid must be a JSON object': {
        'null-grid': None,
        'array-grid': [],
        'string-grid': 'dim',
    },
    "sweep grid field 'dim' must be a nonempty array": {
        'scalar-dim-axis': {'dim': 4},
        'null-dim-axis': {'dim': None},
        'object-dim-axis': {'dim': {}},
        'empty-dim-axis': {'dim': []},
        'boolean-dim-axis': {'dim': True},
        'existing-empty-output': {'dim': []},
    },
    "sweep grid field 'seed' must be a nonempty array": {
        'empty-seed-axis': {'dim': [4], 'seed': []}
    },
    'model config.dim must be a positive integer': {'zero-dimension': {'dim': [4, 0]}},
    'sweep grid seed must be an integer': {
        'boolean-seed': {'seed': [11, True]},
        'string-seed': {'seed': [11, '22']},
    },
    'sweep grid seed must be between': {
        'negative-seed': {'seed': [11, -1]},
        'seed-overflow': {'seed': [11, 2**64]},
    },
    'model config.bound must be two integers': {'bound': {'bound': [30]}},
    'model config.block must be one of': {'block': {'block': ['unknown-block']}},
    "model config has unknown field 'unknown'": {'unknown-field': {'unknown': [1]}},
}


def test_train_visits_byte_pairs_and_resumes(
    monkeypatch, capsys, tiny_settings, tmp_path, snapshot
):
    from tmt.model import Model
    from tmt.runtime import train

    data = tmp_path / 'train.bin'
    data.write_bytes(bytes((0, 255, 65, 10)))
    one, two = (tmp_path / name / 'model.safetensors' for name in ('one', 'two'))
    calls, frozen_steps = [], []
    train_step, step = Model.train_step, Model.step

    def record_train(model, current, target, end, ce_only=False, **kwargs):
        calls.append((current, target, end))
        return train_step(model, current, target, end, ce_only, **kwargs)

    def record_step(model, current, *args, frozen=False, **kwargs):
        if frozen:
            frozen_steps.append(int(current.item()))
        return step(model, current, *args, frozen=frozen, **kwargs)

    monkeypatch.setattr(
        Model, 'sample', lambda *a, **k: pytest.fail('training must not sample')
    )
    monkeypatch.setattr(Model, 'train_step', record_train)
    monkeypatch.setattr(Model, 'step', record_step)
    first = train(tiny_settings, one, str(data), 3, 11)
    second = train(tiny_settings, two, str(data), 3, 11)
    resumed = train(None, one, str(data), 2, 11, resume=True)

    stdout = capsys.readouterr().out
    for message in ('run folder:', 'saved checkpoint:', 'objective:'):
        assert stdout.count(message) == 3, f'expected three occurrences of {message!r}'
    assert 'seed: 11, config:' in stdout
    assert '\r' not in stdout
    expected = [(0, 255, False), (255, 65, False), (65, 10, True)]
    assert calls == expected + expected + expected[:2]
    assert frozen_steps == [10, 10]
    steps = [int(m.optimizer.state['step'].item()) for m in (first, second, resumed)]
    assert steps == [3, 3, 5], 'training and resume optimizer steps differ'
    assert snapshot(first) == snapshot(second)


def test_resume_keeps_earlier_manifest_and_rows(tiny_settings, tmp_path, jsonl):
    from tmt.runtime import train

    data = tmp_path / 'train.bin'
    data.write_bytes(b'abcdef')
    checkpoint = tmp_path / 'run' / 'model.safetensors'
    folder = checkpoint.parent / 'manifest'
    train(tiny_settings, checkpoint, str(data), 3, 11, log_every=2)
    first = json.loads((folder / 'run.json').read_text())
    train(None, checkpoint, str(data), 2, 11, resume=True, log_every=2)
    train(None, checkpoint, str(data), 1, 11, resume=True, log_every=2)

    archives = sorted(folder.glob('run.replaced-*.json'))
    assert len(archives) == 2
    assert json.loads(archives[0].read_text()) == first
    assert json.loads(archives[1].read_text())['optimizer_step'] == 5
    current = json.loads((folder / 'run.json').read_text())
    assert (current['initial_optimizer_step'], current['optimizer_step']) == (5, 6)
    rows = jsonl(folder / 'loss.jsonl')
    assert [r['optimizer_step'] for r in rows] == [2, 3, 5, 6]
    assert [r['completed_updates'] for r in rows] == [2, 3, 2, 1]


def test_save_every_sets_interval_and_updates_manifest(
    monkeypatch, capsys, tiny_settings, tmp_path
):
    import inspect

    from tmt.runtime import TrainingRun, train

    data = tmp_path / 'train.bin'
    data.write_bytes(b'abcdefgh')
    checkpoint = tmp_path / 'run' / 'model.safetensors'
    save, saved = TrainingRun.save_checkpoint, []

    def record(run):
        save(run)
        manifest = json.loads(run.manifest.read_text())
        fields = ('status', 'completed_updates', 'optimizer_step', 'save_every')
        saved.append(tuple(manifest[name] for name in fields))

    monkeypatch.setattr(TrainingRun, 'save_checkpoint', record)
    train(tiny_settings, checkpoint, str(data), 5, 11, save_every=2)
    assert saved == [
        ('running', 2, 2, 2),
        ('running', 4, 4, 2),
        ('running', 5, 5, 2),
    ]
    lines = capsys.readouterr().out.splitlines()
    assert [line for line in lines if line.startswith('saved checkpoint:')] == [
        f'saved checkpoint: {checkpoint}, updates: {updates}, step: {updates}'
        for updates in (2, 4, 5)
    ]
    assert lines[-1] == f'updates: 5, step: 5, checkpoint: {checkpoint}'
    assert inspect.signature(train).parameters['save_every'].default == 500


def test_resume_file_order_uses_seed(tiny_settings, tmp_path):
    import random

    from tmt.checkpoint import new_model, save_checkpoint
    from tmt.runtime import MonitorOptions, TrainingRun

    data = tmp_path / 'data'
    data.mkdir()
    for index in range(8):
        (data / f'{index}.bin').write_bytes(b'ab')
    checkpoint = tmp_path / 'model.safetensors'
    save_checkpoint(checkpoint, new_model(tiny_settings, 11), tiny_settings)
    runs = []
    for unrelated_seed in (7, 37):
        random.seed(unrelated_seed)
        run = TrainingRun(
            settings=None, path=checkpoint, pattern=str(data / '*.bin'),
            updates=1, seed=19, ce_only=False, resume=True, log_every=100,
            monitor=MonitorOptions(),
        )
        run.setup()
        runs.append(run.files)
    expected = sorted(str(path) for path in data.iterdir())
    random.Random(19).shuffle(expected)
    assert runs == [expected, expected]


@pytest.mark.parametrize('resume', [False, True], ids=['fresh', 'resume'])
@pytest.mark.parametrize(
    'stage', ['update', 'terminal', 'sample', 'evaluation', 'save']
)
def test_sigint_saves_complete_update(
    monkeypatch, tiny_settings, tmp_path, snapshot, jsonl, resume, stage
):
    import mlx.core as mx

    from tmt.checkpoint import load_checkpoint
    from tmt.model import Model
    from tmt.runtime import TrainingRun, train

    data = tmp_path / 'train.bin'
    data.write_bytes(b'ab')
    checkpoint = tmp_path / 'run/model.safetensors'
    reference = tmp_path / 'reference/model.safetensors'
    if resume:
        train(tiny_settings, checkpoint, str(data), 2, 11)
        train(tiny_settings, reference, str(data), 2, 11)
    settings = None if resume else tiny_settings
    expected = train(settings, reference, str(data), 1, 11, resume=resume)
    targets = {
        'update': (Model, 'update_gradients'),
        'terminal': (TrainingRun, 'consume_terminal_byte'),
        'sample': (TrainingRun, 'report_sample'),
        'evaluation': (TrainingRun, 'report_evaluation'),
        'save': (mx, 'save_safetensors'),
    }
    owner, name = targets[stage]
    original = getattr(owner, name)

    def interrupt(*args, **kwargs):
        signal.raise_signal(signal.SIGINT)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, interrupt)
    handlers = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        train(
            settings, checkpoint, str(data), 1 if stage == 'save' else 3, 11,
            resume=resume, sample_every=1 if stage == 'sample' else None,
            sample_bytes=2, eval_data=str(data) if stage == 'evaluation' else None,
            eval_every=1 if stage == 'evaluation' else None,
        )
    assert signal.getsignal(signal.SIGINT) == handlers[0]
    assert signal.getsignal(signal.SIGTERM) == handlers[1]
    loaded, _ = load_checkpoint(checkpoint)
    assert snapshot(loaded) == snapshot(expected)
    record = json.loads((checkpoint.parent / 'manifest/run.json').read_text())
    assert record['status'] == 'interrupted'
    assert record['completed_updates'] == 1
    assert record['last_checkpoint'] == str(checkpoint.resolve())
    *earlier, row = jsonl(checkpoint.parent / 'manifest/loss.jsonl')
    assert [r['optimizer_step'] for r in earlier] == ([2] if resume else [])
    assert row['optimizer_step'] == (3 if resume else 1)
    assert row['completed_updates'] == 1


def test_train_in_worker_thread(tiny_settings, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from tmt.runtime import train

    data = tmp_path / 'train.bin'
    data.write_bytes(b'ab')
    handler = signal.getsignal(signal.SIGINT)
    with ThreadPoolExecutor(max_workers=1) as executor:
        model = executor.submit(
            train, tiny_settings, tmp_path / 'model.safetensors', str(data), 1, 11
        ).result(timeout=30)
    assert model.optimizer.state['step'].item() == 1
    assert signal.getsignal(signal.SIGINT) == handler


def test_manifest(monkeypatch, tiny_settings, tmp_path, manifest_fields):
    from tmt.model import Model
    from tmt.runtime import train

    records, calls, paths = [], [], {}
    train_step, original_dump = Model.train_step, json.dump
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'data').mkdir()
    for name, content in (('a.bin', b'ab'), ('b.bin', b'cd'), ('c.bin', b'ef')):
        (tmp_path / 'data' / name).write_bytes(content)
        paths[content[0]] = str((tmp_path / 'data' / name).resolve())

    def capture(record, *args, **kwargs):
        records.append(dict(record))
        return original_dump(record, *args, **kwargs)

    def record_train(model, current, target, end, ce_only=False, **kwargs):
        calls.append((paths[current], ce_only))
        return train_step(model, current, target, end, ce_only, **kwargs)

    checkpoint = tmp_path.resolve() / 'run' / 'model.safetensors'
    with monkeypatch.context() as patches:
        patches.setattr('json.dump', capture)
        patches.setattr(Model, 'train_step', record_train)
        trained = train(tiny_settings, checkpoint, 'data/*.bin', 4, 11)

    assert len(records) == 3
    assert set(records[0]) == manifest_fields
    example_path = Path(__file__).resolve().parents[1] / 'examples' / 'run.example.json'
    example = json.loads(example_path.read_text())
    assert list(records[0]) == list(example)
    assert records[0]['trainable_parameters'] == trained.count()
    assert records[0]['temporal_gradient'] == trained.temporal_gradient
    fixed = {
        'config': tiny_settings,
        'seed': 11,
        'data_glob': 'data/*.bin',
        'updates': 4,
        'objective': 'tmt',
        'resume': False,
        'initial_optimizer_step': 0,
    }
    for record in records:
        assert {k: record[k] for k in fixed} == fixed
        assert set(record['data_files']) == set(paths.values())
        assert record['data_files'] == [path for path, _ in calls[:3]]
    files = records[0]['data_files']
    assert calls == [(path, False) for path in files + files[:1]]
    progress = ('status', 'completed_updates', 'optimizer_step', 'last_checkpoint')
    saved = {
        'completed_updates': 4,
        'optimizer_step': 4,
        'last_checkpoint': str(checkpoint),
    }
    assert [{k: record[k] for k in progress} for record in records] == [
        {
            'status': 'running',
            'completed_updates': 0,
            'optimizer_step': 0,
            'last_checkpoint': None,
        },
        {'status': 'running', **saved},
        {'status': 'complete', **saved},
    ]
    started_at = datetime.fromisoformat(records[0]['started_at'])
    assert started_at.utcoffset() == timezone.utc.utcoffset(None)
    run_json = json.loads((checkpoint.parent / 'manifest' / 'run.json').read_text())
    assert run_json == records[-1]
    assert checkpoint.is_file()


@pytest.mark.parametrize(
    ('error', 'status', 'resume'),
    [
        pytest.param(OSError, 'failed', False, id='filesystem-failure-fresh-run'),
        pytest.param(KeyboardInterrupt, 'interrupted', True, id='interrupt-resume'),
    ],
)
def test_manifest_after_failure(
    monkeypatch, tiny_settings, tmp_path, manifest_fields, jsonl, error, status, resume
):
    from tmt.model import Model
    from tmt.runtime import train

    train_step = Model.train_step
    data = tmp_path / 'train.bin'
    data.write_bytes(b'abcdef')
    checkpoint = tmp_path / 'run' / 'model.safetensors'
    if resume:
        train(tiny_settings, checkpoint, str(data), 2, 11)
        original = checkpoint.read_bytes()
    completed = 0

    def fail(model, current, target, end, ce_only=False, **kwargs):
        nonlocal completed
        if completed == 1:
            raise error('injected update failure')
        result = train_step(model, current, target, end, ce_only, **kwargs)
        completed += 1
        return result

    monkeypatch.setattr(Model, 'train_step', fail)
    settings = None if resume else tiny_settings
    handler = signal.getsignal(signal.SIGINT)
    with pytest.raises(error):
        train(settings, checkpoint, str(data), 3, 17, ce_only=True, resume=resume)
    assert signal.getsignal(signal.SIGINT) == handler

    record = json.loads((checkpoint.parent / 'manifest/run.json').read_text())
    expected = {
        'data_glob': str(data),
        'data_files': [str(data.resolve())],
        'updates': 3,
        'completed_updates': 1,
        'status': status,
        'seed': 17,
        'objective': 'ce_only',
        'resume': resume,
        'initial_optimizer_step': 2 if resume else 0,
        'last_checkpoint': str(checkpoint.resolve()) if resume else None,
        'error': None
        if resume
        else {'type': 'OSError', 'message': 'injected update failure'},
    }
    assert set(record) == manifest_fields
    assert {k: record[k] for k in expected} == expected
    *earlier, loss_row = jsonl(checkpoint.parent / 'manifest/loss.jsonl')
    assert [r['optimizer_step'] for r in earlier] == ([2] if resume else [])
    expected_row = {
        'completed_updates': 1,
        'objective': 'ce_only',
        'interval_updates': 1,
        'optimizer_step': 3 if resume else 1,
    }
    assert {k: loss_row[k] for k in expected_row} == expected_row
    assert math.isfinite(loss_row['loss'])
    assert loss_row['loss'] == loss_row['mean_loss']
    if resume:
        assert checkpoint.read_bytes() == original
    else:
        assert not checkpoint.exists()


def test_short_documents_fail_the_run(tiny_settings, tmp_path):
    from tmt.runtime import train

    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'empty.bin').write_bytes(b'')
    (tmp_path / 'data' / 'one.bin').write_bytes(b'x')
    checkpoint = tmp_path / 'run' / 'model.safetensors'
    with pytest.raises(ValueError, match='no training targets found in data pass'):
        train(tiny_settings, checkpoint, str(tmp_path / 'data/*.bin'), 1, 11)

    record = json.loads((checkpoint.parent / 'manifest/run.json').read_text())
    expected = {
        'status': 'failed',
        'last_checkpoint': None,
        'updates': 1,
        'completed_updates': 0,
        'initial_optimizer_step': 0,
        'error': {
            'type': 'ValueError',
            'message': 'no training targets found in data pass',
        },
    }
    assert {k: record[k] for k in expected} == expected
    assert not checkpoint.exists()


def test_post_update_action_order(monkeypatch, tmp_path):
    from tmt.runtime import MonitorOptions, TrainingRun

    run = TrainingRun(
        settings={},
        path=tmp_path / 'model.safetensors',
        pattern='data/*',
        updates=500,
        seed=11,
        ce_only=False,
        resume=False,
        log_every=500,
        monitor=MonitorOptions(
            sample_every=500,
            eval_data='held-out/*',
            eval_every=500,
        ),
    )
    run.completed = 500
    events = []
    monkeypatch.setattr(run, 'report_loss', lambda: events.append('loss'))
    monkeypatch.setattr(
        run,
        'consume_terminal_byte',
        lambda byte: events.append(('terminal-byte', byte)),
    )
    monkeypatch.setattr(run, 'report_sample', lambda: events.append('sample'))
    monkeypatch.setattr(
        run, 'report_evaluation', lambda: events.append('evaluation')
    )
    monkeypatch.setattr(
        run, 'save_checkpoint', lambda: events.append('checkpoint')
    )

    run.after_update(True, 255)

    assert events == [
        'loss',
        ('terminal-byte', 255),
        'sample',
        'evaluation',
        'checkpoint',
    ]


@pytest.mark.parametrize(
    ('updates', 'interval', 'ce_only', 'ends'),
    [
        pytest.param(5, 2, True, [2, 4, 5], id='final-partial-interval'),
        pytest.param(4, 2, True, [2, 4], id='exact-interval'),
        pytest.param(3, 100, True, [3], id='interval-larger-than-run'),
        pytest.param(5, 2, False, [2, 4, 5], id='tmt-objective'),
    ],
)
def test_loss_rows(
    capsys,
    monkeypatch,
    tiny_settings,
    tmp_path,
    snapshot,
    jsonl,
    updates,
    interval,
    ce_only,
    ends,
):
    from tmt.model import Model
    from tmt.runtime import train

    data = tmp_path / 'train.bin'
    data.write_bytes(b'abcdef')
    objective = 'ce_only' if ce_only else 'tmt'
    losses, components = [], []
    original = Model.train_step

    def capture(model, *args, metrics=None, **kwargs):
        result = original(model, *args, metrics=metrics, **kwargs)
        losses.append(float(result[2].item()))
        components.append(dict(metrics))
        return result

    with monkeypatch.context() as patches:
        ticks = iter([0, 2, 4, 6])
        patches.setattr(Model, 'train_step', capture)
        patches.setattr('tmt.runtime.time.perf_counter', lambda: next(ticks))
        options = {'ce_only': ce_only, 'log_every': interval}
        checkpoint = tmp_path / 'model.safetensors'
        model = train(tiny_settings, checkpoint, str(data), updates, seed=11, **options)
    stdout = capsys.readouterr().out
    rows = jsonl(tmp_path / 'manifest/loss.jsonl')

    keys = ('completed_updates', 'objective', 'loss')
    assert [{k: row[k] for k in keys} for row in rows] == [
        {'completed_updates': end, 'objective': objective, 'loss': losses[end - 1]}
        for end in ends
    ], 'loss report rows differ from captured updates'
    previous = 0
    for number, row in enumerate(rows, 1):
        end = row['completed_updates']
        window, count = slice(previous, end), end - previous
        mean = math.fsum(losses[window]) / count
        assert row['mean_loss'] == pytest.approx(mean, rel=0, abs=5e-8)
        names = {'variance', 'latent_prediction', 'cross_entropy', 'stop'}
        assert set(row['mean_loss_components']) == names
        for name, value in row['mean_loss_components'].items():
            if not ce_only:
                mean_component = math.fsum(c[name] for c in components[window]) / count
                assert value == pytest.approx(mean_component, rel=0, abs=5e-8)
            elif name == 'cross_entropy':
                assert value == pytest.approx(mean, rel=0, abs=5e-8)
            else:
                assert value == 0.0, f'CE-only loss component {name!r} should be zero'
        total = sum(row['mean_loss_components'].values())
        assert total == pytest.approx(row['mean_loss'], rel=0, abs=5e-6)
        expected = {
            'interval_updates': count,
            'elapsed_seconds': number * 2,
            'updates_per_second': count / 2,
            'optimizer_step': end,
        }
        assert {k: row[k] for k in expected} == expected
        previous = end
    assert stdout.count('objective:') == len(ends)
    last = f'updates: {updates}, objective: {objective}, loss: {losses[-1]:.6f}'
    assert last in stdout
    run_json = json.loads((tmp_path / 'manifest/run.json').read_text())
    assert run_json['log_every'] == interval
    reference = tmp_path / 'reference/model.safetensors'
    assert snapshot(model) == snapshot(
        train(tiny_settings, reference, str(data), updates, seed=11, ce_only=ce_only)
    )


def test_train_command_run_folders(monkeypatch, run_cli, tiny_settings, tmp_path):
    (tmp_path / 'train.bin').write_bytes(b'ab')
    (tmp_path / 'model.json').write_text(json.dumps(tiny_settings))
    command_calls = []
    monkeypatch.setattr('tmt.cli.train', lambda *a, **k: command_calls.append((a, k)))

    run_cli(
        'train', '--data', 'train.bin', '--run', 'named', '--updates', '3',
        '--seed', '11',
    )
    named = Path('runs/named/model.safetensors')
    assert command_calls[-1][0][:2] == (tiny_settings, named)
    # The patched function replaces run-folder creation.
    (tmp_path / 'runs' / 'named').mkdir(parents=True)
    run_cli('train', '--data', 'train.bin', '--seed', '11')
    assert command_calls[-1][0][1].parent.parent == Path('runs')
    assert run_cli(
        'train', '--data', 'train.bin', '--run', 'named', '--seed', '11'
    )[0] == 1


def test_sampling(capsys, tiny_settings, tmp_path, snapshot, jsonl):
    from tmt.runtime import train

    data = tmp_path / 'train.bin'
    data.write_bytes(b'abcdefghij')
    checkpoint = tmp_path / 'sampled' / 'model.safetensors'
    plain_checkpoint = tmp_path / 'plain' / 'model.safetensors'
    base = {'seed': 29, 'ce_only': True, 'sample_prompt': 'The ', 'sample_seed': 13}
    first = {'log_every': 4, 'sample_every': 2, 'sample_bytes': 3}
    again = {'log_every': 2, 'sample_every': 1, 'sample_bytes': 2, 'resume': True}

    sampled = train(tiny_settings, checkpoint, str(data), 4, **base, **first)
    plain = train(tiny_settings, plain_checkpoint, str(data), 4, **base, log_every=4)
    assert sampled.compiled is not None
    assert plain.compiled is not None
    assert snapshot(sampled) == snapshot(plain)

    samples = checkpoint.parent / 'manifest' / 'samples.jsonl'
    rows = jsonl(samples)
    keys = ('completed_updates', 'optimizer_step', 'prompt', 'seed', 'bytes')
    assert [{k: row[k] for k in keys} for row in rows] == [
        {
            'completed_updates': updates,
            'optimizer_step': updates,
            'prompt': 'The ',
            'seed': 13,
            'bytes': 3,
        }
        for updates in (2, 4)
    ]
    for row in rows:
        assert len(Path(row['path']).read_bytes()) == row['bytes']

    capsys.readouterr()
    resumed = train(None, checkpoint, str(data), 2, **base, **again)
    assert int(resumed.optimizer.state['step'].item()) == 6
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2:] == [
        f'saved checkpoint: {checkpoint}, updates: 2, step: 6',
        f'updates: 2, step: 6, checkpoint: {checkpoint}',
    ]
    keys = ('completed_updates', 'optimizer_step')
    assert [{k: row[k] for k in keys} for row in jsonl(samples)] == [
        {'completed_updates': 2, 'optimizer_step': 2},
        {'completed_updates': 4, 'optimizer_step': 4},
        {'completed_updates': 1, 'optimizer_step': 5},
        {'completed_updates': 2, 'optimizer_step': 6},
    ]


def test_interval_evaluation(tiny_settings, tmp_path, snapshot, jsonl):
    from tmt.benchmark import evaluate
    from tmt.data import documents
    from tmt.runtime import train

    training = tmp_path / 'training.bin'
    training.write_bytes(b'abcdefghij')
    development = tmp_path / 'development'
    development.mkdir()
    first, second = development / 'a.bin', development / 'b.bin'
    first.write_bytes(b'abcd')
    second.write_bytes(b'uvwxyz')
    pattern = str(development / '*.bin')
    monitored_checkpoint = tmp_path / 'monitored' / 'model.safetensors'
    plain_checkpoint = tmp_path / 'plain' / 'model.safetensors'
    exact_checkpoint = tmp_path / 'exact' / 'model.safetensors'
    base = {'seed': 43, 'ce_only': True}
    monitoring = {'eval_data': pattern, 'eval_every': 2, 'eval_max_bytes': 7}
    path = str(training)

    monitored = train(
        tiny_settings, monitored_checkpoint, path, 5, log_every=5, **base, **monitoring
    )
    plain = train(tiny_settings, plain_checkpoint, path, 5, log_every=5, **base)
    train(tiny_settings, exact_checkpoint, path, 4, log_every=4, **base, **monitoring)

    assert monitored.compiled is not None
    assert plain.compiled is not None
    assert snapshot(plain) == snapshot(monitored)
    folder = monitored_checkpoint.parent / 'manifest'
    rows = jsonl(folder / 'evaluation.jsonl')
    keys = ('completed_updates', 'optimizer_step', 'input_bytes', 'targets', 'windows')
    assert [{k: row[k] for k in keys} for row in rows] == [
        {
            'completed_updates': updates,
            'optimizer_step': updates,
            'input_bytes': 7,
            'targets': 5,
            'windows': [],
        }
        for updates in (2, 4, 5)
    ]
    assert json.loads((folder / 'run.json').read_text())['evaluation'] == {
        'every': 2,
        'data_glob': pattern,
        'max_bytes': 7,
        'data_files': [str(first.resolve()), str(second.resolve())],
    }
    development_documents = list(documents(pattern, 7))
    assert development_documents == [b'abcd', b'uvw']
    final_score = evaluate(monitored, development_documents)
    assert rows[-1]['bpb'] == pytest.approx(final_score['bpb'], rel=0, abs=5e-13)
    exact_rows = jsonl(exact_checkpoint.parent / 'manifest' / 'evaluation.jsonl')
    assert [row['completed_updates'] for row in exact_rows] == [2, 4]


@pytest.mark.parametrize(
    ('grid', 'message', 'existing'),
    [
        pytest.param(grid, message, name == 'existing-empty-output', id=name)
        for message, grids in INVALID_GRIDS.items()
        for name, grid in grids.items()
    ],
)
def test_sweep_rejects_invalid_grid(
    monkeypatch, forbid_call, tiny_settings, tmp_path, grid, message, existing
):
    from tmt.runtime import sweep

    monkeypatch.setattr('tmt.runtime.train', forbid_call)
    (tmp_path / 'model.json').write_text(json.dumps(tiny_settings))
    (tmp_path / 'grid.json').write_text(json.dumps(grid))
    output = tmp_path / 'sweep'
    if existing:
        output.mkdir()
    with pytest.raises(ValueError, match=message):
        args = ('missing/*', 'missing/*', output, tmp_path / 'model.json')
        sweep(tmp_path / 'grid.json', *args, updates=1, seed=11)
    if existing:
        assert list(output.iterdir()) == []
    else:
        assert not output.exists()


@pytest.mark.parametrize(
    ('error_type', 'status'),
    [
        pytest.param(OSError, 'failed', id='filesystem-failure'),
        pytest.param(KeyboardInterrupt, 'interrupted', id='keyboard-interrupt'),
    ],
)
def test_sweep_failure_records_status(
    monkeypatch, tiny_settings, error_type, status, tmp_path
):
    from tmt.runtime import sweep

    grid, config, data = tmp_path / 'grid.json', tmp_path / 'model.json', tmp_path / 'd'
    config.write_text(json.dumps(tiny_settings))
    grid.write_text('{"seed": [11, 22]}')
    data.write_bytes(b'ab')
    output = tmp_path / 'sweep'
    completed_model = SimpleNamespace(count=lambda: 7, temporal_gradient=['tmt-v0'])
    outcomes = iter([completed_model, error_type('injected failure')])

    def next_train(*args, **kwargs):
        result = next(outcomes)
        if isinstance(result, BaseException):
            raise result
        return result

    with monkeypatch.context() as patches, pytest.raises(error_type):
        patches.setattr('tmt.runtime.train', next_train)
        patches.setattr(
            'tmt.benchmark.evaluate', lambda *args: {'targets': 1, 'bpb': 8.0}
        )
        sweep(grid, str(data), str(data), output, config, updates=1, seed=11)

    record = json.loads((output / 'sweep.json').read_text())
    expected = {'status': status, 'current_run': 2, 'completed_runs': 1}
    assert {k: record[k] for k in expected} == expected
    error = {'type': error_type.__name__, 'message': 'injected failure'}
    assert record['error'] == error
    rows = (output / 'results.jsonl').read_bytes()
    first_row = json.loads(rows.splitlines()[0])
    assert first_row['trainable_parameters'] == 7
    assert first_row['temporal_gradient'] == ['tmt-v0']
    with pytest.raises(FileExistsError, match='choose a new output folder'):
        sweep(grid, str(data), str(data), output, config, updates=1, seed=11)
    assert (output / 'results.jsonl').read_bytes() == rows
    sweep(grid, str(data), str(data), tmp_path / 'retry', config, updates=1, seed=11)
    retry = json.loads((tmp_path / 'retry/sweep.json').read_text())
    assert retry['status'] == 'complete'


def test_sweep_argument_dispatch(monkeypatch, run_cli):
    command_calls = []
    monkeypatch.setattr(
        'tmt.cli.sweep', lambda *args, **kwargs: command_calls.append((args, kwargs))
    )
    args = ['sweep', '--grid', 'grid.json', '--data', 'train/*', '--seed', '11']
    args += ['--development', 'dev/*', '--output', 'runs/grid']
    monitoring = ['--sample-every', '2', '--sample-bytes', '3']
    monitoring += ['--eval-data', 'held-out/*', '--eval-every', '5']
    monitoring += ['--save-every', '7']

    for extra, config in (
        ([], 'model.json'),
        (['--config', 'custom.json'], 'custom.json'),
        (monitoring, 'model.json'),
    ):
        assert run_cli(*args, *extra)[0] == 0
        assert command_calls[-1][0][4] == config, 'config path was not dispatched'
    expected = {
        'sample_every': 2,
        'sample_bytes': 3,
        'eval_data': 'held-out/*',
        'eval_every': 5,
        'save_every': 7,
    }
    assert {k: command_calls[-1][1][k] for k in expected} == expected


def test_sweep_runs_grid(
    monkeypatch, capsys, tiny_settings, tmp_path, manifest_fields, jsonl
):
    from tmt.blocks import BLOCKS
    from tmt.checkpoint import load_checkpoint
    from tmt.runtime import sweep

    grid, config = tmp_path / 'grid.json', tmp_path / 'model.json'
    grid.write_text(json.dumps({'block': ['tmt-v0', 'tmt-v0-test'], 'seed': [11, 22]}))
    config.write_text(json.dumps(tiny_settings))
    training, development = tmp_path / 'train.bin', tmp_path / 'development.bin'
    training.write_bytes(b'\x00\x01\x02')
    development.write_bytes(b'\xf0\xf1\xf2\xf3')
    output = tmp_path / 'runs'
    options = {
        'max_bytes': 4,
        'log_every': 1,
        'sample_every': 2,
        'sample_bytes': 3,
        'eval_data': str(development),
        'eval_every': 2,
        'eval_max_bytes': 3,
    }
    monkeypatch.setitem(BLOCKS, 'tmt-v0-test', BLOCKS['tmt-v0'])
    best = sweep(
        grid, str(training), str(development), output, config, 2, 11, **options
    )

    rows = jsonl(output / 'results.jsonl')
    assert [(row['settings']['block'], row['seed']) for row in rows] == [
        ('tmt-v0', 11),
        ('tmt-v0', 22),
        ('tmt-v0-test', 11),
        ('tmt-v0-test', 22),
    ], 'sweep changed product order or per-axis seeds'
    assert best == min(rows, key=lambda row: row['development_bpb'])
    assert json.dumps(best) in capsys.readouterr().out, 'best row was not reported'
    for index, row in enumerate(rows, 1):
        checkpoint = Path(row['checkpoint'])
        folder = checkpoint.parent / 'manifest'
        assert checkpoint.parent.name == f'run-{index:04d}'
        assert row['objective'] == 'tmt'
        assert row['development_targets'] == 3
        assert row['temporal_gradient'] == ['tmt-v0']
        assert row['trainable_parameters'] > 0
        assert json.loads((folder / 'model.json').read_text()) == row['settings']
        manifest = json.loads((folder / 'run.json').read_text())
        assert set(manifest) == manifest_fields, 'run manifest fields differ'
        expected = {
            'seed': row['seed'],
            'updates': row['updates'],
            'objective': row['objective'],
            'data_glob': row['data_glob'],
            'status': 'complete',
            'last_checkpoint': str(checkpoint.resolve()),
            'config': row['settings'],
            'log_every': 1,
            'resume': False,
            'trainable_parameters': row['trainable_parameters'],
            'temporal_gradient': row['temporal_gradient'],
            'data_files': [str(training.resolve())],
            'initial_optimizer_step': 0,
            'completed_updates': row['updates'],
        }
        assert {k: manifest[k] for k in expected} == expected
        assert [r['completed_updates'] for r in jsonl(folder / 'loss.jsonl')] == [1, 2]
        evaluation = jsonl(folder / 'evaluation.jsonl')
        assert [(r['completed_updates'], r['targets']) for r in evaluation] == [(2, 2)]
        samples = jsonl(folder / 'samples.jsonl')
        assert [(r['completed_updates'], r['bytes']) for r in samples] == [(2, 3)]
        assert len(Path(samples[0]['path']).read_bytes()) == 3
        model, settings = load_checkpoint(checkpoint)
        assert settings == row['settings'], 'loaded checkpoint settings differ'
        assert model.count() == row['trainable_parameters']
        assert model.temporal_gradient == row['temporal_gradient']
        assert int(model.optimizer.state['step'].item()) == 2
