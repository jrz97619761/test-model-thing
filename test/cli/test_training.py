import contextlib
import json
import math
from pathlib import Path

import pytest

from cli_support import assert_fields, assert_ordered_records, read_jsonl


def test_train(monkeypatch, tiny_settings, tmp_path):
    import mlx.core as mx
    from tmt.cli import train
    from tmt.main import Model

    root = tmp_path
    data = root / 'train.bin'
    data.write_bytes(bytes((0, 255, 65, 10)))
    calls = []
    train_step, step = Model.train_step, Model.step
    frozen_steps = []

    def record_train(model, current, target, end, ce_only=False, **kwargs):
        calls.append((current, target, end))
        return train_step(model, current, target, end, ce_only, **kwargs)

    def record_step(model, current, *args, frozen=False, **kwargs):
        if frozen:
            frozen_steps.append(int(current.item()))
        return step(model, current, *args, frozen=frozen, **kwargs)

    def forbidden_sample(*args, **kwargs):
        pytest.fail('training must not sample')

    from io import StringIO

    stdout = StringIO()
    monkeypatch.setattr(Model, 'sample', forbidden_sample)
    monkeypatch.setattr(Model, 'train_step', record_train)
    monkeypatch.setattr(Model, 'step', record_step)
    with contextlib.redirect_stdout(stdout):
        first = train(
            tiny_settings, root / 'one' / 'model.safetensors', str(data), 3, 11
        )
        second = train(
            tiny_settings, root / 'two' / 'model.safetensors', str(data), 3, 11
        )
        resumed = train(
            None, root / 'one' / 'model.safetensors', str(data), 2, 11, resume=True
        )

    for message in ('run folder:', 'saved checkpoint:', 'objective:'):
        assert stdout.getvalue().count(message) == 3, (
            f'expected three occurrences of {message!r}'
        )
    assert 'seed: 11, config:' in stdout.getvalue()
    assert '\r' not in stdout.getvalue()

    expected = [(0, 255, False), (255, 65, False), (65, 10, True)]
    assert calls == expected + expected + expected[:2]
    assert frozen_steps == [10, 10]
    optimizer_steps = [
        int(model.optimizer.state['step'].item()) for model in (first, second, resumed)
    ]
    assert optimizer_steps == [3, 3, 5], 'training and resume optimizer steps differ'
    assert mx.array_equal(
        first.decoder.decode.weight, second.decoder.decode.weight
    ).item()


@pytest.mark.parametrize(
    'ce_only', (False, True), ids=['tmt-objective', 'cross-entropy-only']
)
def test_manifest(monkeypatch, tiny_settings, ce_only, tmp_path, manifest_fields):
    from datetime import datetime, timezone
    from tmt.cli import train
    from tmt.main import Model

    original_dump, records = json.dump, []

    def capture(record, *args, **kwargs):
        records.append(dict(record))
        return original_dump(record, *args, **kwargs)

    train_step = Model.train_step
    monkeypatch.chdir(tmp_path)
    root = tmp_path.resolve()
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
    with monkeypatch.context() as patches:
        patches.setattr('tmt.cli.json.dump', capture)
        patches.setattr(Model, 'train_step', record_train)
        trained = train(tiny_settings, checkpoint, 'data/*.bin', 4, 11, ce_only)

    assert len(records) == 2
    assert set(records[0]) == manifest_fields
    assert records[0]['trainable_parameters'] == trained.count(), (
        'initial manifest has the wrong trainable parameter count'
    )
    assert records[0]['temporal_gradient'] == trained.temporal_gradient, (
        'initial manifest has the wrong RTU gradient marker'
    )
    for record in records:
        assert_fields(
            record,
            {
                'config': tiny_settings,
                'seed': 11,
                'data_glob': 'data/*.bin',
            },
            context='manifest record',
        )
        assert set(record['data_files']) == set(paths.values())
        assert record['data_files'] == [path for path, _ in calls[:3]]
        assert_fields(
            record,
            {
                'updates': 4,
                'objective': 'ce_only' if ce_only else 'tmt',
                'resume': False,
                'initial_optimizer_step': 0,
            },
            context='manifest record',
        )
    assert calls == [
        (path, ce_only)
        for path in records[0]['data_files'] + records[0]['data_files'][:1]
    ]
    expected_progress = [
        {'status': 'running', 'completed_updates': 0, 'last_checkpoint': None},
        {
            'status': 'complete',
            'completed_updates': 4,
            'last_checkpoint': str(checkpoint.resolve()),
        },
    ]
    assert_ordered_records(
        records, expected_progress, context='training manifest progress'
    )
    assert datetime.fromisoformat(
        records[0]['started_at']
    ).utcoffset() == timezone.utc.utcoffset(None)
    assert (
        json.loads((checkpoint.parent / 'manifest' / 'run.json').read_text())
        == records[1]
    )
    assert checkpoint.is_file()


@pytest.mark.parametrize(
    ('error_type', 'status'),
    ((OSError, 'failed'), (KeyboardInterrupt, 'interrupted')),
    ids=['filesystem-failure', 'keyboard-interrupt'],
)
@pytest.mark.parametrize('resume', (False, True), ids=['fresh-run', 'resume'])
def test_manifest_partial_progress(
    monkeypatch, tiny_settings, error_type, status, resume, tmp_path, manifest_fields
):
    from tmt.cli import train
    from tmt.main import Model

    train_step = Model.train_step
    root = tmp_path
    data = root / 'train.bin'
    data.write_bytes(b'abcdef')
    checkpoint = root / 'run' / 'model.safetensors'
    original = None
    if resume:
        train(tiny_settings, checkpoint, str(data), 2, 11)
        original = checkpoint.read_bytes()
    completed = 0

    def fail(model, current, target, end, ce_only=False, **kwargs):
        nonlocal completed
        if completed == 1:
            raise error_type('injected update failure')
        result = train_step(model, current, target, end, ce_only, **kwargs)
        completed += 1
        return result

    monkeypatch.setattr(Model, 'train_step', fail)
    with pytest.raises(error_type):
        train(
            None if resume else tiny_settings,
            checkpoint,
            str(data),
            3,
            17,
            True,
            resume,
        )

    record = json.loads((checkpoint.parent / 'manifest/run.json').read_text())
    assert set(record) == manifest_fields
    assert_fields(
        record,
        {
            'data_glob': str(data),
            'data_files': [str(data.resolve())],
            'updates': 3,
            'completed_updates': 1,
            'status': status,
            'seed': 17,
            'objective': 'ce_only',
            'resume': resume,
            'initial_optimizer_step': 2 if resume else 0,
        },
        context='manifest partial progress record',
    )
    assert record['last_checkpoint'] == (str(checkpoint.resolve()) if resume else None)
    metrics = read_jsonl(checkpoint.parent / 'manifest/loss.jsonl')
    assert len(metrics) == 1, (
        'partial-progress manifest has the wrong number of loss rows'
    )
    assert metrics[0]['completed_updates'] == 1, 'loss row has the wrong update count'
    assert metrics[0]['objective'] == 'ce_only', 'loss row has the wrong objective'
    assert math.isfinite(metrics[0]['loss'])
    assert metrics[0]['interval_updates'] == 1
    assert metrics[0]['optimizer_step'] == (3 if resume else 1)
    assert metrics[0]['loss'] == metrics[0]['mean_loss']
    if resume:
        assert checkpoint.read_bytes() == original
    else:
        assert not checkpoint.exists()


@pytest.mark.parametrize(
    ('updates', 'interval', 'expected'),
    ((5, 2, [2, 4, 5]), (4, 2, [2, 4]), (3, 100, [3])),
    ids=['final-partial-interval', 'exact-interval', 'interval-larger-than-run'],
)
def test_train_loss_intervals(
    assert_almost_equal,
    monkeypatch,
    tiny_settings,
    updates,
    interval,
    expected,
    tmp_path,
):
    from io import StringIO
    import mlx.core as mx
    from tmt.cli import train
    from tmt.main import Model

    root = tmp_path
    data = root / 'train.bin'
    data.write_bytes(b'abcdef')
    checkpoint = root / 'model.safetensors'
    losses, stdout = [], StringIO()
    original = Model.train_step

    def capture(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        losses.append(float(result[2].item()))
        return result

    with monkeypatch.context() as patches, contextlib.redirect_stdout(stdout):
        ticks = iter([0, 2, 4, 6])
        patches.setattr(Model, 'train_step', capture)
        patches.setattr('tmt.cli.time.perf_counter', lambda: next(ticks))
        model = train(
            tiny_settings,
            checkpoint,
            str(data),
            updates,
            ce_only=True,
            log_every=interval,
        )
    metrics = read_jsonl(root / 'manifest/loss.jsonl')
    actual_reports = [
        {
            'completed_updates': row['completed_updates'],
            'objective': row['objective'],
            'loss': row['loss'],
        }
        for row in metrics
    ]
    expected_reports = [
        {
            'completed_updates': index,
            'objective': 'ce_only',
            'loss': losses[index - 1],
        }
        for index in expected
    ]
    assert actual_reports == expected_reports, (
        'loss report rows differ from captured updates'
    )
    previous = 0
    for number, row in enumerate(metrics, 1):
        count = row['completed_updates'] - previous
        mean = math.fsum(losses[previous : row['completed_updates']]) / count
        assert_almost_equal(row['mean_loss'], mean)
        assert_almost_equal(row['mean_loss_components']['cross_entropy'], mean)
        for name in ('variance', 'latent_prediction', 'stop'):
            assert row['mean_loss_components'][name] == 0.0, (
                f'CE-only loss component {name!r} should be zero'
            )
        assert_fields(
            row,
            {
                'interval_updates': count,
                'elapsed_seconds': number * 2,
                'updates_per_second': count / 2,
                'optimizer_step': row['completed_updates'],
            },
            context='train loss intervals row',
        )
        previous = row['completed_updates']
    assert stdout.getvalue().count('objective:') == len(expected)
    assert (
        f'updates: {updates}, objective: ce_only, loss: {losses[-1]:.6f}'
        in stdout.getvalue()
    )
    assert json.loads((root / 'manifest/run.json').read_text())['log_every'] == interval
    with contextlib.redirect_stdout(StringIO()):
        reference = train(
            tiny_settings,
            root / 'reference/model.safetensors',
            str(data),
            updates,
            ce_only=True,
        )
    assert mx.array_equal(
        model.decoder.decode.weight, reference.decoder.decode.weight
    ).item()


def test_train_command(monkeypatch, tiny_settings, tmp_path):
    from tmt.cli import main

    root = tmp_path
    data = root / 'train.bin'
    data.write_bytes(b'ab')
    command_calls = []

    def record_command_train(*args, **kwargs):
        command_calls.append((args, kwargs))

    monkeypatch.setattr('tmt.cli.train', record_command_train)
    with contextlib.chdir(root):
        Path('model.json').write_text(json.dumps(tiny_settings))
        main(['train', '--data', str(data), '--run', 'named', '--updates', '3'])
        assert command_calls[-1][0][:2] == (
            tiny_settings,
            Path('runs/named/model.safetensors'),
        )
        Path('runs/named').mkdir(
            parents=True
        )  # The patched function replaces run-folder creation.
        main(['train', '--data', str(data)])
        assert command_calls[-1][0][1].parent.parent == Path('runs')
        with pytest.raises(SystemExit) as exit:
            main(['train', '--data', str(data), '--run', 'named'])
        assert exit.value.code == 1


@pytest.mark.parametrize(
    'args',
    [
        ['--run', 'bad', 'model.safetensors'],
        ['--resume'],
        ['--resume', '--run', 'bad'],
        ['model.safetensors', '--resume', '--config', 'model.json'],
    ],
    ids=[
        'checkpoint-with-run',
        'resume-without-checkpoint',
        'resume-with-run',
        'resume-with-config',
    ],
)
def test_train_argument_errors(cli_workspace, monkeypatch, args):
    from tmt.cli import main

    root, _ = cli_workspace
    monkeypatch.chdir(root)
    with pytest.raises(SystemExit) as exit:
        main(['train', '--data', 'train.bin', *args])
    assert exit.value.code == 2
