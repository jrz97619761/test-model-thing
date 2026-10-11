import contextlib
import json
import math
from pathlib import Path

import pytest


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
        first = train(tiny_settings, root / 'one' / 'model.safetensors', str(data), 3, 11)
        second = train(tiny_settings, root / 'two' / 'model.safetensors', str(data), 3, 11)
        resumed = train(None, root / 'one' / 'model.safetensors', str(data), 2, 11, resume=True)

    assert stdout.getvalue().count('run folder:') == 3
    assert stdout.getvalue().count('saved checkpoint:') == 3
    assert stdout.getvalue().count('objective:') == 3
    assert 'seed: 11, config:' in stdout.getvalue()
    assert '\r' not in stdout.getvalue()

    expected = [(0, 255, False), (255, 65, False), (65, 10, True)]
    assert calls == expected + expected + expected[:2]
    assert frozen_steps == [10, 10]
    assert [int(model.optimizer.state['step'].item()) for model in (first, second, resumed)] == [3, 3, 5]
    assert mx.array_equal(first.decoder.decode.weight, second.decoder.decode.weight).item()


@pytest.mark.parametrize('ce_only', (False, True))
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
    assert (records[0]['trainable_parameters'], records[0]['temporal_gradient']) == (
        trained.count(), trained.temporal_gradient
    )
    for record in records:
        assert (record['config'], record['seed'], record['data_glob']) == (tiny_settings, 11, 'data/*.bin')
        assert set(record['data_files']) == set(paths.values())
        assert record['data_files'] == [path for path, _ in calls[:3]]
        assert (record['updates'], record['objective'], record['resume'], record['initial_optimizer_step']) == (4, 'ce_only' if ce_only else 'tmt', False, 0)
    assert calls == [(path, ce_only) for path in records[0]['data_files'] + records[0]['data_files'][:1]]
    assert [(record['status'], record['completed_updates'], record['last_checkpoint']) for record in records] == [('running', 0, None), ('complete', 4, str(checkpoint.resolve()))]
    assert datetime.fromisoformat(records[0]['started_at']).utcoffset() == timezone.utc.utcoffset(None)
    assert json.loads((checkpoint.parent / 'manifest' / 'run.json').read_text()) == records[1]
    assert checkpoint.is_file()


@pytest.mark.parametrize('error_type,status', ((OSError, 'failed'), (KeyboardInterrupt, 'interrupted')))
@pytest.mark.parametrize('resume', (False, True))
def test_manifest_partial_progress(monkeypatch, tiny_settings, error_type, status, resume, tmp_path, manifest_fields):
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
        train(None if resume else tiny_settings, checkpoint, str(data), 3, 17, True, resume)

    record = json.loads((checkpoint.parent / 'manifest/run.json').read_text())
    assert set(record) == manifest_fields
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


@pytest.mark.parametrize('updates,interval,expected', ((5, 2, [2, 4, 5]), (4, 2, [2, 4]), (3, 100, [3])))
def test_train_loss_intervals(assert_almost_equal, monkeypatch, tiny_settings, updates, interval, expected, tmp_path):
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
        model = train(tiny_settings, checkpoint, str(data), updates,
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
        reference = train(tiny_settings, root / 'reference/model.safetensors',
                          str(data), updates, ce_only=True)
    assert mx.array_equal(model.decoder.decode.weight, reference.decoder.decode.weight).item()


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
        assert command_calls[-1][0][:2] == (tiny_settings, Path('runs/named/model.safetensors'))
        Path('runs/named').mkdir(parents=True)  # The patched function replaces run-folder creation.
        main(['train', '--data', str(data)])
        assert command_calls[-1][0][1].parent.parent == Path('runs')
        with pytest.raises(SystemExit) as exit: main(['train', '--data', str(data), '--run', 'named'])
        assert exit.value.code == 1


@pytest.mark.parametrize('args', [
    ['--run', 'bad', 'model.safetensors'], ['--resume'], ['--resume', '--run', 'bad'],
    ['model.safetensors', '--resume', '--config', 'model.json'],
])
def test_train_argument_errors(cli_workspace, monkeypatch, args):
    from tmt.cli import main

    root, _ = cli_workspace
    monkeypatch.chdir(root)
    with pytest.raises(SystemExit) as exit:
        main(['train', '--data', 'train.bin', *args])
    assert exit.value.code == 2
