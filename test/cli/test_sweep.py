import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


INVALID_GRIDS = [
    (grid, 'sweep grid must be a JSON object') for grid in (None, 4, [], 'dim')
] + [
    ({'dim': axis}, "sweep grid field 'dim' must be a nonempty array")
    for axis in (4, None, '4', {}, [], True)
] + [
    ({'dim': [4], 'seed': []}, "sweep grid field 'seed' must be a nonempty array"),
    ({'dim': [4, 0]}, 'model config.dim must be a positive integer'),
    ({'seed': [11, True]}, 'sweep grid seed must be an integer'),
    ({'seed': [11, -1]}, 'sweep grid seed must be between'),
    ({'seed': [11, 2**64]}, 'sweep grid seed must be between'),
]


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('grid,message', INVALID_GRIDS)
def test_invalid_grid(tmp_path, monkeypatch, forbid_call, tiny_settings, existing, grid, message):
    from tmt.cli import sweep

    monkeypatch.setattr('tmt.cli.train', forbid_call)
    config, grid_path, output = tmp_path / 'model.json', tmp_path / 'grid.json', tmp_path / 'sweep'
    config.write_text(json.dumps(tiny_settings))
    grid_path.write_text(json.dumps(grid))
    if existing:
        output.mkdir()
    with pytest.raises(ValueError, match=message):
        sweep(grid_path, 'missing/*', 'missing/*', output, config, updates=1)
    if existing:
        assert list(output.iterdir()) == []
    else:
        assert not output.exists()


@pytest.mark.parametrize('error_type,status', ((OSError, 'failed'), (KeyboardInterrupt, 'interrupted')))
def test_sweep_failure(monkeypatch, tiny_settings, error_type, status, tmp_path):
    from io import StringIO
    from tmt.cli import sweep

    root = tmp_path
    config, grid, data = root / 'model.json', root / 'grid.json', root / 'data.bin'
    config.write_text(json.dumps(tiny_settings))
    grid.write_text('{"seed": [11, 22]}')
    data.write_bytes(b'ab')
    output = root / 'sweep'
    completed_model = SimpleNamespace(count=lambda: 7, temporal_gradient=['rtu-v1'])
    outcomes = iter([completed_model, error_type('injected failure')])

    def next_train(*args, **kwargs):
        result = next(outcomes)
        if isinstance(result, BaseException):
            raise result
        return result
    with monkeypatch.context() as patches, pytest.raises(error_type):
        patches.setattr('tmt.cli.train', next_train)
        patches.setattr('tmt.benchmark.evaluate', lambda *args: {'targets': 1, 'bpb': 8.0})
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


@pytest.mark.parametrize('ce_only', (False, True))
def test_sweep(monkeypatch, tiny_settings, ce_only, tmp_path, manifest_fields):
    from io import StringIO
    from tmt.cli import load_model, sweep
    from tmt.blocks import BLOCKS

    from tmt.cli import main
    command_calls = []
    with monkeypatch.context() as patches:
        patches.setattr('tmt.cli.sweep', lambda *args, **kwargs: command_calls.append((args, kwargs)))
        args = ['sweep', '--grid', 'grid.json', '--data', 'train/*', '--development', 'dev/*', '--output', 'runs/grid']
        main(args)
        assert command_calls[-1][0][4] == 'model.json'
        main([*args, '--config', 'custom.json'])
        assert command_calls[-1][0][4] == 'custom.json'
        main([*args, '--sample-every', '2', '--sample-bytes', '3',
              '--eval-data', 'held-out/*', '--eval-every', '5'])
        assert command_calls[-1][1]['sample_every'] == 2
        assert command_calls[-1][1]['sample_bytes'] == 3
        assert command_calls[-1][1]['eval_data'] == 'held-out/*'
        assert command_calls[-1][1]['eval_every'] == 5

    root = tmp_path
    grid, config = root / 'grid.json', root / 'model.json'
    grid.write_text(json.dumps({'block': ['rtu', 'rtu-test'], 'seed': [11, 22]}))
    config.write_text(json.dumps(tiny_settings))
    train, development = root / 'train.bin', root / 'development.bin'
    train.write_bytes(b'\x00\x01\x02')
    development.write_bytes(b'\xf0\xf1\xf2\xf3')
    with monkeypatch.context() as patches:
        patches.setitem(BLOCKS, 'rtu-test', BLOCKS['rtu'])
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
            assert (set(manifest), manifest['status'], manifest['last_checkpoint']) == (manifest_fields, 'complete', str(checkpoint.resolve()))
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
