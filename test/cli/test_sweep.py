import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli_support import assert_fields, assert_ordered_records, read_jsonl


INVALID_GRIDS = (
    [(grid, 'sweep grid must be a JSON object') for grid in (None, 4, [], 'dim')]
    + [
        ({'dim': axis}, "sweep grid field 'dim' must be a nonempty array")
        for axis in (4, None, '4', {}, [], True)
    ]
    + [
        ({'dim': [4], 'seed': []}, "sweep grid field 'seed' must be a nonempty array"),
        ({'dim': [4, 0]}, 'model config.dim must be a positive integer'),
        ({'seed': [11, True]}, 'sweep grid seed must be an integer'),
        ({'seed': [11, -1]}, 'sweep grid seed must be between'),
        ({'seed': [11, 2**64]}, 'sweep grid seed must be between'),
    ]
)


@pytest.mark.parametrize(
    'grid,message',
    INVALID_GRIDS,
    ids=[
        'non-object-grid',
        'integer-grid',
        'array-grid',
        'string-grid',
        'scalar-dim-axis',
        'null-dim-axis',
        'string-dim-axis',
        'object-dim-axis',
        'empty-dim-axis',
        'boolean-dim-axis',
        'empty-seed-axis',
        'zero-dimension',
        'boolean-seed',
        'negative-seed',
        'seed-overflow',
    ],
)
@pytest.mark.parametrize('existing', [False, True], ids=['new-output', 'empty-output'])
def test_invalid_grid(
    tmp_path, monkeypatch, forbid_call, tiny_settings, existing, grid, message
):
    from tmt.cli import sweep

    monkeypatch.setattr('tmt.cli.train', forbid_call)
    config, grid_path, output = (
        tmp_path / 'model.json',
        tmp_path / 'grid.json',
        tmp_path / 'sweep',
    )
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


@pytest.mark.parametrize(
    ('error_type', 'status'),
    ((OSError, 'failed'), (KeyboardInterrupt, 'interrupted')),
    ids=['filesystem-failure', 'keyboard-interrupt'],
)
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
        patches.setattr(
            'tmt.benchmark.evaluate', lambda *args: {'targets': 1, 'bpb': 8.0}
        )
        sweep(grid, str(data), str(data), output, config, updates=1)
    record = json.loads((output / 'sweep.json').read_text())
    assert_fields(
        record,
        {
            'status': status,
            'current_run': 2,
            'completed_runs': 1,
        },
        context='sweep failure record',
    )
    assert record['error'] == {
        'type': error_type.__name__,
        'message': 'injected failure',
    }
    first_row = json.loads((output / 'results.jsonl').read_text().splitlines()[0])
    assert first_row['trainable_parameters'] == 7, (
        'first result has the wrong parameter count'
    )
    assert first_row['temporal_gradient'] == ['rtu-v1'], (
        'first result lost the RTU gradient marker'
    )
    rows = (output / 'results.jsonl').read_bytes()
    with pytest.raises(FileExistsError, match='choose a new output folder'):
        sweep(grid, str(data), str(data), output, config, updates=1)
    assert (output / 'results.jsonl').read_bytes() == rows
    with contextlib.redirect_stdout(StringIO()):
        sweep(grid, str(data), str(data), root / 'retry', config, updates=1)
    assert json.loads((root / 'retry/sweep.json').read_text())['status'] == 'complete'


def test_sweep_argument_dispatch(monkeypatch):
    from tmt.cli import main

    command_calls = []
    with monkeypatch.context() as patches:
        patches.setattr(
            'tmt.cli.sweep',
            lambda *args, **kwargs: command_calls.append((args, kwargs)),
        )
        args = [
            'sweep',
            '--grid',
            'grid.json',
            '--data',
            'train/*',
            '--development',
            'dev/*',
            '--output',
            'runs/grid',
        ]
        main(args)
        assert command_calls[-1][0][4] == 'model.json'
        main([*args, '--config', 'custom.json'])
        assert command_calls[-1][0][4] == 'custom.json'
        main(
            [
                *args,
                '--sample-every',
                '2',
                '--sample-bytes',
                '3',
                '--eval-data',
                'held-out/*',
                '--eval-every',
                '5',
            ]
        )
        dispatched_args, dispatched_options = command_calls[-1]
        assert dispatched_options['sample_every'] == 2, (
            'sample interval was not dispatched'
        )
        assert dispatched_options['sample_bytes'] == 3, (
            'sample byte count was not dispatched'
        )
        assert dispatched_options['eval_data'] == 'held-out/*', (
            'evaluation data was not dispatched'
        )
        assert dispatched_options['eval_every'] == 5, (
            'evaluation interval was not dispatched'
        )
        assert dispatched_args[4] == 'model.json', (
            'default config path was not dispatched'
        )


@pytest.mark.parametrize(
    'ce_only', (False, True), ids=['tmt-objective', 'cross-entropy-only']
)
def test_sweep_full_grid(
    monkeypatch, tiny_settings, ce_only, tmp_path, manifest_fields
):
    from io import StringIO
    from tmt.cli import load_model, sweep
    from tmt.blocks import BLOCKS

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
            best = sweep(
                grid,
                str(train),
                str(development),
                output,
                config,
                2,
                11,
                ce_only,
                max_bytes=4,
                log_every=1,
                sample_every=2,
                sample_bytes=3,
                eval_data=str(development),
                eval_every=2,
                eval_max_bytes=3,
            )
        rows = read_jsonl(output / 'results.jsonl')
        actual_runs = [(row['settings']['block'], row['seed']) for row in rows]
        assert actual_runs == [
            ('rtu', 11),
            ('rtu', 22),
            ('rtu-test', 11),
            ('rtu-test', 22),
        ], 'sweep changed product order or per-axis seeds'
        assert best == min(rows, key=lambda row: row['development_bpb']), (
            'sweep returned the wrong best row'
        )
        assert json.dumps(best) in stdout.getvalue(), 'best row was not reported'
        for index, row in enumerate(rows, 1):
            checkpoint = Path(row['checkpoint'])
            assert row['objective'] == ('ce_only' if ce_only else 'tmt'), (
                'objective differs in result row'
            )
            assert row['development_targets'] == 3, (
                'development target count differs in result row'
            )
            assert row['trainable_parameters'] > 0, (
                'result row has no trainable parameters'
            )
            assert row['temporal_gradient'] == ['rtu-v1'], (
                'result row lost the RTU gradient marker'
            )
            assert (
                json.loads((checkpoint.parent / 'manifest' / 'model.json').read_text())
                == row['settings']
            )
            manifest = json.loads(
                (checkpoint.parent / 'manifest' / 'run.json').read_text()
            )
            assert set(manifest) == manifest_fields, 'run manifest fields differ'
            assert manifest['status'] == 'complete', (
                'sweep run manifest is not complete'
            )
            assert manifest['last_checkpoint'] == str(checkpoint.resolve()), (
                'run manifest checkpoint differs'
            )
            assert manifest['config'] == row['settings'], (
                'run manifest config differs from result row'
            )
            assert manifest['trainable_parameters'] == row['trainable_parameters'], (
                'parameter count differs from result row'
            )
            assert manifest['temporal_gradient'] == row['temporal_gradient'], (
                'RTU marker differs from result row'
            )
            assert manifest['log_every'] == 1
            metrics = read_jsonl(checkpoint.parent / 'manifest/loss.jsonl')
            assert_ordered_records(
                metrics,
                [
                    {'completed_updates': 1},
                    {'completed_updates': 2},
                ],
                context=f'sweep run {index} loss rows',
            )
            evaluation = read_jsonl(checkpoint.parent / 'manifest/evaluation.jsonl')
            assert_ordered_records(
                evaluation,
                [
                    {'completed_updates': 2, 'targets': 2},
                ],
                context=f'sweep run {index} evaluations',
            )
            samples = read_jsonl(checkpoint.parent / 'manifest/samples.jsonl')
            assert_ordered_records(
                samples,
                [
                    {'completed_updates': 2, 'bytes': 3},
                ],
                context=f'sweep run {index} samples',
            )
            assert len(Path(samples[0]['path']).read_bytes()) == 3
            for field in ('seed', 'updates', 'objective', 'data_glob'):
                assert manifest[field] == row[field], (
                    f'manifest field {field!r} differs from result row'
                )
            assert manifest['data_files'] == [str(train.resolve())], (
                'run manifest data files differ'
            )
            assert manifest['completed_updates'] == row['updates'], (
                'run manifest update count differs'
            )
            assert manifest['resume'] is False, 'fresh sweep run is marked as resumed'
            assert manifest['initial_optimizer_step'] == 0, (
                'fresh sweep run started from a nonzero step'
            )
            model, settings = load_model(checkpoint, row['seed'])
            assert settings == row['settings'], (
                'loaded checkpoint settings differ from result row'
            )
            assert row['trainable_parameters'] == model.count(), (
                'loaded model parameter count differs'
            )
            assert row['temporal_gradient'] == model.temporal_gradient, (
                'loaded model RTU marker differs'
            )
            assert int(model.optimizer.state['step'].item()) == 2
            assert checkpoint.parent.name == f'run-{index:04d}'
