from tmt.config import group_model_config, load_config
from tmt.records import build_run_manifest, build_sweep_manifest


def test_group_model_config():
    settings = load_config()

    assert group_model_config(settings) == {
        'model': {
            'block': 'tmt-v0',
            'dimensions': {'dim': 512, 'layers': 16, 'spread': 32},
        },
        'hparams': {
            'optimizer': {'rate': 0.0005, 'bound': [40000, 120000]},
            'sampling': {'temp': 0.75},
        },
    }


def test_run_manifest_groups_related_fields():
    settings = load_config()
    record = build_run_manifest(
        config=settings,
        trainable_parameters=123,
        temporal_gradient=['tmt-v0'],
        seed=11,
        data_glob='train/*',
        data_files=['train/a.bin'],
        updates=20,
        log_every=5,
        save_every=10,
        objective='tmt',
        sample={'every': 4, 'prompt': 'A', 'seed': 11, 'bytes': 8},
        evaluation={
            'every': 10,
            'data_glob': 'eval/*',
            'data_files': ['eval/a.bin'],
            'max_bytes': 1024,
        },
        started_at='2026-10-21T00:00:00+00:00',
    )

    assert list(record) == [
        'model', 'hparams', 'data', 'training', 'monitoring', 'progress', 'run'
    ]
    assert record['model'] == {
        'block': 'tmt-v0',
        'dimensions': {'dim': 512, 'layers': 16, 'spread': 32},
        'trainable_parameters': 123,
        'temporal_gradient': ['tmt-v0'],
    }
    assert record['hparams'] == {
        'optimizer': {'rate': 0.0005, 'bound': [40000, 120000]},
        'sampling': {'temp': 0.75},
    }
    assert record['data'] == {
        'train': {'glob': 'train/*', 'files': ['train/a.bin']},
        'evaluation': {
            'glob': 'eval/*',
            'files': ['eval/a.bin'],
            'max_bytes': 1024,
        },
    }
    assert record['training'] == {
        'seed': 11,
        'updates': 20,
        'objective': 'tmt',
        'resume': False,
    }
    assert record['monitoring'] == {
        'log_every': 5,
        'save_every': 10,
        'sample': {'every': 4, 'prompt': 'A', 'seed': 11, 'bytes': 8},
        'evaluation': {'every': 10},
    }
    assert record['progress'] == {
        'completed_updates': 0,
        'initial_optimizer_step': 0,
        'optimizer_step': 0,
    }
    assert record['run'] == {
        'started_at': '2026-10-21T00:00:00+00:00',
        'status': 'running',
        'last_checkpoint': None,
        'error': None,
    }


def test_sweep_manifest_groups_fields_and_file_selections():
    record = build_sweep_manifest(
        config=load_config(),
        grid={'dim': [4, 8]},
        seed=11,
        updates=20,
        objective='tmt',
        data_glob='train/*',
        development_glob='dev/*',
        development_max_bytes=2048,
        log_every=5,
        save_every=10,
        sample_every=4,
        sample_prompt='A',
        sample_seed=11,
        sample_bytes=8,
        eval_data='eval/*',
        eval_every=10,
        eval_max_bytes=1024,
        started_at='2026-10-21T00:00:00+00:00',
        data_files=['train/a.bin'],
        development_files=['dev/a.bin'],
        evaluation_files=['eval/a.bin'],
    )

    assert list(record) == [
        'model', 'hparams', 'grid', 'data', 'training', 'monitoring', 'progress', 'run'
    ]
    assert record['data'] == {
        'train': {'glob': 'train/*', 'files': ['train/a.bin']},
        'development': {
            'glob': 'dev/*',
            'files': ['dev/a.bin'],
            'max_bytes': 2048,
        },
        'evaluation': {
            'glob': 'eval/*',
            'files': ['eval/a.bin'],
            'max_bytes': 1024,
        },
    }
    assert record['training'] == {
        'seed': 11,
        'updates': 20,
        'objective': 'tmt',
        'resume': False,
    }
    assert record['monitoring'] == {
        'log_every': 5,
        'save_every': 10,
        'sample': {'every': 4, 'prompt': 'A', 'seed': 11, 'bytes': 8},
        'evaluation': {'every': 10},
    }
    assert record['progress'] == {'completed_runs': 0, 'current_run': None}
    assert record['run'] == {
        'started_at': '2026-10-21T00:00:00+00:00',
        'status': 'running',
        'error': None,
    }
