import pytest

from tmt.runtime.checkpoint import DummyByteFiles, DEFAULT_CONFIG, DEFAULT_GRID, group_config
from tmt.runtime.runtime import run_manifest, sweep_manifest

def test_run_manifest():
    train_files = DummyByteFiles('data/train/**')
    eval_files = DummyByteFiles('data/eval/**')

    record = run_manifest(
        DEFAULT_CONFIG, train_files, eval_files,
        11, 0, False, (5, 10, 15, 20), ('The ', 11, 8), 1024,
        0, 0, 0, 'Start Time', 'runs/run/model.safetensors', 'running', '', ''
    )

    assert record == {
        'model': group_config(DEFAULT_CONFIG),
        'data': {
            'train': {
                'glob': train_files.pattern,
                'files': train_files.filenames()
            },
            'eval': {
                'glob': eval_files.pattern,
                'files': eval_files.filenames(),
                'max_bytes': 1024
            }
        },
        'training': {
            'seed': 11,
            'updates': 0,
            'resume': False,
        },
        'intervals': {
            'log': 5,
            'save': 10,
            'eval': 15,
            'sample': 20
        },
        'sampling': {
            'prompt': 'The ',
            'seed': 11,
            'length': 8
        },
        'progress': {
            'completed': 0,
            'opt': {
                'initial': 0,
                'current': 0
            }
        },
        'run': {
            'start': 'Start Time',
            'checkpoint': 'runs/run/model.safetensors',
            'status': 'running',
            'error': None
        }
    }

def test_sweep_manifest():
    train_files = DummyByteFiles('data/train/**')
    eval_files = DummyByteFiles('data/eval/**')
    dev_files = DummyByteFiles('data/dev/**')

    record = sweep_manifest(
        DEFAULT_CONFIG, DEFAULT_GRID, train_files, eval_files, dev_files,
        11, 0, (5, 10, 15, 20), ('The ', 11, 8), 1024,
        0, 0, 'Start Time', 'running', '', ''
    )

    assert record == {
        'model': group_config(DEFAULT_CONFIG),
        'grid': DEFAULT_GRID,
        'data': {
            'train': {
                'glob': train_files.pattern,
                'files': train_files.filenames()
            },
            'eval': {
                'glob': eval_files.pattern,
                'files': eval_files.filenames(),
                'max_bytes': 1024
            },
            'dev': {
                'glob': dev_files.pattern,
                'files': dev_files.filenames(),
                'max_bytes': 1024
            }
        },
        'training': {
            'seed': 11,
            'updates': 0
        },
        'intervals': {
            'log': 5,
            'save': 10,
            'eval': 15,
            'sample': 20
        },
        'sampling': {
            'prompt': 'The ',
            'seed': 11,
            'length': 8
        },
        'progress': {
            'completed': 0,
            'current': 0
        },
        'run': {
            'start': 'Start Time',
            'status': 'running',
            'error': None
        }
    }