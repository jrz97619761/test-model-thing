import json

import pytest

from cli_support import assert_checkpoint_rejected


@pytest.mark.parametrize(
    'fault', ['missing', 'extra'], ids=['missing-tensor', 'extra-tensor']
)
def test_checkpoint_rejects_missing_or_extra_tensor(saved_checkpoint, fault):
    from tmt.cli import load_model

    path = saved_checkpoint.path
    mx = saved_checkpoint.mx
    arrays = saved_checkpoint.arrays
    metadata = saved_checkpoint.metadata
    if fault == 'missing':
        name = next(key for key in arrays if key.startswith('m.'))
        arrays.pop(name)
    else:
        name = 'o.extra'
        arrays[name] = next(iter(arrays.values()))
    assert_checkpoint_rejected(path, mx, load_model, arrays, metadata, name)


def test_checkpoint_shapes(saved_checkpoint):
    from tmt.cli import load_model

    path = saved_checkpoint.path
    mx = saved_checkpoint.mx
    arrays = saved_checkpoint.arrays
    metadata = saved_checkpoint.metadata
    name = 'm.decoder.decode.weight'
    arrays[name] = mx.zeros((1,), dtype=arrays[name].dtype)
    assert_checkpoint_rejected(
        path, mx, load_model, arrays, metadata, 'decoder.decode.weight', 'shape'
    )


def test_checkpoint_dtypes(saved_checkpoint):
    from tmt.cli import load_model

    path = saved_checkpoint.path
    mx = saved_checkpoint.mx
    arrays = saved_checkpoint.arrays
    metadata = saved_checkpoint.metadata
    name = next(key for key in arrays if key.startswith('o.') and key.endswith('.m'))
    arrays[name] = mx.zeros(arrays[name].shape, dtype=mx.int32)
    assert_checkpoint_rejected(
        path, mx, load_model, arrays, metadata, name[2:], 'dtype'
    )


def test_checkpoint_round_trip(saved_checkpoint):
    from tmt.cli import load_model

    restored, loaded = load_model(saved_checkpoint.path)

    assert loaded == saved_checkpoint.settings, (
        'loaded settings differ from saved settings'
    )
    assert restored.dim == 4, 'restored model dimension differs'
    assert int(restored.optimizer.state['step'].item()) == 0, 'optimizer step differs'
    assert saved_checkpoint.mx.array_equal(
        restored.decoder.decode.weight,
        saved_checkpoint.model.decoder.decode.weight,
    ).item(), 'decoder weight differs after round trip'


@pytest.mark.parametrize(
    ('fault', 'field'),
    [
        ('metadata', '__metadata__'),
        ('spread', 'spread'),
        ('block', 'block'),
        ('version', 'version'),
    ],
    ids=['missing-metadata', 'missing-spread', 'missing-block', 'invalid-version'],
)
def test_checkpoint_header(saved_checkpoint, fault, field):
    from tmt.cli import load_model

    path = saved_checkpoint.path
    mx = saved_checkpoint.mx
    arrays = saved_checkpoint.arrays
    metadata = saved_checkpoint.metadata
    header = json.loads(metadata['tmt'])
    if fault in ('spread', 'block'):
        header['model'].pop(fault)
    elif fault == 'version':
        header['version'] = True
    invalid = None if fault == 'metadata' else {'tmt': json.dumps(header)}
    assert_checkpoint_rejected(path, mx, load_model, arrays, invalid, field)


def test_checkpoint_truncated_header(saved_checkpoint):
    from tmt.cli import load_model

    path = saved_checkpoint.path
    path.write_bytes((8).to_bytes(8, 'little') + b'{}')
    with pytest.raises(ValueError, match='header'):
        load_model(path)
