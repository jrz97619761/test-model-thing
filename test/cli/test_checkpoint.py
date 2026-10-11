import json

import pytest


def _assert_rejected(path, mx, load_model, arrays, metadata, *parts):
    mx.save_safetensors(path, arrays, metadata=metadata)
    with pytest.raises(ValueError) as error:
        load_model(path)
    message = str(error.value)
    assert all(part in message for part in parts)


@pytest.mark.parametrize('fault', ['missing', 'extra'])
def test_checkpoint_names(saved_checkpoint, fault):
    from tmt.cli import load_model

    path, mx, _, _, arrays, metadata = saved_checkpoint
    if fault == 'missing':
        name = next(key for key in arrays if key.startswith('m.'))
        arrays.pop(name)
    else:
        name = 'o.extra'
        arrays[name] = next(iter(arrays.values()))
    _assert_rejected(path, mx, load_model, arrays, metadata, name)


def test_checkpoint_shapes(saved_checkpoint):
    from tmt.cli import load_model

    (path, mx, _, _, arrays, metadata) = saved_checkpoint
    name = 'm.decoder.decode.weight'
    arrays[name] = mx.zeros((1,), dtype=arrays[name].dtype)
    _assert_rejected(path, mx, load_model, arrays, metadata, 'decoder.decode.weight', 'shape')


def test_checkpoint_dtypes(saved_checkpoint):
    from tmt.cli import load_model

    (path, mx, _, _, arrays, metadata) = saved_checkpoint
    name = next(key for key in arrays if key.startswith('o.') and key.endswith('.m'))
    arrays[name] = mx.zeros(arrays[name].shape, dtype=mx.int32)
    _assert_rejected(path, mx, load_model, arrays, metadata, name[2:], 'dtype')


def test_checkpoint_round_trip(saved_checkpoint):
    from tmt.cli import load_model

    path, mx, settings, model, _, _ = saved_checkpoint
    restored, loaded = load_model(path)
    assert (loaded, restored.dim, int(restored.optimizer.state['step'].item())) == (settings, 4, 0)
    assert mx.array_equal(restored.decoder.decode.weight, model.decoder.decode.weight).item()


@pytest.mark.parametrize('fault,field', [
    ('metadata', '__metadata__'), ('spread', 'spread'), ('block', 'block'), ('version', 'version'),
])
def test_checkpoint_header(saved_checkpoint, fault, field):
    from tmt.cli import load_model

    path, mx, _, _, arrays, metadata = saved_checkpoint
    header = json.loads(metadata['tmt'])
    if fault in ('spread', 'block'):
        header['model'].pop(fault)
    elif fault == 'version':
        header['version'] = True
    invalid = None if fault == 'metadata' else {'tmt': json.dumps(header)}
    _assert_rejected(path, mx, load_model, arrays, invalid, field)


def test_checkpoint_truncated_header(saved_checkpoint):
    from tmt.cli import load_model

    path = saved_checkpoint[0]
    path.write_bytes((8).to_bytes(8, 'little') + b'{}')
    with pytest.raises(ValueError, match='header'):
        load_model(path)
