import json

import pytest

from tmt import validation


@pytest.mark.parametrize(
    ('default', 'value', 'message'),
    [
        (0.5, True, 'invalid JSON type'),
        (0.5, 10**400, 'invalid JSON type'),
        (0.5, float('nan'), 'finite JSON values'),
        (0.5, float('inf'), 'finite JSON values'),
        ((1, 2), 1, 'invalid JSON type'),
        ((1, 2), [float('inf')], 'finite JSON values'),
    ],
    ids=[
        'boolean-for-number',
        'overflow-for-number',
        'nan-json-value',
        'infinite-json-value',
        'scalar-for-array',
        'infinite-array-value',
    ],
)
def test_extended_model_fields(monkeypatch, tmp_path, default, value, message):
    from tmt import config
    from tmt.cli import load_config

    monkeypatch.setitem(config.MODEL_DEFAULTS, 'extra', default)
    settings = load_config()
    assert settings['extra'] == (
        list(default) if isinstance(default, tuple) else default
    )
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'extra': value}))
    with pytest.raises(ValueError, match=f'model config.extra.*{message}'):
        load_config(path)
    checkpoint_metadata = json.dumps(
        {'version': 1, 'model': {**settings, 'extra': value}}
    )
    with pytest.raises(ValueError, match=f'checkpoint.model.extra.*{message}'):
        validation.metadata(checkpoint_metadata)
