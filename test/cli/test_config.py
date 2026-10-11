import inspect
import json

import pytest

from tmt import validation


@pytest.mark.parametrize('default,value,message', [
    (0.5, True, 'invalid JSON type'),
    (0.5, 10**400, 'invalid JSON type'),
    (0.5, float('nan'), 'finite JSON values'),
    (0.5, float('inf'), 'finite JSON values'),
    ((1, 2), 1, 'invalid JSON type'),
    ((1, 2), [float('inf')], 'finite JSON values'),
])
def test_extended_model_fields(monkeypatch, tmp_path, default, value, message):
    from tmt import main
    from tmt.cli import load_config

    parameters = list(inspect.signature(main.Model).parameters.values())
    parameters.append(inspect.Parameter('extra', inspect.Parameter.KEYWORD_ONLY, default=default))

    def extended():
        pass

    extended.__signature__ = inspect.Signature(parameters)
    monkeypatch.setattr(main, 'Model', extended)
    settings = load_config()
    assert settings['extra'] == (list(default) if isinstance(default, tuple) else default)
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'extra': value}))
    with pytest.raises(ValueError, match=f'model config.extra.*{message}'):
        load_config(path)
    with pytest.raises(ValueError, match=f'checkpoint.model.extra.*{message}'):
        validation.validate_metadata(json.dumps({'version': 1, 'model': {**settings, 'extra': value}}))
