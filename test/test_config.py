import json
import inspect
from pathlib import Path

import pytest

from tmt import validation

NAN, INF = float('nan'), float('inf')
CONFIG_CASES = {  # id: (field, invalid value)
    'dim-true': ('dim', True),
    'dim-zero': ('dim', 0),
    'dim-negative': ('dim', -1),
    'dim-float': ('dim', 4.0),
    'dim-string': ('dim', '4'),
    'dim-none': ('dim', None),
    'layers-zero': ('layers', 0),
    'spread-object': ('spread', {}),
    'temp-true': ('temp', True),
    'temp-negative': ('temp', -1),
    'temp-string': ('temp', '1'),
    'temp-nan': ('temp', NAN),
    'temp-inf': ('temp', INF),
    'temp-overflow': ('temp', 10**400),
    'rate-zero': ('rate', 0),
    'rate-negative-inf': ('rate', -INF),
    'block-unknown': ('block', 'unknown-block'),
    'block-int': ('block', 1),
    'bound-int': ('bound', 30),
    'bound-one-value': ('bound', [0]),
    'bound-three-values': ('bound', [0, 1, 2]),
    'bound-negative-start': ('bound', [-1, 1]),
    'bound-equal': ('bound', [1, 1]),
    'bound-reversed': ('bound', [2, 1]),
    'bound-bool': ('bound', [False, 1]),
    'bound-float': ('bound', [0, 1.0]),
}
DEL = object()  # marks a field to delete from the settings
ORDER_CASES = {  # id: (changes, message)
    'missing-before-unknown': (
        {'unknown': 1, 'spread': DEL, 'dim': DEL},
        "model config is missing field 'dim'",
    ),
    'block-before-dim': (
        {'block': 'unknown', 'dim': 0},
        r'model config\.block must be one of',
    ),
    'dim-before-layers-and-spread': (
        {'dim': 0, 'layers': 0, 'spread': 0},
        r'model config\.dim must be a positive integer',
    ),
    'layers-before-spread': (
        {'dim': 4, 'layers': 0, 'spread': 0},
        r'model config\.layers must be a positive integer',
    ),
    'integers-before-temp-and-rate': (
        {'temp': -1, 'rate': 0},
        r'model config\.temp must be a finite nonnegative number',
    ),
    'rate-before-bound': (
        {'temp': 0.75, 'rate': 0, 'bound': [120000, 40000]},
        r'model config\.rate must be a finite positive number',
    ),
    'core-before-json': (
        {'dim': 0, 'extension': NAN},
        r'model config\.dim must be a positive integer',
    ),
    'json-before-extension': (
        {'extension': NAN},
        r'model config\.extension must contain finite JSON values',
    ),
}
EXTENSION_CASES = {  # id: (default, invalid value, message)
    'boolean-for-number': (0.5, True, 'invalid JSON type'),
    'scalar-for-array': ((1, 2), 1, 'invalid JSON type'),
    'nan-json-value': (0.5, NAN, 'finite JSON values'),
}
FILE_FAULTS = {  # id: message fragments
    'missing-tensor': ('m.decoder.stop.bias',),
    'extra-tensor': ('o.extra',),
    'wrong-shape': ('decoder.decode.weight', 'shape'),
    'wrong-dtype': ('decoder.decode.weight.m', 'dtype'),
    'missing-metadata': ('__metadata__',),
    'missing-spread': ('spread',),
    'missing-block': ('block',),
    'unknown-block': ('block',),
    'invalid-version': ('version',),
    'metadata-json': ('invalid JSON',),
    'header-length': ('header',),
    'header-not-json': ('invalid Safetensors header',),
    'header-empty-object': ('nonempty object',),
    'no-tensors': ('no tensors',),
    'truncated-data': ('invalid checkpoint:',),
}


def test_load_config(tmp_path, monkeypatch):
    import inspect

    from tmt import config
    from tmt.cli import load_config
    from tmt.main import Model

    defaults = load_config()
    assert config.load_config() == defaults
    expected = {**config.MODEL_DEFAULTS, 'bound': list(config.MODEL_DEFAULTS['bound'])}
    assert defaults == expected
    parameters = inspect.signature(Model).parameters
    assert set(defaults) == set(parameters), (
        'config fields differ from constructor fields'
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in parameters.values()
    ), 'direct Model constructor fields must remain required'
    with pytest.raises(TypeError, match='required positional arguments'):
        Model(dim=4, layers=1, spread=4, block='rtu')
    with pytest.raises(ValueError, match='unknown block'):
        Model(
            **{**defaults, 'dim': 4, 'layers': 1, 'spread': 4, 'block': 'unknown-block'}
        )

    path = tmp_path / 'model.json'
    path.write_text('{}')
    assert load_config(path) == defaults
    path.write_text('{"dim": 4}')
    settings = load_config(path)
    partial = {'dim': 4, 'layers': 16, 'bound': [40000, 120000]}
    assert {name: settings[name] for name in partial} == partial
    limits = {'dim': 1, 'layers': 1, 'spread': 1, 'temp': 0, 'rate': 1, 'bound': [0, 1]}
    path.write_text(json.dumps(limits))
    settings = load_config(path)
    assert {name: settings[name] for name in limits} == limits
    monkeypatch.setitem(config.MODEL_DEFAULTS, 'extra', 17)
    assert load_config()['extra'] == 17


@pytest.mark.parametrize(
    ('name', 'value'), CONFIG_CASES.values(), ids=list(CONFIG_CASES)
)
def test_invalid_config_values(tmp_path, tiny_settings, name, value):
    from tmt.cli import load_config, new_model

    path = tmp_path / 'model.json'
    path.write_text(json.dumps({name: value}))
    with pytest.raises(ValueError, match=f'model config.{name}'):
        load_config(path)
    settings = dict(tiny_settings, **{name: value})
    with pytest.raises(ValueError, match=f'model config.{name}'):
        new_model(settings, 11)
    raw = json.dumps({'version': 1, 'model': settings})
    with pytest.raises(ValueError, match=f'checkpoint.model.{name}'):
        validation.metadata(raw)


@pytest.mark.parametrize(
    ('changes', 'message'), ORDER_CASES.values(), ids=list(ORDER_CASES)
)
def test_validation_error_order(tiny_settings, monkeypatch, changes, message):
    from tmt import config

    if 'extension' in changes:
        defaults = {**config.MODEL_DEFAULTS, 'extension': 1}
        monkeypatch.setattr(config, 'MODEL_DEFAULTS', defaults)
    settings = {**tiny_settings, **changes}
    settings = {name: item for name, item in settings.items() if item is not DEL}
    with pytest.raises(ValueError, match=message):
        validation.model_settings(settings, 'model config')


@pytest.mark.parametrize(
    ('default', 'value', 'message'), EXTENSION_CASES.values(), ids=list(EXTENSION_CASES)
)
def test_extension_fields(monkeypatch, tmp_path, default, value, message):
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
    raw = json.dumps({'version': 1, 'model': {**settings, 'extra': value}})
    with pytest.raises(ValueError, match=f'checkpoint.model.extra.*{message}'):
        validation.metadata(raw)


@pytest.mark.parametrize(
    ('contents', 'error', 'message'),
    [
        ('{', json.JSONDecodeError, None),
        ('[]', ValueError, 'model config must be a JSON object'),
        ('{"unknown": 1}', ValueError, "unexpected keyword argument 'unknown'"),
    ],
    ids=['invalid-json', 'not-object', 'unknown-field'],
)
def test_load_config_rejects_bad_files(tmp_path, contents, error, message):
    from tmt.cli import load_config

    path = tmp_path / 'model.json'
    path.write_text(contents)
    with pytest.raises(error, match=message):
        load_config(path)


def test_init_writes_templates(run_cli, tmp_path, manifest_fields):
    import inspect

    from tmt.cli import load_config
    from tmt.main import Model

    repo = Path(__file__).resolve().parents[1]
    assert run_cli('init')[0] == 0
    saved = json.loads((tmp_path / 'model.json').read_text())
    assert saved == load_config()
    assert set(saved) == set(inspect.signature(Model).parameters)
    templates = {tmp_path / n for n in ('model.json', 'grid.json', 'run.example.json')}
    assert set(tmp_path.rglob('*.json')) == templates
    for name in ('data/train', 'data/dev', 'runs'):
        assert (tmp_path / name).is_dir(), f'init did not create {name!r}'
    grid = {
        'dim': [4, 8],
        'layers': [1],
        'spread': [4],
        'block': ['rtu'],
        'seed': [11, 22],
    }
    assert json.loads((tmp_path / 'grid.json').read_text()) == grid
    template = json.loads((tmp_path / 'run.example.json').read_text())
    assert template == json.loads((repo / 'examples' / 'run.example.json').read_text())
    assert set(template) == manifest_fields
    expected = {
        'config': saved,
        'seed': 11,
        'trainable_parameters': None,
        'temporal_gradient': None,
        'data_glob': 'data/train/*',
        'data_files': [],
        'updates': 1000,
        'objective': 'tmt',
        'completed_updates': 0,
        'resume': False,
        'initial_optimizer_step': 0,
        'started_at': None,
        'sample': None,
        'evaluation': None,
        'status': 'running',
        'last_checkpoint': None,
    }
    assert {name: template[name] for name in expected} == expected

    explicit = tmp_path / 'custom' / 'settings.json'
    assert run_cli('init', '--output', str(explicit))[0] == 0
    assert json.loads(explicit.read_text()) == saved
    assert set(tmp_path.rglob('*.json')) == templates | {explicit}

    assert load_config(repo / 'examples' / 'model.json') == load_config()
    tiny = json.loads((repo / 'examples' / 'tiny-model.json').read_text())
    assert tiny == {**saved, 'dim': 4, 'layers': 1, 'spread': 4}
    assert json.loads((repo / 'examples' / 'grid.json').read_text()) == grid


@pytest.mark.parametrize(
    'args',
    [('init',), ('init', '--output', 'model.json')],
    ids=['default-output', 'explicit-output'],
)
def test_init_preserves_files(run_cli, tmp_path, args):
    assert run_cli('init')[0] == 0
    (tmp_path / 'model.json').write_text('{"dim": 4}')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert run_cli(*args)[0] == 1
    after = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert after.keys() == before.keys(), 'init failure changed the set of files'
    assert after == before, 'init failure changed existing file contents'


@pytest.mark.parametrize(('fault', 'parts'), FILE_FAULTS.items(), ids=list(FILE_FAULTS))
def test_checkpoint_rejects_invalid_files(checkpoint, fault, parts):
    import mlx.core as mx
    from tmt.cli import load_model

    path, arrays = checkpoint.path, dict(checkpoint.arrays)
    header, tmt, raw = json.loads(checkpoint.metadata['tmt']), None, None
    if fault == 'missing-tensor':
        del arrays[parts[0]]
    elif fault == 'extra-tensor':
        arrays[parts[0]] = arrays['m.decoder.stop.bias']
    elif fault == 'wrong-shape':
        name = 'm.' + parts[0]
        arrays[name] = mx.zeros((1,), dtype=arrays[name].dtype)
    elif fault == 'wrong-dtype':
        name = 'o.' + parts[0]
        arrays[name] = mx.zeros(arrays[name].shape, dtype=mx.int32)
    elif fault in ('missing-spread', 'missing-block'):
        header['model'].pop(parts[0])
    elif fault == 'unknown-block':
        header['model']['block'] = 'unknown'
    elif fault == 'invalid-version':
        header['version'] = True
    elif fault == 'metadata-json':
        tmt = 'x'
    elif fault == 'header-length':
        raw = (8).to_bytes(8, 'little') + b'{}'
    elif fault == 'header-not-json':
        raw = (2).to_bytes(8, 'little') + b'{x'
    elif fault == 'header-empty-object':
        raw = (2).to_bytes(8, 'little') + b'{}'
    elif fault == 'no-tensors':
        body = b'{"__metadata__": {"tmt": "x"}}'
        raw = len(body).to_bytes(8, 'little') + body
    elif fault == 'truncated-data':
        raw = path.read_bytes()[:-16]
    if raw is not None:
        path.write_bytes(raw)
    else:
        metadata = (
            None if fault == 'missing-metadata' else {'tmt': tmt or json.dumps(header)}
        )
        mx.save_safetensors(path, arrays, metadata=metadata)
    with pytest.raises(ValueError) as error:
        load_model(path, 11)
    for part in parts:
        assert part in str(error.value), f'message is missing {part!r}: {error.value}'


def test_checkpoint_loader_names_require_same_seed_signature():
    from tmt.checkpoint import load_checkpoint, load_model

    checkpoint_signature = inspect.signature(load_checkpoint)
    assert inspect.signature(load_model) == checkpoint_signature
    assert checkpoint_signature.parameters['seed'].default is inspect.Parameter.empty
