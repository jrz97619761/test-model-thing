import json
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
    from tmt.config import load_config
    from tmt.model import Model

    defaults = load_config()
    assert config.load_config() == defaults
    expected = {**config.MODEL_DEFAULTS, 'bound': list(config.MODEL_DEFAULTS['bound'])}
    assert defaults == expected
    validation.model_settings(dict(config.MODEL_DEFAULTS), 'model config')
    parameters = inspect.signature(Model).parameters
    assert set(defaults) == set(parameters), (
        'config fields differ from constructor fields'
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in parameters.values()
    ), 'direct Model constructor fields must remain required'
    with pytest.raises(TypeError, match='required positional arguments'):
        Model(dim=4, layers=1, spread=4, block='tmt-v0')
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
    from tmt.checkpoint import new_model
    from tmt.config import load_config

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
    from tmt.config import load_config

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
        ('{', ValueError, r'invalid JSON in .*model\.json: Expecting'),
        ('[]', ValueError, 'model config must be a JSON object'),
        ('{"unknown": 1}', ValueError, "model config has unknown field 'unknown'"),
    ],
    ids=['invalid-json', 'not-object', 'unknown-field'],
)
def test_load_config_rejects_bad_files(tmp_path, contents, error, message):
    from tmt.config import load_config

    path = tmp_path / 'model.json'
    path.write_text(contents)
    with pytest.raises(error, match=message):
        load_config(path)


def test_init_writes_templates(run_cli, tmp_path, manifest_fields):
    import inspect

    from tmt.config import load_config
    from tmt.model import Model

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
        'block': ['tmt-v0'],
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

    from tmt.checkpoint import load_checkpoint

    path, arrays = checkpoint.path, dict(checkpoint.arrays)
    header, tmt, raw = json.loads(checkpoint.metadata['tmt']), None, None
    match fault:
        case 'missing-tensor':
            del arrays[parts[0]]
        case 'extra-tensor':
            arrays[parts[0]] = arrays['m.decoder.stop.bias']
        case 'wrong-shape':
            name = 'm.' + parts[0]
            arrays[name] = mx.zeros((1,), dtype=arrays[name].dtype)
        case 'wrong-dtype':
            name = 'o.' + parts[0]
            arrays[name] = mx.zeros(arrays[name].shape, dtype=mx.int32)
        case 'missing-spread' | 'missing-block':
            header['model'].pop(parts[0])
        case 'unknown-block':
            header['model']['block'] = 'unknown'
        case 'invalid-version':
            header['version'] = True
        case 'metadata-json':
            tmt = 'x'
        case 'header-length':
            raw = (8).to_bytes(8, 'little') + b'{}'
        case 'header-not-json':
            raw = (2).to_bytes(8, 'little') + b'{x'
        case 'header-empty-object':
            raw = (2).to_bytes(8, 'little') + b'{}'
        case 'no-tensors':
            body = b'{"__metadata__": {"tmt": "x"}}'
            raw = len(body).to_bytes(8, 'little') + body
        case 'truncated-data':
            raw = path.read_bytes()[:-16]
    if raw is not None:
        path.write_bytes(raw)
    else:
        metadata = (
            None if fault == 'missing-metadata' else {'tmt': tmt or json.dumps(header)}
        )
        mx.save_safetensors(path, arrays, metadata=metadata)
    with pytest.raises(ValueError) as error:
        load_checkpoint(path)
    for part in parts:
        assert part in str(error.value), f'message is missing {part!r}: {error.value}'


@pytest.mark.parametrize('exists', [False, True], ids=['new', 'existing'])
@pytest.mark.parametrize('failure', ['write', 'interrupt', 'sync', 'replace'])
def test_checkpoint_failure_preserves_destination(
    monkeypatch, checkpoint, exists, failure
):
    import mlx.core as mx

    from tmt import checkpoint as storage

    path = checkpoint.path if exists else checkpoint.path.with_name('new.safetensors')
    original = path.read_bytes() if exists else None
    files = set(path.parent.iterdir())
    error = KeyboardInterrupt if failure == 'interrupt' else OSError

    def fail_write(destination, *args, **kwargs):
        Path(destination).write_bytes(b'partial')
        raise error('injected write failure')

    def fail(*args, **kwargs):
        raise error('injected checkpoint failure')

    match failure:
        case 'write' | 'interrupt':
            monkeypatch.setattr(mx, 'save_safetensors', fail_write)
        case 'sync':
            monkeypatch.setattr(storage.os, 'fsync', fail)
        case _:
            monkeypatch.setattr(storage.os, 'replace', fail)

    with pytest.raises(error):
        storage.save_checkpoint(path, checkpoint.model, checkpoint.settings)
    assert set(path.parent.iterdir()) == files
    if exists:
        assert path.read_bytes() == original
    else:
        assert not path.exists()


@pytest.mark.parametrize('mode', [0o200, 0o600, 0o640, 0o644, 0o660])
def test_checkpoint_replaces_existing_file(checkpoint, snapshot, mode):
    import stat

    from tmt.checkpoint import load_checkpoint, save_checkpoint

    original = checkpoint.path.read_bytes()
    checkpoint.path.chmod(mode)
    checkpoint.model.train_step(65, 66, False)
    save_checkpoint(checkpoint.path, checkpoint.model, checkpoint.settings)
    assert stat.S_IMODE(checkpoint.path.stat().st_mode) == mode
    checkpoint.path.chmod(0o600)
    assert checkpoint.path.read_bytes() != original
    loaded, _ = load_checkpoint(checkpoint.path)
    assert snapshot(loaded) == snapshot(checkpoint.model)
    assert list(checkpoint.path.parent.iterdir()) == [checkpoint.path]


def test_checkpoint_permission_failure_preserves_destination(monkeypatch, checkpoint):
    import stat

    from tmt import checkpoint as storage

    checkpoint.path.chmod(0o640)
    original = checkpoint.path.read_bytes()
    files = set(checkpoint.path.parent.iterdir())

    def fail(*args, **kwargs):
        raise OSError('injected permission failure')

    monkeypatch.setattr(storage.os, 'chmod', fail)
    with pytest.raises(OSError, match='injected permission failure'):
        storage.save_checkpoint(checkpoint.path, checkpoint.model, checkpoint.settings)
    assert checkpoint.path.read_bytes() == original
    assert stat.S_IMODE(checkpoint.path.stat().st_mode) == 0o640
    assert set(checkpoint.path.parent.iterdir()) == files


@pytest.mark.parametrize(
    'failure', [None, 'constructor', 'read', 'validation', 'interrupt']
)
def test_checkpoint_load_preserves_rng(monkeypatch, checkpoint, failure):
    import random

    import mlx.core as mx

    from tmt import checkpoint as storage

    match failure:
        case 'constructor':
            create = storage.new_model

            def fail(settings, seed):
                create(settings, seed)
                raise ValueError('injected constructor failure')

            monkeypatch.setattr(storage, 'new_model', fail)
        case 'read' | 'interrupt':
            def fail(*args, **kwargs):
                error = KeyboardInterrupt if failure == 'interrupt' else ValueError
                raise error('injected tensor read failure')

            monkeypatch.setattr(mx, 'load', fail)
        case 'validation':
            arrays = dict(checkpoint.arrays)
            arrays.pop('m.decoder.stop.bias')
            mx.save_safetensors(checkpoint.path, arrays, metadata=checkpoint.metadata)

    random.seed(817)
    random.random()
    python_state = random.getstate()
    expected_python = random.random()
    random.setstate(python_state)
    mx.random.seed(913)
    pending = mx.random.uniform(shape=(3,))
    mlx_state = mx.array(mx.random.state[0])
    mx.eval(mlx_state)
    expected_mlx = mx.random.normal(shape=(5,))
    mx.random.state[0][...] = mlx_state

    if failure is None:
        storage.load_checkpoint(checkpoint.path)
    else:
        error = KeyboardInterrupt if failure == 'interrupt' else ValueError
        with pytest.raises(error):
            storage.load_checkpoint(checkpoint.path)
    assert random.getstate() == python_state
    assert mx.array_equal(mx.random.state[0], mlx_state).item()
    assert random.random() == expected_python
    assert mx.array_equal(mx.random.normal(shape=(5,)), expected_mlx).item()
    expected_pending = mx.random.uniform(
        shape=(3,), key=mx.random.split(mx.random.key(913))[1]
    )
    assert mx.array_equal(pending, expected_pending).item()


def test_checkpoint_load_preserves_compiled_rng(checkpoint):
    import mlx.core as mx

    from tmt.checkpoint import load_checkpoint

    draw = mx.compile(
        lambda: mx.random.uniform(shape=(5,)),
        inputs=[mx.random.state],
        outputs=[mx.random.state],
    )
    mx.random.seed(817)
    expected = draw().tolist()
    mx.random.seed(817)
    load_checkpoint(checkpoint.path)
    assert draw().tolist() == expected


@pytest.mark.parametrize('value', [0, -1, float('nan')])
def test_positive_rejects_nonpositive_and_nan(value):
    with pytest.raises(ValueError, match='x must be positive'):
        validation.positive(value, 'x')
