import pytest

from pathlib import Path
import json

from tmt.runtime.checkpoint import DEFAULT_CONFIG, load_config, group_config, ungroup_config
from tmt.util.validation import ValidationError, tuplify

def test_config_grouping():
    assert group_config(DEFAULT_CONFIG) == {
        'core': {
            'modules': {
                'block': DEFAULT_CONFIG['block'],
                'loss': DEFAULT_CONFIG['loss']
            },
            'dimensions': {
                'dim': DEFAULT_CONFIG['dim'],
                'layers': DEFAULT_CONFIG['layers'],
                'spread': DEFAULT_CONFIG['spread']
            }
        },
        'hparams': {
            'opt': {
                'lower': DEFAULT_CONFIG['lr'][0],
                'upper': DEFAULT_CONFIG['lr'][1],
                'rate': DEFAULT_CONFIG['lr'][2]
            },
            'temp': DEFAULT_CONFIG['temp']
        },
        'custom': {}
    }

@pytest.mark.parametrize(('custom'), [
    {'opensource': 1},
    {'opensource': (1, 'tmt is open source!')},
    {'opensource': {'why': 'because open source is better', 'open source forever?': True, 'old': (1, 'tmt is open source!')}}
], ids = [
    'scalar', 'tuple', 'dict'
])
def test_config_save_load(tmp_path, custom):
    config = {**DEFAULT_CONFIG, **custom}
    path: Path = tmp_path / 'model.json'

    path.write_text(json.dumps(group_config(config)))
    assert load_config(path) == config

    loaded: dict = tuplify(json.loads(path.read_text()))

    assert loaded['custom'] == custom
    loaded['custom'] = {}

    assert ungroup_config(loaded) == DEFAULT_CONFIG

@pytest.mark.parametrize(('config'), [
    ({'dim': 4}),
    ({'model': {'dimensions': {'dim': 4}}, 'dim': 8}),
    ({
        'dim': 512,
        'layers': 16,
        'spread': 32,
        'temp': 0.75,
        'rate': 0.0005,
        'bound': [40000, 120000],
        'block': 'tmt-v0',
    })
], ids = [
    'flat', 'mixed', 'full-flat'
])
def test_config_reject_flat(tmp_path, config):
    path: Path = tmp_path / 'model.json'
    path.write_text(json.dumps(config))

    with pytest.raises(ValidationError, match = 'grouped config'):
        load_config(path)

@pytest.mark.parametrize(('contents', 'message'), [
    ('{', 'invalid json in path'),
    ('[]', 'config json'),
    ('{"unknown": 1}', 'grouped config'),
], ids = [
    'broken', 'list', 'field'
])
def test_config_reject_bad(tmp_path, contents, message):
    path: Path = tmp_path / 'model.json'
    path.write_text(contents)

    with pytest.raises(ValidationError, match = message):
        load_config(path)

@pytest.mark.parametrize(('contents', 'message'), [
    ('{"core": {"dimensions": {"unknown": 1}}}', 'grouped config -> core'),
    ('{"hparams": {"opt": {"unknown": 1}}}', 'grouped config -> hparams'),
    ('{"core": {}, "dim": 4}', 'grouped config'),
    ('{"core": {}, "other": 1}', 'grouped config'),
    ('{"core": []}', 'should be of type'),
    ('{"core": {"tokenizer": {"merges": []}}}', 'grouped config -> core'),
], ids = [
    'dim',
    'opt',
    'flat',
    'unknown',
    'bad-type',
    'tokenizer',
])
def test_config_reject_malgrouped(tmp_path, contents, message):
    path: Path = tmp_path / 'model.json'

    default = group_config(DEFAULT_CONFIG)
    default.update(json.loads(contents))

    path.write_text(json.dumps(default))

    with pytest.raises(ValidationError, match = message):
        load_config(path)

    