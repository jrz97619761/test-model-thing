from __future__ import annotations

import glob
import json
import os
import random
import stat
import tempfile

from typing import Any, Generator, Iterator, Literal
from pathlib import Path
from contextlib import contextmanager

import tmt.util.validation as validation
import tmt.model.model as model
import tmt.model.blocks as blocks
import tmt.model.tokenizers as tokenizers

# seed

@contextmanager
def preserve_seed() -> Generator[None, None, None]:
    import mlx.core as mx

    pystate = random.getstate()
    mxstate = mx.random.state[0]

    try: yield
    finally:
        random.setstate(pystate)
        mx.random.state[0][...] = mxstate

def seed(s: int) -> None:
    import mlx.core as mx

    validation.seed(s)
    random.seed(s)
    mx.random.seed(s)

# defaults

DEFAULT_CONFIG: blocks.Config = {
    'dim': 512,
    'layers': 16,
    'spread': 32,

    'block': 'tmt0',
    'loss': 'tmt0',

    'temp': 0.75,
    'lr': (40000, 120000, 0.0005)
}

DEFAULT_GRID: dict[str, list[Any]] = {
    'dim': [512, 1024],
    'layers': [16],
    'spread': [32],

    'block': ['tmt0'],
    'loss': ['tmt0'],

    'seed': [11, 22]
}

CONFIG_VERSION = 2

# config handling

def validate_config(config: blocks.Config) -> None:
    validation.instance(config, blocks.Config, 'model config')

    for key, value in config.items():
        validation.require(
            (DEFAULT_CONFIG.get(key) is None) or type(value) == type(DEFAULT_CONFIG[key]),
            f'model config has type mismatch at key \'{key}\''
        ) # custom blocks and losses can have custom config params

    for key, registry in zip(('block', 'loss'), (blocks.BLOCKS, blocks.LOSSES)): validation.key_exists(config[key], registry)
    for key in ('dim', 'layers', 'spread'): validation.positive(config[key], f'config key \'{key}\'')
    validation.positive(config['temp'], 'config key \'temp\'', strict = False)

    for value in config['lr']: validation.positive(value, 'config learning rate key', strict = False)
    validation.require(config['lr'][1] > config['lr'][0], 'invalid learning rate bounds')

    for value, name in zip((config['temp'], config['lr'][2]), ('temp', 'lr rate')): validation.finite(value, name)

def validate_headers(path: validation.FilePath) -> tuple[blocks.Config, tokenizers.Merges | None]:
    with validation.trycatch((UnicodeDecodeError, json.JSONDecodeError), 'failed to decode checkpoint header', True):
        # decode envelope

        with open(path, 'rb') as source:
            size = os.fstat(source.fileno()).st_size

            length = source.read(8)
            validation.require(len(length) == 8, 'truncated checkpoint header length')

            length = int.from_bytes(length, 'little')
            validation.require(length != 0 and length <= size - 8, 'invalid checkpoint header length')

            raw = source.read(length)
            validation.require(len(raw) == length, 'truncated checkpoint header')

        envelope = json.loads(raw.decode('utf-8'))

    # validate header

    validation.instance(envelope, dict, 'checkpoint header')
    validation.require(bool(envelope), 'checkpoint header object must be nonempty')

    meta = envelope.get('__metadata__')
    validation.instance(meta, dict, 'checkpoint metadata')
    validation.instance(meta.get('tmt'), str, 'checkpoint metadata')

    validation.require(any(key != '__metadata__' for key in envelope), 'checkpoint has no tensors')

    # validate config

    with validation.trycatch(ValueError, 'checkpoint config is invalid json'):
        raw_config: dict[str, Any] = json.loads(meta.get('tmt'))

    validation.instance(raw_config, dict, 'checkpoint config')
    validation.instance(raw_config.get('model'), dict, 'checkpoint model config')

    validation.instance(raw_config.get('version'), int, 'checkpoint version')
    validation.require(raw_config.get('version') == CONFIG_VERSION, 'checkpoint version is not 2')

    validation.instance(raw_config.get('tokenizer'), dict, 'tokenizer metadata')

    return {
        key: validation.tuplify(value)
        for key, value in raw_config['model'].items()
    }, raw_config['tokenizer'].get('merges')

def dump_headers(config: blocks.Config, merges: tokenizers.Merges | None):
    return json.dumps({'version': CONFIG_VERSION, 'model': config, 'tokenizer': {'merges': merges}})

def group_config(config: blocks.Config) -> dict[str, Any]:
    validate_config(config)

    return {
        'core': {
            'modules': {
                'block': config['block'],
                'loss': config['loss']
            },
            'dimensions': {
                'dim': config['dim'],
                'layers': config['layers'],
                'spread': config['spread']
            }
        },
        'hparams': {
            'opt': {
                'lower': config['lr'][0],
                'upper': config['lr'][1],
                'rate': config['lr'][2]
            },
            'temp': config['temp']
        },
        'custom': {name: value for name, value in config.items() if name not in DEFAULT_CONFIG}
    }

def ungroup_config(config: dict[str, Any]) -> blocks.Config:
    validation.key_difference(config, ['core', 'hparams'], 'grouped config', strict = False)

    validation.key_difference(config['core'], ['modules', 'dimensions'], 'grouped config -> core')
    validation.key_difference(config['core']['modules'], ['block', 'loss'], 'grouped config -> core -> modules')
    validation.key_difference(config['core']['dimensions'], ['dim', 'layers', 'spread'], 'grouped config -> core -> dimensions')

    validation.key_difference(config['hparams'], ['opt', 'temp'], 'grouped config -> hparams')
    validation.key_difference(config['hparams']['opt'], ['lower', 'upper', 'rate'], 'grouped config -> hparams -> opt')

    flattened = {}

    flattened.update(config['core']['modules'])
    flattened.update(config['core']['dimensions'])

    flattened['temp'] = config['hparams']['temp']
    flattened['lr'] = (
        config['hparams']['opt']['lower'], config['hparams']['opt']['upper'], config['hparams']['opt']['rate']
    )

    for key in config.get('custom', {}):
        validation.require(key not in flattened, 'custom model config attempts to override base config option')

        value = config['custom'][key]
        flattened[key] = validation.tuplify(value)

    return flattened

def load_config(path: validation.FilePath) -> blocks.Config:
    data = read_json(path)
    validation.instance(data, dict, 'config json')

    config = {**DEFAULT_CONFIG, **ungroup_config(data)}
    validate_config(config)

    return config

# tree handling

def flatten(tree: Any, key: str) -> dict[str, Any]:
    import mlx.utils as util
    return {f'{key}.{name}': value for name, value in util.tree_flatten(tree)}

def unflatten(tensors: dict[str, Any], key: str) -> Any:
    import mlx.utils as util

    return util.tree_unflatten([
        (name[len(key) + 1:], value)
        for name, value in tensors.items()
        if name.startswith(f'{key}.')
    ])

# save load

def save(path: validation.FilePath, m: model.Model, config: blocks.Config, merges: tokenizers.Merges | None):
    import mlx.core as mx

    validate_config(config)
    meta = dump_headers(config, merges)

    m.optimizer.opt.init(m.trainable_parameters())
    parameters = flatten(m.parameters(), 'm')
    parameters.update(flatten(m.optimizer.opt.state, 'o'))
    m.optimizer.evaluate(m)

    path = Path(path)
    validation.file_parent(path)

    with tempfile.NamedTemporaryFile(
        dir = path.parent, prefix = f'.{path.name}.', suffix = '.safetensors', delete = False
    ) as file: temp = Path(file.name)

    try:
        mx.save_safetensors(temp, parameters, metadata = {'tmt': meta})
        with temp.open('rb') as file:
            try: mode = stat.S_IMODE(path.stat().st_mode)
            except FileNotFoundError: pass
            else: temp.chmod(mode)

            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok = True)

def load(path: validation.FilePath) -> tuple[model.Model, blocks.Config]:
    import mlx.core as mx

    config, merges = validate_headers(path)
    validate_config(config)

    with preserve_seed():
        seed(0)
        m = model.Model(config, merges)
        m.optimizer.opt.init(m.trainable_parameters())

        expected = flatten(m.parameters(), 'm')
        expected.update(flatten(m.optimizer.opt.state, 'o'))

        with validation.trycatch(Exception, 'invalid checkpoint', True):
            tensors = mx.load(path)

        validation.arrays_match(tensors, expected, 'checkpoint tensors')

        m.update(unflatten(tensors, 'm'))
        m.optimizer.opt.state = unflatten(tensors, 'o')
        m.optimizer.compiled = None

        m.optimizer.evaluate(m)
        return m, config

# jsons

def read_json(path: validation.FilePath) -> Any:
    with open(path) as file:
        with validation.trycatch(json.JSONDecodeError, f'invalid json in path \'{path}\'', True):
            return json.load(file)

def write_json(path: validation.FilePath, obj: Any, exclusive: bool = False) -> Any:
    with open(path, 'x' if exclusive else 'w') as file:
        json.dump(obj, file, indent = 4)
        file.write('\n')

    return obj

def append_jsonl(path: validation.FilePath, row: Any) -> Any:
    with open(path, 'a') as file:
        file.write(json.dumps(row) + '\n')

    return row

# byte data handling

class ByteFiles:
    def __init__(self, pattern: validation.FilePath) -> None:
        self.paths = sorted(glob.glob(pattern, recursive = True))
        validation.require(self.paths, f'no files matched glob \'{pattern!r}\'')

        for path in self.paths:
            if Path(path).is_dir(): continue
            with open(path, 'rb') as file: file.read(1)

        self.pattern = str(pattern)

    def get(self, size: int = -1) -> Iterator[bytes]:
        remaining = size

        for path in self.paths:
            if Path(path).is_dir(): continue

            if remaining == 0: break

            data = Path(path).read_bytes()
            if not data: continue

            if remaining > 0:
                data = data[:remaining]
                remaining -= len(data)

            yield data

    def validate(self, size: int = -1) -> ByteFiles:
        targets = sum(max(len(document), 0) for document in self.get(size))
        validation.require(targets, 'byte documents require at least 1 byte pair')

        return self

    def shuffle(self, seed: int) -> ByteFiles:
        import random
        random.Random(seed).shuffle(self.paths)

        return self

    def filenames(self) -> list[str]:
        return [str(Path(path).resolve()) for path in self.paths]

    @staticmethod
    def load(pattern: validation.FilePath, size: int = -1) -> Iterator[bytes]:
        yield from ByteFiles(pattern).get(size)

class DummyByteFiles:
    def __init__(self, pattern: validation.FilePath) -> None:
        self.pattern = pattern
        
    def filenames(self) -> list[str]: return []

# other

def stamp(mode: Literal['iso', 'file']):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    match mode:
        case 'iso': return now.isoformat()
        case 'file': return now.strftime('%Y%m%dT%H%M%S.%fZ')