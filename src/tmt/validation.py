"""Shared input and array contracts, with no MLX import at module load."""

import json
import math


# Basic values


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_finite_number(value):
    """Accept finite integers and floats, excluding booleans and float overflow."""
    try:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def positive(value, name):
    if value <= 0:
        raise ValueError(f'{name} must be positive')


def keys(actual, expected, label, *, noun='field', extra='unknown'):
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing:
        raise ValueError(f'{label} is missing {noun} {missing[0]!r}')
    if unexpected:
        raise ValueError(f'{label} has {extra} {noun} {unexpected[0]!r}')


# Array contracts


def matches_array(value, shape, dtype):
    """Compare array metadata without evaluation or conversion."""
    import mlx.core as mx

    return isinstance(value, mx.array) and value.shape == shape and value.dtype == dtype


def matches_arrays(actual, expected):
    return actual.keys() == expected.keys() and all(
        matches_array(actual[name], value.shape, value.dtype)
        for name, value in expected.items()
    )


def checkpoint_arrays(actual, expected):
    keys(actual, expected, 'checkpoint', noun='tensor', extra='unexpected')
    for name, template in expected.items():
        array = actual[name]
        label = 'model' if name.startswith('m.') else 'optimizer'
        tensor = name[2:]
        if tuple(array.shape) != tuple(template.shape):
            raise ValueError(
                f'{label} tensor {tensor!r} has shape {tuple(array.shape)}, '
                f'expected {tuple(template.shape)}'
            )
        if array.dtype != template.dtype:
            raise ValueError(
                f'{label} tensor {tensor!r} has dtype {array.dtype}, '
                f'expected {template.dtype}'
            )


# Model settings and checkpoint metadata


def metadata(raw):
    require(isinstance(raw, str), 'checkpoint __metadata__.tmt must be a string')
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f'checkpoint tmt metadata is invalid JSON: {error}') from error
    require(
        isinstance(value, dict) and set(value) == {'version', 'model'},
        "checkpoint tmt metadata must contain exactly 'version' and 'model'",
    )
    require(
        type(value['version']) is int and value['version'] == 1,
        'checkpoint.version must be integer 1',
    )
    return model_settings(value['model'], 'checkpoint.model')


def _model_core_settings(settings, label):
    from tmt.blocks import BLOCKS

    if not isinstance(settings['block'], str) or settings['block'] not in BLOCKS:
        choices = ', '.join(BLOCKS)
        raise ValueError(f'{label}.block must be one of: {choices}')

    for name in ('dim', 'layers', 'spread'):
        require(
            type(settings[name]) is int and settings[name] > 0,
            f'{label}.{name} must be a positive integer',
        )
    for name, strict in (('temp', False), ('rate', True)):
        number = settings[name]
        rule = 'a finite positive number' if strict else 'a finite nonnegative number'
        require(
            is_finite_number(number) and (number > 0 if strict else number >= 0),
            f'{label}.{name} must be {rule}',
        )
    bound = settings['bound']
    require(
        isinstance(bound, list)
        and len(bound) == 2
        and all(type(value) is int for value in bound)
        and 0 <= bound[0] < bound[1],
        f'{label}.bound must be two integers with 0 <= start < end',
    )


def _model_extension_settings(settings, label, defaults):
    core = {'dim', 'layers', 'spread', 'temp', 'rate', 'bound'}
    for name, default in defaults.items():
        if name in core:
            continue
        item = settings[name]
        if isinstance(default, float):
            valid = is_finite_number(item)
        elif isinstance(default, tuple):
            valid = isinstance(item, list)
        else:
            valid = type(item) is type(default)
        require(valid, f'{label}.{name} has an invalid JSON type')


def model_settings(settings, label):
    from tmt.config import MODEL_DEFAULTS

    require(isinstance(settings, dict), f'{label} must be an object')
    keys(settings, MODEL_DEFAULTS, label)
    _model_core_settings(settings, label)

    for name, item in settings.items():
        try:
            json.dumps(item, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'{label}.{name} must contain finite JSON values'
            ) from error

    _model_extension_settings(settings, label, MODEL_DEFAULTS)
    return settings


# Training and sweep options


def seed(value, name='seed'):
    if type(value) is not int:
        raise ValueError(f'{name} must be an integer')
    if not 0 <= value < 2**64:
        raise ValueError(f'{name} must be between 0 and {2**64 - 1}')


def run_name(name):
    require(
        name
        and name not in ('.', '..')
        and not any(char in name for char in ('/', '\\', '\0')),
        'run must be one relative folder name',
    )


def monitoring(
    sample_every, sample_prompt, sample_bytes, eval_data, eval_every, eval_max_bytes
):
    positive(sample_bytes, 'sample-bytes')
    require(sample_prompt, 'sample-prompt must not be empty')
    positive(eval_max_bytes, 'eval-max-bytes')
    if sample_every is not None:
        positive(sample_every, 'sample-every')
    require(
        (eval_data is None) == (eval_every is None),
        'eval-data and eval-every must be used together',
    )
    if eval_every is not None:
        positive(eval_every, 'eval-every')


def training_options(seed_value, sample_seed, updates, log_every):
    seed(seed_value)
    seed(sample_seed, 'sample-seed')
    positive(updates, 'updates')
    positive(log_every, 'log-every')


def sweep_grid(grid, settings):
    require(isinstance(grid, dict), 'sweep grid must be a JSON object')
    for key, axis in grid.items():
        require(
            isinstance(axis, list) and axis,
            f'sweep grid field {key!r} must be a nonempty array',
        )
        for value in axis:
            if key == 'seed':
                seed(value, 'sweep grid seed')
            else:
                candidate = dict(settings)
                candidate[key] = value
                model_settings(candidate, 'model config')


# Block contracts


def block_name(name):
    from tmt.blocks import BLOCKS

    if not isinstance(name, str) or name not in BLOCKS:
        raise ValueError(
            f'unknown block {name!r}; valid blocks: {", ".join(sorted(BLOCKS))}'
        )
    return name


def block_runtime_schema(block, trainable):
    require(
        isinstance(block.state_fields, tuple) and isinstance(block.trace_fields, tuple),
        'block state and trace fields must be tuples',
    )
    fields = block.state_fields + block.trace_fields
    require(
        all(isinstance(name, str) and not name.startswith('_') for name in fields)
        and len(set(fields)) == len(fields),
        'block state and trace fields must be distinct public names in tuples',
    )
    for label in ('temporal_gradient', 'feature_convention'):
        require(
            isinstance(getattr(block, label), str) and getattr(block, label),
            f'block {label} must be a nonempty string',
        )
    import mlx.core as mx

    schema = []
    for name in fields:
        value = getattr(block, name, None)
        require(
            isinstance(value, mx.array),
            f'block runtime field {name!r} must be an array',
        )
        require(name not in trainable, f'block runtime field {name!r} must be frozen')
        schema.append((name, value.shape, value.dtype))
    return tuple(schema)


def block_runtime_values(values, fields, schema):
    require(
        isinstance(values, dict) and set(values) == set(fields),
        f'block state schema must contain exactly {fields!r}',
    )
    field_schema = {name: (shape, dtype) for name, shape, dtype in schema}
    for name, value in values.items():
        shape, dtype = field_schema[name]
        require(
            matches_array(value, shape, dtype),
            f'block runtime field {name!r} must retain shape {shape} and dtype {dtype}',
        )


def block_output(output, input_vector):
    require(
        matches_array(output, input_vector.shape, input_vector.dtype),
        'block output must retain the input vector shape and dtype',
    )
