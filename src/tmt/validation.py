"""Shared input and array contracts, with no MLX import at module load."""

import inspect
import json
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_finite_number(value):
    """Accept finite integers and floats, excluding booleans and float overflow."""
    try:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        return False


def validate_keys(actual, expected, label, *, noun='field', extra='unknown'):
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing:
        raise ValueError(f'{label} is missing {noun} {missing[0]!r}')
    if unexpected:
        raise ValueError(f'{label} has {extra} {noun} {unexpected[0]!r}')


def matches_array(value, shape, dtype):
    """Compare array metadata without evaluation or conversion."""
    import mlx.core as mx

    return isinstance(value, mx.array) and value.shape == shape and value.dtype == dtype


def matches_arrays(actual, expected):
    return actual.keys() == expected.keys() and all(
        matches_array(actual[name], value.shape, value.dtype)
        for name, value in expected.items()
    )


def validate_metadata(raw):
    require(isinstance(raw, str), "checkpoint __metadata__.tmt must be a string")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"checkpoint tmt metadata is invalid JSON: {error}") from error
    require(isinstance(value, dict) and set(value) == {'version', 'model'},
            "checkpoint tmt metadata must contain exactly 'version' and 'model'")
    require(type(value['version']) is int and value['version'] == 1,
            'checkpoint.version must be integer 1')
    return validate_model_settings(value['model'], 'checkpoint.model')


def validate_model_settings(settings, label):
    from tmt.main import Model
    from tmt.blocks import BLOCKS
    parameters = inspect.signature(Model).parameters
    require(isinstance(settings, dict), f'{label} must be an object')
    validate_keys(settings, parameters, label)

    if not isinstance(settings['block'], str) or settings['block'] not in BLOCKS:
        choices = ', '.join(BLOCKS)
        raise ValueError(f'{label}.block must be one of: {choices}')

    for name in ('dim', 'layers', 'spread'):
        require(type(settings[name]) is int and settings[name] > 0,
                f'{label}.{name} must be a positive integer')
    for name, strict in (('temp', False), ('rate', True)):
        number = settings[name]
        rule = 'a finite positive number' if strict else 'a finite nonnegative number'
        require(is_finite_number(number) and (number > 0 if strict else number >= 0),
                f'{label}.{name} must be {rule}')
    bound = settings['bound']
    require(isinstance(bound, list) and len(bound) == 2
            and all(type(value) is int for value in bound) and 0 <= bound[0] < bound[1],
            f'{label}.bound must be two integers with 0 <= start < end')

    for name, item in settings.items():
        try:
            json.dumps(item, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{label}.{name} must contain finite JSON values') from error

    for name, parameter in parameters.items():
        if name in {'dim', 'layers', 'spread', 'temp', 'rate', 'bound'}:
            continue
        default = parameter.default
        if default is inspect.Parameter.empty:
            continue
        item = settings[name]
        if isinstance(default, float):
            valid = is_finite_number(item)
        elif isinstance(default, tuple):
            valid = isinstance(item, list)
        else:
            valid = type(item) is type(default)
        require(valid, f'{label}.{name} has an invalid JSON type')
    return settings


def validate_checkpoint_arrays(actual, expected):
    validate_keys(actual, expected, 'checkpoint', noun='tensor', extra='unexpected')
    for name, template in expected.items():
        array = actual[name]
        label = 'model' if name.startswith('m.') else 'optimizer'
        tensor = name[2:]
        if tuple(array.shape) != tuple(template.shape):
            raise ValueError(
                f'{label} tensor {tensor!r} has shape {tuple(array.shape)}, expected {tuple(template.shape)}'
            )
        if array.dtype != template.dtype:
            raise ValueError(
                f'{label} tensor {tensor!r} has dtype {array.dtype}, expected {template.dtype}'
            )


def positive(value, name):
    if value <= 0:
        raise ValueError(f'{name} must be positive')


def validate_seed(value, name='seed'):
    if type(value) is not int:
        raise ValueError(f'{name} must be an integer')
    if not 0 <= value < 2**64:
        raise ValueError(f'{name} must be between 0 and {2**64 - 1}')


def validate_run_name(name):
    require(name and name not in ('.', '..') and not any(char in name for char in ('/', '\\', '\0')),
            'run must be one relative folder name')


def validate_monitoring(sample_every, sample_prompt, sample_bytes, eval_data,
                        eval_every, eval_max_bytes):
    positive(sample_bytes, 'sample-bytes')
    require(sample_prompt, 'sample-prompt must not be empty')
    positive(eval_max_bytes, 'eval-max-bytes')
    if sample_every is not None:
        positive(sample_every, 'sample-every')
    require((eval_data is None) == (eval_every is None), 'eval-data and eval-every must be used together')
    if eval_every is not None:
        positive(eval_every, 'eval-every')


def validate_training_options(seed, sample_seed, updates, log_every):
    validate_seed(seed)
    validate_seed(sample_seed, 'sample-seed')
    positive(updates, 'updates')
    positive(log_every, 'log-every')


def validate_sweep_grid(grid, settings):
    require(isinstance(grid, dict), 'sweep grid must be a JSON object')
    for key, axis in grid.items():
        require(isinstance(axis, list) and axis, f'sweep grid field {key!r} must be a nonempty array')
        for value in axis:
            if key == 'seed':
                validate_seed(value, 'sweep grid seed')
            else:
                candidate = dict(settings)
                candidate[key] = value
                validate_model_settings(candidate, 'model config')
