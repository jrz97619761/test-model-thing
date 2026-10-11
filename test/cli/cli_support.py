import json


def model_snapshot(model):
    import mlx.core as mx
    import mlx.utils as util

    snapshot = {
        'parameters': {
            name: mx.array(value)
            for name, value in util.tree_flatten(model.trainable_parameters())
        },
        'optimizer': {
            name: mx.array(value)
            for name, value in util.tree_flatten(model.optimizer.state)
        },
        'runtime_state': model.snapshot_state(),
    }
    mx.eval(*snapshot['parameters'].values(), *snapshot['optimizer'].values())
    mx.eval(*(value for block in snapshot['runtime_state'] for value in block.values()))
    return snapshot


def assert_snapshots_equal(actual, expected):
    import mlx.core as mx

    for group in ('parameters', 'optimizer'):
        assert actual[group].keys() == expected[group].keys(), (
            f'{group} tensor names differ'
        )
        for name, expected_value in expected[group].items():
            actual_value = actual[group][name]
            assert mx.array_equal(actual_value, expected_value).item(), (
                f'{group} tensor {name!r} differs'
            )

    assert len(actual['runtime_state']) == len(expected['runtime_state']), (
        'runtime state block count differs'
    )
    for block_index, (actual_block, expected_block) in enumerate(
        zip(actual['runtime_state'], expected['runtime_state'])
    ):
        assert actual_block.keys() == expected_block.keys(), (
            f'runtime state fields differ in block {block_index}'
        )
        for name, expected_value in expected_block.items():
            actual_value = actual_block[name]
            assert mx.array_equal(actual_value, expected_value).item(), (
                f'runtime state tensor {name!r} differs in block {block_index}'
            )


def assert_model_snapshot(model, snapshot):
    assert_snapshots_equal(model_snapshot(model), snapshot)


def next_global_random_value(mx):
    value = mx.random.uniform(shape=(1,))
    mx.eval(value)
    return value


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def assert_checkpoint_rejected(path, mx, load_model, arrays, metadata, *parts):
    import pytest

    mx.save_safetensors(path, arrays, metadata=metadata)
    with pytest.raises(ValueError) as error:
        load_model(path)
    message = str(error.value)
    for part in parts:
        assert part in message, f'error message is missing {part!r}: {message}'


def assert_fields(actual, expected, *, context):
    for field, value in expected.items():
        assert field in actual, f'{context} is missing field {field!r}'
        assert actual[field] == value, f'{context} field {field!r} differs'


def assert_ordered_records(actual, expected, *, context):
    assert len(actual) == len(expected), f'{context} row count differs'
    for index, (row, expected_row) in enumerate(zip(actual, expected)):
        assert_fields(row, expected_row, context=f'{context} row {index}')
