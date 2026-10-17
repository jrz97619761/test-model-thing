import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest


def _array_values(array):
    import mlx.core as mx

    mx.eval(array)
    return [float(value) for value in array.tolist()]


def _flatten_values(value):
    if isinstance(value, list):
        for item in value:
            yield from _flatten_values(item)
    else:
        yield float(value)


def _gradient_summary(array):
    values = list(_flatten_values(array.tolist()))
    return {
        'sum': sum(values),
        'sum_squares': sum(value * value for value in values),
        'max_abs': max((abs(value) for value in values), default=0.0),
    }


def capture_rtu_baseline(config, seed, steps, output_indices, state_rows):
    import mlx.core as mx
    from tmt.main import Model

    random.seed(seed)
    mx.random.seed(seed)
    model = Model(**config)
    mx.eval(model.parameters())

    updates = []
    captured_gradients = []
    original_updategrads = model.updategrads

    def capture_updategrads(grads):
        gradients = {
            f'blocks.{index}.decay': _array_values(block_grads['decay'])
            for index, block_grads in enumerate(grads['blocks'])
        }
        encoder_gradient = grads['encoder']['embed']['weight']
        mx.eval(encoder_gradient)
        gradients['encoder.embed.weight'] = _gradient_summary(encoder_gradient)
        gradients['decoder.stop.bias'] = _array_values(grads['decoder']['stop']['bias'])
        captured_gradients.append(gradients)
        original_updategrads(grads)
        mx.eval(model.parameters(), model.optimizer.state)

    model.updategrads = capture_updategrads

    for step in steps:
        record = {'input': step['input'], 'target': step['target'], 'end': step['end']}
        updates.append(record)
        metrics = {}
        output, stop, loss = model.train_step(
            step['input'], step['target'], step['end'], metrics=metrics
        )
        mx.eval(output, stop, loss, *metrics.values())
        record['logits'] = {
            str(index): float(output[index].item()) for index in output_indices
        }
        record['prediction'] = int(mx.argmax(output).item())
        record['stop'] = float(stop.item())
        record['loss'] = float(loss.item())
        record['components'] = {name: metrics[name] for name in sorted(metrics)}
        record['gradients'] = captured_gradients.pop()
        record['state'] = {
            f'blocks.{index}': {
                'states': _array_values(layer.states),
                'decaytrace': _array_values(layer.decaytrace),
                'embedtrace_rows': {
                    str(byte): _array_values(layer.embedtrace[byte])
                    for byte in state_rows
                },
            }
            for index, layer in enumerate(model.blocks)
        }

    return updates


def test_import_has_no_actions():
    root = Path(__file__).resolve().parents[1]
    code = """
import argparse, builtins
def blocked(*args, **kwargs): raise AssertionError('import performed an action')
builtins.open = blocked
builtins.input = blocked
argparse.ArgumentParser = blocked
import tmt.main
"""
    result = subprocess.run(
        [sys.executable, '-c', code], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_train_step(monkeypatch, model_factory):
    import mlx.core as mx

    model = model_factory()
    mx.eval(model.parameters())
    mx.random.seed(23)
    expected = mx.random.uniform(shape=(1,)).item()
    mx.random.seed(23)

    def reject_sample(*args, **kwargs):
        pytest.fail('train_step sampled')

    monkeypatch.setattr(model, 'sample', reject_sample)
    output, stop, loss = model.train_step(97, 98, False)
    assert (output.shape, stop.shape) == ((256,), (1,))
    assert mx.random.uniform(shape=(1,)).item() == expected
    assert float(loss.item()) == 7.2099761962890625
    assert math.isfinite(float(loss.item()))
    assert int(model.optimizer.state['step'].item()) == 1

    mx.random.seed(29)
    expected = mx.random.uniform(shape=(1,)).item()
    mx.random.seed(29)
    output, stop, loss = model.train_step(98, 99, True)
    assert mx.random.uniform(shape=(1,)).item() == expected
    assert math.isfinite(float(loss.item()))
    assert int(model.optimizer.state['step'].item()) == 2


FIXTURE = Path(__file__).parent / 'fixtures' / 'rtu-baseline.json'


def test_rtu_baseline():
    """Compare seeded RTU updates with the saved fixture.

    After an intentional RTU change, rewrite the fixture and review it:
        TMT_RECORD_RTU_BASELINE=1 python -m pytest test/test_model.py -k rtu_baseline
        git diff test/fixtures/rtu-baseline.json
    """
    fixture = json.loads(FIXTURE.read_text())
    keys = ('config', 'seed', 'steps', 'output_indices', 'state_rows')
    actual_updates = capture_rtu_baseline(*(fixture[key] for key in keys))
    if os.environ.get('TMT_RECORD_RTU_BASELINE'):
        FIXTURE.write_text(
            json.dumps({**fixture, 'updates': actual_updates}, indent=2) + '\n'
        )
        pytest.skip('recorded a new RTU baseline')

    def assert_reference(actual, expected, path='fixture'):
        if isinstance(expected, dict):
            assert actual.keys() == expected.keys(), (
                f'{path}: got fields {list(actual)}, expected {list(expected)}'
            )
            for key in expected:
                assert_reference(actual[key], expected[key], f'{path}.{key}')
        elif isinstance(expected, list):
            assert len(actual) == len(expected), path
            for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                assert_reference(actual_item, expected_item, f'{path}[{index}]')
        elif isinstance(expected, bool):
            assert actual is expected, path
        elif isinstance(expected, (int, str)):
            assert actual == expected, path
        else:
            assert math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6), path

    assert_reference(actual_updates, fixture['updates'])


def test_byte_io(model_factory):
    import mlx.core as mx

    model = model_factory()
    mx.eval(model.parameters())
    encoded = model.encoder(mx.array([0, 255]))
    logits, stop = model.decoder(encoded)
    assert (encoded.shape, logits.shape, stop.shape) == ((2, 4), (2, 256), (2, 1))


def test_recurrence(model_factory):
    import mlx.core as mx

    model = model_factory()
    model.reset()
    layer = model.blocks[0]
    byte = mx.array(65)
    encoded = model.encoder(byte)
    decay = mx.sigmoid(layer.decay)
    expected_state = mx.array(encoded)
    (_, first_states, _), _ = model.step(byte, frozen=True)
    assert mx.allclose(first_states[0]['states'], expected_state, atol=1e-06).item()
    expected_state = decay * expected_state + encoded
    (_, second_states, _), _ = model.step(byte, frozen=True)
    assert mx.allclose(second_states[0]['states'], expected_state, atol=1e-06).item()


@pytest.mark.parametrize('field', ['states', 'decaytrace', 'embedtrace'])
def test_reset(model_factory, field):
    import mlx.core as mx

    model = model_factory()
    layer = model.blocks[0]
    model.train_step(65, 66, False)
    mx.eval(layer.embedtrace)
    assert float(mx.sum(layer.embedtrace[65]).item()) > 0.0
    model.reset()
    assert float(mx.sum(mx.abs(getattr(layer, field))).item()) == 0.0


@pytest.mark.parametrize(
    'target,end,ce_only',
    [
        pytest.param(67, True, False, id='end'),
        pytest.param(67, False, False, id='open'),
        pytest.param(67, False, True, id='ce-only'),
        pytest.param(None, False, False, id='no-target'),
        pytest.param(None, False, True, id='ce-no-target'),
    ],
)
def test_loss(loss_case, target, end, ce_only):
    model, x, output, stop, terms = loss_case
    if target is None:
        expected = 0.0 if ce_only else terms['variance']
    elif ce_only:
        expected = terms['cross_entropy']
    else:
        expected = sum(terms.values()) - terms['stop']
        expected += (float(stop.item()) - float(end)) ** 2
    actual = model.loss(x, output, stop, target, end, ce_only)
    assert float(actual) == pytest.approx(expected, rel=0, abs=1e-5)


@pytest.mark.parametrize(
    'name', ['variance', 'latent_prediction', 'cross_entropy', 'stop']
)
@pytest.mark.parametrize('ce_only', [False, True], ids=['full', 'ce-only'])
def test_loss_components(loss_case, name, ce_only):
    model, x, output, stop, terms = loss_case
    components = {}
    model.loss(x, output, stop, 67, True, ce_only, components)
    expected = 0.0 if ce_only and name != 'cross_entropy' else terms[name]
    assert float(components[name]) == pytest.approx(expected, rel=0, abs=1e-5)


def test_rtu_gradients(monkeypatch, assert_almost_equal):
    import mlx.core as mx
    from tmt.cli import load_config
    from tmt.main import Model

    random.seed(11)
    mx.random.seed(11)
    settings = {**load_config(), 'dim': 4, 'layers': 1, 'spread': 4}
    model = Model(**settings)
    mx.eval(model.parameters())
    captured = []

    def capture(grads):
        captured.append(
            {
                'embed': mx.array(grads['encoder']['embed']['weight']),
                'decay': mx.array(grads['blocks'][0]['decay']),
            }
        )
        mx.eval(*captured[-1].values())

    monkeypatch.setattr(model, 'updategrads', capture)
    model.train_step(65, 66, False, True)
    layer = model.blocks[0]
    previous_state = mx.array(layer.states)
    previous_decaytrace = mx.array(layer.decaytrace)
    previous_embedtrace = mx.array(layer.embedtrace)
    decay = mx.sigmoid(layer.decay)
    mx.eval(previous_state, previous_decaytrace, previous_embedtrace, decay)
    model.train_step(66, 67, True, True)

    expected_decaytrace = (
        decay * previous_decaytrace + decay * (1.0 - decay) * previous_state
    )
    expected_embedtrace = previous_embedtrace * decay + (mx.arange(256) == 66)[
        :, None
    ].astype(mx.float32)
    for field, expected_trace in (
        ('decaytrace', expected_decaytrace),
        ('embedtrace', expected_embedtrace),
    ):
        assert mx.allclose(getattr(layer, field), expected_trace, atol=1e-06).item(), (
            f'{field} does not match the RTU recurrence'
        )

    epsilon = 0.01
    direction = mx.array([1.0, -0.5, 0.25, 0.75])

    def objective(decay_shift=0.0, embed_shift=0.0):
        random.seed(11)
        mx.random.seed(11)
        candidate = Model(**settings)
        params = candidate.trainable_parameters()
        params['blocks'][0]['decay'] = (
            params['blocks'][0]['decay'] + direction * decay_shift
        )
        row = (mx.arange(256) == 65)[:, None].astype(mx.float32)
        params['encoder']['embed']['weight'] = (
            params['encoder']['embed']['weight']
            + row * (direction * embed_shift)[None, :]
        )
        candidate.update(params)
        candidate.step(mx.array(65), frozen=True)
        (x, _, _), (logits, stop) = candidate.step(mx.array(66), frozen=True)
        loss = candidate.loss(x, logits, stop, 67, False, True)
        mx.eval(loss)
        return float(loss.item())

    numerical_decay = (
        objective(decay_shift=epsilon) - objective(decay_shift=-epsilon)
    ) / (2 * epsilon)
    analytic_decay = float(mx.sum(captured[1]['decay'] * direction).item())
    numerical_embed = (
        objective(embed_shift=epsilon) - objective(embed_shift=-epsilon)
    ) / (2 * epsilon)
    analytic_embed = float(mx.sum(captured[1]['embed'][65] * direction).item())
    assert_almost_equal(analytic_decay, numerical_decay, delta=0.002)
    assert_almost_equal(analytic_embed, numerical_embed, delta=0.002)


def test_frozen_step():
    import mlx.core as mx
    import mlx.utils as util
    from tmt.cli import load_config
    from tmt.main import Model

    settings = {**load_config(), 'dim': 4, 'layers': 1, 'spread': 4}
    model = Model(**settings)
    model.train_step(65, 66, False)
    weights = {
        name: mx.array(value)
        for name, value in util.tree_flatten(model.trainable_parameters())
    }
    optimizer = {
        name: mx.array(value)
        for name, value in util.tree_flatten(model.optimizer.state)
    }
    before_state = mx.array(model.blocks[0].states)
    traces = {
        name: mx.array(getattr(model.blocks[0], name))
        for name in ('decaytrace', 'embedtrace')
    }
    mx.eval(*weights.values(), *optimizer.values(), before_state, *traces.values())

    model.step(mx.array(67), frozen=True)
    current_weights = dict(util.tree_flatten(model.trainable_parameters()))
    current_optimizer = dict(util.tree_flatten(model.optimizer.state))
    mx.eval(
        *current_weights.values(),
        *current_optimizer.values(),
        model.blocks[0].states,
        model.blocks[0].decaytrace,
        model.blocks[0].embedtrace,
    )
    current_traces = {name: getattr(model.blocks[0], name) for name in traces}
    for group, expected, actual in (
        ('weights', weights, current_weights),
        ('optimizer', optimizer, current_optimizer),
        ('traces', traces, current_traces),
    ):
        assert actual.keys() == expected.keys(), (
            f'frozen step changed {group} tensor names'
        )
        for name, value in expected.items():
            assert mx.array_equal(value, actual[name]).item(), (
                f'frozen step changed {group} tensor {name!r}'
            )
    assert not mx.array_equal(before_state, model.blocks[0].states).item()
