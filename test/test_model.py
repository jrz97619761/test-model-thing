import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / 'fixtures' / 'rtu-baseline.json'


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
    from tmt.model import Model

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
import tmt.model
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


def test_rtu_baseline():
    """Compare seeded RTU updates with the saved fixture.

    After an intentional RTU change, rewrite the fixture and review it:
        TMT_RECORD_RTU_BASELINE=1 python -m pytest test/test_model.py -k rtu_baseline
        git diff test/fixtures/rtu-baseline.json
    """
    fixture = json.loads(FIXTURE.read_text())
    keys = ('config', 'seed', 'steps', 'output_indices', 'state_rows')
    actual_updates = capture_rtu_baseline(*(fixture[key] for key in keys))
    if os.environ.get('TMT_RECORD_RTU_BASELINE') == '1':
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


@pytest.fixture
def loss_case(model_factory):
    import math

    import mlx.core as mx

    model = model_factory()
    x = [0.0, 0.5, 1.0, 1.5]
    output = mx.arange(256) * 0.001
    stop = mx.array([0.2])
    target = model.encoder(mx.array(67))
    mx.eval(output, stop, target)

    logits = [float(value) for value in output.tolist()]
    x_mean = sum(x) / len(x)
    variance = sum((value - x_mean) ** 2 for value in x) / len(x)
    variance_term = max(0.0, 1.0 - math.sqrt(variance + 1e-4))
    latent = [float(value) for value in target.tolist()]
    latent_term = sum((left - right) ** 2 for left, right in zip(x, latent)) / len(x)
    maximum = max(logits)
    logsumexp = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
    cross_entropy = -logits[67] + logsumexp
    stop_value = float(stop.item())
    terms = {
        'variance': variance_term,
        'latent_prediction': latent_term,
        'cross_entropy': cross_entropy,
        'stop': (stop_value - 1.0) ** 2,
    }
    return model, mx.array(x), output, stop, terms


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
    components = {}
    actual = model.loss(x, output, stop, target, end, ce_only, components)
    assert float(actual) == pytest.approx(expected, rel=0, abs=1e-5)
    scored = target is not None and not ce_only
    expected_components = {
        'variance': 0.0 if ce_only else terms['variance'],
        'latent_prediction': terms['latent_prediction'] if scored else 0.0,
        'cross_entropy': terms['cross_entropy'] if target is not None else 0.0,
        'stop': (0.2 - float(end)) ** 2 if scored else 0.0,
    }
    assert components.keys() == expected_components.keys()
    for name, value in expected_components.items():
        assert float(components[name]) == pytest.approx(value, rel=0, abs=1e-5), name


def test_rtu_gradients(monkeypatch):
    import mlx.core as mx
    from tmt.config import load_config
    from tmt.model import Model

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
    assert analytic_decay == pytest.approx(numerical_decay, rel=0, abs=0.002)
    assert analytic_embed == pytest.approx(numerical_embed, rel=0, abs=0.002)
