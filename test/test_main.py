import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest


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
    result = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True)
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
    import mlx.core as mx
    import mlx.utils as util
    from tmt.main import Model

    fixture_path = Path(__file__).parent / 'fixtures' / 'rtu-baseline.json'
    fixture = json.loads(fixture_path.read_text())
    seed = fixture['seed']
    random.seed(seed)
    mx.random.seed(seed)
    config = fixture['config']
    model = Model(**config)
    mx.eval(model.parameters())

    def values(array):
        nested = array.tolist()
        result = []

        def visit(item):
            if isinstance(item, list):
                for child in item:
                    visit(child)
            else:
                result.append(float(item))

        visit(nested)
        return result

    def summary(array):
        mx.eval(array)
        flattened = values(array)
        return {
            'shape': list(array.shape),
            'sum': sum(flattened),
            'sum_squares': sum(value * value for value in flattened),
            'max_abs': max((abs(value) for value in flattened), default=0.0),
            'head': flattened[:6],
            'tail': flattened[-6:],
        }

    def summary_tree(tree):
        return {str(path): summary(value) for path, value in util.tree_flatten(tree)}

    actual_updates = []
    original_updategrads = model.updategrads

    def capture_updategrads(grads):
        record = actual_updates[-1]
        record['compiled_before_update'] = model.compiled is not None
        record['gradients'] = summary_tree(grads)
        original_updategrads(grads)
        mx.eval(model.parameters(), model.optimizer.state)
        record['optimizer'] = summary_tree(model.optimizer.state)

    model.updategrads = capture_updategrads
    for currb, nextb, end in ((65, 66, False), (66, 67, True), (67, 68, False)):
        record = {'input': currb, 'target': nextb, 'end': end}
        actual_updates.append(record)
        metrics = {}
        output, stop, loss = model.train_step(currb, nextb, end, metrics=metrics)
        mx.eval(output, stop, loss, *metrics.values())
        record['loss'] = float(loss.item())
        record['components'] = metrics
        record['output'] = values(output)
        record['stop'] = values(stop)
        record['state'] = {
            f'blocks.{i}': {
                name: summary(getattr(layer, name))
                for name in ('states', 'decaytrace', 'embedtrace')
            }
            for i, layer in enumerate(model.blocks)
        }
        record['active_embedtrace_rows'] = {
            f'blocks.{i}': {
                str(byte): values(layer.embedtrace[byte])
                for byte in (65, 66, 67)
            }
            for i, layer in enumerate(model.blocks)
        }

    def assert_reference(actual, expected, path='fixture'):
        if isinstance(expected, dict):
            assert actual.keys() == expected.keys(), path
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


@pytest.fixture
def loss_case(model_factory):
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


@pytest.mark.parametrize('target,end,ce_only', [
    pytest.param(67, True, False, id='end'),
    pytest.param(67, False, False, id='open'),
    pytest.param(67, False, True, id='ce-only'),
    pytest.param(None, False, False, id='no-target'),
    pytest.param(None, False, True, id='ce-no-target'),
])
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


@pytest.mark.parametrize('name', ['variance', 'latent_prediction', 'cross_entropy', 'stop'])
@pytest.mark.parametrize('ce_only', [False, True], ids=['full', 'ce-only'])
def test_loss_components(loss_case, name, ce_only):
    model, x, output, stop, terms = loss_case
    components = {}
    model.loss(x, output, stop, 67, True, ce_only, components)
    expected = 0.0 if ce_only and name != 'cross_entropy' else terms[name]
    assert float(components[name]) == pytest.approx(expected, rel=0, abs=1e-5)


def test_rtu_gradients(monkeypatch, assert_almost_equal):
    import mlx.core as mx
    from tmt.main import Model

    random.seed(11)
    mx.random.seed(11)
    model = Model(dim=4, layers=1, spread=4)
    mx.eval(model.parameters())
    captured = []
    def capture(grads):
        captured.append({
            'embed': mx.array(grads['encoder']['embed']['weight']),
            'decay': mx.array(grads['blocks'][0]['decay']),
        })
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

    expected_decaytrace = decay * previous_decaytrace + decay * (1.0 - decay) * previous_state
    expected_embedtrace = previous_embedtrace * decay + (mx.arange(256) == 66)[:, None].astype(mx.float32)
    assert mx.allclose(layer.decaytrace, expected_decaytrace, atol=1e-06).item()
    assert mx.allclose(layer.embedtrace, expected_embedtrace, atol=1e-06).item()

    epsilon = 0.01
    direction = mx.array([1.0, -0.5, 0.25, 0.75])
    def objective(decay_shift=0.0, embed_shift=0.0):
        random.seed(11)
        mx.random.seed(11)
        candidate = Model(dim=4, layers=1, spread=4)
        params = candidate.trainable_parameters()
        params['blocks'][0]['decay'] = params['blocks'][0]['decay'] + direction * decay_shift
        row = (mx.arange(256) == 65)[:, None].astype(mx.float32)
        params['encoder']['embed']['weight'] = (
            params['encoder']['embed']['weight'] + row * (direction * embed_shift)[None, :]
        )
        candidate.update(params)
        candidate.step(mx.array(65), frozen=True)
        (x, _, _), (logits, stop) = candidate.step(mx.array(66), frozen=True)
        loss = candidate.loss(x, logits, stop, 67, False, True)
        mx.eval(loss)
        return float(loss.item())

    numerical_decay = (objective(decay_shift=epsilon) - objective(decay_shift=-epsilon)) / (2 * epsilon)
    analytic_decay = float(mx.sum(captured[1]['decay'] * direction).item())
    numerical_embed = (objective(embed_shift=epsilon) - objective(embed_shift=-epsilon)) / (2 * epsilon)
    analytic_embed = float(mx.sum(captured[1]['embed'][65] * direction).item())
    assert_almost_equal(analytic_decay, numerical_decay, delta=0.002)
    assert_almost_equal(analytic_embed, numerical_embed, delta=0.002)


def test_frozen_step():
    import mlx.core as mx
    import mlx.utils as util
    from tmt.main import Model

    model = Model(dim=4, layers=1, spread=4)
    model.train_step(65, 66, False)
    weights = {name: mx.array(value) for name, value in util.tree_flatten(model.trainable_parameters())}
    optimizer = {name: mx.array(value) for name, value in util.tree_flatten(model.optimizer.state)}
    before_state = mx.array(model.blocks[0].states)
    traces = {name: mx.array(getattr(model.blocks[0], name))
              for name in ('decaytrace', 'embedtrace')}
    mx.eval(*weights.values(), *optimizer.values(), before_state, *traces.values())

    model.step(mx.array(67), frozen=True)
    current_weights = dict(util.tree_flatten(model.trainable_parameters()))
    current_optimizer = dict(util.tree_flatten(model.optimizer.state))
    mx.eval(*current_weights.values(), *current_optimizer.values(), model.blocks[0].states,
            model.blocks[0].decaytrace, model.blocks[0].embedtrace)
    assert weights.keys() == current_weights.keys()
    assert optimizer.keys() == current_optimizer.keys()
    assert all((mx.array_equal(value, current_weights[name]).item() for name, value in weights.items()))
    assert all((mx.array_equal(value, current_optimizer[name]).item() for name, value in optimizer.items()))
    assert not mx.array_equal(before_state, model.blocks[0].states).item()
    assert all((mx.array_equal(value, getattr(model.blocks[0], name)).item() for name, value in traces.items()))
