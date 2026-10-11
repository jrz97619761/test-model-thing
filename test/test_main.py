import math
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


def test_train_step():
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

    import mlx.core as mx
    from tmt.main import Model

    random.seed(11)
    mx.random.seed(11)
    model = Model(dim=4, layers=1, spread=4, temp=0.75, rate=0.0005, bound=(40000, 120000))
    mx.eval(model.parameters())
    mx.random.seed(23)
    expected = mx.random.uniform(shape=(1,)).item()
    mx.random.seed(23)
    with patch.object(model, 'sample', side_effect=AssertionError('train_step sampled')):
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


def test_byte_io_recurrence_and_reset():
    import mlx.core as mx
    from tmt.main import Model

    model = Model(dim=4, layers=1, spread=4)
    mx.eval(model.parameters())
    encoded = model.encoder(mx.array([0, 255]))
    logits, stop = model.decoder(encoded)
    assert (encoded.shape, logits.shape, stop.shape) == ((2, 4), (2, 256), (2, 1))

    model.reset()
    layer = model.blocks[0]
    byte = mx.array(65)
    encoded = model.encoder(byte)
    decay = mx.sigmoid(layer.decay)
    expected_state = mx.array(encoded)
    (_, first_states, _), _ = model.step(byte, frozen=True)
    assert mx.allclose(first_states[0], expected_state, atol=1e-06).item()
    expected_state = decay * expected_state + encoded
    (_, second_states, _), _ = model.step(byte, frozen=True)
    assert mx.allclose(second_states[0], expected_state, atol=1e-06).item()

    model.train_step(65, 66, False)
    mx.eval(layer.embedtrace)
    assert float(mx.sum(layer.embedtrace[65]).item()) > 0.0
    model.reset()
    for layer in model.blocks:
        assert float(mx.sum(mx.abs(layer.states)).item()) == 0.0
        assert float(mx.sum(mx.abs(layer.decaytrace)).item()) == 0.0
        assert float(mx.sum(mx.abs(layer.embedtrace)).item()) == 0.0


def test_loss_terms(assert_almost_equal):
    import mlx.core as mx
    from tmt.main import Model

    model = Model(dim=4, layers=1, spread=4)
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
    full_end = variance_term + latent_term + cross_entropy + (stop_value - 1.0) ** 2
    full_open = variance_term + latent_term + cross_entropy + stop_value ** 2
    assert_almost_equal(float(model.loss(mx.array(x), output, stop, 67, True).item()), full_end, places=5)
    assert_almost_equal(float(model.loss(mx.array(x), output, stop, 67, False).item()), full_open, places=5)
    assert_almost_equal(float(model.loss(mx.array(x), output, stop, 67, False, True).item()), cross_entropy, places=5)
    assert_almost_equal(float(model.loss(mx.array(x), output, stop, None, False).item()), variance_term, places=5)
    assert float(model.loss(mx.array(x), output, stop, None, False, True)) == 0.0
    components = {}
    model.loss(mx.array(x), output, stop, 67, True, components=components)
    for name, expected in (('variance', variance_term), ('latent_prediction', latent_term),
                           ('cross_entropy', cross_entropy), ('stop', (stop_value - 1.0) ** 2)):
        assert_almost_equal(float(components[name].item()), expected, places=5)
    model.loss(mx.array(x), output, stop, 67, False, True, components)
    assert_almost_equal(float(components['cross_entropy'].item()), cross_entropy, places=5)
    assert all((float(components[name].item()) == 0.0 for name in ('variance', 'latent_prediction', 'stop')))


def test_rtu_traces_and_manual_gradients(assert_almost_equal):
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

    with patch.object(model, 'updategrads', side_effect=capture):
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


def test_frozen_step_isolates_weights_and_optimizer():
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
