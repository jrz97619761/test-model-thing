import pytest

import mlx.core as mx

import math
import random

from tmt.model.model import Model
from tmt.model.blocks import LOSSES, Loss

def test_model_recurrence(model_init):
    model: Model = model_init()
    model.reset()

    if model.config['block'] != 'tmt0': pytest.skip('model recurrence test only applies to tmt0')

    layer = model.blocks[0]
    byte = mx.array(65)

    encoded = model.encoder(byte)
    decay = mx.sigmoid(layer.decay)

    expected_state = mx.array(encoded)
    (_, first_states, _), _ = model.step(byte, frozen = True)
    assert mx.allclose(first_states[0]['states'], expected_state, atol = 1e-06).item()

    expected_state = decay * expected_state + encoded
    (_, second_states, _), _ = model.step(byte, frozen = True)
    assert mx.allclose(second_states[0]['states'], expected_state, atol = 1e-06).item()

def test_model_loss(model_init):
    model: Model = model_init()
    lossfn: Loss = LOSSES[model.config['loss']](model.config)

    x = mx.array([0.0, 0.5, 1.0, 1.5])
    output = mx.arange(257) * 0.001
    target = mx.array(67)

    mx.eval(x, output, target)

    model_components: dict[str, mx.array] = {}
    model_loss = model.loss.loss({
        'x': x, 'output': output,
        'target': target, 'target_emb': mx.stop_gradient(model.encoder(target))
    }, model_components)

    loss_components: dict[str, mx.array] = {}
    loss = lossfn.loss({
        'x': x, 'output': output,
        'target': target, 'target_emb': mx.stop_gradient(model.encoder(target))
    }, loss_components)

    assert model_components.keys() == loss_components.keys()

    assert float(model_loss.item()) == pytest.approx(float(loss.item()), rel = 0, abs = 1e-5)
    for name, value in loss_components.items():
        assert float(model_components[name].item()) == pytest.approx(float(value.item()), rel = 0, abs = 1e-5), name

def test_model_stop_token(model_init):
    model: Model = model_init({'loss': 'ceonly'})
    logits, stop = model.decoder(mx.zeros((model.config['dim'], )))

    assert logits.shape == (257, )
    assert float(stop.item()) == pytest.approx(float(mx.softmax(logits)[256].item()))
    assert 'stop' not in model.decoder.parameters()

    forced = mx.full((257, ), -100.0)
    forced[256] = 100.0
    assert int(model.sample(forced).item()) == 256

    x = mx.zeros((model.config['dim'], ))
    expected = float((-forced[256] + mx.logsumexp(forced)).item())

    loss = model.loss.loss({
        'x': x, 'output': forced,
        'target': mx.array(256), 'target_emb': mx.stop_gradient(model.encoder(mx.array(256)))
    }, {})
    assert float(loss.item()) == pytest.approx(expected)

def test_model_train_step(monkeypatch, model_init):
    model: Model = model_init()

    def trigger(*args, **kwargs): pytest.fail('model.sample called')
    monkeypatch.setattr(model, 'sample', trigger)

    # seed 23

    mx.random.seed(23)
    expected = mx.random.uniform(shape = (1, )).item()
    mx.random.seed(23)

    loss, (output, stop) = model.train_step((97, 98, False))
    assert mx.random.uniform(shape = (1, )).item() == expected

    assert (output.shape, stop.shape) == ((257, ), (1, ))

    assert math.isfinite(float(loss.item()))
    assert int(model.optimizer.opt.state['step'].item()) == 1

    # seed 29

    mx.random.seed(29)
    expected = mx.random.uniform(shape = (1, )).item()
    mx.random.seed(29)

    loss, (output, stop) = model.train_step((98, 99, True))
    assert mx.random.uniform(shape = (1, )).item() == expected

    assert math.isfinite(float(loss.item()))
    assert int(model.optimizer.opt.state['step'].item()) == 2