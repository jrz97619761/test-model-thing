import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

import pytest

from tmt.blocks import BLOCKS, Block
from tmt.main import Model


class Stateless(Block):
    def __init__(self, dim, spread):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, enc, x, probe):
        return x + mx.tanh(self.proj(enc)), {}, ()


class MatrixCache(Block):
    state_fields = ('matrix', 'cache')
    trace_fields = ('history',)

    def __init__(self, dim, spread):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.matrix = mx.zeros((dim, dim))
        self.cache = mx.zeros((2, dim))
        self.history = mx.zeros((dim,))
        self.freeze(keys=list(self.state_fields + self.trace_fields), recurse=False)

    def __call__(self, enc, x, probe):
        matrix = 0.5 * self.matrix + enc[:, None] * enc[None, :]
        cache = mx.stack([self.cache[1], x])
        output = x + mx.tanh(self.proj(enc) + matrix.mean(axis=0) + self.cache.mean(axis=0))
        return output, {'matrix': matrix, 'cache': cache}, ()


class NoWeights(Block):
    def __init__(self, dim, spread):
        super().__init__()

    def __call__(self, enc, x, probe):
        return x + mx.tanh(enc), {}, ()


@pytest.fixture(autouse=True)
def register_test_blocks(monkeypatch):
    for name, block in (
        ('stateless-test', Stateless),
        ('matrix-test', MatrixCache),
        ('empty-test', NoWeights),
    ):
        monkeypatch.setitem(BLOCKS, name, block)


def arrays(tree):
    return {name: mx.array(value) for name, value in tree_flatten(tree)}


def equal(left, right):
    assert left.keys() == right.keys()
    assert all(mx.array_equal(left[name], right[name]).item() for name in left)


@pytest.mark.parametrize('block', ['rtu', 'stateless-test', 'matrix-test', 'empty-test'])
def test_training(block, monkeypatch):
    mx.random.seed(11)
    model = Model(dim=4, layers=2, spread=4, block=block)
    initial = arrays(model.trainable_parameters())
    buffers = sum(
        getattr(layer, name).size
        for layer in model.blocks
        for name in layer.state_fields + layer.trace_fields
    )
    parameter_count = sum(value.size for _, value in tree_flatten(model.parameters()))
    assert model.count() == parameter_count - buffers

    gradients = []
    update = model.updategrads

    def capture(grads):
        gradients.append(arrays(grads))
        update(grads)

    monkeypatch.setattr(model, 'updategrads', capture)
    for current, target in [(65, 66), (66, 67), (67, 68)]:
        output, _, loss = model.train_step(current, target, False)
        assert output.shape == (256,)
        assert mx.isfinite(loss).item()
    assert model.optimizer.state['step'].item() == 3
    assert model.compiled is not None
    grads = gradients[-1]
    assert any(mx.any(value != 0).item() for name, value in grads.items() if name.startswith('encoder.'))
    assert any(mx.any(value != 0).item() for name, value in grads.items() if name.startswith('decoder.'))
    if block != 'empty-test':
        assert any(mx.any(value != 0).item() for name, value in grads.items() if name.startswith('blocks.'))
    assert all(mx.all(mx.isfinite(value)).item() for value in grads.values())
    assert any(
        not mx.array_equal(value, initial[name]).item()
        for name, value in arrays(model.trainable_parameters()).items()
    )


@pytest.mark.parametrize('block', ['rtu', 'stateless-test', 'matrix-test', 'empty-test'])
def test_frozen_step(block):
    model = Model(dim=4, layers=2, spread=4, block=block)
    model.train_step(65, 66, False)
    parameters = arrays(model.trainable_parameters())
    optimizer = arrays(model.optimizer.state)
    state = model.snapshot_state()
    traces = [
        {name: mx.array(getattr(layer, name)) for name in layer.trace_fields}
        for layer in model.blocks
    ]

    model.step(mx.array(90), frozen=True)
    model.eval_state()

    equal(parameters, arrays(model.trainable_parameters()))
    equal(optimizer, arrays(model.optimizer.state))
    for layer, saved in zip(model.blocks, traces):
        equal(saved, {name: getattr(layer, name) for name in layer.trace_fields})

    model.restore_state(state)
    for saved, restored in zip(state, model.snapshot_state()):
        equal(saved, restored)


@pytest.mark.parametrize('block', ['rtu', 'stateless-test', 'matrix-test', 'empty-test'])
def test_reset_replay(block):
    model = Model(dim=4, layers=2, spread=4, block=block)
    model.train_step(65, 66, False)
    model.reset()
    (first, _, _), _ = model.step(mx.array(65), frozen=True)
    model.step(mx.array(66), frozen=True)
    model.reset()

    assert all(
        mx.all(getattr(layer, name) == 0).item()
        for layer in model.blocks
        for name in layer.state_fields + layer.trace_fields
    )
    (again, _, _), _ = model.step(mx.array(65), frozen=True)
    assert mx.array_equal(first, again).item()


@pytest.mark.parametrize('block', ['stateless-test', 'matrix-test', 'empty-test'])
def test_monitoring_restores_state(block, monkeypatch):
    from tmt import benchmark
    from tmt.cli import _training_sample

    model = Model(dim=4, layers=2, spread=4, block=block)
    model.train_step(65, 66, False)
    state = model.snapshot_state()
    parameters = arrays(model.parameters())
    optimizer = arrays(model.optimizer.state)

    score = benchmark.evaluate(model, [b'ABCD'], windows=(1, 2))
    assert score['targets'] == 3
    equal(parameters, arrays(model.parameters()))
    equal(optimizer, arrays(model.optimizer.state))
    for saved, restored in zip(state, model.snapshot_state()):
        equal(saved, restored)

    assert len(_training_sample(model, 'A', 3, 11)) == 3
    for saved, restored in zip(state, model.snapshot_state()):
        equal(saved, restored)

    def fail(*args, **kwargs):
        raise RuntimeError('injected')

    monkeypatch.setattr(benchmark, '_nll', fail)
    with pytest.raises(RuntimeError, match='injected'):
        benchmark.evaluate(model, [b'AB'])
    for saved, restored in zip(state, model.snapshot_state()):
        equal(saved, restored)

    monkeypatch.setattr(model, 'sample', fail)
    with pytest.raises(RuntimeError, match='injected'):
        _training_sample(model, 'A', 3, 11)
    for saved, restored in zip(state, model.snapshot_state()):
        equal(saved, restored)
    equal(parameters, arrays(model.parameters()))
    equal(optimizer, arrays(model.optimizer.state))


@pytest.mark.parametrize('block', ['stateless-test', 'matrix-test', 'empty-test'])
def test_checkpoint_round_trip(tmp_path, block):
    from tmt.benchmark import rollout
    from tmt.cli import load_config, load_model, save_checkpoint

    settings = load_config()
    settings.update(dim=4, layers=2, spread=4, block=block)
    model = Model(**settings)
    model.train_step(65, 66, False)
    model.train_step(66, 67, False)
    expected_parameters = arrays(model.parameters())
    expected_optimizer = arrays(model.optimizer.state)

    path = tmp_path / 'candidate.safetensors'
    save_checkpoint(path, model, settings)
    loaded, resolved = load_model(path)

    assert resolved == settings
    equal(expected_parameters, arrays(loaded.parameters()))
    equal(expected_optimizer, arrays(loaded.optimizer.state))
    state = loaded.snapshot_state()
    assert rollout(loaded, b'ABC').shape == (4,)
    assert loaded.blocks[-1].feature_convention == 'final-output'
    loaded.restore_state(state)

    loaded.train_step(67, 68, False)
    model.train_step(67, 68, False)
    equal(arrays(model.parameters()), arrays(loaded.parameters()))
    equal(arrays(model.optimizer.state), arrays(loaded.optimizer.state))
    assert loaded.optimizer.state['step'].item() == 3


@pytest.mark.parametrize(
    ('block', 'message'),
    [
        ('unfrozen-test', 'must be frozen'),
        ('duplicate-test', 'distinct'),
    ],
)
def test_block_schema_validation(monkeypatch, block, message):
    if block == 'unfrozen-test':
        class Unfrozen(MatrixCache):
            def __init__(self, dim, spread):
                super().__init__(dim, spread)
                self.unfreeze(keys=['cache'], recurse=False)

        factory = Unfrozen
    else:
        class Duplicate(MatrixCache):
            trace_fields = ('matrix',)

        factory = Duplicate

    monkeypatch.setitem(BLOCKS, block, factory)
    with pytest.raises(ValueError, match=message):
        Model(dim=4, block=block)


@pytest.mark.parametrize('fault', ['missing', 'extra', 'shape', 'dtype', 'output'])
def test_forward_schema_validation(monkeypatch, fault):
    class Invalid(MatrixCache):
        def __call__(self, enc, x, probe):
            output, state, aux = super().__call__(enc, x, probe)
            if fault == 'missing':
                state.pop('cache')
            elif fault == 'extra':
                state['other'] = enc
            elif fault == 'shape':
                state['matrix'] = enc
            elif fault == 'dtype':
                state['cache'] = state['cache'].astype(mx.float16)
            elif fault == 'output':
                output = output[None, :]
            return output, state, aux

    monkeypatch.setitem(BLOCKS, 'invalid-test', Invalid)
    model = Model(dim=4, layers=1, block='invalid-test')
    with pytest.raises(ValueError, match='block'):
        model.step(mx.array(65), frozen=True)


def test_gradient_hook_schema_validation(monkeypatch):
    class Invalid(MatrixCache):
        def adjust_gradients(self, byte, next_state, aux, probe_grad, grads):
            return grads, None, {'history': mx.zeros((2,))}

    monkeypatch.setitem(BLOCKS, 'invalid-test', Invalid)
    with pytest.raises(ValueError, match='history'):
        Model(dim=4, layers=1, block='invalid-test').train_step(65, 66, False)


def test_feature_shape_validation(monkeypatch):
    from tmt.benchmark import rollout

    model = Model(dim=4, layers=1, block='matrix-test')
    monkeypatch.setattr(model.blocks[0], 'features', lambda x: mx.zeros((4, 4)))
    with pytest.raises(ValueError, match='feature'):
        rollout(model, b'ABC')


def test_custom_gradient(monkeypatch):
    @mx.custom_function
    def product(x, weight):
        return x * weight

    @product.vjp
    def product_vjp(primals, cotangent, output):
        x, weight = primals
        return weight * cotangent, x * cotangent

    class Custom(Block):
        def __init__(self, dim, spread):
            super().__init__()
            self.weight = mx.full((dim,), 0.5)

        def __call__(self, enc, x, probe):
            return x + product(enc, self.weight), {}, ()

    monkeypatch.setitem(BLOCKS, 'custom-test', Custom)
    model = Model(dim=4, layers=1, block='custom-test')
    captured = []

    def capture(grads):
        captured.append(grads)

    monkeypatch.setattr(model, 'updategrads', capture)
    model.train_step(65, 66, False)
    direction = mx.array([1.0, -0.5, 0.25, 0.75])
    analytic = mx.sum(captured[0]['blocks'][0]['weight'] * direction).item()
    weight = mx.array(model.blocks[0].weight)

    def objective(shift):
        model.blocks[0].weight = weight + shift * direction
        (x, _, _), (logits, stop) = model.step(mx.array(65))
        return model.loss(x, logits, stop, 66, False)

    epsilon = 0.001
    numerical = ((objective(epsilon) - objective(-epsilon)) / (2 * epsilon)).item()
    assert analytic == pytest.approx(numerical, abs=0.001)


@pytest.mark.parametrize('fault', ['name', 'shape', 'dtype', 'selector'])
def test_checkpoint_rejects_invalid_tensors(tmp_path, fault):
    import json
    from tmt.cli import load_config, load_model, save_checkpoint

    settings = load_config()
    settings.update(dim=4, layers=2, spread=4, block='matrix-test')
    model = Model(**settings)
    model.train_step(65, 66, False)
    path = tmp_path / 'valid.safetensors'
    save_checkpoint(path, model, settings)
    tensors, metadata = mx.load(path, return_metadata=True)
    name = 'm.blocks.0.matrix'
    if fault == 'name':
        tensors['m.blocks.0.other'] = tensors.pop(name)
    elif fault == 'shape':
        tensors[name] = mx.zeros((4,))
    elif fault == 'dtype':
        tensors[name] = tensors[name].astype(mx.float16)
    elif fault == 'selector':
        record = json.loads(metadata['tmt'])
        record['model']['block'] = 'unknown'
        metadata['tmt'] = json.dumps(record)

    invalid = tmp_path / 'invalid.safetensors'
    mx.save_safetensors(invalid, tensors, metadata=metadata)
    with pytest.raises(ValueError, match='tensor|block'):
        load_model(invalid)


def test_initialization_and_forward_purity():
    mx.random.seed(11)
    reference = Model(dim=4, layers=2, spread=4)
    mx.eval(reference.parameters())
    mx.random.seed(11)
    candidate = Model(dim=4, layers=2, spread=4, block='matrix-test')
    mx.eval(candidate.parameters())
    equal(arrays(reference.encoder.parameters()), arrays(candidate.encoder.parameters()))
    equal(arrays(reference.decoder.parameters()), arrays(candidate.decoder.parameters()))
    before = arrays(candidate.parameters())
    candidate.step(mx.array(65))
    equal(before, arrays(candidate.parameters()))


@pytest.mark.parametrize('fault', ['missing', 'extra', 'shape', 'dtype', 'nonarray', 'encoder'])
def test_invalid_gradients_before_optimizer_update(monkeypatch, forbid_call, fault):
    class Invalid(Stateless):
        def adjust_gradients(self, byte, next_state, aux, probe_grad, grads):
            encoder = None
            if fault == 'missing':
                grads['proj'].pop('weight')
            elif fault == 'extra':
                grads['extra'] = mx.zeros((1,))
            elif fault == 'shape':
                grads['proj']['weight'] = mx.zeros((1,))
            elif fault == 'dtype':
                grads['proj']['weight'] = grads['proj']['weight'].astype(mx.float16)
            elif fault == 'nonarray':
                grads['proj']['weight'] = None
            else:
                encoder = mx.zeros((1,))
            return grads, encoder, {}

    monkeypatch.setitem(BLOCKS, 'invalid-test', Invalid)
    model = Model(dim=4, layers=1, spread=4, block='invalid-test')
    monkeypatch.setattr(model, 'updategrads', forbid_call)
    message = 'block encoder gradient' if fault == 'encoder' else 'block gradient hook'
    with pytest.raises(ValueError, match=message):
        model.train_step(65, 66, False)
