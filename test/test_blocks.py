import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from tmt.blocks import BLOCKS, Block, RTUBlock
from tmt.model import Model


class MatrixCache(Block):
    @property
    def state_fields(self):
        return ('matrix', 'cache')

    @property
    def trace_fields(self):
        return ('history',)

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
        output = x + mx.tanh(
            self.proj(enc) + matrix.mean(axis=0) + self.cache.mean(axis=0)
        )
        return output, {'matrix': matrix, 'cache': cache}, ()


class NoWeights(Block):
    def __init__(self, dim, spread):
        super().__init__()

    def __call__(self, enc, x, probe):
        return x + mx.tanh(enc), {}, ()


class Unfrozen(MatrixCache):
    def __init__(self, dim, spread):
        super().__init__(dim, spread)
        self.unfreeze(keys=['cache'], recurse=False)


class Duplicate(MatrixCache):
    @property
    def trace_fields(self):
        return ('matrix',)


pytestmark = pytest.mark.usefixtures('blocks')


@pytest.fixture
def blocks(monkeypatch):
    monkeypatch.setitem(BLOCKS, 'matrix-test', MatrixCache)
    monkeypatch.setitem(BLOCKS, 'empty-test', NoWeights)


@pytest.fixture
def make_model(tiny_settings):
    return lambda block, **extra: Model(
        **{**tiny_settings, 'layers': 2, 'block': block, **extra}
    )


@pytest.mark.parametrize(
    ('block_type', 'expected'),
    [
        (Block, ((), (), 'one-step-autodiff', 'final-output')),
        (
            RTUBlock,
            (('states',), ('decaytrace', 'embedtrace'), 'rtu-v1', 'recurrent-state'),
        ),
        (
            MatrixCache,
            (('matrix', 'cache'), ('history',), 'one-step-autodiff', 'final-output'),
        ),
    ],
    ids=['base', 'rtu', 'custom-block'],
)
def test_block_contract_properties(block_type, expected):
    block = Block() if block_type is Block else block_type(4, 4)
    fields = ('state_fields', 'trace_fields', 'temporal_gradient', 'feature_source')
    for field, original in zip(fields, expected):
        assert getattr(block, field) == original, f'{field} getter is wrong'
        replacement = ('replacement',) if isinstance(original, tuple) else 'replacement'
        with pytest.raises(AttributeError):
            setattr(block, field, replacement)
        with pytest.raises(AttributeError):
            delattr(block, field)
        assert getattr(block, field) == original, f'{field} changed after edits'
        assert field not in block, f'{field} entered the MLX parameter dictionary'


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('temporal_gradient', 'unknown'),
        ('temporal_gradient', None),
        ('feature_source', 'unknown'),
        ('feature_source', None),
    ],
    ids=[
        'unknown-gradient',
        'non-string-gradient',
        'unknown-feature',
        'non-string-feature',
    ],
)
def test_invalid_block_labels(monkeypatch, tiny_settings, field, value):
    label = property(lambda self: value)
    invalid = type('InvalidLabel', (MatrixCache,), {field: label})
    monkeypatch.setitem(BLOCKS, 'invalid-test', invalid)
    with pytest.raises(ValueError, match=f'block {field} must be one of'):
        Model(**{**tiny_settings, 'block': 'invalid-test'})


@pytest.mark.parametrize('operation', ['commit', 'restore', 'result'])
def test_layer_value_validation_is_lazy(operation):
    layer = RTUBlock(4, 4)
    assert layer._runtime_schema is None

    def check():
        if operation == 'commit':
            layer.commit_state({'states': mx.zeros((2,))})
        elif operation == 'restore':
            saved = {
                'states': mx.zeros((2,)),
                'decaytrace': mx.zeros_like(layer.decaytrace),
                'embedtrace': mx.zeros_like(layer.embedtrace),
            }
            layer.restore_state(saved)
        else:
            enc, output = mx.zeros((4,)), mx.zeros((4,))
            layer.validate_result(enc, output, {'states': mx.zeros((2,))})

    with pytest.raises(ValueError, match='retain shape'):
        check()
    assert layer._runtime_schema is not None


@pytest.mark.parametrize('block', ['rtu', 'matrix-test', 'empty-test'])
def test_training_updates_parameters_and_optimizer(
    block, monkeypatch, make_model, snapshot
):
    mx.random.seed(11)
    model = make_model(block)
    initial = snapshot(model)
    buffers = sum(
        getattr(layer, name).size
        for layer in model.blocks
        for name in layer.state_fields + layer.trace_fields
    )
    parameter_count = sum(value.size for _, value in tree_flatten(model.parameters()))
    assert model.count() == parameter_count - buffers

    gradients = []
    update = model.update_gradients

    def capture(grads):
        gradients.append(dict(tree_flatten(grads)))
        update(grads)

    monkeypatch.setattr(model, 'update_gradients', capture)
    for current, target in [(65, 66), (66, 67), (67, 68)]:
        output, _, loss = model.train_step(current, target, False)
        assert output.shape == (256,)
        assert mx.isfinite(loss).item()
    assert model.optimizer.state['step'].item() == 3
    assert model.compiled is not None
    grads = gradients[-1]
    gradient_groups = ['encoder', 'decoder']
    if block != 'empty-test':
        gradient_groups.append('blocks')
    for group in gradient_groups:
        assert any(
            mx.any(value != 0).item()
            for name, value in grads.items()
            if name.startswith(f'{group}.')
        ), f'{group} gradients are all zero for {block}'
    for name, value in grads.items():
        assert mx.all(mx.isfinite(value)).item(), f'gradient {name!r} is not finite'
    final = snapshot(model)
    assert any(
        final[name] != value
        for name, value in initial.items()
        if name.startswith('parameters.')
    )


@pytest.mark.parametrize('block', ['rtu', 'matrix-test', 'empty-test'])
def test_frozen_step_and_reset(block, make_model, snapshot):
    model = make_model(block)
    model.train_step(65, 66, False)
    if block == 'rtu':
        assert mx.sum(model.blocks[0].embedtrace[65]).item() > 0.0
    fields = {name for layer in model.blocks for name in layer.state_fields}
    saved = model.snapshot_state()
    before = snapshot(model)

    model.step(mx.array(90), frozen=True)
    model.eval_state()
    after = snapshot(model)

    assert after.keys() == before.keys()
    changed = {name for name in before if after[name] != before[name]}
    assert all(name.rsplit('.', 1)[-1] in fields for name in changed), changed
    if fields:
        assert changed, 'frozen step did not change runtime state'
    model.restore_state(saved)
    assert snapshot(model) == before

    model.reset()
    (first, _, _), _ = model.step(mx.array(65), frozen=True)
    model.step(mx.array(66), frozen=True)
    model.reset()
    for index, layer in enumerate(model.blocks):
        for name in layer.state_fields + layer.trace_fields:
            assert mx.all(getattr(layer, name) == 0).item(), (
                f'layer {index} field {name!r} did not reset'
            )
    (again, _, _), _ = model.step(mx.array(65), frozen=True)
    assert mx.array_equal(first, again).item()


@pytest.mark.parametrize('block', ['rtu', 'matrix-test'])
def test_evaluation_restores_state(block, monkeypatch, make_model, snapshot):
    from tmt import benchmark

    model = make_model(block)
    model.train_step(65, 66, False)
    before = snapshot(model)

    score = benchmark.evaluate(model, [b'ABCD'], windows=(1, 2))
    assert score['targets'] == 3
    assert snapshot(model) == before

    def fail(*args, **kwargs):
        raise RuntimeError('injected')

    monkeypatch.setattr(mx, 'logsumexp', fail)
    with pytest.raises(RuntimeError, match='injected'):
        benchmark.evaluate(model, [b'AB'])
    assert snapshot(model) == before


@pytest.mark.parametrize('block', ['rtu', 'matrix-test'])
def test_training_sample_restores_state_and_rng(
    block, monkeypatch, make_model, snapshot
):
    from tmt.runtime import _training_sample

    def next_random(seed):
        mx.random.seed(seed)
        value = mx.random.uniform(shape=(1,))
        mx.eval(value)
        mx.random.seed(seed)
        return value

    def next_global():
        value = mx.random.uniform(shape=(1,))
        mx.eval(value)
        return value

    model = make_model(block)
    model.train_step(65, 66, False, True)
    model.train_step(66, 67, True, True)
    assert model.compiled is not None
    before = snapshot(model)
    expected_next = next_random(817)
    sample_calls = []
    original = model.sample

    def record_sample(*args, **kwargs):
        sample_calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, 'sample', record_sample)
    first = _training_sample(model, 'The ', 5, 31)
    assert isinstance(first, bytes)
    assert len(first) == 5
    assert mx.array_equal(expected_next, next_global()).item()
    assert snapshot(model) == before
    assert sample_calls
    for index, (args, kwargs) in enumerate(sample_calls):
        assert kwargs.get('key') is not None or len(args) >= 2, (
            f'training sample call {index} omitted its explicit MLX random key'
        )

    # Sampling starts from a reset model, so prior state does not change the bytes.
    model.reset()
    reset = snapshot(model)
    assert _training_sample(model, 'The ', 5, 31) == first
    assert snapshot(model) == reset

    def fail(*args, **kwargs):
        raise RuntimeError('injected sampling failure')

    monkeypatch.setattr(model, 'sample', fail)
    expected_next = next_random(991)
    with pytest.raises(RuntimeError, match='injected sampling failure'):
        _training_sample(model, 'A prompt', 4, 7)
    assert mx.array_equal(expected_next, next_global()).item()
    assert snapshot(model) == reset


@pytest.mark.parametrize(
    ('block', 'convention'),
    [('rtu', 'recurrent-state'), ('matrix-test', 'final-output')],
    ids=['rtu', 'matrix-test'],
)
def test_checkpoint_round_trip(block, convention, tmp_path, tiny_settings, snapshot):
    from tmt.benchmark import rollout
    from tmt.checkpoint import load_checkpoint, save_checkpoint

    settings = {**tiny_settings, 'layers': 2, 'block': block}
    model = Model(**settings)
    model.train_step(65, 66, False)
    model.train_step(66, 67, False)
    expected = snapshot(model)

    path = tmp_path / 'candidate.safetensors'
    save_checkpoint(path, model, settings)
    loaded, resolved = load_checkpoint(path)

    assert resolved == settings, 'loaded settings differ from saved settings'
    assert loaded.dim == 4, 'restored model dimension differs'
    assert int(loaded.optimizer.state['step'].item()) == 2, 'optimizer step differs'
    assert mx.array_equal(loaded.decoder.decode.weight, model.decoder.decode.weight)
    assert snapshot(loaded) == expected
    state = loaded.snapshot_state()
    assert rollout(loaded, b'ABC').shape == (4,)
    assert loaded.blocks[-1].feature_source == convention
    loaded.restore_state(state)

    loaded.train_step(67, 68, False)
    model.train_step(67, 68, False)
    assert snapshot(loaded) == snapshot(model)
    assert loaded.optimizer.state['step'].item() == 3


@pytest.mark.parametrize(
    ('factory', 'message'),
    [(Unfrozen, 'must be frozen'), (Duplicate, 'distinct')],
    ids=['unfrozen', 'duplicate'],
)
def test_invalid_block_definitions(monkeypatch, factory, message, tiny_settings):
    monkeypatch.setitem(BLOCKS, 'invalid-test', factory)
    with pytest.raises(ValueError, match=message):
        Model(**{**tiny_settings, 'layers': 16, 'spread': 32, 'block': 'invalid-test'})


@pytest.mark.parametrize(
    'fault',
    ['missing', 'shape', 'dtype', 'output'],
    ids=[
        'missing-state',
        'wrong-state-shape',
        'wrong-state-dtype',
        'wrong-output-shape',
    ],
)
def test_invalid_forward_results(monkeypatch, fault, tiny_settings):
    class Invalid(MatrixCache):
        def __call__(self, enc, x, probe):
            output, state, aux = super().__call__(enc, x, probe)
            if fault == 'missing':
                state.pop('cache')
            elif fault == 'shape':
                state['matrix'] = enc
            elif fault == 'dtype':
                state['cache'] = state['cache'].astype(mx.float16)
            else:
                output = output[None, :]
            return output, state, aux

    monkeypatch.setitem(BLOCKS, 'invalid-test', Invalid)
    model = Model(**{**tiny_settings, 'spread': 32, 'block': 'invalid-test'})
    with pytest.raises(ValueError, match='block'):
        model.step(mx.array(65), frozen=True)


@pytest.mark.parametrize(
    'fault',
    ['missing', 'shape', 'dtype', 'nonarray', 'encoder', 'trace'],
    ids=[
        'missing-gradient',
        'gradient-shape',
        'gradient-dtype',
        'non-array-gradient',
        'encoder-gradient',
        'trace-shape',
    ],
)
def test_invalid_gradient_hooks(monkeypatch, forbid_call, fault, tiny_settings):
    class Invalid(MatrixCache):
        def adjust_gradients(self, byte, next_state, aux, probe_grad, grads):
            encoder, traces = None, {}
            if fault == 'missing':
                grads['proj'].pop('weight')
            elif fault == 'shape':
                grads['proj']['weight'] = mx.zeros((1,))
            elif fault == 'dtype':
                grads['proj']['weight'] = grads['proj']['weight'].astype(mx.float16)
            elif fault == 'nonarray':
                grads['proj']['weight'] = None
            elif fault == 'encoder':
                encoder = mx.zeros((1,))
            else:
                traces = {'history': mx.zeros((2,))}
            return grads, encoder, traces

    monkeypatch.setitem(BLOCKS, 'invalid-test', Invalid)
    model = Model(**{**tiny_settings, 'block': 'invalid-test'})
    monkeypatch.setattr(model, 'update_gradients', forbid_call)
    message = {'encoder': 'block encoder gradient', 'trace': 'history'}.get(
        fault, 'block gradient hook'
    )
    with pytest.raises(ValueError, match=message):
        model.train_step(65, 66, False)


def test_feature_shape_validation(monkeypatch, tiny_settings):
    from tmt.benchmark import rollout

    model = Model(**{**tiny_settings, 'spread': 32, 'block': 'matrix-test'})
    monkeypatch.setattr(model.blocks[0], 'features', lambda x: mx.zeros((4, 4)))
    with pytest.raises(ValueError, match='feature'):
        rollout(model, b'ABC')


def test_initialization_and_forward_purity(tiny_settings, snapshot):
    def shared(model):
        return {
            name: value
            for name, value in snapshot(model).items()
            if name.startswith(('parameters.encoder.', 'parameters.decoder.'))
        }

    mx.random.seed(11)
    settings = {**tiny_settings, 'layers': 2}
    reference = Model(**settings)
    mx.random.seed(11)
    candidate = Model(**{**settings, 'block': 'matrix-test'})
    assert shared(reference)
    assert shared(candidate) == shared(reference)
    before = snapshot(candidate)
    candidate.step(mx.array(65))
    assert snapshot(candidate) == before
