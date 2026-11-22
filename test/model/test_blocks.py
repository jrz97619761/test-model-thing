import pytest

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as util

from tmt.model.blocks import BLOCKS, Block, TMT0Block, FieldNames, Fields, FwdOutput, Config, GradTree, BwdOutput
from tmt.model.model import Model
from tmt.util.validation import ValidationError

from typing import Any

# fixtures

class MatrixCache(Block):
    @property
    def statefields(self) -> FieldNames:
        return ('matrix', 'cache')

    @property
    def tracefields(self) -> FieldNames:
        return ('history', )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.proj = nn.Linear(config['dim'], config['dim'], bias = False)
        
        self.matrix = mx.zeros((config['dim'], config['dim']))
        self.cache = mx.zeros((2, config['dim']))
        self.history = mx.zeros((config['dim'], ))

        self.freezeall()

    def __call__(self, inputs: Fields, probe: mx.array) -> FwdOutput:
        matrix = 0.5 * self.matrix + inputs['enc'][:, None] * inputs['enc'][None, :]
        cache = mx.stack([self.cache[1], inputs['x']])

        output = inputs['x'] + mx.tanh(
            self.proj(inputs['enc']) + matrix.mean(axis = 0) + self.cache.mean(axis = 0)
        )

        return output, {'matrix': matrix, 'cache': cache}, {}

class Unfrozen(MatrixCache):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.unfreeze(keys = ['cache'], recurse = False)

class Duplicate(MatrixCache):
    @property
    def tracefields(self) -> FieldNames:
        return ('matrix', )

class NoWeights(Block):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def __call__(self, inputs: Fields, probe: mx.array) -> FwdOutput:
        return inputs['x'] + mx.tanh(inputs['enc']), {}, {}

pytestmark = pytest.mark.usefixtures('blocks')

@pytest.fixture
def blocks(monkeypatch):
    monkeypatch.setitem(BLOCKS, 'matrixcache', MatrixCache)
    monkeypatch.setitem(BLOCKS, 'noweights', NoWeights)

# tests

BLOCK_OVERRIDES = {'dim': 4, 'spread': 4, 'vocab': 257}

@pytest.mark.parametrize(('blocktype', 'expected'), [
    (Block, ((), ())),
    (TMT0Block, (('states', ), ('decaytrace', 'embedtrace'))),
    (MatrixCache, (('matrix', 'cache'), ('history', ))),
], ids = ['base', 'tmt0', 'matrixcache'])
def test_block_fields(blocktype, expected):
    block = blocktype(BLOCK_OVERRIDES)
    fields = ('statefields', 'tracefields')

    for field, original in zip(fields, expected):
        assert getattr(block, field) == original, f'{field} getter is wrong'
        replacement = ('replacement',) if isinstance(original, tuple) else 'replacement'

        with pytest.raises(AttributeError): setattr(block, field, replacement)
        with pytest.raises(AttributeError): delattr(block, field)

        assert getattr(block, field) == original, f'{field} changed after edits'
        assert field not in block, f'{field} entered the MLX parameter dictionary'

@pytest.mark.parametrize('operation', ['commit', 'result', 'restore'])
def test_block_lazy_validation(operation):
    block = TMT0Block(BLOCK_OVERRIDES)
    assert block.schema is None

    def check():
        match operation:
            case 'commit': block.commit({'states': mx.zeros((2, ))}, {})
            case 'result': block.validate_output(mx.zeros((4, )), mx.zeros((4, )), {'states': mx.zeros((2, ))})
            case 'restore':
                block.restore({
                    'states': mx.zeros((2, )),
                    'decaytrace': mx.zeros_like(block.decaytrace),
                    'embedtrace': mx.zeros_like(block.embedtrace)
                })

    with pytest.raises(ValidationError, match = 'non-matching shape and/or dtype'): check()
    assert block.schema is not None

MODEL_OVERRIDES = lambda block, options = {}: {'block': block, 'layers': 2, **options}

@pytest.mark.parametrize('block', ['tmt0', 'matrixcache', 'noweights'])
def test_block_training(block, monkeypatch, model_init, model_snapshot):
    # init

    model: Model = model_init(MODEL_OVERRIDES(block))
    initial: dict[str, Any] = model_snapshot(model)

    # parameter count

    buffers = sum(getattr(layer, name).size for layer in model.blocks for name in layer.fields())
    paramcount = sum(value.size for _, value in util.tree_flatten(model.parameters()))
    assert model.count() == paramcount - buffers

    # capture

    gradients = []
    update = model.optimizer.apply

    def capture(model, grads):
        gradients.append(dict(util.tree_flatten(grads)))
        update(model, grads)

    monkeypatch.setattr(model.optimizer, 'apply', capture)

    # train

    for current, target in [(65, 66), (66, 67), (67, 68)]:
        loss, (output, _) = model.train_step((current, target, False))
        assert output.shape == (257, )
        assert mx.isfinite(loss).item()

    assert model.optimizer.opt.state['step'].item() == 3
    assert model.optimizer.compiled is not None

    # check

    grads: dict[str, Any] = gradients[-1]

    if block == 'noweights': groups = ['encoder', 'decoder']
    else: groups = ['encoder', 'decoder', 'blocks']

    for group in groups:
        assert any(
            mx.any(value != 0).item() for name, value in grads.items() if name.startswith(f'{group}.')
        ), f'{group} gradients are all zero for {block}'

    for name, value in grads.items():
        assert mx.all(mx.isfinite(value)).item(), f'gradient {name!r} is not finite'

    final: dict[str, Any] = model_snapshot(model)
    assert any(
        final[name] != value
        for name, value in initial.items()
        if name.startswith('parameters.')
    )

@pytest.mark.parametrize('block', ['tmt0', 'matrixcache', 'noweights'])
def test_block_step_reset(block, model_init, model_snapshot):
    model: Model = model_init(MODEL_OVERRIDES(block))

    # train

    model.train_step((65, 66, False))
    if block == 'tmt0': assert mx.sum(model.blocks[0].embedtrace[65]).item() > 0.0

    fields = {name for layer in model.blocks for name in layer.statefields}
    saved = model.snapshot()
    before: dict[str, Any] = model_snapshot(model)

    model.step(mx.array(90), frozen = True)
    model.evaluate()
    after: dict[str, Any] = model_snapshot(model)

    # check

    assert after.keys() == before.keys()

    changed = {name for name in before if after[name] != before[name]}
    assert all(name.rsplit('.', 1)[-1] in fields for name in changed), changed
    if fields: assert changed, 'frozen step did not change runtime state'

    model.restore(saved)
    assert model_snapshot(model) == before

    # reset test

    model.reset()
    (first, _, _), _ = model.step(mx.array(65), frozen = True)
    model.step(mx.array(66), frozen = True)
    model.reset()

    for index, layer in enumerate(model.blocks):
        for name in layer.fields():
            assert mx.all(getattr(layer, name) == 0).item(), (
                f'layer {index} field {name!r} did not reset'
            )

    (again, _, _), _ = model.step(mx.array(65), frozen = True)
    assert mx.array_equal(first, again).item()

@pytest.mark.parametrize('block', ['tmt0', 'matrixcache'])
def test_block_eval_freeze(block, monkeypatch, model_init, model_snapshot):
    from tmt.runtime.benchmark import bpb_score

    model: Model = model_init(MODEL_OVERRIDES(block))
    model.train_step((65, 66, False))
    before: dict[str, Any] = model_snapshot(model)

    score = bpb_score(model, [b'ABCD'], windows = (1, 2))
    assert score['full']['targets'] == 3
    assert model_snapshot(model) == before

    def trigger(*args, **kwargs): raise RuntimeError('injected')
    monkeypatch.setattr(mx, 'logsumexp', trigger)

    with pytest.raises(RuntimeError, match = 'injected'): bpb_score(model, [b'AB'])
    assert model_snapshot(model) == before

@pytest.mark.parametrize('block', ['tmt0', 'matrixcache'])
def test_block_restore_state_rng(block, monkeypatch, model_init, model_snapshot):
    from tmt.runtime.runtime import sample

    def next_random(seed):
        mx.random.seed(seed)

        value = mx.random.uniform(shape=(1, ))
        mx.eval(value)

        mx.random.seed(seed)
        return value

    def next_global():
        value = mx.random.uniform(shape=(1, ))
        mx.eval(value)

        return value

    model: Model = model_init(MODEL_OVERRIDES(block, {'loss': 'ceonly'}))
    model.train_step((65, 66, False))
    model.train_step((66, 67, True))
    assert model.optimizer.compiled is not None

    before: dict[str, Any] = model_snapshot(model)
    expected_next = next_random(817)

    sample_calls: list[tuple[list, dict]] = []
    original = model.sample

    def capture(*args, **kwargs):
        sample_calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, 'sample', capture)

    first = sample(model, 'The ', 5, 31)
    assert isinstance(first, bytes)
    assert len(first) == 5

    assert mx.array_equal(expected_next, next_global()).item()
    assert model_snapshot(model) == before
    assert sample_calls

    for index, (args, kwargs) in enumerate(sample_calls):
        assert kwargs.get('key') is not None or len(args) >= 2, (
            f'training sample call {index} omitted its explicit MLX random key'
        )

    model.reset()
    reset = model_snapshot(model)

    assert sample(model, 'The ', 5, 31) == first
    assert model_snapshot(model) == reset

    def trigger(*args, **kwargs): raise RuntimeError('injected')
    monkeypatch.setattr(model, 'sample', trigger)

    expected_next = next_random(991)
    with pytest.raises(RuntimeError, match = 'injected'): sample(model, 'A prompt ', 4, 7)

    assert mx.array_equal(expected_next, next_global()).item()
    assert model_snapshot(model) == reset

@pytest.mark.parametrize('block', ['tmt0', 'matrixcache'])
def test_block_checkpoint(block, tmp_path, model_init, model_snapshot):
    from tmt.runtime.checkpoint import save, load
    from tmt.runtime.benchmark import cola_rollout

    # train

    model: Model = model_init(MODEL_OVERRIDES(block))
    model.train_step((65, 66, False))
    model.train_step((66, 67, False))

    settings = model.config

    expected: dict[str, Any] = model_snapshot(model)

    # load and save
    
    path = tmp_path / 'candidate.safetensors'
    save(path, model, settings)
    loaded, resolved = load(path)

    # check

    assert resolved == settings, 'loaded settings differ from saved settings'

    assert loaded.config['dim'] == 4, 'restored model dimension differs'    
    assert mx.array_equal(loaded.decoder.decode.weight, model.decoder.decode.weight)

    assert int(loaded.optimizer.opt.state['step'].item()) == 2, 'optimizer step differs'

    assert model_snapshot(loaded) == expected
    state = loaded.snapshot()

    assert cola_rollout(loaded, b'ABC').shape == (4, )
    loaded.restore(state)

    # train again

    loaded.train_step((67, 68, False))
    model.train_step((67, 68, False))

    assert model_snapshot(loaded) == model_snapshot(model)
    assert loaded.optimizer.opt.state['step'].item() == 3

@pytest.mark.parametrize(('blocktype', 'message'), [
    (Unfrozen, 'must be frozen'),
    (Duplicate, 'duplicate block field')
], ids = ['unfrozen', 'duplicate'])
def test_block_invalid_def(monkeypatch, blocktype, message, model_init):
    monkeypatch.setitem(BLOCKS, 'invalid', blocktype)

    with pytest.raises(ValidationError, match = message):
        model_init(MODEL_OVERRIDES('invalid', {'layers': 16, 'spread': 32}))

@pytest.mark.parametrize('fault', [
    'missing', 'shape', 'dtype', 'output'
], ids = [
    'missing-state',
    'wrong-state-shape',
    'wrong-state-dtype',
    'wrong-output-shape'
])
def test_block_invalid_fwd(monkeypatch, fault, model_init):
    class Invalid(MatrixCache):
        def __call__(self, inputs: Fields, probe: mx.array):
            output, state, aux = super().__call__(inputs, probe)

            match fault:
                case 'missing': state.pop('cache')
                case 'shape': state['matrix'] = inputs['enc']
                case 'dtype': state['cache'] = state['cache'].astype(mx.float16)
                case 'output': output = output[None, :]

            return output, state, aux

    monkeypatch.setitem(BLOCKS, 'invalid', Invalid)
    model: Model = model_init(MODEL_OVERRIDES('invalid', {'spread': 32}))

    with pytest.raises(ValidationError, match = 'block'): model.step(mx.array(65), frozen = True)

@pytest.mark.parametrize('fault', [
    'missing', 'shape', 'dtype', 'nonarray', 'encoder', 'trace'
], ids = [
    'missing-gradient',
    'gradient-shape',
    'gradient-dtype',
    'non-array-gradient',
    'encoder-gradient',
    'trace-shape'
])
def test_block_invalid_bwd(monkeypatch, fault, forbid_call, model_init):
    class Invalid(MatrixCache):
        def grads(self, output: FwdOutput, grads: GradTree, probegrad: mx.array) -> BwdOutput:
            enc, traces = {}, {}

            match fault:
                case 'missing': grads['proj'].pop('weight')
                case 'shape': grads['proj']['weight'] = mx.zeros((1,))
                case 'dtype': grads['proj']['weight'] = grads['proj']['weight'].astype(mx.float16)
                case 'nonarray': grads['proj']['weight'] = None
                case 'encoder': enc = {'encoder': mx.zeros((1, ))}
                case 'trace': traces = {'history': mx.zeros((2, ))}

            return grads, traces, enc

    monkeypatch.setitem(BLOCKS, 'invalid', Invalid)
    model: Model = model_init(MODEL_OVERRIDES('invalid'))
    
    message = {
        'encoder': 'encoder gradient', 'trace': 'history', 'nonarray': 'array should be of type'
    }.get(fault, 'adjusted gradient')

    monkeypatch.setattr(model.optimizer, 'apply', forbid_call)
    with pytest.raises(ValidationError, match = message): model.train_step((65, 66, False))

def test_block_feature_shape(monkeypatch, model_init):
    from tmt.runtime.benchmark import cola_rollout

    model: Model = model_init(MODEL_OVERRIDES('matrixcache', {'spread': 32}))
    monkeypatch.setattr(model.blocks[-1], 'features', lambda _: mx.zeros((4, 4)))

    with pytest.raises(ValidationError, match = 'feature'): cola_rollout(model, b'ABC')

def test_block_init_step_pure(model_init, model_snapshot):
    def shared(model: Model):
        return {
            name: value
            for name, value in model_snapshot(model).items()
            if name.startswith(('parameters.encoder.', 'parameters.decoder.'))
        }

    reference: Model = model_init(MODEL_OVERRIDES('tmt0', {'layers': 2}))
    candidate: Model = model_init(MODEL_OVERRIDES('matrixcache', {'layers': 2}))

    assert shared(reference)
    assert shared(candidate) == shared(reference)

    before: dict[str, Any] = model_snapshot(candidate)
    candidate.step(mx.array(65))

    assert model_snapshot(candidate) == before
