import contextlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def evaluation_model():
    import mlx.core as mx
    from tmt.benchmark import evaluate

    class FakeBlock:
        state_fields = ('cache',)
        trace_fields = ('history',)

        def __init__(self):
            self.cache = mx.array([91.])
            self.history = mx.array([[93.]])

        def reset(self):
            self.cache = mx.zeros_like(self.cache)
            self.history = mx.zeros_like(self.history)

        def snapshot_state(self):
            saved = {name: mx.array(getattr(self, name))
                     for name in self.state_fields + self.trace_fields}
            mx.eval(*saved.values())
            return saved

        def restore_state(self, saved):
            for name, value in saved.items():
                setattr(self, name, value)

        def eval_state(self):
            mx.eval(*(getattr(self, name)
                      for name in self.state_fields + self.trace_fields))

    class FakeModel:
        def __init__(self):
            self.blocks = [FakeBlock()]
            self.uniform = False

        def reset(self):
            for block in self.blocks:
                block.reset()

        def snapshot_state(self):
            return [block.snapshot_state() for block in self.blocks]

        def restore_state(self, saved):
            for block, block_saved in zip(self.blocks, saved):
                block.restore_state(block_saved)

        def eval_state(self):
            for block in self.blocks:
                block.eval_state()

        def step(self, current, dummies=None, frozen=False):
            assert frozen
            block = self.blocks[0]
            byte = int(current.item())
            state = (int(block.cache.item()) * 3 + byte) % 17
            block.cache = mx.array([float(state)])
            block.history = block.history + mx.array([[float(byte)]])
            logits = (mx.zeros((256,)) if self.uniform else
                      mx.array([state / 5 if i == byte else 0 for i in range(256)]))
            return ((None, [], []), (logits, mx.zeros((1,))))

    model = FakeModel()
    saved = model.snapshot_state()
    def unchanged():
        current = model.snapshot_state()
        assert len(current) == len(saved)
        for actual, expected in zip(current, saved):
            assert actual.keys() == expected.keys()
            assert all(mx.array_equal(actual[name], expected[name]).item()
                       for name in expected)
    yield model
    unchanged()


@pytest.mark.parametrize('windows', [(0,), (-1,), (1, 0)])
def test_invalid_windows(evaluation_model, windows):
    from tmt.benchmark import evaluate

    with pytest.raises(ValueError, match='windows must be positive'):
        evaluate(evaluation_model, [b'ab'], windows)


@pytest.mark.parametrize('documents', [[], [b''], [b'a', b'b']])
def test_no_targets(evaluation_model, documents):
    from tmt.benchmark import evaluate

    with pytest.raises(ValueError, match='at least one target'):
        evaluate(evaluation_model, documents)


def test_uniform_score(evaluation_model, assert_almost_equal):
    from tmt.benchmark import evaluate

    evaluation_model.uniform = True
    score = evaluate(evaluation_model, [b'aa'])
    assert_almost_equal(score['bpb'], 8.0, delta=1e-6)
    assert score['common_bpb'] is None


def test_context_scores(evaluation_model, assert_almost_equal):
    from tmt.benchmark import evaluate

    model = evaluation_model
    documents, windows = [b'aa', b'abcde', b'', b'xy', b'pqrs'], (1, 2)

    def manual(prefix, target):
        state = sum(byte * 3 ** power for byte, power in zip(prefix, reversed(range(len(prefix))))) % 17
        return math.log(255 + math.exp(state / 5)) - (state / 5 if prefix[-1] == target else 0)

    def total(window=None, common=False):
        return math.fsum(manual(document[max(0, offset - window):offset] if window else document[:offset], document[offset])
                         for document in documents for offset in range(1, len(document)) if not common or offset >= 2)
    result = evaluate(model, documents, windows)
    assert (result['input_bytes'], result['targets'], result['common_targets']) == (13, 9, 5)
    assert_almost_equal(result['bpb'], total() / (9 * math.log(2)), delta=1e-5)
    assert_almost_equal(result['common_bpb'], total(common=True) / (5 * math.log(2)), delta=1e-5)
    for score, window in zip(result['windows'], windows):
        common = total(window, True) / (5 * math.log(2))
        assert_almost_equal(score['bpb'], total(window) / (9 * math.log(2)), delta=1e-5)
        assert_almost_equal(score['common_bpb'], common, delta=1e-5)
        assert_almost_equal(score['delta_bpb'], common - result['common_bpb'], delta=1e-5)


def test_short_context(evaluation_model):
    from tmt.benchmark import evaluate

    score = evaluate(evaluation_model, [b'ab'], (3,))
    assert (score['common_targets'], score['common_bpb'], score['windows'][0]['delta_bpb']) == (0, None, None)


def test_score_failure(evaluation_model, monkeypatch):
    from tmt.benchmark import evaluate

    def fail_score(*args):
        raise RuntimeError('injected score failure')

    monkeypatch.setattr('tmt.benchmark._nll', fail_score)
    with pytest.raises(RuntimeError, match='injected'):
        evaluate(evaluation_model, [b'aa', b'abcde', b'', b'xy', b'pqrs'], (1, 2))


def test_benchmark(assert_almost_equal, monkeypatch, tmp_path):
    import mlx.core as mx
    import mlx.utils as util
    from tmt.benchmark import BYTE_DOCUMENTS, CONTEXT_WINDOWS, mcc, run
    from tmt.cli import new_model, save_checkpoint

    settings = {'dim': 4, 'layers': 1, 'spread': 4, 'temp': 0.75, 'rate': 0.0005,
                'bound': [40000, 120000], 'block': 'rtu'}
    root = tmp_path
    model = new_model(settings, 11)
    checkpoint = root / 'model.safetensors'
    save_checkpoint(checkpoint, model, settings)
    fields = ('states', 'decaytrace', 'embedtrace')
    before = {name: mx.array(value) for name, value in util.tree_flatten(model.parameters()) if not any(field in name for field in fields)}
    optimizer_before = dict(util.tree_flatten(model.optimizer.state))
    mx.eval(*before.values(), *optimizer_before.values())
    data = root / 'cola.tsv'
    data.write_text('src\t1\tx\talpha\nsrc\t0\tx\tbeta\nsrc\t0\tx\tgamma\nsrc\t1\tx\tdelta\n')
    from io import StringIO
    stdout = StringIO()
    states = [mx.array(getattr(model.blocks[0], field)) for field in fields]
    with contextlib.redirect_stdout(stdout):
        byte_only = run(str(checkpoint), model=model)
    assert byte_only['epochs'] == []
    assert byte_only['byte_scores']['targets'] == 390
    assert '104 targets' in stdout.getvalue()
    assert all((mx.array_equal(value, getattr(model.blocks[0], field)).item() for value, field in zip(states, fields)))
    with contextlib.redirect_stdout(stdout):
        result = run(str(checkpoint), 1, 0.5, str(data), model=model, seed=11)
    assert byte_only['byte_scores'] == result['byte_scores']
    assert result['feature_convention'] == 'recurrent-state'
    assert 'CoLA feature convention: recurrent-state' in stdout.getvalue()
    for partition in ('train', 'held'):
        score = result['epochs'][0][partition]
        assert sum((score[key] for key in ('tp', 'tn', 'fp', 'fn'))) == 2
    with monkeypatch.context() as patches:
        patches.setattr('tmt.benchmark.evaluate', lambda *args: {
            'bpb': 8.0, 'targets': 1, 'common_bpb': 8.0, 'common_targets': 1, 'windows': [],
        })
        patches.setattr(model.blocks[-1], 'features', lambda x: mx.zeros((model.dim, 1)))
        with pytest.raises(ValueError, match='CoLA features must be an MLX array with shape'):
            run(str(checkpoint), 1, 0.5, str(data), model=model, seed=11)
    with pytest.raises(FileNotFoundError): run(str(checkpoint), data=str(root / 'missing.tsv'), model=model)

    scores = result['byte_scores']
    assert (sum(map(len, BYTE_DOCUMENTS)), scores['input_bytes'], scores['targets'], scores['common_targets']) == (393, 393, 390, 104)
    assert tuple((item['window'] for item in scores['windows'])) == CONTEXT_WINDOWS
    held = result['epochs'][0]['held']
    denominator = math.sqrt((held['tp'] + held['fp']) * (held['tp'] + held['fn']) * (held['tn'] + held['fp']) * (held['tn'] + held['fn']))
    assert_almost_equal(held['mcc'], (held['tp'] * held['tn'] - held['fp'] * held['fn']) * 100 / denominator if denominator else 0)
    assert mcc(1, 1, 0, 0) == 100.0
    after = dict(util.tree_flatten(model.parameters()))
    assert all((mx.array_equal(value, after[name]).item() for name, value in before.items()))
    optimizer_after = dict(util.tree_flatten(model.optimizer.state))
    assert all((mx.array_equal(value, optimizer_after[name]).item() for name, value in optimizer_before.items()))


@pytest.mark.parametrize('epochs', [0, -1])
@pytest.mark.parametrize('supply_model', [False, True])
def test_invalid_cola_epochs(monkeypatch, forbid_call, epochs, supply_model):
    from tmt.benchmark import run

    monkeypatch.setattr('tmt.cli.load_model', forbid_call)
    monkeypatch.setattr('tmt.benchmark.evaluate', forbid_call)
    model = SimpleNamespace(freeze=forbid_call) if supply_model else None
    with pytest.raises(ValueError, match='epochs must be positive with cola-data'):
        run('missing.safetensors', epochs, data='missing.tsv', model=model)
