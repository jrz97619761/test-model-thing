import math
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def evaluation_model():
    import mlx.core as mx

    class FakeBlock:
        @property
        def state_fields(self):
            return ('cache',)

        @property
        def trace_fields(self):
            return ('history',)

        def __init__(self):
            self.cache = mx.array([91.0])
            self.history = mx.array([[93.0]])

        def reset(self):
            self.cache = mx.zeros_like(self.cache)
            self.history = mx.zeros_like(self.history)

        def snapshot_state(self):
            saved = {
                name: mx.array(getattr(self, name))
                for name in self.state_fields + self.trace_fields
            }
            mx.eval(*saved.values())
            return saved

        def restore_state(self, saved):
            for name, value in saved.items():
                setattr(self, name, value)

        def eval_state(self):
            mx.eval(
                *(getattr(self, name) for name in self.state_fields + self.trace_fields)
            )

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
            logits = (
                mx.zeros((256,))
                if self.uniform
                else mx.array(
                    [state / 5 if index == byte else 0 for index in range(256)]
                )
            )
            return ((None, [], []), (logits, mx.zeros((1,))))

    model = FakeModel()
    saved = model.snapshot_state()

    def unchanged():
        current = model.snapshot_state()
        assert len(current) == len(saved), 'block count changed'
        for block_index, (actual, expected) in enumerate(zip(current, saved)):
            assert actual.keys() == expected.keys(), (
                f'block {block_index} fields changed'
            )
            for name, expected_value in expected.items():
                assert mx.array_equal(actual[name], expected_value).item(), (
                    f'block {block_index} field {name!r} changed'
                )

    yield model
    unchanged()


@pytest.mark.parametrize(
    ('documents', 'windows', 'message'),
    [
        ([b'ab'], (0,), 'windows must be positive'),
        ([b'ab'], (1, 0), 'windows must be positive'),
        ([], (), 'at least one target'),
        ([b''], (), 'at least one target'),
        ([b'a', b'b'], (), 'at least one target'),
    ],
    ids=[
        'zero-window',
        'mixed-valid-and-zero',
        'no-documents',
        'empty-document',
        'single-byte-documents',
    ],
)
def test_evaluate_rejects_invalid_input(evaluation_model, documents, windows, message):
    from tmt.benchmark import evaluate

    with pytest.raises(ValueError, match=message):
        evaluate(evaluation_model, documents, windows)


def test_uniform_score(evaluation_model):
    from tmt.benchmark import evaluate

    evaluation_model.uniform = True
    score = evaluate(evaluation_model, [b'aa'])
    assert score['bpb'] == pytest.approx(8.0, rel=0, abs=1e-6)
    assert score['common_bpb'] is None


def test_context_scores(evaluation_model):
    from tmt.benchmark import evaluate

    model = evaluation_model
    documents, windows = [b'aa', b'abcde', b'', b'xy', b'pqrs'], (1, 2)

    def predicted_state(prefix):
        state = 0
        for byte in prefix:
            state = (state * 3 + byte) % 17
        return state

    def manual_nll(prefix, target):
        state = predicted_state(prefix)
        target_logit = state / 5 if prefix[-1] == target else 0
        return math.log(255 + math.exp(state / 5)) - target_logit

    def total_nll(window=None, common_only=False):
        losses = []
        for document in documents:
            for target_offset in range(1, len(document)):
                if common_only and target_offset < 2:
                    continue
                start = max(0, target_offset - window) if window is not None else 0
                prefix = document[start:target_offset]
                losses.append(manual_nll(prefix, document[target_offset]))
        return math.fsum(losses)

    def approx(expected):
        return pytest.approx(expected, rel=0, abs=1e-5)

    result = evaluate(model, documents, windows)
    input_bytes = sum(len(document) for document in documents)
    target_count = sum(max(len(document) - 1, 0) for document in documents)
    common_target_count = sum(
        1
        for document in documents
        for target_offset in range(1, len(document))
        if target_offset >= max(windows)
    )
    assert result['input_bytes'] == input_bytes == 13
    assert result['targets'] == target_count == 9
    assert result['common_targets'] == common_target_count == 5
    assert result['bpb'] == approx(total_nll() / (target_count * math.log(2)))
    common_baseline_bpb = total_nll(common_only=True) / (
        common_target_count * math.log(2)
    )
    assert result['common_bpb'] == approx(common_baseline_bpb)
    for score, window in zip(result['windows'], windows):
        window_bpb = total_nll(window) / (target_count * math.log(2))
        window_common_bpb = total_nll(window, common_only=True) / (
            common_target_count * math.log(2)
        )
        assert score['bpb'] == approx(window_bpb)
        assert score['common_bpb'] == approx(window_common_bpb)
        assert score['delta_bpb'] == approx(window_common_bpb - common_baseline_bpb)


def test_short_context(evaluation_model):
    from tmt.benchmark import evaluate

    score = evaluate(evaluation_model, [b'ab'], (3,))
    assert score['common_targets'] == 0
    assert score['common_bpb'] is None
    assert score['windows'][0]['delta_bpb'] is None


def test_benchmark_run(checkpoint, tmp_path, capsys):
    from mlx.utils import tree_flatten
    from tmt.benchmark import BYTE_DOCUMENTS, CONTEXT_WINDOWS, mcc, run

    model = checkpoint.model
    runtime = ('states', 'decaytrace', 'embedtrace')

    def state():
        parameters = {
            name: value.tolist()
            for name, value in tree_flatten(model.parameters())
            if not any(field in name for field in runtime)
        }
        optimizer = {
            name: value.tolist() for name, value in tree_flatten(model.optimizer.state)
        }
        return parameters, optimizer

    before = state()
    data = tmp_path / 'cola.tsv'
    data.write_text(
        'src\t1\tx\talpha\nsrc\t0\tx\tbeta\nsrc\t0\tx\tgamma\nsrc\t1\tx\tdelta\n'
    )
    result = run(str(checkpoint.path), 1, 0.5, str(data), model=model, seed=11)
    output = capsys.readouterr().out
    assert '(390 targets)' in output
    assert '(104 targets)' in output
    assert set(result) == {'byte_scores', 'feature_source', 'epoch_scores'}
    assert result['feature_source'] == 'recurrent-state'
    assert 'CoLA feature convention: recurrent-state' in output
    for partition in ('train', 'held'):
        score = result['epoch_scores'][0][partition]
        assert sum(score[key] for key in ('tp', 'tn', 'fp', 'fn')) == 2

    scores = result['byte_scores']
    fixture_bytes = sum(len(document) for document in BYTE_DOCUMENTS)
    assert scores['input_bytes'] == fixture_bytes == 393
    assert scores['targets'] == 390
    assert scores['common_targets'] == 104
    assert tuple(item['window'] for item in scores['windows']) == CONTEXT_WINDOWS
    held = result['epoch_scores'][0]['held']
    denominator = math.sqrt(
        (held['tp'] + held['fp'])
        * (held['tp'] + held['fn'])
        * (held['tn'] + held['fp'])
        * (held['tn'] + held['fn'])
    )
    expected_mcc = (
        (held['tp'] * held['tn'] - held['fp'] * held['fn']) * 100 / denominator
        if denominator
        else 0
    )
    assert held['mcc'] == pytest.approx(expected_mcc, rel=0, abs=5e-8)
    assert mcc(1, 1, 0, 0) == 100.0
    assert state() == before


def test_benchmark_run_without_cola(checkpoint):
    from tmt.benchmark import run

    result = run(str(checkpoint.path), model=checkpoint.model, seed=11)
    assert set(result) == {'byte_scores', 'epoch_scores'}
    assert result['epoch_scores'] == []


@pytest.mark.parametrize(
    ('text', 'error'),
    [
        ('src\t1\tx', 'row 4 has 3 columns; expected 4'),
        ('src\t1\tx\talpha\textra', 'row 4 has 5 columns; expected 4'),
        ('src\t1\tx\talpha\t', 'row 4 has 5 columns; expected 4'),
        ('\tsrc\t1\tx\talpha', 'row 4 has 5 columns; expected 4'),
        ('src\t1\tx\t', 'row 4 has an empty sentence'),
        ('src\t1\tx\t   ', 'row 4 has an empty sentence'),
        (
            'src\t2\tx\tgamma\nsrc\t2\tx\tdelta',
            r'row 4 has invalid label .2.; expected 0 or 1',
        ),
        ('src\tno\tx\tgamma', r'row 4 has invalid label .no.; expected 0 or 1'),
    ],
    ids=[
        'three-columns',
        'five-columns',
        'trailing-tab',
        'leading-tab',
        'empty-sentence',
        'blank-sentence',
        'label-2',
        'label-no',
    ],
)
def test_cola_rows_rejected_before_model_work(
    tmp_path, monkeypatch, forbid_call, text, error
):
    from tmt.benchmark import run

    data = tmp_path / 'cola.tsv'
    data.write_text(
        'src\t1\tx\talpha\n\nsrc\t0\tx\tbeta\n' + text + '\n', encoding='utf-8'
    )
    monkeypatch.setattr('tmt.checkpoint.load_checkpoint', forbid_call)
    monkeypatch.setattr('tmt.benchmark.evaluate', forbid_call)
    with pytest.raises(ValueError, match=f'CoLA TSV {error}'):
        run('missing.safetensors', data=str(data), model=None, seed=11)


def test_cola_preserves_sentence_bytes(tmp_path):
    from tmt.benchmark import cola

    data = tmp_path / 'cola.tsv'
    data.write_bytes('\r\n  \r\n\t1\t\t  café  \r\nsrc\t0\tx\tbeta'.encode('utf-8'))
    assert cola(str(data)) == [('  café  '.encode('utf-8'), 1), (b'beta', 0)]


@pytest.mark.parametrize(
    ('epochs', 'rows', 'error', 'message'),
    [
        (0, None, ValueError, 'epochs must be positive with cola-data'),
        (-1, None, ValueError, 'epochs must be positive with cola-data'),
        (1, None, FileNotFoundError, None),
        (
            1,
            'src\t1\tx\talpha',
            FileNotFoundError,
            'at least two valid four-column rows',
        ),
    ],
    ids=['epochs-0', 'epochs-negative', 'missing-file', 'one-row'],
)
def test_run_rejects_bad_cola_inputs(
    tmp_path, monkeypatch, forbid_call, epochs, rows, error, message
):
    from tmt.benchmark import run

    data = tmp_path / 'cola.tsv'
    if rows:
        data.write_text(rows + '\n', encoding='utf-8')
    monkeypatch.setattr('tmt.checkpoint.load_checkpoint', forbid_call)
    monkeypatch.setattr('tmt.benchmark.evaluate', forbid_call)
    model = SimpleNamespace(freeze=forbid_call)
    with pytest.raises(error, match=message):
        run('missing.safetensors', epochs, data=str(data), model=model, seed=11)


@pytest.mark.parametrize(
    ('args', 'code', 'error'),
    [
        (['--help'], 0, None),
        (
            [
                'missing.safetensors', '--cola-data', 'cola.tsv', '--epochs', '1',
                '--seed', '11',
            ],
            1,
            'CoLA TSV row 3 has 3 columns; expected 4',
        ),
        (['missing.safetensors', '--seed', '-1'], 1, 'seed must be between'),
    ],
    ids=['help', 'bad-row', 'invalid-seed'],
)
def test_benchmark_entry_points(tmp_path, args, code, error):
    (tmp_path / 'cola.tsv').write_text(
        'src\t1\tx\talpha\nsrc\t0\tx\tbeta\nsrc\t1\tx\n', encoding='utf-8'
    )

    def run(*command):
        result = subprocess.run(
            [sys.executable, *command, *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode, result.stdout, result.stderr

    returncode, stdout, stderr = run('-m', 'tmt', 'benchmark')
    assert run('-m', 'tmt.benchmark') == (returncode, stdout, stderr)
    assert returncode == code, stderr
    assert 'Traceback' not in stderr
    assert [path.name for path in tmp_path.iterdir()] == ['cola.tsv']
    if error is None:
        assert stdout.startswith('## benchmark')
    else:
        assert error in stderr
        assert stdout == ''
