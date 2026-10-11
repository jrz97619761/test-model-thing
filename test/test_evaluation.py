import contextlib
import math
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    'windows',
    [(0,), (-1,), (1, 0)],
    ids=['zero-window', 'negative-window', 'mixed-valid-and-zero'],
)
def test_invalid_windows(evaluation_model, windows):
    from tmt.benchmark import evaluate

    with pytest.raises(ValueError, match='windows must be positive'):
        evaluate(evaluation_model, [b'ab'], windows)


@pytest.mark.parametrize(
    'documents',
    [[], [b''], [b'a', b'b']],
    ids=['no-documents', 'empty-document', 'single-byte-documents'],
)
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
    assert_almost_equal(
        result['bpb'], total_nll() / (target_count * math.log(2)), delta=1e-5
    )
    common_baseline_bpb = total_nll(common_only=True) / (
        common_target_count * math.log(2)
    )
    assert_almost_equal(result['common_bpb'], common_baseline_bpb, delta=1e-5)
    for score, window in zip(result['windows'], windows):
        window_bpb = total_nll(window) / (target_count * math.log(2))
        window_common_bpb = total_nll(window, common_only=True) / (
            common_target_count * math.log(2)
        )
        assert_almost_equal(score['bpb'], window_bpb, delta=1e-5)
        assert_almost_equal(score['common_bpb'], window_common_bpb, delta=1e-5)
        assert_almost_equal(
            score['delta_bpb'], window_common_bpb - common_baseline_bpb, delta=1e-5
        )


def test_short_context(evaluation_model):
    from tmt.benchmark import evaluate

    score = evaluate(evaluation_model, [b'ab'], (3,))
    assert score['common_targets'] == 0
    assert score['common_bpb'] is None
    assert score['windows'][0]['delta_bpb'] is None


def test_score_calculation_failure_restores_state(evaluation_model, monkeypatch):
    from tmt.benchmark import evaluate

    def fail_score(*args):
        raise RuntimeError('injected score failure')

    monkeypatch.setattr('mlx.core.logsumexp', fail_score)
    with pytest.raises(RuntimeError, match='injected score failure'):
        evaluate(evaluation_model, [b'aa', b'abcde', b'', b'xy', b'pqrs'], (1, 2))


def test_benchmark_results(assert_almost_equal, tmp_path, saved_checkpoint):
    import mlx.utils as util
    from tmt.benchmark import BYTE_DOCUMENTS, CONTEXT_WINDOWS, mcc, run

    checkpoint = saved_checkpoint.path
    mx = saved_checkpoint.mx
    model = saved_checkpoint.model
    fields = ('states', 'decaytrace', 'embedtrace')
    before = {
        name: mx.array(value)
        for name, value in util.tree_flatten(model.parameters())
        if not any(field in name for field in fields)
    }
    optimizer_before = dict(util.tree_flatten(model.optimizer.state))
    mx.eval(*before.values(), *optimizer_before.values())
    data = tmp_path / 'cola.tsv'
    data.write_text(
        'src\t1\tx\talpha\nsrc\t0\tx\tbeta\nsrc\t0\tx\tgamma\nsrc\t1\tx\tdelta\n'
    )
    from io import StringIO

    stdout = StringIO()
    states = [mx.array(getattr(model.blocks[0], field)) for field in fields]
    with contextlib.redirect_stdout(stdout):
        byte_only = run(str(checkpoint), model=model)
    assert byte_only['epochs'] == []
    assert byte_only['byte_scores']['targets'] == 390
    assert '104 targets' in stdout.getvalue()
    for saved, field in zip(states, fields):
        assert mx.array_equal(saved, getattr(model.blocks[0], field)).item(), (
            f'benchmark changed RTU field {field!r}'
        )
    with contextlib.redirect_stdout(stdout):
        result = run(str(checkpoint), 1, 0.5, str(data), model=model, seed=11)
    assert byte_only['byte_scores'] == result['byte_scores']
    assert result['feature_convention'] == 'recurrent-state'
    assert 'CoLA feature convention: recurrent-state' in stdout.getvalue()
    for partition in ('train', 'held'):
        score = result['epochs'][0][partition]
        classification_count = sum(score[key] for key in ('tp', 'tn', 'fp', 'fn'))
        assert classification_count == 2

    scores = result['byte_scores']
    fixture_bytes = sum(len(document) for document in BYTE_DOCUMENTS)
    assert scores['input_bytes'] == fixture_bytes == 393
    assert scores['targets'] == 390
    assert scores['common_targets'] == 104
    actual_windows = tuple(item['window'] for item in scores['windows'])
    assert actual_windows == CONTEXT_WINDOWS
    held = result['epochs'][0]['held']
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
    assert_almost_equal(held['mcc'], expected_mcc)
    assert mcc(1, 1, 0, 0) == 100.0
    after = dict(util.tree_flatten(model.parameters()))
    optimizer_after = dict(util.tree_flatten(model.optimizer.state))
    for group, expected, actual in (
        ('model', before, after),
        ('optimizer', optimizer_before, optimizer_after),
    ):
        for name, value in expected.items():
            assert mx.array_equal(value, actual[name]).item(), (
                f'benchmark changed {group} tensor {name!r}'
            )


def test_benchmark_rejects_invalid_cola_features(
    monkeypatch, tmp_path, saved_checkpoint
):
    import mlx.core as mx
    from tmt.benchmark import run

    checkpoint = saved_checkpoint.path
    model = saved_checkpoint.model
    data = tmp_path / 'cola.tsv'
    data.write_text(
        'src\t1\tx\talpha\nsrc\t0\tx\tbeta\nsrc\t0\tx\tgamma\nsrc\t1\tx\tdelta\n'
    )

    monkeypatch.setattr(
        'tmt.benchmark.evaluate',
        lambda *args: {
            'bpb': 8.0,
            'targets': 1,
            'common_bpb': 8.0,
            'common_targets': 1,
            'windows': [],
        },
    )
    monkeypatch.setattr(
        model.blocks[-1], 'features', lambda x: mx.zeros((model.dim, 1))
    )
    with pytest.raises(
        ValueError, match='CoLA features must be an MLX array with shape'
    ):
        run(str(checkpoint), 1, 0.5, str(data), model=model, seed=11)


def test_benchmark_rejects_missing_cola_file(tmp_path):
    from tmt.benchmark import run

    missing_data = tmp_path / 'missing.tsv'
    with pytest.raises(FileNotFoundError):
        run('unused.safetensors', data=str(missing_data))


@pytest.mark.parametrize('epochs', [0, -1])
@pytest.mark.parametrize('supply_model', [False, True])
def test_invalid_cola_epochs(monkeypatch, forbid_call, epochs, supply_model):
    from tmt.benchmark import run

    monkeypatch.setattr('tmt.cli.load_model', forbid_call)
    monkeypatch.setattr('tmt.benchmark.evaluate', forbid_call)
    model = SimpleNamespace(freeze=forbid_call) if supply_model else None
    with pytest.raises(ValueError, match='epochs must be positive with cola-data'):
        run('missing.safetensors', epochs, data='missing.tsv', model=model)
