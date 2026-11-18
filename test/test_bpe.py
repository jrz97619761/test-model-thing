import json
import math

import pytest

from tmt.data import BPETokenizer


class FakeModel:
    """Frozen BPE model with uniform logits that records replayed tokens."""

    def __init__(self):
        self.tokenizer = BPETokenizer([[97, 98], [99, 100]])
        self.state = [42]
        self.replays = []

    def snapshot_state(self):
        return list(self.state)

    def restore_state(self, state):
        self.state = state

    def eval_state(self):
        pass

    def reset(self):
        self.state = []
        self.replays.append(self.state)

    def step(self, current, frozen=False):
        import mlx.core as mx

        assert frozen
        self.state.append(int(current.item()))
        return ((None, [], []), (mx.zeros((259,)), None))

    def sample(self, logits, key=None):
        import mlx.core as mx

        return mx.array(257)


def test_merged_token_gradients_and_checkpoint_round_trip(
    tmp_path, tiny_settings, snapshot
):
    import mlx.core as mx

    from tmt.checkpoint import load_checkpoint, new_model, save_checkpoint

    settings = {**tiny_settings, 'tokenizer': {'merges': [[97, 98]]}}
    model = new_model(settings, 11)
    assert model.encoder.embed.weight.shape == (258, 4)
    assert model.decoder.decode.weight.shape == (258, 4)
    assert model.blocks[0].embedtrace.shape == (258, 4)
    before = mx.array(model.encoder.embed.weight[257])
    model.train_step(257, 257, False)
    model.train_step(257, 256, True)
    assert not mx.array_equal(before, model.encoder.embed.weight[257]).item()
    assert mx.any(model.blocks[0].embedtrace[257] != 0).item()
    path = tmp_path / 'model.safetensors'
    save_checkpoint(path, model, settings)
    loaded, restored = load_checkpoint(path)
    assert restored == settings
    assert snapshot(loaded) == snapshot(model)
    assert loaded.tokenizer.encode(b'abab') == [257, 257]


def test_bpe_cli_train_resume_evaluate_generate(workspace, run_cli):
    from tmt.checkpoint import load_checkpoint

    tokenizer = {'merges': [[97, 98], [257, 257]]}
    (workspace / 'tokenizer.json').write_text(json.dumps(tokenizer))
    (workspace / 'train.bin').write_bytes(b'ababab xyz ababab xyz')
    args = ['--data', 'train.bin', '--seed', '11', '--updates', '2']
    code, _, error = run_cli(
        'train', '--run', 'bpe', '--bpe', 'tokenizer.json', *args,
        '--sample-every', '1', '--sample-bytes', '3',
        '--eval-data', 'dev.bin', '--eval-every', '1',
    )
    assert code == 0, error
    path = workspace / 'runs/bpe/model.safetensors'
    assert load_checkpoint(path)[1]['tokenizer'] == tokenizer
    manifest = json.loads((workspace / 'runs/bpe/manifest/run.json').read_text())
    assert manifest['model']['tokenizer'] == tokenizer
    (workspace / 'tokenizer.json').unlink()
    code, _, error = run_cli('train', str(path), '--resume', *args)
    assert code == 0, error
    assert load_checkpoint(path)[0].optimizer.state['step'].item() == 4
    code, _, error = run_cli(
        'evaluate', str(path), '--data', 'dev.bin', '--windows', '1', '2',
        '--output', 'score.json',
    )
    assert code == 0, error
    assert json.loads((workspace / 'score.json').read_text())['scored_bytes'] == 2
    code, _, error = run_cli(
        'generate', str(path), '--prompt', 'ab', '--bytes', '3',
        '--seed', '11', '--output', 'sample.bin',
    )
    assert code == 0, error
    assert len((workspace / 'sample.bin').read_bytes()) <= 3
    code, _, error = run_cli('benchmark', str(path), '--seed', '11')
    assert code == 0, error
    code, _, error = run_cli('train', str(path), '--resume', '--bpe', 'x', *args)
    assert code == 2
    assert '--bpe cannot be used with --resume' in error


def test_bpe_evaluation_scores_bytes_and_exact_context_windows():
    from tmt.benchmark import evaluate

    model = FakeModel()
    score = evaluate(model, [b'abcdx'], windows=(1, 2))
    assert model.state == [42]
    assert (score['input_bytes'], score['targets'], score['scored_bytes']) == (5, 2, 3)
    assert (score['common_targets'], score['common_scored_bytes']) == (2, 3)
    assert score['bpb'] == pytest.approx(2 * math.log2(259) / 3)
    assert all(
        window['bpb'] == pytest.approx(score['bpb'])
        for window in score['windows']
    )
    assert model.replays == [[257, 258, 120], [98], [100], [257], [258]]


def test_bpe_evaluation_rejects_documents_without_token_pairs():
    from tmt.benchmark import evaluate

    with pytest.raises(ValueError, match='target token pair'):
        evaluate(FakeModel(), [b'ab'])


def test_generation_and_sample_truncate_to_byte_limit(tmp_path, monkeypatch):
    from tmt import runtime

    model = FakeModel()
    monkeypatch.setattr(runtime, 'load_checkpoint', lambda path: (model, {}))
    output = tmp_path / 'output.bin'
    assert runtime.generate('ignored', 'ab', output, 7, count=3) == b'aba'
    assert output.read_bytes() == b'aba'
    assert model.state == [257, 257, 257]
    model.state = [42]
    assert runtime._training_sample(model, 'ab', 3, 7) == b'aba'
    assert model.state == [42]


def test_stop_ends_generation_and_training_sample(tmp_path, monkeypatch):
    import mlx.core as mx

    from tmt import runtime

    model = FakeModel()
    tokens = iter([257, 256, 257, 256])
    monkeypatch.setattr(model, 'sample', lambda *args, **kwargs: mx.array(next(tokens)))
    monkeypatch.setattr(runtime, 'load_checkpoint', lambda path: (model, {}))
    output = tmp_path / 'output.bin'
    assert runtime.generate('ignored', 'ab', output, 7, count=9) == b'ab'
    assert output.read_bytes() == b'ab'
    assert model.state == [257, 257]
    model.state = [42]
    assert runtime._training_sample(model, 'ab', 9, 7) == b'ab'
    assert model.state == [42]
