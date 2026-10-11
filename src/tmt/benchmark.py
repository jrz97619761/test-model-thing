# Keep the documented module command as an alias, before any MLX imports.
if __name__ == '__main__':
    import sys
    from tmt.cli import main

    raise SystemExit(main(['benchmark', *sys.argv[1:]]))

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

from .main import Model
from . import validation

# Keep these bytes and windows fixed for checkpoint comparisons.
BYTE_FIXTURE = 'tmt-byte-fixtures-v1'
BYTE_DOCUMENTS = (
    b'The small boat crossed the lake before sunset. A light shone from the shore. '
    b'The crew followed the light and reached the dock. By morning, the boat was ready '
    b'for another trip across the lake.\n',
    b'The key is amber. Keep the key while you read the next sentences. '
    b'A bird sits on the fence. A train passes the station. Rain falls on the roof. '
    b'The key is still amber.\n',
    b'Raw bytes: \x00\xff\nUTF-8: caf\xc3\xa9 \xf0\x9f\x98\x80\r\n',
)
CONTEXT_WINDOWS = (1, 8, 32, 128)


class Classification(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2)

    def __call__(self, x: mx.array):
        return self.proj(x)


def cola(filepath: str):
    data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for row_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) != 4:
                raise ValueError(
                    f'CoLA TSV row {row_number} has {len(parts)} columns; expected 4.'
                )
            try:
                label = int(parts[1])
            except ValueError as error:
                raise ValueError(
                    f'CoLA TSV row {row_number} has invalid label {parts[1]!r}; '
                    'expected 0 or 1.'
                ) from error
            if label not in (0, 1):
                raise ValueError(
                    f'CoLA TSV row {row_number} has invalid label {parts[1]!r}; '
                    'expected 0 or 1.'
                )
            if not parts[3].strip():
                raise ValueError(f'CoLA TSV row {row_number} has an empty sentence.')
            data.append((parts[3].encode('utf-8'), label))

    return data


def mcc(tp, tn, fp, fn):
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    score = (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0
    return score * 100


def rollout(model: Model, b_s: bytes):
    model.reset()

    x = None
    for b in b_s:
        (x, _, _), _ = model.step(mx.array(b), frozen=True)
    if x is None:
        return None

    features = model.blocks[-1].features(x)
    if not isinstance(features, mx.array) or tuple(features.shape) != (model.dim,):
        actual_shape = (
            tuple(features.shape)
            if isinstance(features, mx.array)
            else type(features).__name__
        )
        raise ValueError(
            f'CoLA features must be an MLX array with shape ({model.dim},); '
            f'got {actual_shape}'
        )
    model.eval_state()
    mx.eval(features)
    return features


def evaluate(model, documents, windows=()):
    """Score next bytes with frozen steps and restore live model state."""
    documents, windows = tuple(documents), tuple(windows)
    if any(window <= 0 for window in windows):
        raise ValueError('evaluation windows must be positive')
    input_bytes = sum(len(document) for document in documents)
    targets = sum(max(len(document) - 1, 0) for document in documents)
    if targets == 0:
        raise ValueError(
            'evaluation requires at least one target byte pair '
            '(a document with at least two bytes)'
        )
    common_start = max(windows) if windows else None
    snapshot = model.snapshot_state()
    full_nll = common_nll = 0.0
    common_targets = 0

    def evaluation_step(current):
        (_, _, _), (logits, _) = model.step(mx.array(current), frozen=True)
        return logits

    def negative_log_likelihood(logits, target):
        return float((mx.logsumexp(logits) - logits[target]).item())

    def bits_per_byte(nll_sum, target_count):
        return nll_sum / (math.log(2.0) * target_count)

    try:
        for document in documents:
            model.reset()
            for offset, current in enumerate(document):
                logits = evaluation_step(current)
                target_offset = offset + 1
                if target_offset < len(document):
                    loss = negative_log_likelihood(logits, document[target_offset])
                    full_nll += loss
                    if common_start is not None and target_offset >= common_start:
                        common_nll += loss
                        common_targets += 1
            model.eval_state()

        scores = []
        for window in windows:
            window_nll = window_common_nll = 0.0
            for document in documents:
                for target_offset in range(1, len(document)):
                    model.reset()
                    start = max(0, target_offset - window)
                    logits = None
                    for current in document[start:target_offset]:
                        logits = evaluation_step(current)
                    loss = negative_log_likelihood(logits, document[target_offset])
                    window_nll += loss
                    if common_start is not None and target_offset >= common_start:
                        window_common_nll += loss
            common_bpb = (
                bits_per_byte(window_common_nll, common_targets)
                if common_targets
                else None
            )
            scores.append(
                {
                    'window': window,
                    'bpb': bits_per_byte(window_nll, targets),
                    'common_bpb': common_bpb,
                    'delta_bpb': common_bpb - bits_per_byte(common_nll, common_targets)
                    if common_targets
                    else None,
                }
            )
    finally:
        model.restore_state(snapshot)
        model.eval_state()

    return {
        'input_bytes': input_bytes,
        'targets': targets,
        'bpb': bits_per_byte(full_nll, targets),
        'common_targets': common_targets,
        'common_bpb': bits_per_byte(common_nll, common_targets)
        if common_targets
        else None,
        'windows': scores,
    }


def benchmark(
    model: Model,
    data: list,
    train: bool,
    head: Classification,
    optimizer: opt.AdamW,
    lossfn,
):
    tp, tn, fp, fn = 0, 0, 0, 0

    for b_s, label in data:
        if len(b_s) == 0:
            continue

        state = rollout(model, b_s)
        if state is None:
            continue

        if train:
            (_, choice), grads = mx.value_and_grad(lossfn, argnums=0)(
                head.trainable_parameters(), state, label
            )

            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state)

        else:
            choice = head(state)
        predicted = mx.argmax(choice).item()

        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 0 and label == 0:
            tn += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 1:
            fn += 1

    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'mcc': mcc(tp, tn, fp, fn)}


def run(
    path: str,
    epochs: int = 1,
    split: float = 0.5,
    data: str | None = None,
    *,
    model=None,
    seed=11,
):
    validation.seed(seed)

    rows = None
    if data is not None and epochs < 1:
        raise ValueError('epochs must be positive with cola-data')
    if data is not None:
        rows = cola(data)
        if rows == [] or len(rows) < 2:
            raise FileNotFoundError(
                f'CoLA TSV {data!r} must contain at least two valid four-column rows.'
            )

    if model is None:
        from tmt.cli import load_model

        model, _ = load_model(path, seed)
    model.freeze()
    byte_scores = evaluate(model, BYTE_DOCUMENTS, CONTEXT_WINDOWS)
    print(f'Frozen BPB: {byte_scores["bpb"]:.6f} ({byte_scores["targets"]} targets)')
    print(
        f'Common-target BPB: {byte_scores["common_bpb"]:.6f} '
        f'({byte_scores["common_targets"]} targets)'
    )
    for score in byte_scores['windows']:
        print(
            f'Window {score["window"]} BPB: {score["bpb"]:.6f}, '
            f'common BPB: {score["common_bpb"]:.6f}, '
            f'delta: {score["delta_bpb"]:+.6f}'
        )

    if data is None:
        return {'byte_scores': byte_scores, 'epochs': []}

    split = int(len(rows) * (min(max(split, 0.0), 1.0)))
    train, held = rows[:split], rows[split:]

    mx.random.seed(seed)
    head = Classification(model.dim)
    optimizer = opt.AdamW(learning_rate=1e-3)

    def lossfn(params, state: mx.array, target: int):
        head.update(params)
        choice = head(state)

        loss = nn.losses.cross_entropy(choice[None, :], mx.array([target])).mean()
        return loss, choice

    print(f'CoLA feature convention: {model.blocks[-1].feature_convention}')
    print('Starting benchmark.')

    epoch_scores = []
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1} / {epochs}')
        train_score = benchmark(model, train, True, head, optimizer, lossfn)
        held_score = benchmark(model, held, False, head, optimizer, lossfn)
        epoch_scores.append(
            {'epoch': epoch + 1, 'train': train_score, 'held': held_score}
        )
        for name, score in (('train', train_score), ('held', held_score)):
            print(
                f'Epoch {epoch + 1} {name}: T+ {score["tp"]}, T- {score["tn"]}, '
                f'F+ {score["fp"]}, F- {score["fn"]}, '
                f'MCC x 100: {score["mcc"]:.4f}'
            )
    return {
        'byte_scores': byte_scores,
        'feature_convention': model.blocks[-1].feature_convention,
        'epochs': epoch_scores,
    }
