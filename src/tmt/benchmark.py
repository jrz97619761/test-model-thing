"""Frozen byte evaluation, the fixed byte suite, and optional CoLA scores."""

from __future__ import annotations

# Keep the documented module command as an alias, before any MLX imports.
if __name__ == '__main__':
    import sys

    from .cli import console_main

    raise SystemExit(console_main(['benchmark', *sys.argv[1:]]))

import math
from collections.abc import Callable, Iterable
from os import PathLike
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

from . import validation
from .checkpoint import preserved_rng
from .model import Model
from .records import write_json

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


def load_cola(filepath: str | PathLike[str]) -> list[tuple[bytes, int]]:
    data = []

    with open(filepath, encoding='utf-8') as file:
        for row_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) != 4:
                raise ValueError(
                    f'CoLA TSV row {row_number} has {len(parts)} columns; expected 4'
                )
            try:
                label = int(parts[1])
            except ValueError as error:
                raise ValueError(
                    f'CoLA TSV row {row_number} has invalid label {parts[1]!r}; '
                    'expected 0 or 1'
                ) from error
            if label not in (0, 1):
                raise ValueError(
                    f'CoLA TSV row {row_number} has invalid label {parts[1]!r}; '
                    'expected 0 or 1'
                )
            if not parts[3].strip():
                raise ValueError(f'CoLA TSV row {row_number} has an empty sentence')
            data.append((parts[3].encode('utf-8'), label))

    return data


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    score = (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0
    return score * 100


def rollout(model: Model, sentence: bytes) -> mx.array | None:
    model.reset()

    x = None
    for byte in sentence:
        (x, _, _), _ = model.step(mx.array(byte), frozen=True)
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


def evaluate(
    model: Model, documents: Iterable[bytes], windows: Iterable[int] = ()
) -> dict[str, Any]:
    """Score next bytes with frozen steps and restore live model state."""
    documents, windows = tuple(documents), tuple(windows)
    if any(window <= 0 for window in windows):
        raise ValueError('evaluation windows must be positive')
    input_bytes = sum(len(document) for document in documents)
    targets = validation.evaluation_targets(documents)
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


def score_cola(
    model: Model,
    data: list[tuple[bytes, int]],
    train: bool,
    head: nn.Linear,
    optimizer: opt.AdamW,
    loss_fn: Callable[..., tuple[mx.array, mx.array]],
) -> dict[str, Any]:
    tp, tn, fp, fn = 0, 0, 0, 0

    for sentence, label in data:
        state = rollout(model, sentence)
        if state is None:
            continue

        if train:
            (_, choice), grads = mx.value_and_grad(loss_fn, argnums=0)(
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
    path: str | PathLike[str],
    epochs: int = 1,
    split: float = 0.5,
    data: str | None = None,
    *,
    model: Model | None = None,
    seed: int,
    output: str | None = None,
) -> dict[str, Any]:
    validation.seed(seed)
    validation.require(0 <= split <= 1, 'split must be between 0 and 1')
    if output is not None:
        validation.output_folder(output)

    rows = None
    if data is not None and epochs < 1:
        raise ValueError('epochs must be positive with cola-data')
    if data is not None:
        rows = load_cola(data)
        if len(rows) < 2:
            raise ValueError(
                f'CoLA TSV {data!r} must contain at least two valid four-column rows'
            )

    if model is None:
        from .checkpoint import load_checkpoint

        model, _ = load_checkpoint(path)
    model.freeze()
    optimizer_step = int(model.optimizer.state['step'].item())
    byte_scores = evaluate(model, BYTE_DOCUMENTS, CONTEXT_WINDOWS)
    print(f'frozen BPB: {byte_scores["bpb"]:.6f}, targets: {byte_scores["targets"]}')
    print(
        f'common BPB: {byte_scores["common_bpb"]:.6f}, '
        f'targets: {byte_scores["common_targets"]}'
    )
    for score in byte_scores['windows']:
        print(
            f'window {score["window"]}: BPB {score["bpb"]:.6f}, '
            f'common BPB {score["common_bpb"]:.6f}, delta {score["delta_bpb"]:+.6f}'
        )

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        if output is not None:
            record = {
                **result,
                'checkpoint': str(path),
                'optimizer_step': optimizer_step,
                'seed': seed,
            }
            write_json(output, record)
        return result

    if data is None:
        return finish({'byte_scores': byte_scores, 'epoch_scores': []})

    split = int(len(rows) * split)
    train, held = rows[:split], rows[split:]

    with preserved_rng():
        mx.random.seed(seed)
        head = nn.Linear(model.dim, 2)
    optimizer = opt.AdamW(learning_rate=1e-3)

    def loss_fn(params, state: mx.array, target: int):
        head.update(params)
        choice = head(state)

        loss = nn.losses.cross_entropy(choice[None, :], mx.array([target])).mean()
        return loss, choice

    print(f'CoLA feature source: {model.blocks[-1].feature_source}')

    epoch_scores = []
    for epoch in range(epochs):
        train_score = score_cola(model, train, True, head, optimizer, loss_fn)
        held_score = score_cola(model, held, False, head, optimizer, loss_fn)
        epoch_scores.append(
            {'epoch': epoch + 1, 'train': train_score, 'held': held_score}
        )
        for name, score in (('train', train_score), ('held', held_score)):
            print(
                f'epoch {epoch + 1}/{epochs} {name}: '
                f'T+ {score["tp"]}, T- {score["tn"]}, '
                f'F+ {score["fp"]}, F- {score["fn"]}, '
                f'MCC x 100: {score["mcc"]:.4f}'
            )
    return finish(
        {
            'byte_scores': byte_scores,
            'feature_source': model.blocks[-1].feature_source,
            'epoch_scores': epoch_scores,
        }
    )
