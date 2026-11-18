"""Frozen byte evaluation, the fixed byte suite, and optional CoLA scores."""

from __future__ import annotations

# Keep the documented module command as an alias, before any MLX imports.
if __name__ == '__main__':
    import sys

    from .cli import console_main

    raise SystemExit(console_main(['benchmark', *sys.argv[1:]]))

import math
from collections.abc import Callable, Iterable, Sequence
from os import PathLike
from typing import Any, NamedTuple

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
    for token in model.tokenizer.encode(sentence):
        (x, _, _), _ = model.step(mx.array(token), frozen=True)
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


class _Target(NamedTuple):
    """One scored next token and its byte span in the source document."""

    document: bytes
    token: int
    start: int
    size: int


def _targets(
    model: Model, documents: tuple[bytes, ...], encoded: list[Sequence[int]]
) -> list[_Target]:
    """List every token after the first in each document, in document order."""
    targets = []
    for document, tokens in zip(documents, encoded):
        start = 0
        for index, token in enumerate(tokens):
            size = len(model.tokenizer.decode(token))
            if index:
                targets.append(_Target(document, token, start, size))
            start += size
    return targets


def evaluate(
    model: Model, documents: Iterable[bytes], windows: Iterable[int] = ()
) -> dict[str, Any]:
    """Score next tokens in bits per target byte and restore live model state.

    Byte mode scores one byte per target. A window encodes its byte suffix again,
    so its BPE tokens can differ from the full-document tokens.
    """
    documents, windows = tuple(documents), tuple(windows)
    if any(window <= 0 for window in windows):
        raise ValueError('evaluation windows must be positive')
    encoded = [model.tokenizer.encode(document) for document in documents]
    targets = _targets(model, documents, encoded)
    if not targets:
        raise ValueError(
            'evaluation requires at least one target token pair '
            '(a document with at least two tokens)'
        )
    common_start = max(windows, default=None)
    common = [
        common_start is not None and target.start >= common_start
        for target in targets
    ]
    common_targets = sum(common)
    scored_bytes = sum(target.size for target in targets)
    common_bytes = sum(target.size for target, chosen in zip(targets, common) if chosen)
    snapshot = model.snapshot_state()

    def evaluation_step(current):
        _, (logits, _) = model.step(mx.array(current), frozen=True)
        return logits

    def negative_log_likelihood(logits, target):
        return float((mx.logsumexp(logits) - logits[target]).item())

    def bits_per_byte(losses, common_only=False):
        if common_only:
            if not common_bytes:
                return None
            nll = sum(loss for loss, chosen in zip(losses, common) if chosen)
            return nll / (math.log(2.0) * common_bytes)
        return sum(losses) / (math.log(2.0) * scored_bytes)

    try:
        full = []
        for tokens in encoded:
            model.reset()
            logits = None
            for index, current in enumerate(tokens):
                if index:
                    full.append(negative_log_likelihood(logits, current))
                # The final token is consumed without a target, as in training.
                logits = evaluation_step(current)
            model.eval_state()

        by_window = []
        for window in windows:
            losses = []
            for target in targets:
                model.reset()
                context = target.document[max(0, target.start - window):target.start]
                for current in model.tokenizer.encode(context):
                    logits = evaluation_step(current)
                losses.append(negative_log_likelihood(logits, target.token))
            by_window.append(losses)
    finally:
        model.restore_state(snapshot)
        model.eval_state()

    common_bpb = bits_per_byte(full, common_only=True)
    scores = []
    for window, losses in zip(windows, by_window):
        window_common_bpb = bits_per_byte(losses, common_only=True)
        scores.append(
            {
                'window': window,
                'bpb': bits_per_byte(losses),
                'common_bpb': window_common_bpb,
                'delta_bpb': None
                if window_common_bpb is None
                else window_common_bpb - common_bpb,
            }
        )
    return {
        'input_bytes': sum(len(document) for document in documents),
        'targets': len(targets),
        'scored_bytes': scored_bytes,
        'bpb': bits_per_byte(full),
        'common_targets': common_targets,
        'common_scored_bytes': common_bytes,
        'common_bpb': common_bpb,
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
