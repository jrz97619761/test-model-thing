import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

import tmt.util.validation as validation
import tmt.runtime.checkpoint as checkpoint
import tmt.model.model as model
import tmt.model.blocks as blocks

import math
from typing import Any, Iterable, Sequence

# CoLA

type CoLAData = list[tuple[bytes, int]]

class CoLAHead(nn.Module):
    def __init__(self, config: blocks.Config) -> None:
        super().__init__()
        self.head = nn.Linear(config['dim'], 2)
        self.opt = opt.AdamW(learning_rate = 1e-3)

    def __call__(self, x: mx.array) -> mx.array: return self.head(x)

    def loss(self, params, states: mx.array, label: int):
        self.update(params)
        choice = self(states)

        loss = nn.losses.cross_entropy(choice[None, :], mx.array([label])).mean()
        return loss, choice

def cola_load(path: validation.FilePath) -> CoLAData:
    data = []

    with open(path, encoding = 'utf-8') as file:
        for row, line in enumerate(file, start = 1):
            if not line.strip(): continue
            split = line.rstrip('\r\n').split('\t')

            validation.require(len(split) == 4, f'CoLA row {row} has {len(split)} columns, expected 4')
            with validation.trycatch(ValueError, f'CoLA row {row} has invalid label {split[1]!r}, expected 0 or 1'):
                label = int(split[1])
            validation.require(label in (0, 1), f'CoLA row {row} has invalid label {split[1]!r}, expected 0 or 1')
            validation.require(split[3].strip(), f'CoLA row {row} has empty sentence')

            data.append((split[3].encode('utf-8'), label))

    validation.require(len(data) >= 2, 'CoLA data must contain at least two valid 4-column rows')
    return data

def cola_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    score = (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0
    return score * 100

def cola_rollout(model: model.Model, sentence: bytes) -> mx.array:
    model.reset()
    for token in model.tokenizer.encode(sentence): (x, _, _), _ = model.step(mx.array(token), frozen = True)

    features = model.blocks[-1].features(x)
    validation.instance(features, mx.array, 'CoLA features')

    validation.require(
        tuple(features.shape) == (model.config['dim'], ),
        f'CoLA features should have shape {(model.config['dim'], )}, got {tuple(features.shape)}'
    )

    model.evaluate()
    mx.eval(features)
    return features

def cola_score(model: model.Model, data: CoLAData, head: CoLAHead, train: bool) -> dict[str, Any]:
    tp = tn = fp = fn = 0

    for sentence, label in data:
        if len(sentence) == 0: continue
        state = cola_rollout(model, sentence)

        if train:
            (_, choice), grads = mx.value_and_grad(head.loss, argnums = 0)(
                head.trainable_parameters(), state, label
            )

            head.opt.update(head, grads)
            mx.eval(head.parameters(), head.opt.state)

        else: choice = head(state)
        pred = mx.argmax(choice).item()

        match (pred, label):
            case (1, 1): tp += 1
            case (0, 0): tn += 1
            case (1, 0): fp += 1
            case (0, 1): fn += 1

    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'mcc': cola_mcc(tp, tn, fp, fn)}

# bpb

BPB_SENTENCES: tuple[bytes, ...] = (
    b'The small boat crossed the lake before sunset. A light shone from the shore. '
    b'The crew followed the light and reached the dock. By morning, the boat was ready '
    b'for another trip across the lake.\n',
    b'The key is amber. Keep the key while you read the next sentences. '
    b'A bird sits on the fence. A train passes the station. Rain falls on the roof. '
    b'The key is still amber.\n',
    b'Raw bytes: \x00\xff\nUTF-8: caf\xc3\xa9 \xf0\x9f\x98\x80\r\n',
)

BPB_WINDOWS: tuple[int, ...] = (1, 8, 32, 128)

type BPBTarget = tuple[bytes, int, int, int]

def bpb_targets(model: model.Model, documents: Iterable[bytes], encoded: list[Sequence[int]]) -> list[BPBTarget]:
    targets = []

    for document, tokens in zip(documents, encoded):
        start = 0
        for index, token in enumerate(tokens):
            size = len(model.tokenizer.decode(token))
            if index: targets.append((document, token, start, size))

            start += size

    return targets

def bpb_score(model: model.Model, documents: Iterable[bytes], windows: tuple[int, ...] = ()) -> dict[str, Any]:
    validation.require(all(window > 0 for window in windows), 'bit-per-byte eval window <= 0')

    encoded = [model.tokenizer.encode(document) for document in documents]
    targets = bpb_targets(model, documents, encoded)
    validation.require(targets, 'bit-per-byte eval requires at least 1 byte pair')

    common_start = max(windows, default = None)
    common = [
        common_start is not None and target[2] >= common_start
        for target in targets
    ]
    common_targets = sum(common)

    scored_bytes = sum(target[3] for target in targets)
    common_bytes = sum(target[3] for target, chosen in zip(targets, common) if chosen)

    snapshot = model.snapshot()

    def eval(byte) -> mx.array:
        (_, _, _), (logits, _) = model.step(mx.array(byte), frozen = True)
        return logits

    def nll(logits: mx.array, target: mx.array) -> float:
        return float((mx.logsumexp(logits) - logits[target]).item())

    def bpb(losses: list[float], common_only = False) -> float | None:
        if common_only:
            if not common_bytes: return None

            nll = sum(loss for loss, chosen in zip(losses, common) if chosen)
            return nll / (math.log(2.0) * common_bytes)

        return sum(losses) / (math.log(2.0) * scored_bytes)

    try:
        full = []
        for tokens in encoded:
            model.reset()
            logits = None

            for index, current in enumerate(tokens):
                if index: full.append(nll(logits, current))
                logits = eval(current)

            model.evaluate()

        by_window = []
        for window in windows:
            losses = []

            for target in targets:
                model.reset()
                context = target[0][max(0, target[2] - window) : target[2]]

                for current in model.tokenizer.encode(context): logits = eval(current)
                losses.append(nll(logits, target[1]))

            by_window.append(losses)
        
    finally:
        model.restore(snapshot)
        model.evaluate()

    bpb_common = bpb(full, common_only = True)

    scores = []
    for window, losses in zip(windows, by_window):
        bpb_common_window = bpb(losses, common_only = True)
        scores.append({
            'window': window,

            'bpb': bpb(losses),
            'bpb_common': bpb_common_window,
            'bpb_delta': None if bpb_common_window is None else bpb_common_window - bpb_common
        })

    return {
        'inputs': sum(len(document) for document in documents),
        'full': {
            'targets': len(targets),
            'scored_bytes': scored_bytes,
            'bpb': bpb(full)
        },
        'common': {
            'targets': common_targets,
            'scored_bytes': common_bytes,
            'bpb': bpb_common,
        } if common_start is not None else None,
        'windows': scores,
    }

# benchmark

def benchmark(
        model: model.Model, cola: validation.FilePath,
        seed: int, epochs: int = 1, split: float = 0.5
    ):
    validation.seed(seed)
    validation.require(0 <= split <= 1, 'split must be between 0 and 1')
    validation.positive(epochs, 'epochs')

    model.freeze()
    bpb = bpb_score(model, BPB_SENTENCES, BPB_WINDOWS)

    data = cola_load(cola)

    split = int(len(data) * split)
    train, held = data[:split], data[split:]

    with checkpoint.preserve_seed():
        checkpoint.seed(seed)
        head = CoLAHead(model.config)

    cola_scores = []
    for epoch in range(epochs):
        train_score = cola_score(model, train, head, train = True)
        held_score = cola_score(model, held, head, train = False)
        
        cola_scores.append({'epoch': epoch + 1, 'train': train_score, 'held': held_score})

    return {'cola': cola_scores, 'bpb': bpb}