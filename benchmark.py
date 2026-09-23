import argparse, math, os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

from main import Model

class Classification(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2)

    def __call__(self, x: mx.array): return self.proj(x)

def cola(filepath: str):
    data = []

    try:
        with open(filepath, 'r', encoding = 'utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4: data.append((parts[3].encode('utf-8'), int(parts[1])))

    except FileNotFoundError: pass
    return data

def mcc(tp, tn, fp, fn):
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    score = (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0
    return score * 100

def rollout(model: Model, b_s: bytes):
    model.reset()

    for b in b_s: _, _ = model.step(mx.array(b), frozen = True)
    state = model.blocks[-1].states

    if state is not None: mx.eval(state)
    return state

def benchmark(model: Model, data: list, train: bool, head: Classification, optimizer: opt.AdamW, lossfn):
    tp, tn, fp, fn = 0, 0, 0, 0

    for i, (b_s, label) in enumerate(data):
        if len(b_s) == 0: continue
            
        state = rollout(model, b_s)
        if state is None: continue

        if train:
            (_, choice), grads = mx.value_and_grad(lossfn, argnums = 0)(head.trainable_parameters(), state, label)

            optimizer.update(head, grads)
            mx.eval(head.parameters(), optimizer.state)

        else: choice = head(state)
        predicted = mx.argmax(choice).item()

        if predicted == 1 and label == 1: tp += 1
        elif predicted == 0 and label == 0: tn += 1
        elif predicted == 1 and label == 0: fp += 1
        elif predicted == 0 and label == 1: fn += 1

        if i > 0 and i % 500 == 0: print(f'[{i} / {len(data) - 1}] {'train' if train else 'held'}: T+ {tp}, T- {tn}, F+ {fp}, F- {fn} ({mcc(tp, tn, fp, fn):.4f})')

    print(f'[{i} / {len(data) - 1}] {'train' if train else 'held'}: T+ {tp}, T- {tn}, F+ {fp}, F- {fn} ({mcc(tp, tn, fp, fn):.4f})')
        
def run(path: str, epochs: int, split: float, data: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model checkpoint not found at {path!r}.')

    model = Model(dim = 512, layers = 16, spread = 32, temp = 0.75, rate = 5e-4, bound = (40000, 120000))
    model.load(path)
    model.freeze()

    head = Classification(model.dim)
    optimizer = opt.AdamW(learning_rate = 1e-3)

    rows = cola(data)
    if rows == [] or len(rows) < 2:
        raise FileNotFoundError('Invalid or missing CoLA dataset. Download it again from https://nyu-mll.github.io/CoLA/.')

    split = int(len(rows) * (min(max(split, 0.0), 1.0)))
    train, held = rows[:split], rows[split:]

    def lossfn(params, state: mx.array, target: int):
        head.update(params)
        choice = head(state)

        loss = nn.losses.cross_entropy(choice[None, :], mx.array([target])).mean()
        return loss, choice

    print('Starting benchmark.')

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1} / {epochs}')
        benchmark(model, train, True, head, optimizer, lossfn)
        benchmark(model, held, False, head, optimizer, lossfn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'CoLA benchmark for test-model-thing')

    parser.add_argument('path')
    parser.add_argument('epochs', type = int)
    parser.add_argument('split', type = float)
    
    parser.add_argument('--data', default = 'CoLA/original/raw/in_domain_train.tsv')

    args = parser.parse_args()
    run(args.path, args.epochs, args.split, args.data)