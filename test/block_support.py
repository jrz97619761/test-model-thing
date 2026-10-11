import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from tmt.blocks import Block


class Stateless(Block):
    def __init__(self, dim, spread):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, enc, x, probe):
        return x + mx.tanh(self.proj(enc)), {}, ()


class MatrixCache(Block):
    @property
    def state_fields(self):
        return ('matrix', 'cache')

    @property
    def trace_fields(self):
        return ('history',)

    def __init__(self, dim, spread):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.matrix = mx.zeros((dim, dim))
        self.cache = mx.zeros((2, dim))
        self.history = mx.zeros((dim,))
        self.freeze(keys=list(self.state_fields + self.trace_fields), recurse=False)

    def __call__(self, enc, x, probe):
        matrix = 0.5 * self.matrix + enc[:, None] * enc[None, :]
        cache = mx.stack([self.cache[1], x])
        output = x + mx.tanh(
            self.proj(enc) + matrix.mean(axis=0) + self.cache.mean(axis=0)
        )
        return output, {'matrix': matrix, 'cache': cache}, ()


class NoWeights(Block):
    def __init__(self, dim, spread):
        super().__init__()

    def __call__(self, enc, x, probe):
        return x + mx.tanh(enc), {}, ()


def arrays(tree):
    return {name: mx.array(value) for name, value in tree_flatten(tree)}


def equal(left, right):
    assert set(left) == set(right), f'field sets differ: {set(left) ^ set(right)!r}'
    for name, expected in left.items():
        assert mx.array_equal(expected, right[name]).item(), f'array {name!r} differs'
