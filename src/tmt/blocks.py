import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .validation import matches_array, require


class Block(nn.Module):
    """Byte-step block with fixed, frozen runtime arrays and optional temporal gradients."""

    state_fields = ()
    trace_fields = ()
    temporal_gradient = 'one-step-autodiff'
    feature_convention = 'final-output'

    def validate(self):
        require(isinstance(self.state_fields, tuple) and isinstance(self.trace_fields, tuple),
                'block state and trace fields must be tuples')
        fields = self.state_fields + self.trace_fields
        require(
            all(isinstance(name, str) and not name.startswith('_') for name in fields)
            and len(set(fields)) == len(fields),
            'block state and trace fields must be distinct public names in tuples',
        )
        for label in ('temporal_gradient', 'feature_convention'):
            require(isinstance(getattr(self, label), str) and getattr(self, label),
                    f'block {label} must be a nonempty string')
        trainable = dict(tree_flatten(self.trainable_parameters()))
        for name in fields:
            require(isinstance(getattr(self, name, None), mx.array),
                    f'block runtime field {name!r} must be an array')
            require(name not in trainable, f'block runtime field {name!r} must be frozen')
        self._runtime_schema = tuple((name, getattr(self, name).shape, getattr(self, name).dtype)
                                     for name in fields)

    def _validate_values(self, values, fields):
        require(isinstance(values, dict) and set(values) == set(fields),
                f'block state schema must contain exactly {fields!r}')
        schema = {name: (shape, dtype) for name, shape, dtype in self._runtime_schema}
        for name, value in values.items():
            shape, dtype = schema[name]
            require(matches_array(value, shape, dtype),
                    f'block runtime field {name!r} must retain shape {shape} and dtype {dtype}')

    def validate_result(self, enc, x, state):
        require(matches_array(x, enc.shape, enc.dtype),
                'block output must retain the input vector shape and dtype')
        self._validate_values(state, self.state_fields)

    def reset(self):
        for name in self.state_fields + self.trace_fields:
            setattr(self, name, mx.zeros_like(getattr(self, name)))

    def probe(self):
        return mx.array(0.0)

    def adjust_gradients(self, byte, next_state, aux, probe_grad, grads):
        return grads, None, {}

    def features(self, x):
        return x

    def snapshot_state(self):
        saved = {name: mx.array(getattr(self, name)) for name in self.state_fields + self.trace_fields}
        mx.eval(*saved.values())
        return saved

    def restore_state(self, saved):
        self._validate_values(saved, self.state_fields + self.trace_fields)
        for name, value in saved.items():
            setattr(self, name, value)
        self.eval_state()

    def commit_state(self, state, traces=None):
        self._validate_values(state, self.state_fields)
        require(traces is None or isinstance(traces, dict), 'block trace update must be a dictionary')
        if traces:
            self._validate_values(traces, self.trace_fields)
        for name, value in {**state, **(traces or {})}.items():
            setattr(self, name, mx.stop_gradient(value))

    def eval_state(self):
        mx.eval(*(getattr(self, name) for name in self.state_fields + self.trace_fields))


class Layer(Block):
    state_fields = ('states',)
    trace_fields = ('decaytrace', 'embedtrace')
    temporal_gradient = 'rtu-v1'
    feature_convention = 'recurrent-state'

    def __init__(self, dim: int, spread: int):
        super().__init__()
        halflives = mx.exp(mx.linspace(0.0, math.log(float(spread)), dim))
        retention = mx.exp(-math.log(2.0) / halflives)
        self.decay = mx.log(retention) - mx.log1p(-retention)
        self.states = mx.zeros((dim, ))
        self.decaytrace = mx.zeros((dim, ))
        self.embedtrace = mx.zeros((256, dim))
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias = False)
        self.silu = nn.SiLU()
        self.freeze(keys = ['states', 'decaytrace', 'embedtrace'], recurse = False)

    def __call__(self, enc, x, probe):
        decay = mx.sigmoid(self.decay)
        state = (decay * self.states) + enc + probe
        return x + self.silu(self.weights(self.norm(state))), {'states': state}, decay

    def probe(self):
        return mx.zeros_like(self.states)

    def adjust_gradients(self, byte, next_state, aux, probe_grad, grads):
        decay = aux
        c_range = (mx.arange(256) == byte)[:, None].astype(mx.float32)
        decay_embedtrace = self.embedtrace * decay
        embedtrace = decay_embedtrace + c_range
        decaytrace = (decay * self.decaytrace) + (decay * (1.0 - decay) * self.states)
        grads['decay'] = probe_grad * decaytrace
        return grads, probe_grad * decay_embedtrace, {'decaytrace': decaytrace, 'embedtrace': embedtrace}

    def features(self, x):
        return self.states


BLOCKS = {'rtu': Layer}


def validate_block(name):
    if not isinstance(name, str) or name not in BLOCKS:
        raise ValueError(f'unknown block {name!r}; valid blocks: {", ".join(sorted(BLOCKS))}')
    return name
