import math
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from . import validation


type FeatureSource = Literal['final-output', 'recurrent-state']
type TemporalGradient = Literal['one-step-autodiff', 'rtu-v1']
type GradientValue = mx.array | dict[str, GradientValue] | list[GradientValue]
type GradientTree = dict[str, GradientValue]
type RuntimeSchema = tuple[tuple[str, tuple[int, ...], mx.Dtype], ...]


class Block(nn.Module):
    """Byte-step block with fixed, frozen runtime arrays."""

    def __init__(self) -> None:
        super().__init__()
        self._runtime_schema: RuntimeSchema | None = None

    @property
    def state_fields(self) -> tuple[str, ...]:
        return ()

    @property
    def trace_fields(self) -> tuple[str, ...]:
        return ()

    @property
    def temporal_gradient(self) -> TemporalGradient:
        return 'one-step-autodiff'

    @property
    def feature_source(self) -> FeatureSource:
        return 'final-output'

    def validate(self) -> None:
        trainable = dict(tree_flatten(self.trainable_parameters()))
        self._runtime_schema = validation.block_runtime_schema(self, trainable)

    def _validate_values(
        self, values: dict[str, mx.array], fields: tuple[str, ...]
    ) -> None:
        schema = self._runtime_schema
        if schema is None:
            self.validate()
            schema = self._runtime_schema
        assert schema is not None
        validation.block_runtime_values(values, fields, schema)

    def validate_result(
        self, enc: mx.array, x: mx.array, state: dict[str, mx.array]
    ) -> None:
        validation.block_output(x, enc)
        self._validate_values(state, self.state_fields)

    def reset(self) -> None:
        for name in self.state_fields + self.trace_fields:
            setattr(self, name, mx.zeros_like(getattr(self, name)))

    def probe(self) -> mx.array:
        return mx.array(0.0)

    def adjust_gradients(
        self,
        byte: mx.array,
        next_state: dict[str, mx.array],
        aux: Any,
        probe_grad: mx.array,
        grads: GradientTree,
    ) -> tuple[GradientTree, mx.array | None, dict[str, mx.array]]:
        return grads, None, {}

    def features(self, x: mx.array) -> mx.array:
        return x

    def snapshot_state(self) -> dict[str, mx.array]:
        saved = {
            name: mx.array(getattr(self, name))
            for name in self.state_fields + self.trace_fields
        }
        mx.eval(*saved.values())
        return saved

    def restore_state(self, saved: dict[str, mx.array]) -> None:
        self._validate_values(saved, self.state_fields + self.trace_fields)
        for name, value in saved.items():
            setattr(self, name, value)
        self.eval_state()

    def commit_state(
        self,
        state: dict[str, mx.array],
        traces: dict[str, mx.array] | None = None,
    ) -> None:
        self._validate_values(state, self.state_fields)
        validation.require(
            traces is None or isinstance(traces, dict),
            'block trace update must be a dictionary',
        )
        if traces:
            self._validate_values(traces, self.trace_fields)
        for name, value in {**state, **(traces or {})}.items():
            setattr(self, name, mx.stop_gradient(value))

    def eval_state(self) -> None:
        mx.eval(
            *(getattr(self, name) for name in self.state_fields + self.trace_fields)
        )


class Layer(Block):
    @property
    def state_fields(self) -> tuple[str, ...]:
        return ('states',)

    @property
    def trace_fields(self) -> tuple[str, ...]:
        return ('decaytrace', 'embedtrace')

    @property
    def temporal_gradient(self) -> TemporalGradient:
        return 'rtu-v1'

    @property
    def feature_source(self) -> FeatureSource:
        return 'recurrent-state'

    def __init__(self, dim: int, spread: int) -> None:
        super().__init__()
        halflives = mx.exp(mx.linspace(0.0, math.log(float(spread)), dim))
        retention = mx.exp(-math.log(2.0) / halflives)
        self.decay = mx.log(retention) - mx.log1p(-retention)
        self.states = mx.zeros((dim,))
        self.decaytrace = mx.zeros((dim,))
        self.embedtrace = mx.zeros((256, dim))
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias=False)
        self.silu = nn.SiLU()
        self.freeze(keys=['states', 'decaytrace', 'embedtrace'], recurse=False)

    def __call__(
        self, enc: mx.array, x: mx.array, probe: mx.array
    ) -> tuple[mx.array, dict[str, mx.array], mx.array]:
        decay = mx.sigmoid(self.decay)
        state = (decay * self.states) + enc + probe
        return x + self.silu(self.weights(self.norm(state))), {'states': state}, decay

    def probe(self) -> mx.array:
        return mx.zeros_like(self.states)

    def adjust_gradients(
        self,
        byte: mx.array,
        next_state: dict[str, mx.array],
        aux: mx.array,
        probe_grad: mx.array,
        grads: GradientTree,
    ) -> tuple[GradientTree, mx.array | None, dict[str, mx.array]]:
        decay = aux
        c_range = (mx.arange(256) == byte)[:, None].astype(mx.float32)
        decay_embedtrace = self.embedtrace * decay
        embedtrace = decay_embedtrace + c_range
        decaytrace = (decay * self.decaytrace) + (decay * (1.0 - decay) * self.states)
        grads['decay'] = probe_grad * decaytrace
        return (
            grads,
            probe_grad * decay_embedtrace,
            {'decaytrace': decaytrace, 'embedtrace': embedtrace},
        )

    def features(self, x: mx.array) -> mx.array:
        return self.states


BLOCKS = {'rtu': Layer}
