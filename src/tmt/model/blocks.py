from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as util

import math
from typing import Any

import tmt.util.validation as validation

# types

type GradValue = mx.array | dict[str, GradValue] | list[GradValue]
type GradTree = dict[str, GradValue]

type FieldNames = tuple[str, ...]
type Fields = dict[str, mx.array | None]

type FwdOutput = tuple[mx.array, Fields, Fields]
type BwdOutput = tuple[GradTree, Fields, Fields]

type ConfigValue = int | float | bool | tuple[ConfigValue, ...]
type Config = dict[str, ConfigValue]

type BlockSchema = dict[str, tuple[tuple[int, ...], mx.Dtype]]

# base classes

class Block(nn.Module):
    def __init__(self, config: Config, aux: Config) -> None:
        super().__init__()
        self.config = config
        self.aux = aux
        self.schema = None

    # implementable

    @property
    def statefields(self) -> FieldNames: return ()

    @property
    def tracefields(self) -> FieldNames: return ()

    def probe(self) -> mx.array: return mx.array(0.0)

    def features(self, x: mx.array) -> mx.array: return x

    def __call__(self, inputs: Fields, probe: mx.array) -> FwdOutput: return inputs['x'], {}, {}

    def grads(self, output: FwdOutput, grads: GradTree, probegrad: mx.array) -> BwdOutput:
        return grads, {}, {}

    # field handling

    def fields(self) -> FieldNames: return self.statefields + self.tracefields

    def freezeall(self) -> None:
        self.freeze(keys = list(self.fields()), recurse = False)

    def reset(self) -> None:
        for field in self.fields(): setattr(self, field, mx.zeros_like(getattr(self, field)))

    def snapshot(self) -> Fields:
        saved: Fields = {field: mx.array(getattr(self, field)) for field in self.fields()}
        mx.eval(*saved.values())

        return saved

    def restore(self, fields: Fields) -> None:
        self.validate(self.fields(), fields)
        
        for field, data in fields.items(): setattr(self, field, data)
        self.evaluate()

    def commit(self, states: Fields, traces: Fields) -> None:
        self.validate(self.statefields, states)
        self.validate(self.tracefields, traces, strict = False)

        for field, data in {**states, **traces}.items(): setattr(self, field, mx.stop_gradient(data))

    def evaluate(self) -> None:
        mx.eval(*(getattr(self, field) for field in self.fields()))

    # validation

    def create_schema(self) -> BlockSchema:
        validation.instance(self.statefields, tuple, 'block state fields')
        validation.instance(self.tracefields, tuple, 'block trace fields')

        for name in self.fields():
            validation.instance(name, str, 'block field name')
            validation.require(not name.startswith('_'), f'block field \'{name!r}\' should be public')

        validation.require(len(set(self.fields())) == len(self.fields()), 'duplicate block fields should not exist')

        self.schema = {}
        trainable = dict(util.tree_flatten(self.trainable_parameters()))

        for name in self.fields():
            value = getattr(self, name, None)

            validation.instance(value, mx.array, f'block field \'{name!r}\'')
            validation.require(name not in trainable, f'block field \'{name!r}\' must be frozen')

            self.schema[name] = (value.shape, value.dtype)

        return self.schema

    def validate(self, names: FieldNames, fields: Fields, strict: bool = True):
        if not strict and not fields: return

        validation.require(set(names) == set(fields), f'block field names do not match block fields. names: {set(names)}, fields: {set(fields)}')
        if self.schema is None: self.create_schema()

        for name, value in fields.items():
            shape, dtype = self.schema[name]
            validation.array_shape(value, shape, dtype, f'block field: {name}')

    def validate_output(self, x: mx.array, enc: mx.array, states: Fields):
        validation.array_match(x, enc, 'model block output x')
        self.validate(self.statefields, states)

class Loss:
    def __init__(self, config: Config, aux: Config) -> None:
        self.config = config
        self.aux = aux

    def loss(self, fields: Fields, metrics: Fields) -> mx.array: return mx.array(0.0)

# implementations

class TMT0Block(Block):
    def __init__(self, config: Config, aux: Config) -> None:
        super().__init__(config, aux)

        # decay
        halflives = mx.exp(mx.linspace(0.0, math.log(float(config['spread'])), config['dim']))
        retention = mx.exp(-math.log(2.0) / halflives)
        self.decay = mx.log(retention) - mx.log1p(-retention)

        # states and traces
        self.states = mx.zeros((config['dim'],))
        self.decaytrace = mx.zeros((config['dim'],))
        self.embedtrace = mx.zeros((aux['vocab'], config['dim']))

        self.freezeall()

        # modules
        self.norm = nn.LayerNorm(config['dim'])
        self.weights = nn.Linear(config['dim'], config['dim'], bias = False)
        self.silu = nn.SiLU()
    
    @property
    def statefields(self) -> FieldNames: return ('states', )

    @property
    def tracefields(self) -> FieldNames: return ('decaytrace', 'embedtrace')

    def probe(self) -> mx.array: return mx.zeros_like(self.states)

    def features(self, x: mx.array) -> mx.array: return self.states

    def __call__(self, inputs: Fields, probe: mx.array) -> FwdOutput:
        decay = mx.sigmoid(self.decay)
        state = (decay * self.states) + inputs['enc'] + probe
        pred = inputs['x'] + self.silu(self.weights(self.norm(state)))

        return pred, {'states': state}, {'decay': decay}

    def grads(self, output: FwdOutput, grads: GradTree, probegrad: mx.array) -> BwdOutput:
        decay = output[2]['decay']

        c_range = (mx.arange(self.aux['vocab']) == output[0])[:, None].astype(mx.float32)
        decay_embedtrace = self.embedtrace * decay

        embedtrace = decay_embedtrace + c_range
        decaytrace = (decay * self.decaytrace) + (decay * (1.0 - decay) * self.states)

        grads['decay'] = probegrad * decaytrace
        return grads, {'decaytrace': decaytrace, 'embedtrace': embedtrace}, {'encoder': probegrad * decay_embedtrace}

class CEOnlyLoss(Loss):
    def __init__(self, config: Config, aux: Config): super().__init__(config, aux)

    def loss(self, fields: Fields, metrics: Fields) -> mx.array:
        output, target = fields['output'], fields['target']

        if target is not None: metrics['ce'] = -output[target] + mx.logsumexp(output)
        else: metrics['ce'] = mx.array(0.0)

        return metrics['ce']

class TMT0Loss(Loss):
    def __init__(self, config: Config, aux: Config): super().__init__(config, aux)

    def loss(self, fields: Fields, metrics: Fields) -> mx.array:
        x, output, target, target_emb = fields['x'], fields['output'], fields['target'], fields['target_emb']

        loss = metrics['variance'] = mx.maximum(0.0, 1.0 - mx.sqrt(mx.var(x) + 1e-4))
        
        metrics['latentpred'] = mx.mean(mx.square(x - target_emb))
        metrics['crossentropy'] = -output[target] + mx.logsumexp(output)

        loss = loss + metrics['latentpred']
        loss = loss - output[target] + mx.logsumexp(output) # keep this order - reyvvon

        return loss

# register

BLOCKS: dict[str, Any] = {
    'tmt0': TMT0Block
}

LOSSES: dict[str, Any] = {
    'ceonly': CEOnlyLoss,
    'tmt0': TMT0Loss
}