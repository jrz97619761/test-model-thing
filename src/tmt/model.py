"""Byte encoder, decoder, block stack, objective loss, and optimizer updates."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten

from . import validation
from .blocks import BLOCKS, Block


class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)

    def __call__(self, x: mx.array):
        return self.embed(x)


class Decoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def __call__(self, x: mx.array):
        return self.decode(x), mx.sigmoid(self.stop(x))


class Model(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: int,
        spread: int,
        temp: float,
        rate: float,
        bound: tuple[int, int],
        block: str,
    ):
        super().__init__()
        validation.block_name(block)
        self.dim = dim
        self.temp = temp

        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)

        self.blocks = [BLOCKS[block](dim, spread) for _ in range(layers)]
        for item in self.blocks:
            validation.require(
                isinstance(item, Block), 'block factories must return a Block module'
            )
            item.validate()

        def learning_rate(step: mx.array):
            progress = mx.clip(
                (step.astype(mx.float32) + 1.0 - float(bound[0]))
                / float(bound[1] - bound[0]),
                0.0,
                1.0,
            )
            return rate * (1.0 - 0.9 * progress)

        self.optimizer = opt.AdamW(learning_rate=learning_rate)
        self.compiled = None

    def sample(self, output: mx.array, key: mx.array | None = None):
        probs = mx.softmax(output)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8)) / mx.log(mx.array(256.0))

        temp = mx.maximum(0.1, self.temp * (1.0 - self.temp * entropy)).item()
        return mx.random.categorical(output / temp, key=key)

    def step(
        self,
        byte: mx.array,
        dummies: list[mx.array] | None = None,
        frozen: bool = False,
    ):
        if dummies is None:
            dummies = [block.probe() for block in self.blocks]

        enc = self.encoder(byte)
        x = enc

        states, auxiliary = [], []

        for i, block in enumerate(self.blocks):
            x, state, aux = block(enc, x, dummies[i])
            block.validate_result(enc, x, state)
            if frozen:
                block.commit_state(state)

            states.append(state)
            auxiliary.append(aux)

        return (x, states, auxiliary), self.decoder(x)

    def update_gradients(self, grads):
        if self.compiled is None:
            self.optimizer.update(self, grads)
            mx.eval(self.parameters(), self.optimizer.state)
            state = [self.state, self.optimizer.state]

            def update(gradients):
                self.optimizer.update(self, gradients)
                return self.optimizer.state['step']

            self.compiled = mx.compile(update, inputs=state, outputs=state)

        else:
            self.compiled(grads)
            mx.eval(self.parameters(), self.optimizer.state)

    def loss(
        self,
        x: mx.array,
        output: mx.array,
        stop: mx.array,
        target: int | None,
        end: bool,
        ce_only: bool = False,
        components: dict | None = None,
    ) -> mx.array:
        terms = {
            name: mx.array(0.0)
            for name in ('variance', 'latent_prediction', 'cross_entropy', 'stop')
        }
        if not ce_only:
            loss = mx.maximum(0.0, 1.0 - mx.sqrt(mx.var(x) + 1e-4))
            terms['variance'] = loss
            if target is not None:
                n = mx.array(target)
                target_embedding = mx.stop_gradient(self.encoder(n))

                terms['latent_prediction'] = mx.mean(mx.square(x - target_embedding))
                terms['cross_entropy'] = -output[n] + mx.logsumexp(output)
                terms['stop'] = mx.mean(
                    mx.square(stop - mx.array([1.0 if end else 0.0]))
                )
                loss = loss + terms['latent_prediction']
                # Keep this order: seeded tests pin the float32 loss bits.
                loss = loss - output[n] + mx.logsumexp(output)
                loss = loss + terms['stop']
        else:
            if target is not None:
                loss = -output[mx.array(target)] + mx.logsumexp(output)
            else:
                loss = mx.array(0.0)
            terms['cross_entropy'] = loss

        if components is not None:
            components.update(terms)

        return loss

    def train_step(
        self,
        current: int,
        target: int | None,
        end: bool,
        ce_only: bool = False,
        metrics: dict | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        byte = mx.array(current)
        params = self.trainable_parameters()

        def forward(params, dummies: list[mx.array]):
            self.update(params)
            (x, states, auxiliary), (output, stop) = self.step(byte, dummies)
            components = {}
            loss = self.loss(x, output, stop, target, end, ce_only, components)
            return loss, (states, auxiliary, output, stop, components)

        (loss, (states, auxiliary, output, stop, components)), (grads, probe_grads) = (
            mx.value_and_grad(forward, argnums=(0, 1))(
                params, [block.probe() for block in self.blocks]
            )
        )

        self.update(params)

        for i, block in enumerate(self.blocks):
            local = grads['blocks'][i] if 'blocks' in grads else {}
            templates = dict(tree_flatten(local))
            adjusted, encoder_addition, traces = block.adjust_gradients(
                byte, states[i], auxiliary[i], probe_grads[i], local
            )
            actual = dict(tree_flatten(adjusted))
            validation.require(
                validation.matches_arrays(actual, templates),
                'block gradient hook must retain local gradient names, '
                'shapes, and dtypes',
            )
            if 'blocks' in grads:
                grads['blocks'][i] = adjusted
            if encoder_addition is not None:
                embedding = grads['encoder']['embed']['weight']
                validation.require(
                    validation.matches_array(
                        encoder_addition, embedding.shape, embedding.dtype
                    ),
                    'block encoder gradient must retain embedding shape and dtype',
                )
                grads['encoder']['embed']['weight'] += encoder_addition
            block.commit_state(states[i], traces)

        self.eval_state()

        if metrics is not None:
            mx.eval(loss, *components.values())
            metrics.update(
                {name: float(value.item()) for name, value in components.items()}
            )
        self.update_gradients(grads)
        return output, stop, loss

    def __call__(
        self, current: int, target: int | None, end: bool, frozen: bool, ce_only: bool
    ):
        if frozen:
            _, (output, stop) = self.step(mx.array(current), frozen=True)

            self.eval_state()
            return self.sample(output).item(), stop.item()

        output, stop, _ = self.train_step(current, target, end, ce_only)
        return self.sample(output).item(), stop.item()

    def reset(self):
        for block in self.blocks:
            block.reset()
        self.eval_state()

    def snapshot_state(self):
        return [block.snapshot_state() for block in self.blocks]

    def restore_state(self, snapshot):
        validation.require(
            len(snapshot) == len(self.blocks),
            'model state snapshot must contain every block',
        )
        for block, saved in zip(self.blocks, snapshot):
            block.check_runtime_values(saved, block.state_fields + block.trace_fields)
        for block, saved in zip(self.blocks, snapshot):
            block.restore_state(saved)

    def eval_state(self):
        for block in self.blocks:
            block.eval_state()

    @property
    def temporal_gradient(self):
        return [block.temporal_gradient for block in self.blocks]

    def count(self) -> int:
        """Count currently trainable tensor elements, before a full-model freeze."""
        return sum(value.size for _, value in tree_flatten(self.trainable_parameters()))
