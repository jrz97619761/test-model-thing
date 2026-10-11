import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten

from .blocks import BLOCKS, Block, Layer, validate_block

class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)

    def __call__(self, x: mx.array): return self.embed(x)

class Decoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def __call__(self, x: mx.array): return self.decode(x), mx.sigmoid(self.stop(x))

class Model(nn.Module):
    def __init__(self, dim: int = 512, layers: int = 16, spread: int = 32, temp: float = 0.75, rate: float = 0.0005, bound: tuple[int, int] = (40000, 120000), block: str = "rtu"):
        super().__init__()
        validate_block(block)
        self.dim = dim
        self.layers = layers
        self.temp = temp

        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)

        self.blocks = [BLOCKS[block](dim, spread) for _ in range(layers)]
        for layer in self.blocks:
            if not isinstance(layer, Block):
                raise ValueError('block factories must return a Block module')
            layer.validate()

        def lrfn(step: mx.array):
            progress = mx.clip((step.astype(mx.float32) + 1.0 - float(bound[0])) / float(bound[1] - bound[0]), 0.0, 1.0)
            return rate * (1.0 - 0.9 * progress)

        self.optimizer = opt.AdamW(learning_rate = lrfn)
        self.compiled = None

    def sample(self, output: mx.array, key: mx.array | None = None):
        probs = mx.softmax(output)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8)) / mx.log(mx.array(256.0))

        temp = mx.maximum(0.1, self.temp * (1.0 - self.temp * entropy)).item()
        return mx.random.categorical(output / temp, key=key)

    def step(self, c: mx.array, dummies: list[mx.array] | None = None, frozen: bool = False):
        if dummies is None: dummies = [layer.probe() for layer in self.blocks]

        enc = self.encoder(c)
        x = enc

        states, auxiliary = [], []

        for i, layer in enumerate(self.blocks):
            x, state, aux = layer(enc, x, dummies[i])
            layer.validate_result(enc, x, state)
            if frozen: layer.commit_state(state)

            states.append(state)
            auxiliary.append(aux)

        return (x, states, auxiliary), self.decoder(x)

    def updategrads(self, grads):
        if self.compiled is None:
            self.optimizer.update(self, grads)
            mx.eval(self.parameters(), self.optimizer.state)
            state = [self.state, self.optimizer.state]

            def update(gradients):
                self.optimizer.update(self, gradients)
                return self.optimizer.state["step"]

            self.compiled = mx.compile(update, inputs = state, outputs = state)

        else:
            self.compiled(grads)
            mx.eval(self.parameters(), self.optimizer.state)

    def loss(self, x: mx.array, output: mx.array, stop: mx.array, nextb: int | None, end: bool, ce_only: bool = False,
             components: dict | None = None) -> mx.array:
        terms = {name: mx.array(0.0) for name in ('variance', 'latent_prediction', 'cross_entropy', 'stop')}
        if not ce_only:
            loss = mx.maximum(0.0, 1.0 - mx.sqrt(mx.var(x) + 1e-4))
            terms['variance'] = loss
            if nextb is not None:
                n = mx.array(nextb)
                tgt = mx.stop_gradient(self.encoder(n))

                terms['latent_prediction'] = mx.mean(mx.square(x - tgt))
                terms['cross_entropy'] = -output[n] + mx.logsumexp(output)
                terms['stop'] = mx.mean(mx.square(stop - mx.array([1.0 if end else 0.0])))
                loss = loss + terms['latent_prediction']
                loss = loss - output[n] + mx.logsumexp(output)
                loss = loss + terms['stop']
        else:
            if nextb is not None: loss = -output[mx.array(nextb)] + mx.logsumexp(output)
            else: loss = 0.0
            terms['cross_entropy'] = mx.array(loss)

        if components is not None:
            components.update(terms)

        return loss

    def train_step(self, currb: int, nextb: int | None, end: bool, ce_only: bool = False,
                   metrics: dict | None = None) -> tuple[mx.array, mx.array, mx.array]:
        c = mx.array(currb)
        p = self.trainable_parameters()

        def fwd(params, dummies: list[mx.array]):
            self.update(params)
            (x, states, auxiliary), (output, stop) = self.step(c, dummies)
            components = {}
            loss = self.loss(x, output, stop, nextb, end, ce_only, components)
            return loss, (states, auxiliary, output, stop, components)

        (loss, (states, auxiliary, output, stop, components)), (grads, dlds_s) = mx.value_and_grad(
            fwd, argnums = (0, 1)
        )(p, [layer.probe() for layer in self.blocks])

        self.update(p)

        block_grads = grads.get('blocks', [{} for _ in self.blocks])
        for i, layer in enumerate(self.blocks):
            local = block_grads[i]
            templates = dict(tree_flatten(local))
            adjusted, encoder_addition, traces = layer.adjust_gradients(
                c, states[i], auxiliary[i], dlds_s[i], local
            )
            actual = dict(tree_flatten(adjusted))
            if actual.keys() != templates.keys() or any(
                not isinstance(actual[name], mx.array)
                or actual[name].shape != value.shape or actual[name].dtype != value.dtype
                for name, value in templates.items()
            ):
                raise ValueError('block gradient hook must retain local gradient names, shapes, and dtypes')
            block_grads[i] = adjusted
            if encoder_addition is not None:
                embedding = grads['encoder']['embed']['weight']
                if (not isinstance(encoder_addition, mx.array)
                        or encoder_addition.shape != embedding.shape or encoder_addition.dtype != embedding.dtype):
                    raise ValueError('block encoder gradient must retain embedding shape and dtype')
                grads['encoder']['embed']['weight'] += encoder_addition
            layer.commit_state(states[i], traces)

        self.eval_state()

        if metrics is not None:
            mx.eval(loss, *components.values())
            metrics.update({name: float(value.item()) for name, value in components.items()})
        self.updategrads(grads)
        return output, stop, loss

    def __call__(self, currb: int, nextb: int | None, end: bool, frozen: bool, ce_only: bool):
        c = mx.array(currb)

        if frozen:
            _, (output, stop) = self.step(c, frozen = True)

            self.eval_state()
            return self.sample(output).item(), stop.item()

        output, stop, _ = self.train_step(currb, nextb, end, ce_only)
        return self.sample(output).item(), stop.item()

    def reset(self):
        for layer in self.blocks:
            layer.reset()
        self.eval_state()

    def snapshot_state(self):
        return [layer.snapshot_state() for layer in self.blocks]

    def restore_state(self, snapshot):
        if len(snapshot) != len(self.blocks):
            raise ValueError('model state snapshot must contain every block')
        for layer, saved in zip(self.blocks, snapshot):
            layer._validate_values(saved, layer.state_fields + layer.trace_fields)
        for layer, saved in zip(self.blocks, snapshot):
            layer.restore_state(saved)

    def eval_state(self):
        for layer in self.blocks:
            layer.eval_state()

    @property
    def temporal_gradient(self):
        return [layer.temporal_gradient for layer in self.blocks]

    def count(self) -> int:
        """Count currently trainable tensor elements, before a full-model freeze."""
        return sum(value.size for _, value in tree_flatten(self.trainable_parameters()))
