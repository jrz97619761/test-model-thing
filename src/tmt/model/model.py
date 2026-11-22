import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util

import tmt.model.blocks as blocks
import tmt.model.tokenizers as tokenizers
import tmt.util.validation as validation

type ModelInputs = tuple[int, int, bool]

class Encoder(nn.Module):
    def __init__(self, config: blocks.Config) -> None:
        super().__init__()
        self.embed = nn.Embedding(config['vocab'], config['dim'])

    def __call__(self, x: mx.array) -> mx.array: return self.embed(x)

class Decoder(nn.Module):
    def __init__(self, config: blocks.Config) -> None:
        super().__init__()
        self.decode = nn.Linear(config['dim'], config['vocab'])

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = self.decode(x)
        return logits, mx.softmax(logits)[256:257]

class Optimizer:
    def __init__(self, config: blocks.Config) -> None:
        self.config = config

        self.opt = opt.AdamW(learning_rate = self.lr)
        self.compiled = None

    def lr(self, step: mx.array) -> mx.array:
        progress = mx.clip(
            (step.astype(mx.float32) + 1.0 - float(self.config['lr'][0]))
            / float(self.config['lr'][1] - self.config['lr'][0]),
            0.0, 1.0
        )
        return self.config['lr'][2] * (1.0 - 0.9 * progress)

    def apply(self, model: Model, grads: blocks.GradTree) -> None:
        if self.compiled is None:
            self.opt.update(model, grads)
            self.evaluate(model)

            state = [model.state, self.opt.state]

            def update(gradients):
                self.opt.update(model, gradients)
                return self.opt.state['step']

            self.compiled = mx.compile(update, inputs = state, outputs = state)

        else:
            self.compiled(grads)
            self.evaluate(model)

    def evaluate(self, model: Model) -> None: mx.eval(model.parameters(), self.opt.state)

class Model(nn.Module):
    def __init__(self, config: blocks.Config) -> None:
        super().__init__()

        self.tokenizer = tokenizers.Tokenizer() if config.get('tokenizer') is None else tokenizers.BPE(config['tokenizer'])
        config['vocab'] = self.tokenizer.size

        self.config = config

        validation.key_exists(config['block'], blocks.BLOCKS)
        validation.key_exists(config['loss'], blocks.LOSSES)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.optimizer = Optimizer(config)

        self.blocks: list[blocks.Block] = [blocks.BLOCKS[config['block']](config) for _ in range(config['layers'])]
        self.loss: blocks.Loss = blocks.LOSSES[config['loss']](config)

        for block in self.blocks:
            validation.instance(block, blocks.Block, 'block')
            block.create_schema()

        validation.instance(self.loss, blocks.Loss, 'loss')

        mx.eval(self.parameters())

    # utility

    def count(self) -> int:
        return sum(value.size for _, value in util.tree_flatten(self.trainable_parameters()))

    def evaluate(self) -> None:
        for block in self.blocks: block.evaluate()

    def reset(self) -> None:
        for block in self.blocks: block.reset()
        self.evaluate()

    def snapshot(self) -> list[blocks.Fields]:
        return [block.snapshot() for block in self.blocks]

    def restore(self, snapshot: list[blocks.Fields]):
        validation.iterable_length(snapshot, self.blocks, 'model snapshot')
        for block, fields in zip(self.blocks, snapshot): block.restore(fields)

    # train and inference

    def sample(self, output: mx.array, key: mx.array | None = None) -> mx.array:
        probs = mx.softmax(output)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8)) / mx.log(mx.array(float(self.config['vocab'])))

        temp = mx.maximum(0.1, self.config['temp'] * (1.0 - self.config['temp'] * entropy)).item()
        return mx.random.categorical(output / temp, key = key)

    def step(self, byte: mx.array, probes: list[mx.array] = [], frozen: bool = False) -> tuple[blocks.FwdOutput, tuple[mx.array, mx.array]]:
        x = enc = self.encoder(byte)
        states, auxiliary = [], []

        if not probes: probes = [block.probe() for block in self.blocks]
        for i, block in enumerate(self.blocks):
            (x, state, aux) = block({'enc': enc, 'x': x}, probes[i])
            block.validate_output(x, enc, state)

            if frozen: block.commit(state, {})

            states.append(state)
            auxiliary.append(aux)

        return (x, states, auxiliary), self.decoder(x)

    def train_step(self, inputs: ModelInputs, metrics: dict[str, float] | None = None) -> tuple[mx.array, ...]:
        current, target, _ = inputs

        byte = mx.array(current)
        params = self.trainable_parameters()

        def fwd(params, probes: list[mx.array]):
            self.update(params)
            (x, states, auxiliary), (output, stop) = self.step(byte, probes)

            terms = {}
            tgt = mx.array(target)

            loss = self.loss.loss({
                'x': x, 'output': output,
                'target': tgt, 'target_emb': mx.stop_gradient(self.encoder(tgt))
            }, terms)

            return loss, (states, auxiliary, output, stop, terms)

        (loss, (states, auxiliary, output, stop, terms)), (grads, probegrads) = (
            mx.value_and_grad(fwd, argnums = (0, 1))(
                params, [block.probe() for block in self.blocks]
            )
        )

        self.update(params)

        for i, block in enumerate(self.blocks):
            grad = grads['blocks'][i]
            templates = dict(util.tree_flatten(grad))

            adjusted, traces, aux = block.grads((byte, states[i], auxiliary[i]), grad, probegrads[i])
            validation.arrays_match(dict(util.tree_flatten(adjusted)), templates, 'adjusted gradient')

            grads['blocks'][i] = adjusted

            if aux.get('encoder') is not None:
                validation.array_match(aux['encoder'], grads['encoder']['embed']['weight'], 'encoder gradient')
                grads['encoder']['embed']['weight'] += aux['encoder']

            block.commit(states[i], traces)

        self.evaluate()

        mx.eval(loss, *terms.values())
        if metrics is not None: metrics.update({name: float(value.item()) for name, value in terms.items()})

        self.optimizer.apply(self, grads)
        return loss, (output, stop)

    def __call__(self, inputs: ModelInputs, frozen: bool) -> tuple[mx.array, ...]:
        if frozen:
            _, (output, stop) = self.step(mx.array(inputs[0]), frozen = True)
            self.evaluate()
        else: _, (output, stop) = self.train_step(inputs)

        return self.sample(output).item(), stop.item()