import argparse, glob, itertools, math, os, random, sys, time
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util

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

class Layer(nn.Module):
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

    def __call__(self, enc: mx.array, x: mx.array, dummy: mx.array):
        decay = mx.sigmoid(self.decay)
        state = (decay * self.states) + enc + dummy

        return x + self.silu(self.weights(self.norm(state))), state, decay

class Model(nn.Module):
    def __init__(self, dim: int, layers: int, spread: int, temp: float, lr: float, lrbegin: int, lrend: int):
        super().__init__()
        self.dim = dim
        self.layercount = layers
        self.temp = temp

        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)

        self.layers = [Layer(dim, spread) for _ in range(layers)]

        def lrfn(step: mx.array):
            progress = mx.clip(
                (step.astype(mx.float32) + 1.0 - float(lrbegin)) / float(lrend - lrbegin),
                0.0, 1.0
            )
            return lr * (1.0 - 0.9 * progress)

        self.optimizer = opt.AdamW(learning_rate = lrfn)

    def sample(self, output: mx.array):
        probs = mx.softmax(output)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8)) / mx.log(mx.array(256.0))

        temp = mx.maximum(0.1, self.temp * (1.0 - self.temp * entropy)).item()
        return mx.random.categorical(output / temp)

    def evaluate(self): mx.eval(*[layer.states for layer in self.layers])

    def reset(self):
        for layer in self.layers:
            layer.states = mx.zeros((self.dim, ))

            layer.decaytrace = mx.zeros((self.dim, ))
            layer.embedtrace = mx.zeros((256, self.dim))

        self.evaluate()

    def step(self, c: mx.array, dummies: mx.array | None = None, frozen: bool = False):
        if dummies is None: dummies = [mx.zeros((self.dim, )) for _ in range(self.layercount)]

        enc = self.encoder(c)
        x = enc
            
        states, decays = [], []

        for i, layer in enumerate(self.layers):
            x, state, decay = layer(enc, x, dummies[i])
            if frozen: layer.states = mx.stop_gradient(state)

            states.append(state)
            decays.append(decay)

        return (x, states, decays), self.decoder(x)

    def __call__(self, currb: int, nextb: int | None, end: bool, frozen: bool):
        c = mx.array(currb)

        if frozen:
            _, (output, stop) = self.step(c, frozen = True)

            self.evaluate()
            return self.sample(output).item(), stop.item()

        p = self.trainable_parameters()

        def fwd(params, dummies: list[mx.array]):
            self.update(params)

            (x, states, decays), (output, stop) = self.step(c, dummies)

            loss = mx.maximum(0.0, 1.0 - mx.sqrt(mx.var(x) + 1e-4))
            if nextb is not None:
                n = mx.array(nextb)
                tgt = mx.stop_gradient(self.encoder(n))

                loss = loss + mx.mean(mx.square(x - tgt))
                loss = loss - output[n] + mx.logsumexp(output)

                loss = loss + mx.mean(mx.square(stop - mx.array([1.0 if end else 0.0])))
                
            # loss = variance loss + pred mse loss + crossentropy loss + stop mse loss
            return loss, (states, decays, output, stop)

        (_, (states, decays, output, stop)), (grads, dlds_s) = mx.value_and_grad(
            fwd, argnums = (0, 1)
        )(p, [mx.zeros((self.dim, )) for _ in range(self.layercount)])

        self.update(p)

        for i, layer in enumerate(self.layers):
            dlds = dlds_s[i]

            embedtrace = (layer.embedtrace * decays[i]) + (mx.arange(256) == c)[:, None].astype(mx.float32)
            grads["encoder"]["embed"]["weight"] += dlds * (layer.embedtrace * decays[i])
            
            decaytrace = (decays[i] * layer.decaytrace) + (decays[i] * (1.0 - decays[i]) * layer.states)
            grads["layers"][i]["decay"] = dlds * decaytrace

            layer.states = mx.stop_gradient(states[i])

            layer.decaytrace = mx.stop_gradient(decaytrace)
            layer.embedtrace = mx.stop_gradient(embedtrace)
            
            mx.eval(layer.states, layer.decaytrace, layer.embedtrace)

        self.optimizer.update(self, grads)
        mx.eval(self.parameters(), self.optimizer.state)

        return self.sample(output).item(), stop.item()

    def save(self, path: str):
    data = {}
    for k, v in util.tree_flatten(self.parameters()): data[f"m.{k}"] = v
    for k, v in util.tree_flatten(self.optimizer.state): data[f"o.{k}"] = v

    for i, layer in enumerate(self.layers):
        data[f"state.{i}"] = layer.states
        data[f"decaytrace.{i}"] = layer.decaytrace
        data[f"embedtrace.{i}"] = layer.embedtrace

    directory, filename = os.path.split(path)
    if not filename.endswith('.safetensors'):
        filename += '.safetensors'
    tmp = os.path.join(directory, 'temporary-' + filename)

    mx.save_safetensors(tmp, data)
    os.replace(tmp, path)

    def load(self, path: str):
        if not os.path.exists(path): return

        data, model, opts = mx.load(path), {}, {}
        
        for k, v in data.items():
            if k.startswith("m."): model[k[2:]] = v
            elif k.startswith("o."): opts[k[2:]] = v
            elif k.startswith("state."): self.layers[int(k.split('.')[1])].states = v
            elif k.startswith("decaytrace."): self.layers[int(k.split('.')[1])].decaytrace = v
            elif k.startswith("embedtrace."): self.layers[int(k.split('.')[1])].embedtrace = v
            
        if model: self.update(util.tree_unflatten(list(model.items())))
        if opts: self.optimizer.state = util.tree_unflatten(list(opts.items()))

    def count(self) -> int:
        per_layer = self.dim * self.dim + 3 * self.dim
        return 256 * self.dim + self.layercount * per_layer + 256 * self.dim + 256 + self.dim + 1

class Runtime:
    def __init__(self, path: str, threshold: float, **kwargs):
        self.model = Model(**kwargs)
        self.path = path
        self.threshold = threshold

        self.step = 0
        self.prevtime = None

    def save(self):
        self.step += 1
        if self.step % 500 == 0: self.model.save(self.path)

    def call(self, c: int, n: int | None, end: bool, save: bool, frozen: bool):
        outputs = self.model(c, n, end, frozen)

        if save: self.save()
        return outputs

    def write(self, b: int):
        sys.stdout.buffer.write(bytes([b]))
        sys.stdout.flush()

    def chat(self, save: bool, frozen: bool):
        while True:
            text = input(f'\n[{self.now()} | {0 if self.prevtime is None else time.time() - self.prevtime:.4f}s]\nUser >> ')
            self.prevtime = time.time()

            data = (text + '\n').encode('utf-8')
            
            for i, (c, n) in enumerate(itertools.pairwise(data)):
                b, _ = self.call(c, n, i == len(data) - 2, save, frozen)

            print(f'\n[{self.now()}]\nModel >> ', end = '', flush = True)

            b = data[-1]
            while True:
                b, stop = self.call(b, None, False, save, frozen)
                self.write(b)

                if stop > self.threshold:
                    print()
                    break

    def train(self, save: bool, frozen: bool, dataset: str):
        files = glob.glob(dataset, recursive = True)

        if not files:
            raise FileNotFoundError(
                f'Could not find training files with the following glob: {dataset!r}. Try downloading a dataset first.'
            )

        random.shuffle(files)

        while True:
            for file in files:
                with open(file, 'r', encoding = 'utf-8', errors = 'ignore') as f:
                    for line in f:
                        data = line.encode('utf-8')
                        if len(data) < 2: continue

                        for i, (c, n) in enumerate(itertools.pairwise(data)):
                            b, _ = self.call(c, n, i == len(data) - 2, save, frozen)
                            self.write(b)

    def now(self): return datetime.now().strftime('%d/%m/%Y, %H:%M:%S')

    def __call__(self, mode: str, dataset: str, save: bool, frozen: bool):
        self.model.load(self.path)
        print()

        try:
            match mode:
                case 'train': self.train(save, frozen, dataset)
                case 'chat': self.chat(save, frozen)

        finally:
            if save: self.model.save(self.path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'test-model-thing')
    parser.add_argument('path')
    parser.add_argument('mode', choices = ['train', 'chat'])

    parser.add_argument('--frozen', action = 'store_true')
    parser.add_argument('--no-save', action = 'store_false')
    parser.add_argument('--dataset', default = 'wikipedia_clean/**/wiki_*')

    args = parser.parse_args()

    runtime = Runtime(path = args.path, threshold = 0.35, dim = 512, layers = 16, spread = 32, temp = 0.75, lr = 5e-4, lrbegin = 40000, lrend = 120000)
    print(f'parameters: {runtime.model.count():,}')

    runtime(args.mode, args.dataset, args.no_save, args.frozen)
