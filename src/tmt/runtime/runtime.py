import tmt.model.model as model
import tmt.model.blocks as blocks
import tmt.runtime.checkpoint as checkpoint
import tmt.runtime.benchmark as benchmark
import tmt.util.validation as validation

import mlx.core as mx

from pathlib import Path
from typing import Any

import itertools
import time
import threading
import signal

# util

def sample(model: model.Model, prompt: str, steps: int, seed: int):
    validation.seed(seed, 'sampling seed')
    snapshot = model.snapshot()

    try:
        model.reset()
        for token in model.tokenizer.encode(prompt.encode('utf-8')):
            _, (logits, _) = model.step(mx.array(token), frozen = True)

        key = mx.random.key(seed)
        generated = bytearray()

        while len(generated) < steps:
            key, sample_key = mx.random.split(key)

            token = int(model.sample(logits, key = sample_key).item())
            if token == 256: break

            generated.extend(model.tokenizer.decode(token)[:steps - len(generated)])
            _, (logits, _) = model.step(mx.array(token), frozen = True)

        model.evaluate()
        return bytes(generated)

    finally:
        model.restore(snapshot)
        model.evaluate()

def generate(model: model.Model, prompt: str, steps: int, seed: int, temp: float = -1.0):
    validation.seed(seed)
    validation.finite(temp, 'temperature')

    validation.require(prompt, 'prompt cannot be empty')
    validation.require(steps >= 0, 'cannot have negative generation length')

    model.reset()
    if temp > 0.0: model.config['temp'] = temp

    generated = bytearray()
    with checkpoint.preserve_seed():
        mx.random.seed(seed)

        for token in model.tokenizer.encode(prompt.encode('utf-8')):
            _, (logits, _) = model.step(mx.array(token), frozen = True)

        for _ in range(steps):
            token = int(model.sample(logits).item())
            if token == 256: break

            generated.extend(model.tokenizer.decode(token)[:steps - len(generated)])
            _, (logits, _) = model.step(mx.array(token), frozen = True)

    return bytes(generated)

# training run

def run_manifest(
    config: blocks.Config,
    train_files: checkpoint.ByteFiles, eval_files: checkpoint.ByteFiles,

    seed: int, updates: int, resume: bool,
    intervals: tuple[int, int, int, int], sampling: tuple[str, int, int], max_bytes: int,
    
    completed: int, initial_opt_step: int, opt_step: int,
    start: str, checkpoint_path: str, status: str, error_type: str, error_msg: str
) -> dict[str, Any]:
    log_interval, save_interval, eval_interval, sample_interval = intervals
    sample_prompt, sample_seed, sample_length = sampling
    
    return {
        'model': checkpoint.group_config(config),
        'data': {
            'train': {
                'glob': train_files.pattern,
                'files': train_files.filenames()
            },
            'eval': {
                'glob': eval_files.pattern,
                'files': eval_files.filenames(),
                'max_bytes': max_bytes
            }
        },
        'training': {
            'seed': seed,
            'updates': updates,
            'resume': resume,
        },
        'intervals': {
            'log': log_interval,
            'save': save_interval,
            'eval': eval_interval,
            'sample': sample_interval
        },
        'sampling': {
            'prompt': sample_prompt,
            'seed': sample_seed,
            'length': sample_length
        },
        'progress': {
            'completed': completed,
            'opt': {
                'initial': initial_opt_step,
                'current': opt_step
            }
        },
        'run': {
            'start': start,
            'checkpoint': checkpoint_path,
            'status': status,
            'error': {
                'type': error_type,
                'msg': error_msg
            } if error_type else None
        }
    }

class TrainingRun:
    def __init__(
            self,
            path: validation.FilePath, data: validation.FilePath, eval_data: validation.FilePath,
            config: blocks.Config | None, seed: int, updates: int,
            intervals: tuple[int, int, int, int], sampling: tuple[str, int, int],
            resume: bool = False, max_bytes: int = -1, new: bool = False
        ):
        TrainingRun.validate(seed, updates, intervals, sampling)

        # options

        self.path = path
        self.seed = seed
        self.updates = updates

        self.log_interval, self.save_interval, self.eval_interval, self.sample_interval = intervals
        self.sample_prompt, self.sample_seed, self.sample_length = sampling

        self.resume = resume
        self.max_bytes = max_bytes

        # file handling

        self.files = checkpoint.ByteFiles(data).shuffle(self.seed)
        self.eval_files = checkpoint.ByteFiles(eval_data).validate(self.max_bytes)

        if resume: self.model, self.config = checkpoint.load(self.path)
        else:
            validation.require(config is not None, 'empty config upon initializing new model')
            checkpoint.validate_config(config)

            with checkpoint.preserve_seed():
                checkpoint.seed(0)
                self.model = model.Model(config)
                self.config = config

        # directories

        self.checkpoint = Path(path)
        self.checkpoint.parent.mkdir(parents = True, exist_ok = not new)

        self.manifest_dir = self.checkpoint.parent / 'manifest'
        self.manifest_dir.mkdir(parents = True, exist_ok = True)

        self.sample_dir = self.manifest_dir / 'samples' / checkpoint.stamp('file')

        # progress

        self.completed = self.logged = self.evaluated = 0
        self.initial_opt_step = self.opt_step

        self.loss = self.loss_sum = 0.0
        self.loss_components = {}

        self.stamp = (checkpoint.stamp('iso'), checkpoint.stamp('file'))
        self.start = self.last_report = time.perf_counter()

        self.status = 'running'
        self.error_type = self.error_msg = ''

        self.interrupted = False

    @property
    def opt_step(self) -> int: return int(self.model.optimizer.opt.state['step'].item())

    @staticmethod
    def validate(seed: int, updates: int, intervals: tuple[int, int, int, int], sampling: tuple[str, int, int]):
        validation.seed(seed)
        validation.seed(sampling[1])
        validation.require(sampling[0], 'sample prompt is empty')

        for name, value in zip(
            ('updates', 'sample_length', 'log_interval', 'save_interval', 'eval_interval', 'sample_interval'),
            (updates, sampling[2], *intervals)
        ): validation.positive(value, name)

    # logging and saving

    def report(self, file: str, data: dict[str, Any]) -> dict[str, Any]:
        checkpoint.append_jsonl(self.manifest_dir / file, {
            'completed': self.completed,
            'opt_step': self.opt_step,

            **data
        })

        return data

    def report_loss(self) -> None:
        now = time.perf_counter()
        interval = self.completed - self.logged

        self.report('loss.jsonl', {
            'loss': self.loss,
            'loss_mean': self.loss_sum / interval,
            'loss_components': {
                name: value / interval
                for name, value in self.loss_components.items()
            },

            'interval': interval,
            'elapsed': now - self.start,
            'updates_per_sec': interval / max(now - self.last_report, 1e-9)
        })

        self.logged = self.completed
        self.last_report = now

        self.loss_sum = 0.0
        self.loss_components.clear()

    def report_sample(self) -> None:
        sampled = sample(self.model, self.sample_prompt, self.sample_length, self.sample_seed)
        step = self.opt_step

        self.sample_dir.mkdir(parents = True, exist_ok = True)

        destination = self.sample_dir / f'step_{step:012d}.bin'
        destination.write_bytes(sampled)

        self.report('samples.jsonl', {
            'prompt': self.sample_prompt,
            'seed': self.sample_seed,
            'length': len(sampled),

            'path': str(destination.resolve())
        })

    def report_eval(self) -> None:
        bpb = benchmark.bpb_score(self.model, list(self.eval_files.get(self.max_bytes)))
        self.report('eval.jsonl', bpb)

        self.evaluated = self.completed

    def save(self) -> None:
        checkpoint.save(self.checkpoint, self.model, self.config)
        checkpoint.write_json(self.manifest_dir / 'manifest.json', run_manifest(
            self.config, self.files, self.eval_files,
            self.seed, self.updates, self.resume,

            (self.log_interval, self.save_interval, self.eval_interval, self.sample_interval),
            (self.sample_prompt, self.sample_seed, self.sample_length),

            self.max_bytes, self.completed, self.initial_opt_step, self.opt_step,
            self.stamp[0], str(self.checkpoint.resolve()),
            self.status, self.error_type, self.error_msg
        ))

    # training

    def check_interrupt(self) -> None:
        if self.interrupted:
            self.save()
            raise KeyboardInterrupt

    def check_intervals(self) -> None:
        if self.completed % self.log_interval == 0: self.report_loss()
        if self.completed % self.eval_interval == 0: self.report_eval()
        if self.completed % self.sample_interval == 0: self.report_sample()
        if self.completed % self.save_interval == 0: self.save()

    def single_pass(self) -> None:
        start = self.completed

        for data in self.files.get():
            self.check_interrupt()
            data = self.model.tokenizer.encode(data)

            self.model.reset()
            if len(data) == 1: self.consume(data[0])

            for index in range(len(data)):
                end = (index == len(data) - 1)
                components: dict[str, float] = {}

                _, (_, loss) = self.model.train_step(
                    (data[index], 256 if end else data[index + 1], end),
                    components
                )

                self.completed += 1

                self.loss = float(loss.item())
                self.loss_sum += self.loss

                for name, value in components.items():
                    total = self.loss_components.get(name, 0.0)
                    self.loss_components[name] = total + value

                self.check_intervals()

                if self.completed > self.updates: break

        self.check_interrupt()
        validation.require(self.completed != start, 'failed to locate valid training targets')        

    def run(self) -> model.Model:
        def interrupt(*_): self.interrupted = True

        # interrupt handler

        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handler = signal.getsignal(signum)
                if handler is not None:
                    handlers[signum] = handler
                    signal.signal(signum, interrupt)

        # training run

        try:
            if self.resume and (self.manifest_dir / 'manifest.json').exists():
                (self.manifest_dir / 'manifest.json').rename(self.manifest_dir / f'manifest_old_{self.stamp[1]}.json')

            for name in ('loss.jsonl', 'samples.jsonl', 'eval.jsonl'):
                with open(self.manifest_dir / name, 'a' if self.resume else 'w'): pass

            checkpoint.write_json(self.manifest_dir / 'manifest.json', run_manifest(
                self.config, self.files, self.eval_files,
                self.seed, self.updates, self.resume,

                (self.log_interval, self.save_interval, self.eval_interval, self.sample_interval),
                (self.sample_prompt, self.sample_seed, self.sample_length),

                self.max_bytes, self.completed, self.initial_opt_step, self.opt_step,
                self.stamp[0], str(self.checkpoint.resolve()) if self.resume else None,
                self.status, self.error_type, self.error_msg
            ))

            while self.completed < self.updates: self.single_pass()

            if self.completed != self.evaluated: self.report_eval()
            self.check_interrupt()
            self.save()

            if self.interrupted: raise KeyboardInterrupt
            self.status = 'complete'

        except KeyboardInterrupt:
            self.status = 'interrupted'
            raise

        except Exception as error:
            self.status = 'failed'

            self.error_type = type(error).__name__,
            self.error_msg = str(error)

            raise

        finally:
            try:
                checkpoint.write_json(self.manifest_dir / 'manifest.json', run_manifest(
                    self.config, self.files, self.eval_files,
                    self.seed, self.updates, self.resume,

                    (self.log_interval, self.save_interval, self.eval_interval, self.sample_interval),
                    (self.sample_prompt, self.sample_seed, self.sample_length),

                    self.max_bytes, self.completed, self.initial_opt_step, self.opt_step,
                    self.stamp[0], str(self.checkpoint.resolve()),
                    self.status, self.error_type, self.error_msg
                ))

                if self.completed != self.logged: self.report_loss()

            finally:
                for signum, handler in handlers.items(): signal.signal(signum, handler)

        return self.model

def inspect(path: validation.FilePath) -> dict[str, Any]:
    model, config = checkpoint.load(path)

    summary = {
        'opt_step': int(model.optimizer.opt.state['step'].item()),
        'param_count': model.count(),
        'config': checkpoint.group_config(config),
        'run': None
    }

    manifest = Path(path).parent / 'manifest' / 'manifest.json'
    if manifest.is_file():
        data = checkpoint.read_json(manifest)
        validation.instance(data, dict, 'run manifest')
        for key in ('run', 'training', 'progress'): validation.instance(data.get(key), dict, f'run manifest -> {key}')

        summary['run'] = {
            'status': data['run'].get('status'),
            'seed': data['training'].get('seed'),
            'completed': data['progress'].get('completed')
        }

    return summary

# sweep run

def validate_grid(grid: Any, config: blocks.Config) -> None:
    validation.instance(grid, dict, 'sweep grid')
    for key, axis in grid.items():
        validation.instance(key, str, 'sweep grid key')
        validation.require(key != 'tokenizer', 'sweep grid cannot set tokenizer, use train --bpe')

        validation.instance(axis, list, f'sweep grid axis {key!r}')
        validation.require(axis, f'sweep grid axis {key!r} must be nonempty')

        for value in axis:
            if key == 'seed': validation.seed(value, 'sweep grid seed')
            else:
                candidate = config
                candidate[key] = value
                checkpoint.validate_config(candidate)

def sweep_manifest(
    config: blocks.Config, grid: dict[str, list[Any]],
    train_files: checkpoint.ByteFiles, eval_files: checkpoint.ByteFiles, dev_files: checkpoint.ByteFiles,

    seed: int, updates: int, intervals: tuple[int, int, int, int], sampling: tuple[str, int, int], max_bytes: int,
    completed: int, current: int, start: str, status: str, error_type: str, error_msg: str
) -> dict[str, Any]:
    log_interval, save_interval, eval_interval, sample_interval = intervals
    sample_prompt, sample_seed, sample_length = sampling
    
    return {
        'model': checkpoint.group_config(config),
        'grid': grid,
        'data': {
            'train': {
                'glob': train_files.pattern,
                'files': train_files.filenames()
            },
            'eval': {
                'glob': eval_files.pattern,
                'files': eval_files.filenames(),
                'max_bytes': max_bytes
            },
            'dev': {
                'glob': dev_files.pattern,
                'files': dev_files.filenames(),
                'max_bytes': max_bytes
            }
        },
        'training': {
            'seed': seed,
            'updates': updates
        },
        'intervals': {
            'log': log_interval,
            'save': save_interval,
            'eval': eval_interval,
            'sample': sample_interval
        },
        'sampling': {
            'prompt': sample_prompt,
            'seed': sample_seed,
            'length': sample_length
        },
        'progress': {
            'completed': completed,
            'current': current
        },
        'run': {
            'start': start,
            'status': status,
            'error': {
                'type': error_type,
                'msg': error_msg
            } if error_type else None
        }
    }

type SweepCandidate = tuple[int, int, blocks.Config]
type SweepExecution = tuple[model.Model, Path, dict[str, Any]]

def sweep_candidate(index: int, seed: int, keys: list[str], values: tuple[Any, ...], config: blocks.Config) -> SweepCandidate:
    overrides = dict(zip(keys, values))
    run_seed = overrides.pop('seed', seed)

    run_config = config
    run_config.update(overrides)

    return (index, run_seed, run_config)

def sweep_run(
    candidate: SweepCandidate, output: Path,
    data: checkpoint.ByteFiles, eval_data: checkpoint.ByteFiles, dev_data: checkpoint.ByteFiles,

    updates: int, intervals: tuple[int, int, int, int], sampling: tuple[str, int, int], max_bytes: int
) -> SweepExecution:
    run_path = output / f'run_{candidate[0]:04d}'
    run_path.mkdir(parents = True, exist_ok = True)

    manifest_dir = run_path / 'manifest'
    manifest_dir.mkdir(parents = True, exist_ok = True)
    checkpoint.write_json(manifest_dir / 'model.json', checkpoint.group_config(candidate[2]))

    checkpoint_dir = run_path / 'model.safetensors'

    model = TrainingRun(
        checkpoint_dir, data.pattern, eval_data.pattern,
        candidate[2], candidate[1],
        updates, intervals, sampling, False, max_bytes
    ).run()

    score = benchmark.bpb_score(model, list(dev_data.get(max_bytes)))
    return (model, checkpoint_dir, score)

def sweep_result(
    candidate: SweepCandidate, execution: SweepExecution,
    data: checkpoint.ByteFiles, dev_data: checkpoint.ByteFiles,
    updates: int, max_bytes: int
) -> dict[str, Any]:
    index, seed, config = candidate
    model, checkpoint_dir, score = execution

    return {
        'model': checkpoint.group_config(config),
        'param_count': model.count(),
        'data': {
            'train': {'glob': data.pattern},
            'dev': {'glob': dev_data.pattern, 'max_bytes': max_bytes}
        },
        'training': {
            'seed': seed,
            'updates': updates
        },
        'eval': score,
        'run': {
            'index': index,
            'checkpoint': str(checkpoint_dir.resolve())
        }
    }

def sweep(
    data: validation.FilePath, eval_data: validation.FilePath, dev_data: validation.FilePath,
    grid_path: validation.FilePath, config_path: validation.FilePath, output_path: validation.FilePath,

    seed: int, updates: int, intervals: tuple[int, int, int, int], sampling: tuple[str, int, int],
    max_bytes: int = -1
) -> dict[str, Any]:    
    TrainingRun.validate(seed, updates, intervals, sampling)
    config = checkpoint.load_config(config_path)

    grid = checkpoint.read_json(grid_path)
    validate_grid(grid, config)
    keys = list(grid)

    train_files = checkpoint.ByteFiles(data)
    eval_files = checkpoint.ByteFiles(eval_data).validate(max_bytes)
    dev_files = checkpoint.ByteFiles(dev_data).validate(max_bytes)

    output = Path(output_path)
    validation.require(
        not (output.exists() and any(output.iterdir())),
        f'sweep output folder {output} is not empty, choose a different output folder'
    )

    output.mkdir(parents = True, exist_ok = True)
    rows = []

    completed = current = 0

    start = checkpoint.stamp('iso')
    status = 'running'
    error_type = error_msg = ''

    checkpoint.write_json(output / 'sweep.json', sweep_manifest(
        config, grid,
        train_files, eval_files, dev_files,
        seed, updates, intervals, sampling, max_bytes,
        completed, current, start, status, error_type, error_msg
    ))

    try:
        for index, values in enumerate(
            itertools.product(*(grid[key] for key in keys)), 1
        ):
            current = index
            candidate = sweep_candidate(index, seed, keys, values, config)

            run = sweep_run(
                candidate, output,
                train_files, eval_files, dev_files,
                updates, intervals, sampling, max_bytes
            )

            row = sweep_result(candidate, run, train_files, dev_files, updates, max_bytes)
            checkpoint.append_jsonl(output / 'results.jsonl', row)
            rows.append(row)

            completed += 1
            checkpoint.write_json(output / 'sweep.json', sweep_manifest(
                config, grid,
                train_files, eval_files, dev_files,
                seed, updates, intervals, sampling, max_bytes,
                completed, current, start, status, error_type, error_msg
            ))

        status = 'complete'

    except KeyboardInterrupt:
        status = 'interrupted'
        raise

    except Exception as error:
        status = 'failed'

        error_type = type(error).__name__,
        error_msg = str(error)

        raise

    finally:
        checkpoint.write_json(output / 'sweep.json', sweep_manifest(
            config, grid,
            train_files, eval_files, dev_files,
            seed, updates, intervals, sampling, max_bytes,
            completed, current, start, status, error_type, error_msg
        ))

    return min(rows, key = lambda row: row['eval']['bpb'])