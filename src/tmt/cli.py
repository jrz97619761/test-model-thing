import argparse
import glob
import itertools
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import validation


def load_config(path=None):
    from tmt.config import MODEL_DEFAULTS

    overrides = {}
    if path is not None:
        with open(path) as file:
            overrides = json.load(file)

    if not isinstance(overrides, dict):
        raise ValueError('model config must be a JSON object')
    for name in overrides:
        if name not in MODEL_DEFAULTS:
            raise ValueError(
                f'invalid model config: got an unexpected keyword argument {name!r}'
            )
    resolved = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in {**MODEL_DEFAULTS, **overrides}.items()
    }
    return validation.model_settings(resolved, 'model config')


def new_model(settings, seed):
    validation.seed(seed)
    validation.model_settings(settings, 'model config')
    import random
    import mlx.core as mx
    from tmt.main import Model

    random.seed(seed)
    mx.random.seed(seed)
    model = Model(**settings)
    mx.eval(model.parameters())
    return model


def _read_header(path):
    try:
        with open(path, 'rb') as source:
            size = os.fstat(source.fileno()).st_size
            length_bytes = source.read(8)
            if len(length_bytes) != 8:
                raise ValueError(
                    'invalid checkpoint: truncated Safetensors header length'
                )
            length = int.from_bytes(length_bytes, 'little')
            if length == 0 or length > size - 8:
                raise ValueError(
                    'invalid checkpoint: invalid Safetensors header length'
                )
            raw = source.read(length)
            if len(raw) != length:
                raise ValueError('invalid checkpoint: truncated Safetensors header')
        envelope = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f'invalid checkpoint: invalid Safetensors header: {error}'
        ) from error
    if not isinstance(envelope, dict) or not envelope:
        raise ValueError(
            'invalid checkpoint: Safetensors header must be a nonempty object'
        )
    metadata = envelope.get('__metadata__')
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {'tmt'}
        or not isinstance(metadata.get('tmt'), str)
    ):
        raise ValueError(
            "invalid checkpoint: __metadata__ must contain exactly string 'tmt'"
        )
    if not any(name != '__metadata__' for name in envelope):
        raise ValueError('invalid checkpoint: checkpoint contains no tensors')
    return envelope


def _flatten(tree, prefix):
    import mlx.utils as util

    return {f'{prefix}.{name}': value for name, value in util.tree_flatten(tree)}


def save_checkpoint(path, model, settings):
    import mlx.core as mx

    resolved = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in settings.items()
    }
    raw = json.dumps({'version': 1, 'model': resolved})
    validation.metadata(raw)
    model.optimizer.init(model.trainable_parameters())
    tensors = _flatten(model.parameters(), 'm')
    tensors.update(_flatten(model.optimizer.state, 'o'))
    mx.eval(model.parameters(), model.optimizer.state)
    mx.save_safetensors(path, tensors, metadata={'tmt': raw})


def load_model(path, seed=11):
    validation.seed(seed)
    import mlx.core as mx
    import mlx.utils as util

    envelope = _read_header(path)
    settings = validation.metadata(envelope['__metadata__']['tmt'])
    model = new_model(settings, seed)
    model.optimizer.init(model.trainable_parameters())
    expected = _flatten(model.parameters(), 'm')
    expected.update(_flatten(model.optimizer.state, 'o'))
    try:
        tensors = mx.load(path)
    except Exception as error:
        raise ValueError(f'invalid checkpoint: {error}') from error
    validation.checkpoint_arrays(tensors, expected)

    parameters = {
        name[2:]: value for name, value in tensors.items() if name.startswith('m.')
    }
    optimizer = {
        name[2:]: value for name, value in tensors.items() if name.startswith('o.')
    }
    model.update(util.tree_unflatten(list(parameters.items())))
    model.optimizer.state = util.tree_unflatten(list(optimizer.items()))
    model.compiled = None
    mx.eval(model.parameters(), model.optimizer.state)
    return model, settings


def _input_files(pattern):
    """Resolve a selection and check that every selected file is readable."""
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f'no files matched {pattern!r}')
    for filename in files:
        with open(filename, 'rb') as file:
            file.read(1)
    return files


def documents(pattern, max_bytes=None):
    """Yield sorted raw files with an optional prefix across all file bytes."""
    files = _input_files(pattern)
    remaining = max_bytes
    for path in files:
        if remaining == 0:
            break
        data = Path(path).read_bytes()
        if remaining is not None:
            data = data[:remaining]
            remaining -= len(data)
        yield data


def _write_manifest(path, record):
    with open(path, 'w') as file:
        json.dump(record, file, indent=2)
        file.write('\n')


def _append_jsonl(path, row):
    with open(path, 'a') as file:
        file.write(json.dumps(row) + '\n')


def _training_sample(model, prompt, count, seed):
    """Sample with a local RNG key and restore recurrent state after any exit."""
    validation.seed(seed, 'sample-seed')
    import mlx.core as mx

    snapshot = model.snapshot_state()
    try:
        model.reset()
        for byte in prompt.encode('utf-8'):
            _, (logits, _) = model.step(mx.array(byte), frozen=True)
        key = mx.random.key(seed)
        generated = bytearray()
        for _ in range(count):
            key, sample_key = mx.random.split(key)
            byte = int(model.sample(logits, key=sample_key).item())
            generated.append(byte)
            _, (logits, _) = model.step(mx.array(byte), frozen=True)
        model.eval_state()
        return bytes(generated)
    finally:
        model.restore_state(snapshot)
        model.eval_state()


def train(
    settings,
    path,
    pattern,
    updates=1000,
    seed=11,
    ce_only=False,
    resume=False,
    log_every=100,
    sample_every=None,
    sample_prompt='The ',
    sample_seed=11,
    sample_bytes=256,
    eval_data=None,
    eval_every=None,
    eval_max_bytes=8192,
    *,
    new_run=False,
):
    """Train on adjacent raw-byte pairs and record the run manifest."""
    validation.training_options(seed, sample_seed, updates, log_every)
    validation.monitoring(
        sample_every, sample_prompt, sample_bytes, eval_data, eval_every, eval_max_bytes
    )
    import mlx.core as mx

    files = _input_files(pattern)
    evaluation_data = (
        None if eval_data is None else list(documents(eval_data, eval_max_bytes))
    )
    if resume:
        model, settings = load_model(path, seed)
    else:
        model = new_model(settings, seed)
    random.shuffle(files)

    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=not new_run)
    manifest_dir = checkpoint.parent / 'manifest'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / 'run.json'
    metrics = manifest_dir / 'loss.jsonl'
    record = {
        'config': settings,
        'trainable_parameters': model.count(),
        'temporal_gradient': model.temporal_gradient,
        'seed': seed,
        'data_glob': pattern,
        'data_files': [str(Path(filename).resolve()) for filename in files],
        'updates': updates,
        'log_every': log_every,
        'objective': 'ce_only' if ce_only else 'tmt',
        'completed_updates': 0,
        'resume': resume,
        'initial_optimizer_step': int(model.optimizer.state['step'].item())
        if resume
        else 0,
        'sample': None
        if sample_every is None
        else {
            'every': sample_every,
            'prompt': sample_prompt,
            'seed': sample_seed,
            'bytes': sample_bytes,
        },
        'evaluation': None
        if eval_every is None
        else {
            'every': eval_every,
            'data_glob': eval_data,
            'max_bytes': eval_max_bytes,
            'data_files': [
                str(Path(name).resolve())
                for name in sorted(glob.glob(eval_data, recursive=True))
            ],
        },
        'started_at': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'last_checkpoint': str(checkpoint.resolve()) if resume else None,
    }

    def consume(byte):
        model.step(mx.array(byte), frozen=True)
        model.eval_state()

    def save():
        save_checkpoint(checkpoint, model, settings)
        record['last_checkpoint'] = str(checkpoint.resolve())
        print(f'saved checkpoint: {checkpoint}, updates: {completed}')

    completed = 0
    last_loss = None
    logged = 0
    evaluated = 0
    loss_sum = 0.0
    component_sums = {}
    started = last_report = time.perf_counter()
    sample_dir = (
        checkpoint.parent
        / 'samples'
        / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    )

    def report_loss():
        nonlocal logged, last_report, loss_sum
        now = time.perf_counter()
        interval_updates = completed - logged
        loss = float(last_loss.item())
        row = {
            'completed_updates': completed,
            'optimizer_step': int(model.optimizer.state['step'].item()),
            'objective': record['objective'],
            'loss': loss,
            'mean_loss': loss_sum / interval_updates,
            'mean_loss_components': {
                name: value / interval_updates for name, value in component_sums.items()
            },
            'interval_updates': interval_updates,
            'elapsed_seconds': now - started,
            'updates_per_second': interval_updates / max(now - last_report, 1e-9),
        }
        _append_jsonl(metrics, row)
        print(
            f'updates: {completed}, objective: {record["objective"]}, '
            f'loss: {loss:.6f}, '
            f'mean loss: {row["mean_loss"]:.6f}, step: {row["optimizer_step"]}, '
            f'updates/s: {row["updates_per_second"]:.2f}',
            flush=True,
        )
        logged = completed
        last_report = now
        loss_sum = 0.0
        component_sums.clear()

    def report_sample():
        sample = _training_sample(model, sample_prompt, sample_bytes, sample_seed)
        step = int(model.optimizer.state['step'].item())
        sample_dir.mkdir(parents=True, exist_ok=True)
        destination = sample_dir / f'step-{step:012d}.bin'
        destination.write_bytes(sample)
        _append_jsonl(
            manifest_dir / 'samples.jsonl',
            {
                'completed_updates': completed,
                'optimizer_step': step,
                'prompt': sample_prompt,
                'seed': sample_seed,
                'bytes': len(sample),
                'path': str(destination.resolve()),
            },
        )
        print(f'sample at update {completed}, step {step}: {sample!r}', flush=True)

    def report_evaluation():
        nonlocal evaluated
        from tmt.benchmark import evaluate

        result = evaluate(model, evaluation_data)
        _append_jsonl(
            manifest_dir / 'evaluation.jsonl',
            {
                'completed_updates': completed,
                'optimizer_step': int(model.optimizer.state['step'].item()),
                **result,
            },
        )
        print(
            f'evaluation at update {completed}: BPB {result["bpb"]:.6f}, '
            f'targets: {result["targets"]}',
            flush=True,
        )
        evaluated = completed

    status = 'failed'
    try:
        metrics.write_text('')
        (manifest_dir / 'samples.jsonl').write_text('')
        (manifest_dir / 'evaluation.jsonl').write_text('')
        _write_manifest(manifest, record)
        print(
            f'run folder: {checkpoint.parent}, seed: {seed}, '
            f'config: {json.dumps(settings)}'
        )
        while completed < updates:
            before_pass = completed
            for filename in files:
                data = Path(filename).read_bytes()
                if not data:
                    continue
                model.reset()
                if len(data) == 1:
                    consume(data[0])
                    continue
                for index in range(len(data) - 1):
                    end = index == len(data) - 2
                    components = {}
                    _, _, last_loss = model.train_step(
                        data[index], data[index + 1], end, ce_only, metrics=components
                    )
                    completed += 1
                    loss_sum += float(last_loss.item())
                    for name, value in components.items():
                        component_sums[name] = component_sums.get(name, 0.0) + value
                    if completed % log_every == 0:
                        report_loss()
                    if end:
                        consume(data[-1])
                    if sample_every is not None and completed % sample_every == 0:
                        report_sample()
                    if eval_every is not None and completed % eval_every == 0:
                        report_evaluation()
                    if completed % 500 == 0:
                        save()
                    if completed >= updates:
                        break
                if completed >= updates:
                    break
            if completed == before_pass:
                raise ValueError('no training targets found in data pass')
        if eval_every is not None and completed != evaluated:
            report_evaluation()
        save()
        status = 'complete'
    except KeyboardInterrupt:
        status = 'interrupted'
        raise
    finally:
        record['status'] = status
        record['completed_updates'] = completed
        _write_manifest(manifest, record)
        if last_loss is not None and completed != logged:
            report_loss()

    print(f'updates: {completed}, checkpoint: {checkpoint}')
    return model


def evaluate_command(args):
    validation.positive(args.max_bytes, 'max-bytes')
    for window in args.windows:
        validation.positive(window, 'evaluation windows')
    from tmt.benchmark import evaluate as score

    model, _ = load_model(args.checkpoint)
    data = list(documents(args.data, args.max_bytes))
    result = score(model, data, args.windows)
    print(
        f'input bytes: {result["input_bytes"]}, targets: {result["targets"]}, '
        f'BPB: {result["bpb"]}'
    )
    print(
        f'common targets: {result["common_targets"]}, '
        f'common BPB: {result["common_bpb"]}'
    )
    for window in result['windows']:
        print(
            f'window {window["window"]}: BPB {window["bpb"]}, '
            f'common BPB {window["common_bpb"]}, delta {window["delta_bpb"]}'
        )
    if args.output is not None:
        with open(args.output, 'w') as file:
            json.dump(result, file, indent=2)
            file.write('\n')


def generate(
    checkpoint, prompt, output, seed=11, count=512, temperature=None, log_every=1000
):
    """Replay a UTF-8 prompt and write a fixed count of sampled bytes."""
    validation.seed(seed)
    if not prompt:
        raise ValueError('generation needs a nonempty prompt')
    if count < 0:
        raise ValueError('bytes must be nonnegative')
    validation.positive(log_every, 'log-every')
    if temperature is not None and (not math.isfinite(temperature) or temperature < 0):
        raise ValueError('temperature must be a finite nonnegative number')

    import mlx.core as mx

    model, _ = load_model(checkpoint, seed)
    mx.eval(model.parameters())
    model.reset()
    if temperature is not None:
        model.temp = temperature
    mx.random.seed(seed)

    prompt = prompt.encode('utf-8')
    for byte in prompt:
        _, (logits, _) = model.step(mx.array(byte), frozen=True)
    generated = bytearray()
    for _ in range(count):
        byte = int(model.sample(logits).item())
        generated.append(byte)
        _, (logits, _) = model.step(mx.array(byte), frozen=True)
        if len(generated) % log_every == 0:
            print(f'generated bytes: {len(generated)}/{count}', flush=True)

    with open(output, 'wb') as file:
        file.write(generated)
    print(f'Generated {len(generated)} bytes to {output}')
    return bytes(generated)


def sweep(
    grid_path,
    pattern,
    development,
    output,
    config='model.json',
    updates=1000,
    seed=11,
    ce_only=False,
    max_bytes=8192,
    log_every=100,
    sample_every=None,
    sample_prompt='The ',
    sample_seed=11,
    sample_bytes=256,
    eval_data=None,
    eval_every=None,
    eval_max_bytes=8192,
):
    """Run each model and seed pair in JSON product order."""
    validation.training_options(seed, sample_seed, updates, log_every)
    validation.positive(max_bytes, 'max-bytes')
    validation.monitoring(
        sample_every, sample_prompt, sample_bytes, eval_data, eval_every, eval_max_bytes
    )
    from tmt.benchmark import evaluate

    settings = load_config(config)
    with open(grid_path) as file:
        grid = json.load(file)
    validation.sweep_grid(grid, settings)
    keys = list(grid)
    _input_files(pattern)
    if eval_data is not None:
        list(documents(eval_data, eval_max_bytes))
    development_data = list(documents(development, max_bytes))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f'sweep output folder {output} is not empty; choose a new output folder'
        )
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / 'results.jsonl'
    rows = []

    sweep_manifest = output / 'sweep.json'
    record = {
        'config': settings,
        'grid': grid,
        'seed': seed,
        'updates': updates,
        'objective': 'ce_only' if ce_only else 'tmt',
        'data_glob': pattern,
        'development_glob': development,
        'development_max_bytes': max_bytes,
        'log_every': log_every,
        'sample_every': sample_every,
        'sample_prompt': sample_prompt,
        'sample_seed': sample_seed,
        'sample_bytes': sample_bytes,
        'eval_data': eval_data,
        'eval_every': eval_every,
        'eval_max_bytes': eval_max_bytes,
        'started_at': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'completed_runs': 0,
        'current_run': None,
        'error': None,
    }
    _write_manifest(sweep_manifest, record)
    try:
        for index, values in enumerate(
            itertools.product(*(grid[key] for key in keys)), 1
        ):
            record['current_run'] = index
            overrides = dict(zip(keys, values))
            run_seed = overrides.pop('seed', seed)
            run_settings = dict(settings)
            run_settings.update(overrides)
            validation.model_settings(run_settings, 'model config')
            run_path = output / f'run-{index:04d}'
            run_path.mkdir(parents=True, exist_ok=True)
            manifest_dir = run_path / 'manifest'
            manifest_dir.mkdir(parents=True, exist_ok=True)
            with open(manifest_dir / 'model.json', 'w') as file:
                json.dump(run_settings, file, indent=2)
                file.write('\n')
            checkpoint = run_path / 'model.safetensors'
            model = train(
                run_settings,
                checkpoint,
                pattern,
                updates,
                run_seed,
                ce_only,
                resume=False,
                log_every=log_every,
                sample_every=sample_every,
                sample_prompt=sample_prompt,
                sample_seed=sample_seed,
                sample_bytes=sample_bytes,
                eval_data=eval_data,
                eval_every=eval_every,
                eval_max_bytes=eval_max_bytes,
            )
            score = evaluate(model, development_data)
            row = {
                'run': index,
                'settings': run_settings,
                'trainable_parameters': model.count(),
                'temporal_gradient': model.temporal_gradient,
                'seed': run_seed,
                'updates': updates,
                'objective': 'ce_only' if ce_only else 'tmt',
                'data_glob': pattern,
                'development_glob': development,
                'development_max_bytes': max_bytes,
                'development_targets': score['targets'],
                'development_bpb': score['bpb'],
                'checkpoint': str(checkpoint.resolve()),
            }
            with open(results_path, 'a') as file:
                file.write(json.dumps(row) + '\n')
            rows.append(row)
            record['completed_runs'] = len(rows)
            _write_manifest(sweep_manifest, record)
        record['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as error:
        record['status'] = (
            'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        )
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        _write_manifest(sweep_manifest, record)

    best = min(rows, key=lambda row: row['development_bpb'])
    print(f'best row: {json.dumps(best)}')
    return best


def init_config(path=None):
    settings = load_config()
    templates = (
        {path: settings}
        if path is not None
        else {
            'model.json': settings,
            'grid.json': {
                'dim': [4, 8],
                'layers': [1],
                'spread': [4],
                'block': ['rtu'],
                'seed': [11, 22],
            },
            'run.example.json': {
                'config': settings,
                'seed': 11,
                'trainable_parameters': None,
                'temporal_gradient': None,
                'data_glob': 'data/train/*',
                'data_files': [],
                'updates': 1000,
                'log_every': 100,
                'objective': 'tmt',
                'completed_updates': 0,
                'resume': False,
                'initial_optimizer_step': 0,
                'started_at': None,
                'sample': None,
                'evaluation': None,
                'status': 'running',
                'last_checkpoint': None,
            },
        }
    )
    for destination, value in templates.items():
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, 'x') as file:
            json.dump(value, file, indent=2)
            file.write('\n')
        print(f'Wrote template to {destination}')
    if path is None:
        for directory in ('data/train', 'data/dev', 'runs'):
            Path(directory).mkdir(parents=True, exist_ok=True)


def version_text():
    from importlib.metadata import distribution

    installed = distribution('test-model-thing')
    direct_url = installed.read_text('direct_url.json')
    commit = None
    if direct_url:
        commit = json.loads(direct_url).get('vcs_info', {}).get('commit_id')
    return f'tmt {installed.version}, commit {commit or "unknown"}'


def command_manual(topic=None):
    from importlib.resources import files

    manual = files('tmt').joinpath('commands.md').read_text(encoding='utf-8')
    if topic is None:
        return manual
    if topic not in (
        'init',
        'train',
        'sweep',
        'evaluate',
        'benchmark',
        'generate',
        'help',
    ):
        raise ValueError(f'unknown help topic: {topic}')
    heading = f'## {topic}\n'
    section = manual.split(heading, 1)[1].split('\n## ', 1)[0]
    return heading + section.rstrip() + '\n'


def build_parser():
    training_options = argparse.ArgumentParser(add_help=False)
    training_options.add_argument(
        '--data',
        required=True,
        help='quoted raw-file train glob, one document per file',
    )
    training_options.add_argument(
        '--updates',
        type=int,
        default=1000,
        help='target-byte optimizer updates per run, default: 1000',
    )
    training_options.add_argument(
        '--log-every',
        type=int,
        default=100,
        help='positive loss report interval in updates, default: 100',
    )
    training_options.add_argument(
        '--seed', type=int, default=11, help='Python and MLX seed, default: 11'
    )
    training_options.add_argument(
        '--ce-only',
        action='store_true',
        help='next-byte cross entropy only; default: full TMT objective',
    )
    training_options.add_argument(
        '--sample-every',
        type=int,
        help='positive sample interval in updates; default: disabled',
    )
    training_options.add_argument(
        '--sample-prompt',
        default='The ',
        help='fixed nonempty sample prompt, default: "The "',
    )
    training_options.add_argument(
        '--sample-seed', type=int, default=11, help='fixed sample seed, default: 11'
    )
    training_options.add_argument(
        '--sample-bytes',
        type=int,
        default=256,
        help='positive sample byte count, default: 256',
    )
    training_options.add_argument(
        '--eval-data', help='quoted held-out byte glob; requires --eval-every'
    )
    training_options.add_argument(
        '--eval-every',
        type=int,
        help='positive held-out BPB interval in updates; requires --eval-data',
    )
    training_options.add_argument(
        '--eval-max-bytes',
        type=int,
        default=8192,
        help='positive held-out input byte budget, default: 8192',
    )

    parser = argparse.ArgumentParser(
        prog='tmt',
        description='TMT experiment commands',
        epilog='Use tmt help for the complete command manual.',
    )
    parser.add_argument('--version', action='version', version=version_text())
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='write default model settings')
    init.add_argument(
        '--output',
        help='write only model JSON to PATH; default: initialize current folder',
    )
    training = commands.add_parser(
        'train', parents=[training_options], help='train a model on byte files'
    )
    training.add_argument(
        'checkpoint', nargs='?', help='explicit checkpoint path; required with --resume'
    )
    training.add_argument(
        '--run',
        help='one relative folder name under runs/; '
        'default: UTC timestamp with microseconds; excludes checkpoint and --resume',
    )
    settings = training.add_mutually_exclusive_group()
    settings.add_argument(
        '--config',
        help='fresh model JSON, default: model.json in current folder; '
        'excludes --resume',
    )
    settings.add_argument(
        '--resume',
        action='store_true',
        help='load checkpoint settings and add --updates with a new data pass; '
        'no exact cursor/RNG resume',
    )
    evaluation = commands.add_parser('evaluate', help='score next-byte predictions')
    evaluation.add_argument('checkpoint', help='strict model and optimizer checkpoint')
    evaluation.add_argument(
        '--data', required=True, help='quoted raw-file glob, one document per file'
    )
    evaluation.add_argument(
        '--max-bytes',
        type=int,
        default=8192,
        help='positive global input-byte prefix across sorted files, default: 8192',
    )
    evaluation.add_argument(
        '--windows',
        type=int,
        nargs='+',
        default=[],
        help='context lengths in input bytes; default: no windows',
    )
    evaluation.add_argument(
        '--output', help='optional JSON score file; default: terminal only'
    )
    benchmark = commands.add_parser('benchmark', help='run byte and CoLA benchmarks')
    benchmark.add_argument(
        'checkpoint', help='strict checkpoint; fixed byte suite always runs'
    )
    benchmark.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='positive CoLA classifier epochs, default: 1',
    )
    benchmark.add_argument(
        '--split',
        type=float,
        default=0.5,
        help='contiguous CoLA train fraction, default: 0.5',
    )
    benchmark.add_argument(
        '--cola-data', help='optional four-column CoLA TSV; default: byte suite only'
    )
    benchmark.add_argument(
        '--seed',
        type=int,
        default=11,
        help='model load and CoLA head seed, default: 11',
    )
    generation = commands.add_parser(
        'generate', help='generate a fixed number of bytes'
    )
    generation.add_argument(
        'checkpoint', help='strict checkpoint for frozen generation'
    )
    generation.add_argument(
        '--prompt', required=True, help='nonempty text encoded as UTF-8'
    )
    generation.add_argument(
        '--output', required=True, help='raw-byte output file without the prompt'
    )
    generation.add_argument(
        '--seed', type=int, default=11, help='sample seed, default: 11'
    )
    generation.add_argument(
        '--bytes', type=int, default=512, help='number of output bytes, default: 512'
    )
    generation.add_argument(
        '--log-every',
        type=int,
        default=1000,
        help='positive progress interval in output bytes, default: 1000',
    )
    generation.add_argument(
        '--temperature',
        type=float,
        help='finite nonnegative sampler temperature; effective minimum: 0.1; '
        'default: checkpoint value',
    )
    sweeping = commands.add_parser(
        'sweep', parents=[training_options], help='run a sequential model grid'
    )
    sweeping.add_argument(
        '--grid',
        required=True,
        help='JSON object with nonempty arrays of model values or seed',
    )
    sweeping.add_argument(
        '--development', required=True, help='quoted raw-file development glob'
    )
    sweeping.add_argument(
        '--output', required=True, help='sweep folder for run-NNNN and results.jsonl'
    )
    sweeping.add_argument(
        '--config',
        default='model.json',
        help='base model JSON, default: model.json in current folder',
    )
    sweeping.add_argument(
        '--max-bytes',
        type=int,
        default=8192,
        help='positive global development input-byte prefix, default: 8192',
    )
    commands.add_parser('help', help='read the bundled command manual or one topic')
    return parser


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    help_flags = ('-h', '--help')
    if argv and (
        argv[0] == 'help'
        or (len(argv) > 1 and any(flag in argv for flag in help_flags))
    ):
        if argv[0] == 'help':
            if any(flag in argv for flag in help_flags):
                topic = 'help'
            else:
                if len(argv) > 2:
                    parser.error('help accepts at most one topic')
                topic = argv[1] if len(argv) == 2 else None
        else:
            topic = argv[1] if argv[0] in help_flags else argv[0]
        try:
            print(command_manual(topic), end='')
        except ValueError as error:
            parser.error(str(error))
        return
    args = parser.parse_args(argv)
    if hasattr(args, 'seed'):
        validation.seed(args.seed)
    if args.command in ('train', 'sweep'):
        validation.seed(args.sample_seed, 'sample-seed')
        validation.positive(args.updates, 'updates')
        monitoring = {
            name: getattr(args, name)
            for name in (
                'sample_every',
                'sample_prompt',
                'sample_seed',
                'sample_bytes',
                'eval_data',
                'eval_every',
                'eval_max_bytes',
            )
        }
        validation.monitoring(
            args.sample_every,
            args.sample_prompt,
            args.sample_bytes,
            args.eval_data,
            args.eval_every,
            args.eval_max_bytes,
        )
    if args.command in ('train', 'sweep', 'generate'):
        validation.positive(args.log_every, 'log-every')
    if args.command == 'init':
        init_config(args.output)
    elif args.command == 'train':
        if args.checkpoint is not None and args.run is not None:
            parser.error('checkpoint and --run cannot be used together')
        if args.resume and (args.checkpoint is None or args.run is not None):
            parser.error('--resume requires an explicit checkpoint and forbids --run')
        if args.run is not None:
            validation.run_name(args.run)
        if not glob.glob(args.data, recursive=True):
            raise FileNotFoundError(f'no files matched {args.data!r}')
        settings = None if args.resume else load_config(args.config or 'model.json')
        new_run = args.checkpoint is None
        if new_run:
            name = args.run or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            folder = Path('runs') / name
            if folder.exists():
                raise FileExistsError(
                    f'run folder {folder} already exists; choose a new run name'
                )
            args.checkpoint = folder / 'model.safetensors'
        train(
            settings,
            args.checkpoint,
            args.data,
            args.updates,
            args.seed,
            args.ce_only,
            args.resume,
            args.log_every,
            new_run=new_run,
            **monitoring,
        )
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'benchmark':
        from tmt.benchmark import run

        run(args.checkpoint, args.epochs, args.split, args.cola_data, seed=args.seed)
    elif args.command == 'generate':
        generate(
            args.checkpoint,
            args.prompt,
            args.output,
            args.seed,
            args.bytes,
            args.temperature,
            args.log_every,
        )
    elif args.command == 'sweep':
        sweep(
            args.grid,
            args.data,
            args.development,
            args.output,
            args.config,
            args.updates,
            args.seed,
            args.ce_only,
            args.max_bytes,
            args.log_every,
            **monitoring,
        )


def main(argv=None):
    try:
        return _main(argv)
    except (OSError, ValueError) as error:
        print(f'tmt: error: {error}', file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
