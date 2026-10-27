"""Training, sampling, generation, and sweep runtime operations."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import random
import signal
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import validation
from .checkpoint import load_checkpoint, new_model, preserved_rng
from .checkpoint import save_checkpoint as write_checkpoint
from .config import group_model_config, load_config
from .data import ByteFiles, DataOptions, documents
from .records import (
    RunManifest,
    RunStatus,
    SweepManifest,
    append_jsonl,
    build_run_manifest,
    build_sweep_manifest,
    read_json,
    write_json,
)

if TYPE_CHECKING:
    import mlx.core as mx

    from .model import Model


def _training_sample(model: Model, prompt: str, count: int, seed: int) -> bytes:
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


@dataclass(frozen=True)
class MonitorOptions:
    """Sampling, evaluation timing, and reporting settings for a run."""

    sample_every: int | None = None
    sample_prompt: str = 'The '
    sample_seed: int = 11
    sample_bytes: int = 256
    eval_every: int | None = None
    log_every: int = 100
    save_every: int = 500


@dataclass(frozen=True)
class TrainingOptions:
    """Update budget and objective settings for one run."""

    updates: int
    seed: int
    ce_only: bool = False
    resume: bool = False


@dataclass
class TrainingRun:
    """Mutable state and actions for one training run."""

    settings: dict[str, Any] | None
    path: str | PathLike[str]
    data: DataOptions
    options: TrainingOptions
    monitor: MonitorOptions
    new_run: bool = False
    files: list[str] = field(default_factory=list)
    evaluation_data: list[bytes] | None = None
    model: Model = field(init=False)
    checkpoint: Path = field(init=False)
    manifest_dir: Path = field(init=False)
    manifest: Path = field(init=False)
    metrics: Path = field(init=False)
    record: RunManifest | None = None
    completed: int = 0
    last_loss: mx.array | None = None
    logged: int = 0
    evaluated: int = 0
    loss_sum: float = 0.0
    component_sums: dict[str, float] = field(default_factory=dict)
    started: float = 0.0
    last_report: float = 0.0
    sample_dir: Path = field(init=False)
    stamp: str = ''
    status: RunStatus = 'running'
    interrupted: bool = False

    def setup(self) -> None:
        """Load inputs and model, then create the run manifest paths."""
        self.files = ByteFiles(self.data.pattern).paths
        evaluation_files = []
        if self.data.eval_data is not None:
            selection = ByteFiles(self.data.eval_data)
            evaluation_files = selection.paths
            self.evaluation_data = list(
                selection.documents(self.data.eval_max_bytes)
            )
            validation.evaluation_targets(self.evaluation_data)
        if self.options.resume:
            self.model, self.settings = load_checkpoint(self.path)
        else:
            self.model = new_model(self.settings, self.options.seed)
        random.Random(self.options.seed).shuffle(self.files)

        self.checkpoint = Path(self.path)
        self.checkpoint.parent.mkdir(parents=True, exist_ok=not self.new_run)
        self.manifest_dir = self.checkpoint.parent / 'manifest'
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.manifest_dir / 'run.json'
        self.metrics = self.manifest_dir / 'loss.jsonl'
        self.record = build_run_manifest(
            config=self.settings,
            trainable_parameters=self.model.count(),
            temporal_gradient=self.model.temporal_gradient,
            seed=self.options.seed,
            data_glob=str(self.data.pattern),
            data_files=[str(Path(filename).resolve()) for filename in self.files],
            updates=self.options.updates,
            log_every=self.monitor.log_every,
            objective='ce_only' if self.options.ce_only else 'tmt',
            resume=self.options.resume,
            initial_optimizer_step=self.optimizer_step,
            sample=None
            if self.monitor.sample_every is None
            else {
                'every': self.monitor.sample_every,
                'prompt': self.monitor.sample_prompt,
                'seed': self.monitor.sample_seed,
                'bytes': self.monitor.sample_bytes,
            },
            evaluation=None
            if self.monitor.eval_every is None
            else {
                'every': self.monitor.eval_every,
                'data_glob': str(self.data.eval_data),
                'max_bytes': self.data.eval_max_bytes,
                'data_files': [str(Path(name).resolve()) for name in evaluation_files],
            },
            started_at=datetime.now(timezone.utc).isoformat(),
            last_checkpoint=(
                str(self.checkpoint.resolve()) if self.options.resume else None
            ),
            optimizer_step=self.optimizer_step,
            save_every=self.monitor.save_every,
        )
        self.started = self.last_report = time.perf_counter()
        self.stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.sample_dir = self.checkpoint.parent / 'samples' / self.stamp

    @property
    def optimizer_step(self) -> int:
        """Total optimizer updates in the model, including any before a resume."""
        return int(self.model.optimizer.state['step'].item())

    def report_loss(self) -> None:
        """Write and print the current objective loss row."""
        if self.record is None or self.last_loss is None:
            raise RuntimeError('training run is not ready to report loss')
        now = time.perf_counter()
        interval_updates = self.completed - self.logged
        loss = float(self.last_loss.item())
        row = {
            'completed_updates': self.completed,
            'optimizer_step': self.optimizer_step,
            'objective': 'ce_only' if self.options.ce_only else 'tmt',
            'loss': loss,
            'mean_loss': self.loss_sum / interval_updates,
            'mean_loss_components': {
                name: value / interval_updates
                for name, value in self.component_sums.items()
            },
            'interval_updates': interval_updates,
            'elapsed_seconds': now - self.started,
            'updates_per_second': interval_updates / max(now - self.last_report, 1e-9),
        }
        append_jsonl(self.metrics, row)
        print(
            f'updates: {self.completed}, '
            f'objective: {"ce_only" if self.options.ce_only else "tmt"}, '
            f'loss: {loss:.6f}, '
            f'mean loss: {row["mean_loss"]:.6f}, step: {row["optimizer_step"]}, '
            f'updates/s: {row["updates_per_second"]:.2f}',
            flush=True,
        )
        self.logged = self.completed
        self.last_report = now
        self.loss_sum = 0.0
        self.component_sums.clear()

    def report_sample(self) -> None:
        """Write a seeded training sample and its manifest row."""
        sample = _training_sample(
            self.model,
            self.monitor.sample_prompt,
            self.monitor.sample_bytes,
            self.monitor.sample_seed,
        )
        step = self.optimizer_step
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        destination = self.sample_dir / f'step-{step:012d}.bin'
        destination.write_bytes(sample)
        append_jsonl(
            self.manifest_dir / 'samples.jsonl',
            {
                'completed_updates': self.completed,
                'optimizer_step': step,
                'prompt': self.monitor.sample_prompt,
                'seed': self.monitor.sample_seed,
                'bytes': len(sample),
                'path': str(destination.resolve()),
            },
        )
        print(
            f'sample at update {self.completed}, step {step}: {sample!r}', flush=True
        )

    def report_evaluation(self) -> None:
        """Score held-out documents and write an evaluation row."""
        if self.evaluation_data is None:
            raise RuntimeError('evaluation data is not configured')
        from .benchmark import evaluate

        result = evaluate(self.model, self.evaluation_data)
        append_jsonl(
            self.manifest_dir / 'evaluation.jsonl',
            {
                'completed_updates': self.completed,
                'optimizer_step': self.optimizer_step,
                **result,
            },
        )
        print(
            f'evaluation at update {self.completed}: BPB {result["bpb"]:.6f}, '
            f'targets: {result["targets"]}',
            flush=True,
        )
        self.evaluated = self.completed

    def save_checkpoint(self) -> None:
        """Save model and optimizer state, then rewrite the manifest."""
        if self.record is None:
            raise RuntimeError('training run is not ready to save')
        write_checkpoint(self.checkpoint, self.model, self.settings)
        self.record['run']['last_checkpoint'] = str(self.checkpoint.resolve())
        self.record['progress']['completed_updates'] = self.completed
        self.record['progress']['optimizer_step'] = self.optimizer_step
        write_json(self.manifest, self.record)
        print(
            f'saved checkpoint: {self.checkpoint}, updates: {self.completed}, '
            f'step: {self.optimizer_step}'
        )

    def consume_terminal_byte(self, byte: int) -> None:
        """Advance frozen recurrent state over a document's final byte."""
        import mlx.core as mx

        self.model.step(mx.array(byte), frozen=True)
        self.model.eval_state()

    def check_interrupt(self) -> None:
        """Save at a complete update boundary after a SIGINT request."""
        if self.interrupted:
            self.save_checkpoint()
            raise KeyboardInterrupt

    def after_update(self, end: bool, terminal_byte: int) -> None:
        """Run every action triggered by an optimizer update in stable order."""
        if self.completed % self.monitor.log_every == 0:
            self.report_loss()
        if end:
            self.consume_terminal_byte(terminal_byte)
        if (
            self.monitor.sample_every is not None
            and self.completed % self.monitor.sample_every == 0
        ):
            self.report_sample()
        if (
            self.monitor.eval_every is not None
            and self.completed % self.monitor.eval_every == 0
        ):
            self.report_evaluation()
        if self.completed % self.monitor.save_every == 0:
            self.save_checkpoint()

    def one_data_pass(self) -> None:
        """Visit the selected files once, in the run's fixed shuffled order."""
        before_pass = self.completed
        for filename in self.files:
            self.check_interrupt()
            data = Path(filename).read_bytes()
            if not data:
                continue
            self.model.reset()
            if len(data) == 1:
                self.consume_terminal_byte(data[0])
                continue
            for index in range(len(data) - 1):
                end = index == len(data) - 2
                components: dict[str, float] = {}
                _, _, self.last_loss = self.model.train_step(
                    data[index],
                    data[index + 1],
                    end,
                    self.options.ce_only,
                    metrics=components,
                )
                self.completed += 1
                self.loss_sum += float(self.last_loss.item())
                for name, value in components.items():
                    total = self.component_sums.get(name, 0.0)
                    self.component_sums[name] = total + value
                self.after_update(end, data[-1])
                self.check_interrupt()
                if self.completed >= self.options.updates:
                    break
            if self.completed >= self.options.updates:
                break
        self.check_interrupt()
        if self.completed == before_pass:
            raise ValueError('no training targets found in data pass')

    def finalize(self) -> None:
        """Write final run status and any pending partial loss row."""
        if self.record is None:
            raise RuntimeError('training run is not ready to finalize')
        self.record['run']['status'] = self.status
        self.record['progress']['completed_updates'] = self.completed
        self.record['progress']['optimizer_step'] = self.optimizer_step
        write_json(self.manifest, self.record)
        if self.last_loss is not None and self.completed != self.logged:
            self.report_loss()

    def run(self) -> Model:
        """Run data passes and preserve the run's failure and interrupt status."""
        if self.record is None:
            raise RuntimeError('training run must be set up before it can run')

        def request_interrupt(signum, frame):
            self.interrupted = True

        previous_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handler = signal.getsignal(signum)
                if handler is not None:
                    previous_handlers[signum] = handler
                    signal.signal(signum, request_interrupt)
        try:
            if self.options.resume and self.manifest.exists():
                self.manifest.rename(
                    self.manifest_dir / f'run.replaced-{self.stamp}.json'
                )
            for name in ('loss.jsonl', 'samples.jsonl', 'evaluation.jsonl'):
                # A resume appends to the rows of earlier invocations.
                with open(
                    self.manifest_dir / name,
                    'a' if self.options.resume else 'w',
                ):
                    pass
            write_json(self.manifest, self.record)
            print(
                f'run folder: {self.checkpoint.parent}, '
                f'seed: {self.options.seed}, '
                f'config: {json.dumps(self.settings)}'
            )
            while self.completed < self.options.updates:
                self.one_data_pass()
            if (
                self.monitor.eval_every is not None
                and self.completed != self.evaluated
            ):
                self.report_evaluation()
            self.check_interrupt()
            self.save_checkpoint()
            if self.interrupted:
                raise KeyboardInterrupt
            self.status = 'complete'
        except KeyboardInterrupt:
            self.status = 'interrupted'
            raise
        except Exception as error:
            self.status = 'failed'
            self.record['run']['error'] = {
                'type': type(error).__name__,
                'message': str(error),
            }
            raise
        finally:
            try:
                self.finalize()
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)

        print(
            f'updates: {self.completed}, step: {self.optimizer_step}, '
            f'checkpoint: {self.checkpoint}'
        )
        return self.model


def train(
    settings: dict[str, Any] | None,
    path: str | PathLike[str],
    pattern: str | PathLike[str],
    updates: int,
    seed: int,
    *,
    ce_only: bool = False,
    resume: bool = False,
    log_every: int = 100,
    sample_every: int | None = None,
    sample_prompt: str = 'The ',
    sample_seed: int = 11,
    sample_bytes: int = 256,
    eval_data: str | None = None,
    eval_every: int | None = None,
    eval_max_bytes: int = 8192,
    new_run: bool = False,
    save_every: int = 500,
) -> Model:
    """Train on adjacent raw-byte pairs and record the run manifest."""
    monitor = MonitorOptions(
        sample_every=sample_every,
        sample_prompt=sample_prompt,
        sample_seed=sample_seed,
        sample_bytes=sample_bytes,
        eval_every=eval_every,
        log_every=log_every,
        save_every=save_every,
    )
    data = DataOptions(
        pattern=pattern,
        eval_data=eval_data,
        eval_max_bytes=eval_max_bytes,
    )
    options = TrainingOptions(
        updates=updates,
        seed=seed,
        ce_only=ce_only,
        resume=resume,
    )
    validation.training_options(seed, updates, log_every)
    validation.positive(save_every, 'save-every')
    validation.monitoring(
        monitor.sample_every,
        monitor.sample_prompt,
        monitor.sample_seed,
        monitor.sample_bytes,
        data.eval_data,
        monitor.eval_every,
        data.eval_max_bytes,
    )
    run = TrainingRun(
        settings=settings,
        path=path,
        data=data,
        options=options,
        monitor=monitor,
        new_run=new_run,
    )
    run.setup()
    return run.run()


def evaluate_checkpoint(
    checkpoint: str | PathLike[str],
    pattern: str | PathLike[str],
    max_bytes: int,
    windows: Sequence[int],
) -> dict[str, Any]:
    """Load a checkpoint and evaluate it on selected byte documents."""
    validation.positive(max_bytes, 'max-bytes')
    for window in windows:
        validation.positive(window, 'evaluation windows')
    from .benchmark import evaluate

    data = list(documents(pattern, max_bytes))
    validation.evaluation_targets(data)
    model, _ = load_checkpoint(checkpoint)
    return {
        **evaluate(model, data, windows),
        'checkpoint': str(checkpoint),
        'optimizer_step': int(model.optimizer.state['step'].item()),
    }


def inspect_checkpoint(checkpoint: str | PathLike[str]) -> dict[str, Any]:
    """Load a checkpoint and summarize its state and any run manifest beside it."""
    model, settings = load_checkpoint(checkpoint)
    summary: dict[str, Any] = {
        'optimizer_step': int(model.optimizer.state['step'].item()),
        'trainable_parameters': model.count(),
        'config': settings,
        'run': None,
    }
    manifest = Path(checkpoint).parent / 'manifest' / 'run.json'
    if manifest.is_file():
        record = read_json(manifest)
        validation.require(
            isinstance(record, dict)
            and all(
                isinstance(record.get(name), dict)
                for name in ('run', 'training', 'progress')
            ),
            'run manifest must contain grouped run, training, and progress objects',
        )
        summary['run'] = {
            'status': record['run'].get('status'),
            'objective': record['training'].get('objective'),
            'seed': record['training'].get('seed'),
            'completed_updates': record['progress'].get('completed_updates'),
        }
    return summary


def generate(
    checkpoint: str | PathLike[str],
    prompt: str,
    output: str | PathLike[str],
    seed: int,
    *,
    count: int = 512,
    temperature: float | None = None,
    log_every: int = 1000,
) -> bytes:
    """Replay a UTF-8 prompt and write a fixed count of sampled bytes."""
    validation.seed(seed)
    validation.require(prompt, 'generation needs a nonempty prompt')
    validation.require(count >= 0, 'bytes must be nonnegative')
    validation.positive(log_every, 'log-every')
    validation.require(
        temperature is None or (math.isfinite(temperature) and temperature >= 0),
        'temperature must be a finite nonnegative number',
    )
    validation.output_folder(output)

    import mlx.core as mx

    model, _ = load_checkpoint(checkpoint)
    model.reset()
    if temperature is not None:
        model.temp = temperature

    generated = bytearray()
    with preserved_rng():
        mx.random.seed(seed)
        for byte in prompt.encode('utf-8'):
            _, (logits, _) = model.step(mx.array(byte), frozen=True)
        for _ in range(count):
            byte = int(model.sample(logits).item())
            generated.append(byte)
            _, (logits, _) = model.step(mx.array(byte), frozen=True)
            if len(generated) % log_every == 0:
                print(f'generated bytes: {len(generated)}/{count}', flush=True)

    with open(output, 'wb') as file:
        file.write(generated)
    print(f'wrote output: {output}, bytes: {len(generated)}')
    return bytes(generated)


@dataclass(frozen=True)
class SweepCandidate:
    """Resolved model settings and seed for one grid row."""

    index: int
    settings: dict[str, Any]
    seed: int


@dataclass(frozen=True)
class SweepExecution:
    """Model, checkpoint, and development score for one completed run."""

    model: Model
    checkpoint: Path
    score: dict[str, Any]


def _resolve_sweep_candidate(
    index: int,
    values: tuple[Any, ...],
    keys: list[str],
    settings: dict[str, Any],
    seed: int,
) -> SweepCandidate:
    """Resolve one grid point while preserving key and product order."""
    overrides = dict(zip(keys, values))
    run_seed = overrides.pop('seed', seed)
    run_settings = dict(settings)
    run_settings.update(overrides)
    validation.model_settings(run_settings, 'model config')
    return SweepCandidate(index=index, settings=run_settings, seed=run_seed)


def _run_sweep_candidate(
    candidate: SweepCandidate,
    output: Path,
    data: DataOptions,
    options: TrainingOptions,
    monitor: MonitorOptions,
    development_data: list[bytes],
    evaluate: Callable[[Model, list[bytes]], dict[str, Any]],
) -> SweepExecution:
    """Train and score one already-resolved sweep candidate."""
    run_path = output / f'run-{candidate.index:04d}'
    run_path.mkdir(parents=True, exist_ok=True)
    manifest_dir = run_path / 'manifest'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_dir / 'model.json', group_model_config(candidate.settings))
    checkpoint = run_path / 'model.safetensors'
    candidate_options = dataclasses.replace(options, seed=candidate.seed)
    model = train(
        candidate.settings,
        checkpoint,
        data.pattern,
        candidate_options.updates,
        candidate_options.seed,
        ce_only=candidate_options.ce_only,
        resume=candidate_options.resume,
        log_every=monitor.log_every,
        sample_every=monitor.sample_every,
        sample_prompt=monitor.sample_prompt,
        sample_seed=monitor.sample_seed,
        sample_bytes=monitor.sample_bytes,
        eval_data=data.eval_data,
        eval_every=monitor.eval_every,
        eval_max_bytes=data.eval_max_bytes,
        save_every=monitor.save_every,
    )
    score = evaluate(model, development_data)
    return SweepExecution(model=model, checkpoint=checkpoint, score=score)


def _sweep_result(
    candidate: SweepCandidate,
    run: SweepExecution,
    data: DataOptions,
    development: str | PathLike[str],
    max_bytes: int,
    options: TrainingOptions,
) -> dict[str, Any]:
    """Build the stable results row for one completed sweep candidate."""
    model = run.model
    score = run.score
    checkpoint = run.checkpoint
    grouped_model = group_model_config(candidate.settings)
    model_group = dict(grouped_model['model'])
    extensions = model_group.pop('extensions', None)
    model_group['trainable_parameters'] = model.count()
    model_group['temporal_gradient'] = model.temporal_gradient
    if extensions is not None:
        model_group['extensions'] = extensions
    return {
        'model': model_group,
        'hparams': grouped_model['hparams'],
        'data': {
            'train': {'glob': str(data.pattern)},
            'development': {'glob': str(development), 'max_bytes': max_bytes},
        },
        'training': {
            'seed': candidate.seed,
            'updates': options.updates,
            'objective': 'ce_only' if options.ce_only else 'tmt',
        },
        'evaluation': {'targets': score['targets'], 'bpb': score['bpb']},
        'run': {
            'index': candidate.index,
            'checkpoint': str(checkpoint.resolve()),
        },
    }


def sweep(
    grid_path: str | PathLike[str],
    pattern: str | PathLike[str],
    development: str | PathLike[str],
    output: str | PathLike[str],
    config: str | PathLike[str],
    updates: int,
    seed: int,
    *,
    ce_only: bool = False,
    max_bytes: int = 8192,
    log_every: int = 100,
    sample_every: int | None = None,
    sample_prompt: str = 'The ',
    sample_seed: int = 11,
    sample_bytes: int = 256,
    eval_data: str | None = None,
    eval_every: int | None = None,
    eval_max_bytes: int = 8192,
    save_every: int = 500,
) -> dict[str, Any]:
    """Run each model and seed pair in JSON product order."""
    monitor = MonitorOptions(
        sample_every=sample_every,
        sample_prompt=sample_prompt,
        sample_seed=sample_seed,
        sample_bytes=sample_bytes,
        eval_every=eval_every,
        log_every=log_every,
        save_every=save_every,
    )
    data = DataOptions(
        pattern=pattern,
        eval_data=eval_data,
        eval_max_bytes=eval_max_bytes,
    )
    options = TrainingOptions(
        updates=updates,
        seed=seed,
        ce_only=ce_only,
    )
    validation.training_options(seed, updates, monitor.log_every)
    validation.positive(max_bytes, 'max-bytes')
    validation.positive(monitor.save_every, 'save-every')
    validation.monitoring(
        monitor.sample_every,
        monitor.sample_prompt,
        monitor.sample_seed,
        monitor.sample_bytes,
        data.eval_data,
        monitor.eval_every,
        data.eval_max_bytes,
    )
    from .benchmark import evaluate

    settings = load_config(config)
    grid = read_json(grid_path)
    validation.sweep_grid(grid, settings)
    keys = list(grid)
    train_files = ByteFiles(data.pattern).paths
    evaluation_files = []
    if data.eval_data is not None:
        evaluation_selection = ByteFiles(data.eval_data)
        evaluation_files = evaluation_selection.paths
        validation.evaluation_targets(
            evaluation_selection.documents(data.eval_max_bytes)
        )
    development_selection = ByteFiles(development)
    development_data = list(development_selection.documents(max_bytes))
    validation.evaluation_targets(development_data)
    output_path = Path(output)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f'sweep output folder {output_path} is not empty; '
            'choose a new output folder'
        )
    output_path.mkdir(parents=True, exist_ok=True)
    results_path = output_path / 'results.jsonl'
    rows: list[dict[str, Any]] = []

    sweep_manifest = output_path / 'sweep.json'
    record: SweepManifest = build_sweep_manifest(
        config=settings,
        grid=grid,
        seed=seed,
        updates=updates,
        objective='ce_only' if ce_only else 'tmt',
        data_glob=str(pattern),
        development_glob=str(development),
        development_max_bytes=max_bytes,
        log_every=log_every,
        save_every=save_every,
        started_at=datetime.now(timezone.utc).isoformat(),
        sample_every=monitor.sample_every,
        sample_prompt=monitor.sample_prompt,
        sample_seed=monitor.sample_seed,
        sample_bytes=monitor.sample_bytes,
        eval_data=None if data.eval_data is None else str(data.eval_data),
        eval_every=monitor.eval_every,
        eval_max_bytes=data.eval_max_bytes,
        data_files=[str(Path(filename).resolve()) for filename in train_files],
        evaluation_files=[
            str(Path(filename).resolve()) for filename in evaluation_files
        ],
        development_files=[
            str(Path(filename).resolve())
            for filename in development_selection.paths
        ],
    )
    write_json(sweep_manifest, record)
    try:
        for index, values in enumerate(
            itertools.product(*(grid[key] for key in keys)), 1
        ):
            record['progress']['current_run'] = index
            candidate = _resolve_sweep_candidate(index, values, keys, settings, seed)
            run = _run_sweep_candidate(
                candidate,
                output_path,
                data,
                options,
                monitor,
                development_data,
                evaluate,
            )
            row = _sweep_result(
                candidate,
                run,
                data,
                development,
                max_bytes,
                options,
            )
            append_jsonl(results_path, row)
            rows.append(row)
            record['progress']['completed_runs'] = len(rows)
            write_json(sweep_manifest, record)
        record['run']['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as error:
        record['run']['status'] = (
            'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        )
        record['run']['error'] = {
            'type': type(error).__name__,
            'message': str(error),
        }
        raise
    finally:
        write_json(sweep_manifest, record)

    best = min(rows, key=lambda row: row['evaluation']['bpb'])
    print(f'best row: {json.dumps(best)}')
    return best
